"""Unit tests for the scan concurrency gate and restart recovery.

Every scan here runs against a stub merger, so no models are loaded; the DB is
the throwaway SQLite file conftest points the settings at.
"""
import asyncio
import json
import uuid

import pytest
from sqlalchemy import delete

from backend.app import main
from backend.app.api.routes import analyze
from backend.app.config import settings
from backend.app.db.models import Finding, Scan
from backend.app.db.session import SessionLocal, init_db
from backend.app.services.scan_runner import get_scan_semaphore, reset_scan_semaphore, run_scan

_REQUEST = {"code_snippet": "def f():\n    return 1\n", "language": "python"}


@pytest.fixture(scope="module", autouse=True)
def _tables():
    init_db()


@pytest.fixture
def make_scan():
    """Create Scan rows and drop them again afterwards — a stray "queued" row
    would be re-queued by the recovery that runs in a later test's app lifespan."""
    created: list[str] = []

    def _make(status: str = "queued") -> str:
        job_id = uuid.uuid4().hex
        with SessionLocal() as session:
            session.add(
                Scan(
                    id=job_id,
                    status=status,
                    mode="snippet",
                    request_json=json.dumps(_REQUEST),
                )
            )
            session.commit()
        created.append(job_id)
        return job_id

    yield _make

    with SessionLocal() as session:
        session.execute(delete(Finding).where(Finding.scan_id.in_(created)))
        session.execute(delete(Scan).where(Scan.id.in_(created)))
        session.commit()


@pytest.fixture
def single_slot_gate(monkeypatch):
    """One scan at a time, on a semaphore built fresh for this test's loop."""
    monkeypatch.setattr(settings, "MAX_CONCURRENT_SCANS", 1)
    reset_scan_semaphore()
    yield
    reset_scan_semaphore()


class NoFindingsRouter:
    """The LLM reviews every scan now (not only ones with retrieval matches), so
    scans need a router; this one never finds anything and never hits a network."""

    mock = False

    async def generate(self, system, user, **kwargs):
        return {"findings": []}, "stub"


_ROUTER = NoFindingsRouter()


def _scan(job_id: str) -> Scan | None:
    with SessionLocal() as session:
        scan = session.get(Scan, job_id)
        if scan is not None:
            session.expunge(scan)
        return scan


def _status(job_id: str) -> str | None:
    scan = _scan(job_id)
    return scan.status if scan is not None else None


class GatedMerger:
    """Blocks inside the guarded section until released, and records how many
    scans were inside it at once."""

    def __init__(self):
        self.entered = asyncio.Event()
        self.proceed = asyncio.Event()
        self.active = 0
        self.max_active = 0
        self.calls = 0

    async def analyze_code(self, code, language=None):
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.entered.set()
        try:
            await self.proceed.wait()
        finally:
            self.active -= 1
        return {"ghost_hunter_findings": [], "team_memory_findings": [], "is_vulnerable": False}


def test_second_scan_stays_queued_until_the_gate_frees_up(make_scan, single_slot_gate):
    first, second = make_scan(), make_scan()

    async def scenario():
        merger = GatedMerger()
        second_started = asyncio.Event()

        async def run_second():
            # Set before awaiting: run_scan then runs synchronously up to the
            # semaphore, so once this resumes the waiter, it really is waiting.
            second_started.set()
            await run_scan(second, merger, _ROUTER)

        task_a = asyncio.create_task(run_scan(first, merger, _ROUTER))
        task_b = asyncio.create_task(run_second())
        await asyncio.wait_for(merger.entered.wait(), timeout=10)
        await asyncio.wait_for(second_started.wait(), timeout=10)

        observed = (_status(first), _status(second), get_scan_semaphore().locked())

        merger.proceed.set()
        await asyncio.wait_for(asyncio.gather(task_a, task_b), timeout=10)
        return observed, merger.max_active, merger.calls

    (status_a, status_b, gate_held), max_active, calls = asyncio.run(scenario())

    # While the first scan holds the only permit the second has not started.
    assert status_a == "running"
    assert status_b == "queued"
    assert gate_held is True
    # ...and the guarded section was never entered twice.
    assert max_active == 1
    assert calls == 2
    assert _status(first) == "completed"
    assert _status(second) == "completed"


class BoomMerger:
    async def analyze_code(self, code, language=None):
        raise RuntimeError("model exploded")


def test_failed_scan_is_sanitized_and_frees_the_gate(make_scan, single_slot_gate):
    failing, following = make_scan(), make_scan()

    async def scenario():
        await run_scan(failing, BoomMerger(), _ROUTER)
        merger = GatedMerger()
        merger.proceed.set()
        # Would hang if the failed scan had leaked its permit.
        await asyncio.wait_for(run_scan(following, merger, _ROUTER), timeout=10)

    asyncio.run(scenario())

    scan = _scan(failing)
    assert scan.status == "failed"
    assert scan.error == "Analysis failed"
    assert _status(following) == "completed"


def test_cancelled_scan_is_marked_failed_and_frees_the_gate(make_scan, single_slot_gate):
    cancelled, following = make_scan(), make_scan()

    async def scenario():
        merger = GatedMerger()
        task = asyncio.create_task(run_scan(cancelled, merger, _ROUTER))
        await asyncio.wait_for(merger.entered.wait(), timeout=10)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        merger.proceed.set()
        await asyncio.wait_for(run_scan(following, merger, _ROUTER), timeout=10)

    asyncio.run(scenario())

    scan = _scan(cancelled)
    assert scan.status == "failed"  # not stuck "running"
    assert scan.error == "interrupted"
    assert _status(following) == "completed"


def test_recovery_fails_running_scans_and_requeues_queued_ones(make_scan, monkeypatch):
    running, queued, done = make_scan("running"), make_scan("queued"), make_scan("completed")
    invoked: list[str] = []

    async def fake_run_scan_job(job_id: str) -> None:
        invoked.append(job_id)
        with SessionLocal() as session:
            session.get(Scan, job_id).status = "completed"
            session.commit()

    monkeypatch.setattr(analyze, "run_scan_job", fake_run_scan_job)

    async def scenario():
        first = await main.recover_orphaned_scans()
        # The row is still "queued" (nothing has run yet), so a second recovery
        # would re-schedule it if the guard didn't hold.
        second = await main.recover_orphaned_scans()
        await asyncio.gather(*first, *second)
        return len(first), len(second)

    scheduled, rescheduled = asyncio.run(scenario())

    assert (scheduled, rescheduled) == (1, 0)
    assert invoked == [queued]
    assert _status(queued) == "completed"

    killed = _scan(running)
    assert killed.status == "failed"
    assert killed.error == "interrupted by server restart"
    assert _status(done) == "completed"  # untouched


def test_recovery_of_a_scan_that_vanished_does_not_raise(make_scan, monkeypatch):
    job_id = make_scan("queued")
    # Stubbed so the recovered task resolves the singletons without loading models.
    monkeypatch.setattr(analyze, "get_merger", lambda: GatedMerger())
    monkeypatch.setattr(analyze, "get_llm_router", lambda: None)

    async def scenario():
        tasks = await main.recover_orphaned_scans()
        # recover_orphaned_scans never awaits, so the task has not started yet:
        # the row disappears between being picked up and being run.
        with SessionLocal() as session:
            session.execute(delete(Scan).where(Scan.id == job_id))
            session.commit()
        await asyncio.gather(*tasks, return_exceptions=True)
        return tasks

    tasks = asyncio.run(scenario())

    assert len(tasks) == 1
    # run_scan swallows the missing row; the fire-and-forget task ends cleanly.
    assert tasks[0].exception() is None
