"""Feedback endpoint: developers up/down-vote findings; votes tune future retrieval."""
from fastapi import APIRouter, Depends, HTTPException

from backend.app.api.deps import require_api_key
from backend.app.core.feedback_store import get_net_votes
from backend.app.db.models import Feedback, Finding
from backend.app.db.session import SessionLocal
from backend.app.models.schemas import FeedbackRequest, FeedbackResponse

router = APIRouter()


@router.post("/", response_model=FeedbackResponse, dependencies=[Depends(require_api_key)])
async def submit_feedback(payload: FeedbackRequest):
    """Record (or update) a vote on a finding. One vote per finding; re-voting overwrites."""
    session = SessionLocal()
    try:
        finding = session.get(Finding, payload.finding_id)
        if finding is None:
            raise HTTPException(status_code=404, detail="Finding not found")

        existing = (
            session.query(Feedback).filter(Feedback.finding_id == payload.finding_id).one_or_none()
        )
        if existing is None:
            session.add(
                Feedback(
                    finding_id=finding.id,
                    scan_id=finding.scan_id,
                    point_id=finding.point_id,
                    collection=finding.collection,
                    vote=payload.vote,
                )
            )
        else:
            existing.vote = payload.vote
        session.commit()

        net = get_net_votes([finding.point_id]).get(finding.point_id, 0)
        return FeedbackResponse(
            status="ok",
            finding_id=finding.id,
            point_id=finding.point_id,
            net_votes=net,
        )
    finally:
        session.close()
