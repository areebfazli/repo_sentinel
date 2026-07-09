const API_HOST = 'http://127.0.0.1:8000';
const ANALYZE_URL = `${API_HOST}/api/v1/analyze/`;
const POLL_INTERVAL_MS = 1500;
const POLL_MAX_ATTEMPTS = 80; // ~120s

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Render Markdown to sanitized HTML. The report embeds LLM- and team-comment-
// authored text (attacker-influenceable), and marked does not sanitize, so we
// run the output through DOMPurify before it ever touches innerHTML.
function renderMarkdownSafe(md) {
    const html = marked.parse(md || '');
    return typeof DOMPurify !== 'undefined' ? DOMPurify.sanitize(html) : escapeHtml(html);
}

function getApiKey() {
    return localStorage.getItem('reposentinel_api_key') || '';
}

function apiHeaders(extra = {}) {
    const headers = { ...extra };
    const key = getApiKey();
    if (key) headers['X-RepoSentinel-Key'] = key;
    return headers;
}

// Expose the latest findings so the feedback UI (Phase 5) can reference them.
window.repoSentinelFindings = [];

document.addEventListener('DOMContentLoaded', () => {
    const codeInput = document.getElementById('codeInput');
    const scanBtn = document.getElementById('scanBtn');
    const btnText = scanBtn.querySelector('.btn-text');
    const loader = scanBtn.querySelector('.loader');

    const welcomeState = document.getElementById('welcomeState');
    const resultsState = document.getElementById('resultsState');

    marked.setOptions({ breaks: true, gfm: true });

    // Language selector — "Auto" sends no language so the backend doesn't
    // filter the CVE search to the wrong partition (a wrong guess = false clean).
    const langSelect = document.createElement('select');
    langSelect.id = 'langSelect';
    langSelect.style.cssText = 'margin-bottom:10px; padding:6px 10px; border-radius:8px;';
    langSelect.innerHTML = `
        <option value="">Language: Auto</option>
        <option value="python">Python</option>
        <option value="javascript">JavaScript / TypeScript</option>
        <option value="go">Go</option>
        <option value="java">Java</option>`;
    if (codeInput && codeInput.parentNode) {
        codeInput.parentNode.insertBefore(langSelect, codeInput);
    }

    // Let the user set an API key via the Settings nav link (needed when the
    // backend enforces REPOSENTINEL_API_KEY; unset for local dev).
    document.querySelectorAll('.nav-link').forEach((link) => {
        if (link.textContent.trim().toLowerCase() === 'settings') {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const current = getApiKey();
                const value = window.prompt('RepoSentinel API key (leave blank for none):', current);
                if (value !== null) localStorage.setItem('reposentinel_api_key', value.trim());
            });
        }
    });

    const showSpinner = (message) => {
        resultsState.innerHTML = `
            <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; gap:20px; color:var(--text-secondary);">
                <div class="loader" style="width:40px; height:40px; border-width:4px; border-top-color:var(--accent-purple);"></div>
                <p>${message}</p>
            </div>`;
    };

    const showError = (message) => {
        resultsState.innerHTML = `
            <div style="text-align:center; padding:2rem;">
                <h3 style="color:var(--accent-red); margin-bottom:1rem;">Scan Failed</h3>
                <p>${message}</p>
                <p style="font-size:0.8rem; margin-top:2rem;">Is the FastAPI backend running?</p>
            </div>`;
    };

    const renderResult = (result) => {
        window.repoSentinelFindings = result.findings || [];
        let html = renderMarkdownSafe(result.report_markdown);
        html += renderFindingCards(result.findings || []);
        resultsState.innerHTML = html;
        resultsState.style.animation = 'slideUp 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
        wireFeedbackButtons(resultsState);
    };

    const pollJob = async (jobId) => {
        const url = `${ANALYZE_URL}${jobId}`;
        for (let attempt = 0; attempt < POLL_MAX_ATTEMPTS; attempt++) {
            await sleep(POLL_INTERVAL_MS);
            const resp = await fetch(url, { headers: apiHeaders() });
            if (!resp.ok) throw new Error(`Poll error: ${resp.status}`);
            const data = await resp.json();
            if (data.status === 'completed') return data.result;
            if (data.status === 'failed') throw new Error(data.error || 'Scan failed');
            showSpinner(data.status === 'running' ? 'Analyzing code…' : 'Queued…');
        }
        throw new Error('Timed out waiting for the scan to finish.');
    };

    scanBtn.addEventListener('click', async () => {
        const code = codeInput.value.trim();
        if (!code) {
            codeInput.style.animation = 'shake 0.5s ease';
            setTimeout(() => (codeInput.style.animation = ''), 500);
            return;
        }

        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
        scanBtn.style.pointerEvents = 'none';
        welcomeState.classList.add('hidden');
        resultsState.classList.remove('hidden');
        showSpinner('Queuing scan…');

        try {
            const body = { code_snippet: code };
            if (langSelect.value) body.language = langSelect.value;
            const resp = await fetch(ANALYZE_URL, {
                method: 'POST',
                headers: apiHeaders({ 'Content-Type': 'application/json' }),
                body: JSON.stringify(body),
            });
            if (resp.status === 401) throw new Error('Unauthorized — set your API key in Settings.');
            if (!(resp.status === 202 || resp.ok)) throw new Error(`API Error: ${resp.status}`);

            const { job_id: jobId } = await resp.json();
            showSpinner('Running semantic vector search…');
            const result = await pollJob(jobId);
            renderResult(result);
        } catch (error) {
            console.error('Scan failed:', error);
            showError(error.message);
        } finally {
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
            scanBtn.style.pointerEvents = 'auto';
        }
    });
});

function renderFindingCards(findings) {
    if (!findings.length) return '';
    const cards = findings
        .map((f) => {
            const ref = f.cve_id || (f.team_pr_id ? `PR ${f.team_pr_id}` : '');
            const badge = f.source === 'cve' ? '🌐 CVE' : '🏠 Team';
            const sev = f.severity ? ` · ${f.severity}` : '';
            return `
            <div class="finding-card" data-finding-id="${f.finding_id}" style="border:1px solid var(--border, #333); border-radius:10px; padding:12px 14px; margin-top:10px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-size:0.8rem; color:var(--text-secondary);">${badge}${sev}</div>
                    <div class="feedback" style="display:flex; gap:8px;">
                        <button class="fb-up" data-vote="1" title="Helpful" style="cursor:pointer; background:none; border:none; font-size:1rem;">👍</button>
                        <button class="fb-down" data-vote="-1" title="Not relevant" style="cursor:pointer; background:none; border:none; font-size:1rem;">👎</button>
                    </div>
                </div>
                <div style="font-weight:600; margin:4px 0;">${escapeHtml(f.title)}</div>
                <div style="font-size:0.8rem; color:var(--text-secondary);">
                    ${ref ? escapeHtml(ref) + ' · ' : ''}similarity ${Number(f.similarity_score).toFixed(2)}
                </div>
            </div>`;
        })
        .join('');
    return `<h3 style="margin-top:24px;">Matched memories</h3>${cards}`;
}

function wireFeedbackButtons(container) {
    container.querySelectorAll('.finding-card').forEach((card) => {
        const findingId = Number(card.getAttribute('data-finding-id'));
        card.querySelectorAll('button[data-vote]').forEach((btn) => {
            btn.addEventListener('click', async () => {
                const vote = Number(btn.getAttribute('data-vote'));
                try {
                    const resp = await fetch(`${API_HOST}/api/v1/feedback/`, {
                        method: 'POST',
                        headers: apiHeaders({ 'Content-Type': 'application/json' }),
                        body: JSON.stringify({ finding_id: findingId, vote }),
                    });
                    if (!resp.ok) throw new Error(`Feedback error: ${resp.status}`);
                    // Highlight the chosen vote.
                    card.querySelectorAll('button[data-vote]').forEach((b) => (b.style.opacity = '0.35'));
                    btn.style.opacity = '1';
                } catch (err) {
                    console.error('Feedback failed:', err);
                }
            });
        });
    });
}

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) =>
        ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]
    );
}

// Animations
const style = document.createElement('style');
style.textContent = `
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
`;
document.head.appendChild(style);
