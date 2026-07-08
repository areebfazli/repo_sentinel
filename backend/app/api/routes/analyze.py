from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from backend.app.core.rag_merger import RagMerger
from backend.app.core.report_generator import ReportGenerator
from backend.app.models.schemas import AnalyzeRequest, AnalyzeResponse

router = APIRouter()

# Dependency injection for the heavy ML models so we don't reload them on every request
def get_merger():
    # In production, this would be attached to app.state
    # For now, we'll instantiate it if needed, but ideally it's shared
    if not hasattr(get_merger, "instance"):
        get_merger.instance = RagMerger()
    return get_merger.instance

def get_report_generator():
    if not hasattr(get_report_generator, "instance"):
        get_report_generator.instance = ReportGenerator()
    return get_report_generator.instance

@router.post("/", response_model=AnalyzeResponse)
async def analyze_pr_code(
    request: AnalyzeRequest,
    merger: RagMerger = Depends(get_merger),
    report_gen: ReportGenerator = Depends(get_report_generator)
):
    """
    Analyzes a code snippet from a Pull Request.
    1. Embeds and searches against the Ghost Hunter (CVE) vector DB.
    2. Embeds and searches against the Team Memory vector DB.
    3. Reranks the findings.
    4. Generates a unified Markdown report via LLM.
    """
    try:
        # Step 1 & 2: RAG Merger (Dual Vector Search)
        raw_findings = await merger.analyze_code(request.code_snippet)
        
        # Step 3: LLM Report Generation
        final_markdown = report_gen.generate_pr_comment(
            code_snippet=request.code_snippet, 
            merged_findings=raw_findings
        )
        
        return AnalyzeResponse(
            is_vulnerable=raw_findings["is_vulnerable"],
            report_markdown=final_markdown,
            ghost_hunter_matches=len(raw_findings["ghost_hunter_findings"]),
            team_memory_matches=len(raw_findings["team_memory_findings"])
        )
        
    except Exception:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail="Analysis failed") from None
