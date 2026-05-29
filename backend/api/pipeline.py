"""
Pipeline API router.

Endpoints:
  GET  /pipeline/status   — how many transcripts processed, is it running
  POST /pipeline/run      — kick off walk-forward pipeline, streams SSE events
  GET  /pipeline/runs     — history of all processing runs
  GET  /pipeline/runs/:id — single run detail with agent trace
"""
from __future__ import annotations
import asyncio
import json
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from ..db.session import get_db
from ..db.models import ProcessingRun, Transcript, RunStatus
from ..agents.pipeline import run_pipeline_streaming, TRANSCRIPTS_DIR

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

_pipeline_running = False


@router.get("/status")
async def pipeline_status(db: AsyncSession = Depends(get_db)):
    total_files = len([
        f for f in os.listdir(TRANSCRIPTS_DIR) if f.endswith(".pdf")
    ]) if os.path.exists(TRANSCRIPTS_DIR) else 0

    processed = (await db.execute(select(func.count(Transcript.id)))).scalar_one()

    last_run = (await db.execute(
        select(ProcessingRun).order_by(desc(ProcessingRun.run_started_at)).limit(1)
    )).scalar_one_or_none()

    from ..utils.llm_config import current_provider_info
    return {
        "total_transcripts": total_files,
        "processed":         processed,
        "pending":           max(0, total_files - processed),
        "is_running":        _pipeline_running,
        "last_run_at":       last_run.run_started_at.isoformat() if last_run else None,
        "llm":               current_provider_info(),
    }


@router.get("/run")  # GET so EventSource works
async def run_pipeline(resume: bool = True, max: int = 0):
    """
    Stream walk-forward pipeline as Server-Sent Events.
    EventSource requires GET — POST would silently fail in the browser.
    """
    global _pipeline_running
    if _pipeline_running:
        raise HTTPException(status_code=409, detail="Pipeline already running")

    if not os.getenv("GROQ_API_KEY") and \
       not os.getenv("ANTHROPIC_API_KEY") and \
       not os.getenv("OPENAI_API_KEY") and \
       not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500,
            detail="No LLM API key set. Add GROQ_API_KEY (or other provider key) to .env")

    async def event_stream():
        global _pipeline_running
        _pipeline_running = True
        try:
            async for line in run_pipeline_streaming(
                transcripts_dir=TRANSCRIPTS_DIR,
                resume=resume,
                max_transcripts=max if max > 0 else None,
            ):
                yield f"data: {line}\n\n"
                await asyncio.sleep(0)
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
        finally:
            _pipeline_running = False

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
            "Connection":                  "keep-alive",
        },
    )


@router.get("/runs")
async def list_runs(db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(
        select(ProcessingRun).order_by(desc(ProcessingRun.run_started_at)).limit(100)
    )).scalars().all()
    return [
        {
            "id":                  r.id,
            "status":              r.status.value,
            "transcript_filename": r.transcript_filename,
            "run_started_at":      r.run_started_at.isoformat() if r.run_started_at else None,
            "run_ended_at":        r.run_ended_at.isoformat() if r.run_ended_at else None,
            "claims_extracted":    r.claims_extracted,
            "claims_resolved":     r.claims_resolved,
            "error_message":       r.error_message,
        }
        for r in rows
    ]


@router.get("/runs/{run_id}")
async def get_run(run_id: int, db: AsyncSession = Depends(get_db)):
    row = (await db.execute(
        select(ProcessingRun).where(ProcessingRun.id == run_id)
    )).scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "id":                  row.id,
        "status":              row.status.value,
        "transcript_filename": row.transcript_filename,
        "run_started_at":      row.run_started_at.isoformat() if row.run_started_at else None,
        "run_ended_at":        row.run_ended_at.isoformat() if row.run_ended_at else None,
        "claims_extracted":    row.claims_extracted,
        "claims_resolved":     row.claims_resolved,
        "error_message":       row.error_message,
        "agent_trace":         row.agent_trace or [],
    }