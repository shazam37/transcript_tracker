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
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from ..db.session import get_db
from ..db.models import ProcessingRun, Transcript, RunStatus
from ..agents.pipeline import run_pipeline_streaming, TRANSCRIPTS_DIR

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

# Simple in-memory lock — prevents concurrent pipeline runs
_pipeline_running = False


@router.get("/status")
async def pipeline_status(db: AsyncSession = Depends(get_db)):
    total_files = len([
        f for f in os.listdir(TRANSCRIPTS_DIR) if f.endswith(".pdf")
    ]) if os.path.exists(TRANSCRIPTS_DIR) else 0

    processed = (await db.execute(select(func.count(Transcript.id)))).scalar_one()

    last_run_row = (await db.execute(
        select(ProcessingRun)
        .order_by(desc(ProcessingRun.run_started_at))
        .limit(1)
    )).scalar_one_or_none()

    return {
        "total_transcripts": total_files,
        "processed":         processed,
        "pending":           max(0, total_files - processed),
        "is_running":        _pipeline_running,
        "last_run_at":       last_run_row.run_started_at.isoformat() if last_run_row else None,
    }


@router.post("/run")
async def run_pipeline(
    resume: bool = True,
    db: AsyncSession = Depends(get_db),
):
    """
    Stream the walk-forward pipeline as Server-Sent Events.
    Each event is a JSON object with an "event" type field.

    Event types:
      pipeline_start    — pipeline beginning
      pipeline_info     — metadata (total count, skip count)
      transcript_start  — starting a transcript
      agent_trace       — step-by-step agent reasoning
      claim_extracted   — a new claim was found
      claim_resolved    — an open claim was resolved
      transcript_done   — transcript finished with stats
      transcript_error  — transcript failed (non-fatal)
      pipeline_done     — all transcripts complete
      error             — fatal error
    """
    global _pipeline_running
    if _pipeline_running:
        raise HTTPException(status_code=409, detail="Pipeline already running")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    async def event_stream():
        global _pipeline_running
        _pipeline_running = True
        try:
            async for line in run_pipeline_streaming(
                transcripts_dir=TRANSCRIPTS_DIR,
                api_key=api_key,
                resume=resume,
            ):
                yield f"data: {line}\n\n"
                await asyncio.sleep(0)   # yield control to event loop
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
