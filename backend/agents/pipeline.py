"""
Pipeline runner: orchestrates walk-forward processing across all 31 transcripts.

Walk-forward invariant (hard rule):
  When processing transcript N, only transcripts 1..N-1 are used for resolution.
  This is enforced by:
    1. Processing in strict chronological order (sorted by embedded ISO date)
    2. IngestAgent snapshots open claims BEFORE ExtractionAgent adds new ones
    3. Each transcript gets its own DB transaction — no bleed between runs

Resume semantics:
  The ProcessingRun table tracks every transcript processed.
  If the pipeline crashes at transcript 17, restart with resume=True
  and it picks up at 18. Transcripts 1-17 are not re-processed.

Streaming:
  Each transcript emits SSE events via an async generator.
  The FastAPI /pipeline/run endpoint streams these to the frontend
  so you can watch claims being extracted in real time.
"""

from __future__ import annotations
import asyncio
import os
import json
import traceback
from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import select

from ..db.models import ProcessingRun, Transcript, RunStatus
from ..db.session import AsyncSessionLocal
from ..ingestion.parser import get_sorted_transcript_files
from .graph import build_graph
from .state import AgentState

TRANSCRIPTS_DIR = os.getenv(
    "TRANSCRIPTS_DIR",
    os.path.join(os.path.dirname(__file__), "../../data/raw_transcripts"),
)


def _evt(kind: str, **data) -> str:
    return json.dumps({"event": kind, "ts": datetime.utcnow().isoformat(), **data})


async def _already_processed() -> set[str]:
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Transcript.filename))
        return {row[0] for row in result.fetchall()}


async def _create_run(filename: str) -> int:
    async with AsyncSessionLocal() as db:
        run = ProcessingRun(
            status=RunStatus.RUNNING,
            transcript_filename=filename,
            run_started_at=datetime.utcnow(),
        )
        db.add(run)
        await db.commit()
        await db.refresh(run)
        return run.id


async def _finish_run(run_id: int, extracted: int, resolved: int,
                      trace: list, error: str | None = None):
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(ProcessingRun).where(ProcessingRun.id == run_id)
        )
        run = result.scalar_one_or_none()
        if run:
            run.status           = RunStatus.FAILED if error else RunStatus.COMPLETED
            run.run_ended_at     = datetime.utcnow()
            run.claims_extracted = extracted
            run.claims_resolved  = resolved
            run.agent_trace      = trace
            run.error_message    = error
            await db.commit()


async def run_pipeline_streaming(
    transcripts_dir: str = TRANSCRIPTS_DIR,
    resume: bool = True,
    max_transcripts: int | None = None,
) -> AsyncGenerator[str, None]:
    """
    Walk-forward SSE generator.

    Each transcript:
      1. Emits transcript_start immediately
      2. Dispatches graph.ainvoke to thread pool (non-blocking)
      3. Polls for completion every 0.5s, yielding keepalive comments
      4. On completion, emits all trace/claim/resolution events
    """
    yield _evt("pipeline_start", message="Starting walk-forward pipeline")

    all_files = get_sorted_transcript_files(transcripts_dir)
    if not all_files:
        yield _evt("error", message=f"No PDF files found in {transcripts_dir}")
        return

    yield _evt("pipeline_info", total_transcripts=len(all_files))

    already_done = await _already_processed() if resume else set()
    pending  = [f for f in all_files if f not in already_done]
    if max_transcripts:
        pending = pending[:max_transcripts]
    skipped  = len(all_files) - len(pending)

    if skipped:
        yield _evt("pipeline_info",
                   message=f"Resuming — skipping {skipped} already-processed transcripts")

    if not pending:
        yield _evt("pipeline_done",
                   message="All transcripts already processed",
                   total=len(all_files))
        return

    # Build graph once — reused across all transcripts
    graph = build_graph()

    for idx, filename in enumerate(pending, start=skipped + 1):
        pdf_path = os.path.join(transcripts_dir, filename)
        run_id   = await _create_run(filename)

        yield _evt("transcript_start",
                   index=idx, total=len(all_files),
                   filename=filename, run_id=run_id)

        extracted_claims, resolutions, trace, error_msg = [], [], [], None

        try:
            initial: AgentState = {
                "transcript_path":   pdf_path,
                "processing_order":  idx,
                "run_id":            run_id,
                "transcript_id":     None,
                "transcript_date":   None,
                "event_type":        None,
                "fiscal_period":     None,
                "speech_blocks":     None,
                "open_claims":       None,
                "extracted_claims":  [],
                "extraction_errors": [],
                "resolutions":       [],
                "resolution_errors": [],
                "audit_entries":     [],
                "run_summary":       None,
                "error":             None,
                "agent_trace":       [],
            }

            # ── Run graph as a concurrent async task ──────────────────────
            # LLM calls use ainvoke (truly async), so the graph runs on the
            # main event loop without blocking. The keepalive loop below runs
            # between awaited LLM calls, keeping the SSE connection alive.
            task = asyncio.create_task(graph.ainvoke(initial))

            dot_count = 0
            while not task.done():
                await asyncio.sleep(0.5)
                dot_count += 1
                # Emit a visible tick event every 2s so the frontend stays alive
                if dot_count % 4 == 0:
                    yield _evt("tick", elapsed=dot_count // 2, run_id=run_id,
                               index=idx, total=len(all_files), filename=filename)

            final = await task   # re-raises if task threw

            trace            = final.get("agent_trace", [])
            extracted_claims = final.get("extracted_claims", [])
            resolutions      = final.get("resolutions", [])
            all_errors       = (final.get("extraction_errors", []) +
                                final.get("resolution_errors", []))

            # Emit trace lines
            for line in trace:
                yield _evt("agent_trace", message=line, run_id=run_id)

            # Emit extracted claims
            for c in extracted_claims:
                yield _evt("claim_extracted",
                           run_id=run_id,
                           claim_type=c.get("claim_type"),
                           speaker=c.get("speaker_name"),
                           summary=c.get("claim_summary", "")[:120],
                           hedge_level=c.get("hedge_level"))

            # Emit resolutions
            for r in resolutions:
                yield _evt("claim_resolved",
                           run_id=run_id,
                           claim_id=r.get("claim_id"),
                           new_status=r.get("new_status"),
                           reasoning=r.get("reasoning", "")[:200])

            # Emit individual errors so the frontend can display them
            for err in all_errors[:10]:  # cap at 10 per transcript to avoid flooding
                yield _evt("agent_error", run_id=run_id, message=str(err)[:300])

            yield _evt("transcript_done",
                       index=idx, filename=filename,
                       claims_extracted=len(extracted_claims),
                       claims_resolved=len(resolutions),
                       error_count=len(all_errors))

        except Exception as e:
            error_msg = str(e)
            tb = traceback.format_exc()
            yield _evt("transcript_error", filename=filename,
                       error=error_msg, traceback=tb[:500])

        finally:
            await _finish_run(run_id,
                              extracted=len(extracted_claims),
                              resolved=len(resolutions),
                              trace=trace,
                              error=error_msg)

    yield _evt("pipeline_done",
               message="Walk-forward pipeline complete",
               total=len(all_files))