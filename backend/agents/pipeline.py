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
import os
import re
import json
from datetime import datetime
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import AsyncSession
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


async def _get_already_processed(db: AsyncSession) -> set[str]:
    result = await db.execute(select(Transcript.filename))
    return {row[0] for row in result.fetchall()}


async def run_pipeline_streaming(
    transcripts_dir: str = TRANSCRIPTS_DIR,
    api_key: Optional[str] = None,
    resume: bool = True,
) -> AsyncGenerator[str, None]:
    """
    Walk-forward pipeline as an async SSE generator.
    Yields JSON-encoded event strings consumed by the /pipeline/run endpoint.
    """

    def event(kind: str, **data) -> str:
        return json.dumps({"event": kind, "ts": datetime.utcnow().isoformat(), **data})

    yield event("pipeline_start", message="Starting walk-forward pipeline")

    all_files = get_sorted_transcript_files(transcripts_dir)
    if not all_files:
        yield event("error", message=f"No PDF files found in {transcripts_dir}")
        return

    yield event("pipeline_info", total_transcripts=len(all_files))

    async with AsyncSessionLocal() as db:
        already_processed = await _get_already_processed(db) if resume else set()

        pending = [f for f in all_files if f not in already_processed]
        skipped = len(all_files) - len(pending)

        if skipped:
            yield event("pipeline_info", message=f"Resuming — skipping {skipped} already-processed transcripts")

        if not pending:
            yield event("pipeline_done", message="All transcripts already processed", total=len(all_files))
            return

        # Build graph once (agents are stateless across calls)
        graph = build_graph(db=db, api_key=api_key)

        for idx, filename in enumerate(pending, start=skipped + 1):
            pdf_path = os.path.join(transcripts_dir, filename)

            # Create a ProcessingRun row for this transcript
            run = ProcessingRun(
                status=RunStatus.RUNNING,
                transcript_filename=filename,
                run_started_at=datetime.utcnow(),
            )
            db.add(run)
            await db.flush()
            run_id = run.id

            yield event(
                "transcript_start",
                index=idx,
                total=len(all_files),
                filename=filename,
                run_id=run_id,
            )

            try:
                initial_state: AgentState = {
                    "transcript_path":    pdf_path,
                    "processing_order":   idx,
                    "run_id":             run_id,
                    "transcript_id":      None,
                    "transcript_date":    None,
                    "event_type":         None,
                    "fiscal_period":      None,
                    "speech_blocks":      None,
                    "open_claims":        None,
                    "extracted_claims":   [],
                    "extraction_errors":  [],
                    "resolutions":        [],
                    "resolution_errors":  [],
                    "audit_entries":      [],
                    "run_summary":        None,
                    "error":              None,
                    "agent_trace":        [],
                }

                final_state = await graph.ainvoke(initial_state)

                # Emit per-agent trace
                for trace_line in final_state.get("agent_trace", []):
                    yield event("agent_trace", message=trace_line, run_id=run_id)

                # Emit extracted claims
                for claim in final_state.get("extracted_claims", []):
                    yield event(
                        "claim_extracted",
                        run_id=run_id,
                        claim_type=claim.get("claim_type"),
                        speaker=claim.get("speaker_name"),
                        summary=claim.get("claim_summary", "")[:120],
                        hedge_level=claim.get("hedge_level"),
                    )

                # Emit resolutions
                for res in final_state.get("resolutions", []):
                    yield event(
                        "claim_resolved",
                        run_id=run_id,
                        claim_id=res.get("claim_id"),
                        new_status=res.get("new_status"),
                        reasoning=res.get("reasoning", "")[:200],
                    )

                # Update run
                run.status          = RunStatus.COMPLETED
                run.run_ended_at    = datetime.utcnow()
                run.claims_extracted = len(final_state.get("extracted_claims", []))
                run.claims_resolved  = len(final_state.get("resolutions", []))
                run.agent_trace      = final_state.get("agent_trace", [])
                await db.commit()

                yield event(
                    "transcript_done",
                    index=idx,
                    filename=filename,
                    claims_extracted=run.claims_extracted,
                    claims_resolved=run.claims_resolved,
                    errors=final_state.get("extraction_errors", []) + final_state.get("resolution_errors", []),
                )

            except Exception as e:
                await db.rollback()
                run.status        = RunStatus.FAILED
                run.error_message = str(e)
                run.run_ended_at  = datetime.utcnow()
                await db.commit()

                yield event("transcript_error", filename=filename, error=str(e))

    yield event("pipeline_done", message="Walk-forward pipeline complete", total=len(all_files))
