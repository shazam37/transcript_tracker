"""
IngestAgent: PDF → DB Transcript row + open claims list.

Responsibilities:
  1. Parse the PDF into structured SpeechBlocks
  2. Persist a Transcript row (idempotent — skip if already exists)
  3. Load all currently OPEN claims from DB (for the Resolver Agent to use)

Why load open claims here (not in ResolutionAgent)?
  We want to snapshot the open-claim set BEFORE extraction of new claims.
  This enforces walk-forward: new claims from THIS transcript are not
  eligible for resolution until the NEXT transcript is processed.
"""

from __future__ import annotations
import json
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..db.models import Transcript, Claim, ResolutionStatus, ProcessingRun, RunStatus
from ..ingestion.parser import parse_transcript


async def run_ingest_agent(state: dict, db: AsyncSession) -> dict:
    """
    Node function for the LangGraph IngestAgent.
    Reads: transcript_path, processing_order, run_id
    Writes: transcript_id, transcript_date, event_type, speech_blocks, open_claims
    """
    updates: dict = {"agent_trace": []}

    try:
        path  = state["transcript_path"]
        order = state["processing_order"]
        run_id = state["run_id"]

        updates["agent_trace"].append(f"[IngestAgent] Parsing {path}")

        # Parse PDF
        parsed = parse_transcript(path)

        # Check if already processed
        existing = await db.execute(
            select(Transcript).where(Transcript.filename == parsed.filename)
        )
        transcript_row = existing.scalar_one_or_none()

        if transcript_row is None:
            transcript_row = Transcript(
                filename=parsed.filename,
                transcript_date=parsed.transcript_date,
                event_type=parsed.event_type,
                fiscal_period=parsed.fiscal_period,
                processing_order=order,
                full_text=parsed.full_text[:50000],  # cap at 50k chars
                speaker_metadata={
                    "corporate": parsed.corporate_participants,
                    "analysts":  parsed.analyst_participants,
                },
                run_id=run_id,
            )
            db.add(transcript_row)
            await db.flush()  # get the ID
            updates["agent_trace"].append(
                f"[IngestAgent] Persisted transcript id={transcript_row.id} "
                f"date={parsed.transcript_date} event={parsed.event_type}"
            )
        else:
            updates["agent_trace"].append(
                f"[IngestAgent] Transcript already exists id={transcript_row.id}, skipping insert"
            )

        # Snapshot currently OPEN claims BEFORE we process this transcript
        # This is the walk-forward enforcement point
        open_result = await db.execute(
            select(Claim).where(Claim.resolution_status == ResolutionStatus.OPEN)
        )
        open_claims = open_result.scalars().all()

        open_claims_dicts = [
            {
                "id":              c.id,
                "speaker_name":    c.speaker_name,
                "speaker_title":   c.speaker_title,
                "raw_quote":       c.raw_quote,
                "claim_summary":   c.claim_summary,
                "claim_type":      c.claim_type.value,
                "hedge_level":     c.hedge_level.value,
                "timeframe":       c.timeframe,
                "timeframe_date":  c.timeframe_date,
                "transcript_date": c.transcript.transcript_date if c.transcript else None,
                "tags":            c.tags or [],
            }
            for c in open_claims
        ]

        # Build speech blocks for extraction
        speech_blocks = [
            {
                "speaker_name":  b.speaker_name,
                "speaker_title": b.speaker_title,
                "text":          b.text,
                "section":       b.section,
                "is_management": b.is_management,
            }
            for b in parsed.speech_blocks
        ]

        updates.update({
            "transcript_id":   transcript_row.id,
            "transcript_date": parsed.transcript_date,
            "event_type":      parsed.event_type,
            "fiscal_period":   parsed.fiscal_period,
            "speech_blocks":   speech_blocks,
            "open_claims":     open_claims_dicts,
        })
        updates["agent_trace"].append(
            f"[IngestAgent] Loaded {len(open_claims_dicts)} open claims for resolution check"
        )

    except Exception as e:
        updates["error"] = f"IngestAgent failed: {e}"
        updates["agent_trace"].append(f"[IngestAgent] ERROR: {e}")

    return updates
