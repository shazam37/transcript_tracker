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

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from ..db.models import Transcript, Claim, ResolutionStatus
from ..ingestion.parser import parse_transcript


async def run_ingest_agent(state: dict, db: AsyncSession) -> dict:
    updates: dict = {"agent_trace": [], "extracted_claims": [], "extraction_errors": [],
                     "resolutions": [], "resolution_errors": [], "audit_entries": []}
    try:
        path   = state["transcript_path"]
        order  = state["processing_order"]
        run_id = state["run_id"]

        updates["agent_trace"].append(f"[IngestAgent] Parsing {path}")

        # ── 1. Parse PDF ──────────────────────────────────────────────────
        parsed = parse_transcript(path)

        # ── 2. Persist transcript row (idempotent) ────────────────────────
        existing = (await db.execute(
            select(Transcript).where(Transcript.filename == parsed.filename)
        )).scalar_one_or_none()

        if existing is None:
            transcript_row = Transcript(
                filename=parsed.filename,
                transcript_date=parsed.transcript_date,
                event_type=parsed.event_type,
                fiscal_period=parsed.fiscal_period,
                processing_order=order,
                full_text=parsed.full_text[:50000],
                speaker_metadata={
                    "corporate": parsed.corporate_participants,
                    "analysts":  parsed.analyst_participants,
                },
                run_id=run_id,
            )
            db.add(transcript_row)
            await db.flush()  # assigns transcript_row.id
            updates["agent_trace"].append(
                f"[IngestAgent] Persisted transcript id={transcript_row.id} "
                f"date={parsed.transcript_date} event={parsed.event_type}"
            )
        else:
            transcript_row = existing
            updates["agent_trace"].append(
                f"[IngestAgent] Already exists id={transcript_row.id}, skipping insert"
            )

        # ── 3. Snapshot OPEN claims (walk-forward enforcement) ────────────
        # selectinload(Claim.transcript) avoids lazy-load MissingGreenlet hang.
        open_claims_rows = (await db.execute(
            select(Claim)
            .where(Claim.resolution_status == ResolutionStatus.OPEN)
            .options(selectinload(Claim.transcript))
        )).scalars().all()

        open_claims_dicts = [
            {
                "id":             c.id,
                "speaker_name":   c.speaker_name,
                "speaker_title":  c.speaker_title,
                "raw_quote":      c.raw_quote,
                "claim_summary":  c.claim_summary,
                "claim_type":     c.claim_type.value,
                "hedge_level":    c.hedge_level.value,
                "timeframe":      c.timeframe,
                "timeframe_date": c.timeframe_date,
                "transcript_date": c.transcript.transcript_date if c.transcript else None,
                "tags":           c.tags or [],
            }
            for c in open_claims_rows
        ]

        # ── 4. Build speech block dicts ───────────────────────────────────
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
            f"[IngestAgent] {len(open_claims_dicts)} open claims snapshotted, "
            f"{len(speech_blocks)} speech blocks ready"
        )

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        updates["error"] = f"IngestAgent failed: {e}"
        updates["agent_trace"].append(f"[IngestAgent] ERROR: {e}\n{tb}")

    return updates