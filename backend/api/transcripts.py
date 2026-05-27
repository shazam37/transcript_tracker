"""Transcripts API router."""
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from ..db.session import get_db
from ..db.models import Transcript, Claim

router = APIRouter(prefix="/transcripts", tags=["transcripts"])


@router.get("")
async def list_transcripts(db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(
        select(Transcript).order_by(Transcript.processing_order)
    )).scalars().all()

    # Claim counts per transcript
    counts = (await db.execute(
        select(Claim.transcript_id, func.count(Claim.id))
        .group_by(Claim.transcript_id)
    )).fetchall()
    count_map = {row[0]: row[1] for row in counts}

    return [
        {
            "id":               t.id,
            "filename":         t.filename,
            "transcript_date":  t.transcript_date,
            "event_type":       t.event_type,
            "fiscal_period":    t.fiscal_period,
            "processing_order": t.processing_order,
            "claim_count":      count_map.get(t.id, 0),
            "processed_at":     t.processed_at.isoformat() if t.processed_at else None,
        }
        for t in rows
    ]


@router.get("/{transcript_id}")
async def get_transcript(transcript_id: int, db: AsyncSession = Depends(get_db)):
    row = (await db.execute(
        select(Transcript)
        .where(Transcript.id == transcript_id)
        .options(selectinload(Transcript.claims))
    )).scalar_one_or_none()

    if not row:
        raise HTTPException(status_code=404, detail="Transcript not found")

    claims = [
        {
            "id":               c.id,
            "speaker_name":     c.speaker_name,
            "claim_summary":    c.claim_summary,
            "claim_type":       c.claim_type.value,
            "hedge_level":      c.hedge_level.value,
            "resolution_status": c.resolution_status.value,
            "timeframe":        c.timeframe,
        }
        for c in row.claims
    ]

    return {
        "id":               row.id,
        "filename":         row.filename,
        "transcript_date":  row.transcript_date,
        "event_type":       row.event_type,
        "fiscal_period":    row.fiscal_period,
        "processing_order": row.processing_order,
        "speaker_metadata": row.speaker_metadata,
        "claims":           claims,
    }
