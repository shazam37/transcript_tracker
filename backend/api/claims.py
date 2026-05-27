"""Claims API router."""
from __future__ import annotations
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import selectinload

from ..db.session import get_db
from ..db.models import Claim, Evidence, ClaimAuditLog, Transcript, ResolutionStatus
from ..schemas.schemas import ClaimOut, ClaimDetailOut, ClaimFilter, EvidenceOut, AuditLogOut

router = APIRouter(prefix="/claims", tags=["claims"])


def _build_filters(q, f: ClaimFilter):
    """Apply ClaimFilter to a SQLAlchemy query."""
    if f.status:
        q = q.where(Claim.resolution_status == f.status)
    if f.claim_type:
        q = q.where(Claim.claim_type == f.claim_type)
    if f.speaker:
        q = q.where(Claim.speaker_name.ilike(f"%{f.speaker}%"))
    if f.tag:
        q = q.where(Claim.tags.contains([f.tag]))
    if f.from_date:
        q = q.join(Transcript, Claim.transcript_id == Transcript.id).where(
            Transcript.transcript_date >= f.from_date
        )
    if f.to_date:
        q = q.join(Transcript, Claim.transcript_id == Transcript.id).where(
            Transcript.transcript_date <= f.to_date
        )
    if f.hedge_level:
        q = q.where(Claim.hedge_level == f.hedge_level)
    if f.search:
        term = f"%{f.search}%"
        q = q.where(
            or_(
                Claim.claim_summary.ilike(term),
                Claim.raw_quote.ilike(term),
                Claim.speaker_name.ilike(term),
            )
        )
    return q


@router.get("", response_model=dict)
async def list_claims(
    status:      Optional[str] = None,
    claim_type:  Optional[str] = None,
    speaker:     Optional[str] = None,
    tag:         Optional[str] = None,
    from_date:   Optional[str] = None,
    to_date:     Optional[str] = None,
    hedge_level: Optional[str] = None,
    search:      Optional[str] = None,
    page:        int = Query(1, ge=1),
    page_size:   int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    f = ClaimFilter(
        status=status, claim_type=claim_type, speaker=speaker,
        tag=tag, from_date=from_date, to_date=to_date,
        hedge_level=hedge_level, search=search,
        page=page, page_size=page_size,
    )

    base_q = select(Claim).join(Transcript, Claim.transcript_id == Transcript.id)
    base_q = _build_filters(base_q, f)

    # Count
    count_q = select(func.count()).select_from(base_q.subquery())
    total   = (await db.execute(count_q)).scalar_one()

    # Page
    items_q = (
        base_q
        .options(selectinload(Claim.transcript))
        .order_by(Transcript.transcript_date.asc(), Claim.id.asc())
        .offset((f.page - 1) * f.page_size)
        .limit(f.page_size)
    )
    rows = (await db.execute(items_q)).scalars().all()

    claims_out = []
    for c in rows:
        d = ClaimOut.model_validate(c).model_dump()
        d["transcript_date"] = c.transcript.transcript_date if c.transcript else None
        d["event_type"]      = c.transcript.event_type if c.transcript else None
        claims_out.append(d)

    return {
        "total":    total,
        "page":     f.page,
        "page_size": f.page_size,
        "items":    claims_out,
    }


@router.get("/{claim_id}", response_model=dict)
async def get_claim(claim_id: int, db: AsyncSession = Depends(get_db)):
    q = (
        select(Claim)
        .where(Claim.id == claim_id)
        .options(
            selectinload(Claim.transcript),
            selectinload(Claim.evidence_list).selectinload(Evidence.source_transcript),
            selectinload(Claim.audit_log).selectinload(ClaimAuditLog.source_transcript),
        )
    )
    claim = (await db.execute(q)).scalar_one_or_none()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")

    evidence = [
        {
            "id":               e.id,
            "speaker_name":     e.speaker_name,
            "raw_quote":        e.raw_quote,
            "evidence_summary": e.evidence_summary,
            "supports_claim":   e.supports_claim,
            "transcript_date":  e.source_transcript.transcript_date if e.source_transcript else None,
            "event_type":       e.source_transcript.event_type if e.source_transcript else None,
        }
        for e in claim.evidence_list
    ]

    audit = [
        {
            "id":                     a.id,
            "changed_at":             a.changed_at.isoformat() if a.changed_at else None,
            "old_status":             a.old_status.value,
            "new_status":             a.new_status.value,
            "reasoning":              a.reasoning,
            "agent_name":             a.agent_name,
            "source_transcript_date": a.source_transcript.transcript_date if a.source_transcript else None,
            "source_transcript_name": a.source_transcript.filename if a.source_transcript else None,
        }
        for a in claim.audit_log
    ]

    out = ClaimOut.model_validate(claim).model_dump()
    out["transcript_date"] = claim.transcript.transcript_date if claim.transcript else None
    out["event_type"]      = claim.transcript.event_type if claim.transcript else None
    out["evidence"]        = evidence
    out["audit_trail"]     = audit
    return out
