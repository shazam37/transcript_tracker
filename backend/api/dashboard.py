"""Dashboard and stats API router."""
from __future__ import annotations
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from fastapi import APIRouter, Depends

from ..db.session import get_db
from ..db.models import Claim, Transcript, Evidence, ResolutionStatus, ClaimType

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("")
async def get_dashboard(db: AsyncSession = Depends(get_db)):
    # ── Resolution status counts ──────────────────────────────────────────
    status_rows = (await db.execute(
        select(Claim.resolution_status, func.count(Claim.id))
        .group_by(Claim.resolution_status)
    )).fetchall()
    counts = {row[0].value if hasattr(row[0], 'value') else row[0]: row[1] for row in status_rows}

    total          = sum(counts.values())
    materialized   = counts.get("materialized", 0)
    partial        = counts.get("partially_materialized", 0)
    not_mat        = counts.get("not_materialized", 0)
    superseded     = counts.get("superseded", 0)
    unresolvable   = counts.get("unresolvable", 0)
    open_count     = counts.get("open", 0)

    resolved_denom = materialized + partial + not_mat + superseded
    full_hit       = round(materialized / resolved_denom, 3) if resolved_denom else 0.0
    directional    = round((materialized + partial) / resolved_denom, 3) if resolved_denom else 0.0

    # ── By type ───────────────────────────────────────────────────────────
    type_rows = (await db.execute(
        select(Claim.claim_type, Claim.resolution_status, func.count(Claim.id))
        .group_by(Claim.claim_type, Claim.resolution_status)
    )).fetchall()

    type_map: dict = {}
    for ctype, status, cnt in type_rows:
        key = ctype.value if hasattr(ctype, 'value') else ctype
        st  = status.value if hasattr(status, 'value') else status
        if key not in type_map:
            type_map[key] = {"claim_type": key, "total": 0, "open": 0, "materialized": 0, "resolution_rate": 0}
        type_map[key]["total"] += cnt
        if st == "open": type_map[key]["open"] += cnt
        if st == "materialized": type_map[key]["materialized"] += cnt

    for v in type_map.values():
        resolved = v["total"] - v["open"]
        v["resolution_rate"] = round(resolved / v["total"], 3) if v["total"] else 0

    # ── By speaker ────────────────────────────────────────────────────────
    speaker_rows = (await db.execute(
        select(Claim.speaker_name, Claim.resolution_status, func.count(Claim.id))
        .where(Claim.resolution_status != ResolutionStatus.OPEN)
        .group_by(Claim.speaker_name, Claim.resolution_status)
    )).fetchall()

    spk_map: dict = {}
    for name, status, cnt in speaker_rows:
        st = status.value if hasattr(status, 'value') else status
        if name not in spk_map:
            spk_map[name] = {"speaker_name": name, "total": 0, "materialized": 0,
                             "partially_materialized": 0, "not_materialized": 0, "hit_rate": 0}
        spk_map[name]["total"] += cnt
        if st in spk_map[name]: spk_map[name][st] += cnt

    for v in spk_map.values():
        v["hit_rate"] = round(v["materialized"] / v["total"], 3) if v["total"] else 0

    by_speaker = sorted(spk_map.values(), key=lambda x: -x["total"])[:10]

    # ── Timeline: claims made + resolved per transcript ───────────────────
    timeline_rows = (await db.execute(
        select(
            Transcript.transcript_date,
            Transcript.event_type,
            func.count(Claim.id).label("claims_made")
        )
        .join(Claim, Claim.transcript_id == Transcript.id)
        .group_by(Transcript.transcript_date, Transcript.event_type)
        .order_by(Transcript.transcript_date)
    )).fetchall()

    resolution_rows = (await db.execute(
        select(Claim.resolution_date, func.count(Claim.id).label("resolved"))
        .where(Claim.resolution_date.isnot(None))
        .group_by(Claim.resolution_date)
    )).fetchall()
    res_map = {row[0]: row[1] for row in resolution_rows}

    timeline = [
        {
            "date":            row[0],
            "event_type":      row[1],
            "claims_made":     row[2],
            "claims_resolved": res_map.get(row[0], 0),
        }
        for row in timeline_rows
    ]

    # ── Totals ────────────────────────────────────────────────────────────
    total_transcripts = (await db.execute(select(func.count(Transcript.id)))).scalar_one()
    total_evidence    = (await db.execute(select(func.count(Evidence.id)))).scalar_one()

    return {
        "resolution_stats": {
            "total": total, "open": open_count,
            "materialized": materialized,
            "partially_materialized": partial,
            "not_materialized": not_mat,
            "superseded": superseded,
            "unresolvable": unresolvable,
            "full_hit_rate": full_hit,
            "directional_accuracy": directional,
        },
        "by_type":           sorted(type_map.values(), key=lambda x: -x["total"]),
        "by_speaker":        by_speaker,
        "total_transcripts": total_transcripts,
        "total_evidence":    total_evidence,
        "timeline":          timeline,
    }
