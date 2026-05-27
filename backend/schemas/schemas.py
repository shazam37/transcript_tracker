"""Pydantic v2 schemas — API request/response contracts."""
from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict


# ── Enums (re-exported as strings for OpenAPI) ────────────────────────────────

class ClaimTypeEnum(str):
    pass

# ── Transcript ────────────────────────────────────────────────────────────────

class TranscriptOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id:               int
    filename:         str
    transcript_date:  str
    event_type:       str
    fiscal_period:    Optional[str]
    processing_order: int
    processed_at:     Optional[datetime]
    claim_count:      Optional[int] = None


# ── Claim ─────────────────────────────────────────────────────────────────────

class ClaimOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id:                   int
    transcript_id:        int
    transcript_date:      Optional[str] = None
    event_type:           Optional[str] = None
    speaker_name:         str
    speaker_title:        Optional[str]
    raw_quote:            str
    claim_summary:        str
    claim_type:           str
    hedge_level:          str
    timeframe:            Optional[str]
    timeframe_date:       Optional[str]
    tags:                 Optional[list] = []
    section:              Optional[str]
    resolution_status:    str
    resolution_reasoning: Optional[str]
    resolution_date:      Optional[str]
    created_at:           Optional[datetime]


class ClaimDetailOut(ClaimOut):
    """Extended claim with evidence and full audit trail."""
    evidence:    list[EvidenceOut] = []
    audit_trail: list[AuditLogOut] = []


class ClaimFilter(BaseModel):
    status:      Optional[str] = None
    claim_type:  Optional[str] = None
    speaker:     Optional[str] = None
    tag:         Optional[str] = None
    from_date:   Optional[str] = None
    to_date:     Optional[str] = None
    hedge_level: Optional[str] = None
    search:      Optional[str] = None   # full-text on summary + quote
    page:        int = 1
    page_size:   int = 50


# ── Evidence ──────────────────────────────────────────────────────────────────

class EvidenceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id:                int
    speaker_name:      Optional[str]
    raw_quote:         str
    evidence_summary:  Optional[str]
    supports_claim:    bool
    transcript_date:   Optional[str] = None


# ── Audit log ─────────────────────────────────────────────────────────────────

class AuditLogOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id:                      int
    changed_at:              datetime
    old_status:              str
    new_status:              str
    reasoning:               str
    agent_name:              Optional[str]
    source_transcript_date:  Optional[str] = None
    source_transcript_name:  Optional[str] = None


# ── Stats ─────────────────────────────────────────────────────────────────────

class ResolutionStats(BaseModel):
    total:                    int
    open:                     int
    materialized:             int
    partially_materialized:   int
    not_materialized:         int
    superseded:               int
    unresolvable:             int
    full_hit_rate:            float   # materialized / resolved (excl open/unresolvable)
    directional_accuracy:     float   # (materialized + partial) / resolved


class TypeBreakdown(BaseModel):
    claim_type:       str
    total:            int
    open:             int
    materialized:     int
    resolution_rate:  float


class SpeakerStats(BaseModel):
    speaker_name:           str
    total:                  int
    materialized:           int
    partially_materialized: int
    not_materialized:       int
    hit_rate:               float


class DashboardOut(BaseModel):
    resolution_stats:    ResolutionStats
    by_type:             list[TypeBreakdown]
    by_speaker:          list[SpeakerStats]
    total_transcripts:   int
    total_evidence:      int
    timeline:            list[dict]   # [{date, claims_made, claims_resolved}]


# ── Pipeline ──────────────────────────────────────────────────────────────────

class PipelineStatusOut(BaseModel):
    total_transcripts:   int
    processed:           int
    pending:             int
    is_running:          bool
    last_run_at:         Optional[datetime]


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id:                  int
    run_started_at:      Optional[datetime]
    run_ended_at:        Optional[datetime]
    status:              str
    transcript_filename: Optional[str]
    claims_extracted:    int
    claims_resolved:     int
    error_message:       Optional[str]


# ── Forward refs ──────────────────────────────────────────────────────────────
ClaimDetailOut.model_rebuild()
