"""
Database models using SQLAlchemy 2.0 async ORM.

Schema design rationale:
─────────────────────────────────────────────────────────────────────
TRANSCRIPTS
  Source of truth for every document ingested.
  Immutable once created — we never re-parse a processed transcript.

CLAIMS
  A forward-looking statement extracted from a transcript.
  Immutable fields: everything set at extraction time.
  Mutable fields: resolution_status, resolution_* — only updated by the
  Resolver Agent and only ever monotonically progress through the lifecycle.

EVIDENCE
  A quote from a *later* transcript that speaks to a claim.
  Many-to-many with Claims via claim_evidence join table.
  Walk-forward constraint: evidence.transcript_date > claim.transcript_date

CLAIM_AUDIT_LOG
  Append-only log of every state transition on every claim.
  Critical for the demo: "show me every step of how claim X was resolved".
  Never deleted, never updated. The only write is INSERT.

PROCESSING_RUNS
  One row per pipeline run. Tracks timing, cost, errors.
  Lets you see "how long did processing 31 transcripts take" and replay.

Why PostgreSQL over SQLite:
  - JSONB for flexible metadata (claim tags, speaker metadata)
  - Native full-text search via tsvector for claim search
  - Row-level locking for safe concurrent agent operations
  - Production-ready if this scales to 30k+ docs

Walk-forward enforcement (DB level):
  The DB does NOT enforce walk-forward — that's the pipeline's job.
  But the transcript processing_order column makes the order auditable.
"""

from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Enum, ForeignKey, Integer,
    String, Text, Table, UniqueConstraint, Index, func
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ── Enums ────────────────────────────────────────────────────────────────────

class ClaimType(str, PyEnum):
    FINANCIAL_GUIDANCE    = "financial_guidance"
    OPERATIONAL_MILESTONE = "operational_milestone"
    MARKET_OUTLOOK        = "market_outlook"
    STRATEGIC_INITIATIVE  = "strategic_initiative"
    ESG_COMMITMENT        = "esg_commitment"
    MACRO_ASSUMPTION      = "macro_assumption"
    HEDGED_STATEMENT      = "hedged_statement"


class HedgeLevel(str, PyEnum):
    HARD        = "hard"
    SOFT        = "soft"
    SPECULATIVE = "speculative"


class ResolutionStatus(str, PyEnum):
    OPEN                   = "open"
    MATERIALIZED           = "materialized"
    PARTIALLY_MATERIALIZED = "partially_materialized"
    NOT_MATERIALIZED       = "not_materialized"
    SUPERSEDED             = "superseded"
    UNRESOLVABLE           = "unresolvable"


class RunStatus(str, PyEnum):
    PENDING    = "pending"
    RUNNING    = "running"
    COMPLETED  = "completed"
    FAILED     = "failed"


# ── Join table ────────────────────────────────────────────────────────────────

claim_evidence = Table(
    "claim_evidence",
    Base.metadata,
    Column("claim_id",    Integer, ForeignKey("claims.id"),   primary_key=True),
    Column("evidence_id", Integer, ForeignKey("evidence.id"), primary_key=True),
)


# ── Main tables ───────────────────────────────────────────────────────────────

class Transcript(Base):
    __tablename__ = "transcripts"

    id               = Column(Integer, primary_key=True)
    filename         = Column(String(500), nullable=False, unique=True)
    transcript_date  = Column(String(10), nullable=False)   # ISO date YYYY-MM-DD
    event_type       = Column(String(100), nullable=False)  # "Q1 Earnings Call" etc.
    fiscal_period    = Column(String(50))                   # "Q1 2024", "FY 2023"
    processing_order = Column(Integer, nullable=False)      # Chronological index (1-based)
    full_text        = Column(Text)                         # Raw extracted text
    speaker_metadata = Column(JSONB, default=dict)          # {corporate: [...], analysts: [...]}
    processed_at     = Column(DateTime, default=datetime.utcnow)
    run_id           = Column(Integer, ForeignKey("processing_runs.id"), nullable=True)

    claims           = relationship("Claim", back_populates="transcript", cascade="all, delete-orphan")
    processing_run   = relationship("ProcessingRun", back_populates="transcripts")

    __table_args__ = (
        Index("ix_transcripts_date", "transcript_date"),
    )


class Claim(Base):
    __tablename__ = "claims"

    id               = Column(Integer, primary_key=True)
    transcript_id    = Column(Integer, ForeignKey("transcripts.id"), nullable=False)

    # Immutable extraction fields
    speaker_name     = Column(String(200), nullable=False)
    speaker_title    = Column(String(300))
    raw_quote        = Column(Text, nullable=False)         # Verbatim from transcript
    claim_summary    = Column(Text, nullable=False)         # 1-sentence AI summary
    claim_type       = Column(Enum(ClaimType), nullable=False)
    hedge_level      = Column(Enum(HedgeLevel), nullable=False)
    timeframe        = Column(String(100))                  # e.g. "FY2024", "H2 2023"
    timeframe_date   = Column(String(10), nullable=True)    # ISO date best estimate
    tags             = Column(JSONB, default=list)          # ["revenue", "margin", "europe"]
    section          = Column(String(50))                   # "management_discussion" | "qa"
    extraction_run_id = Column(Integer, ForeignKey("processing_runs.id"), nullable=True)
    created_at       = Column(DateTime, default=datetime.utcnow)

    # Mutable resolution fields (updated by Resolver Agent only)
    resolution_status    = Column(Enum(ResolutionStatus), default=ResolutionStatus.OPEN, nullable=False)
    resolution_reasoning = Column(Text)
    resolution_date      = Column(String(10))               # ISO date when resolved
    resolved_by_run_id   = Column(Integer, ForeignKey("processing_runs.id"), nullable=True)

    # Relationships
    transcript    = relationship("Transcript", back_populates="claims")
    evidence_list = relationship("Evidence", secondary=claim_evidence, back_populates="claims")
    audit_log     = relationship("ClaimAuditLog", back_populates="claim", order_by="ClaimAuditLog.changed_at")

    __table_args__ = (
        Index("ix_claims_transcript",       "transcript_id"),
        Index("ix_claims_status",           "resolution_status"),
        Index("ix_claims_type",             "claim_type"),
        Index("ix_claims_timeframe_date",   "timeframe_date"),
        Index("ix_claims_speaker",          "speaker_name"),
    )


class Evidence(Base):
    """
    A quote from a later transcript that resolves or partially resolves a claim.
    Walk-forward: evidence.transcript.transcript_date > claim.transcript.transcript_date
    """
    __tablename__ = "evidence"

    id                 = Column(Integer, primary_key=True)
    source_transcript_id = Column(Integer, ForeignKey("transcripts.id"), nullable=False)
    speaker_name       = Column(String(200))
    raw_quote          = Column(Text, nullable=False)
    evidence_summary   = Column(Text)
    supports_claim     = Column(Boolean, nullable=False)    # True=materialized, False=contradicts
    created_at         = Column(DateTime, default=datetime.utcnow)

    source_transcript  = relationship("Transcript")
    claims             = relationship("Claim", secondary=claim_evidence, back_populates="evidence_list")


class ClaimAuditLog(Base):
    """
    Immutable append-only audit trail for every status change on every claim.
    This is the core explainability / auditability mechanism.
    Even if you deleted the evidence, the audit log tells you:
      - When did this claim change status?
      - Which transcript triggered it?
      - What was the before/after state?
      - What reasoning did the agent give?
    """
    __tablename__ = "claim_audit_log"

    id               = Column(Integer, primary_key=True)
    claim_id         = Column(Integer, ForeignKey("claims.id"), nullable=False)
    changed_at       = Column(DateTime, default=datetime.utcnow, nullable=False)
    source_transcript_id = Column(Integer, ForeignKey("transcripts.id"), nullable=True)
    old_status       = Column(Enum(ResolutionStatus), nullable=False)
    new_status       = Column(Enum(ResolutionStatus), nullable=False)
    reasoning        = Column(Text, nullable=False)
    agent_name       = Column(String(100))                  # Which agent made this change
    run_id           = Column(Integer, ForeignKey("processing_runs.id"), nullable=True)
    evidence_id      = Column(Integer, ForeignKey("evidence.id"), nullable=True)

    claim            = relationship("Claim", back_populates="audit_log")
    source_transcript = relationship("Transcript", foreign_keys=[source_transcript_id])
    evidence         = relationship("Evidence")
    run              = relationship("ProcessingRun")

    __table_args__ = (
        Index("ix_audit_claim_id",    "claim_id"),
        Index("ix_audit_changed_at",  "changed_at"),
    )


class ProcessingRun(Base):
    """
    Tracks every pipeline execution. One row per transcript per pipeline invocation.
    Lets you replay: "what did the system know after processing transcript 7?"
    """
    __tablename__ = "processing_runs"

    id             = Column(Integer, primary_key=True)
    run_started_at = Column(DateTime, default=datetime.utcnow)
    run_ended_at   = Column(DateTime, nullable=True)
    status         = Column(Enum(RunStatus), default=RunStatus.PENDING)
    transcript_filename = Column(String(500), nullable=True)
    claims_extracted    = Column(Integer, default=0)
    claims_resolved     = Column(Integer, default=0)
    error_message       = Column(Text, nullable=True)
    agent_trace         = Column(JSONB, default=list)  # Step-by-step agent reasoning log

    transcripts    = relationship("Transcript", back_populates="processing_run")
