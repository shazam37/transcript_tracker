"""
Shared state for the LangGraph pipeline.

The pipeline has 4 agents that operate in sequence per transcript:
  1. IngestAgent     — parse PDF, persist Transcript row
  2. ExtractionAgent — extract Claims from management speech
  3. ResolutionAgent — resolve open claims against new evidence
  4. AuditAgent      — write audit log entries, update run stats

State flows through the graph as a TypedDict.
Each agent reads what it needs and writes its outputs — no side effects
outside the state until the final DB-write step.

Why LangGraph instead of a simple sequential script?
  - Conditional routing: if extraction finds 0 claims, skip resolution
  - Retry logic per node: if extraction fails on one block, retry without
    aborting the entire pipeline
  - Streaming: the graph can emit events as claims are extracted, enabling
    a live progress UI
  - Checkpointing: if the process crashes mid-transcript, resume from
    the last completed node (not re-run from scratch)
  - Observability: every node transition is logged with input/output,
    making the agent's reasoning fully traceable
"""

from __future__ import annotations
from typing import TypedDict, Annotated, Optional
import operator


class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    transcript_path:      str                  # Full path to PDF
    processing_order:     int                  # Chronological index (1-based)
    run_id:               int                  # ProcessingRun.id

    # ── After IngestAgent ─────────────────────────────────────────────────
    transcript_id:        Optional[int]        # DB row id
    transcript_date:      Optional[str]        # ISO date
    event_type:           Optional[str]
    fiscal_period:        Optional[str]
    speech_blocks:        Optional[list[dict]] # [{speaker, title, text, section, is_mgmt}]
    open_claims:          Optional[list[dict]] # Claims currently OPEN from earlier transcripts

    # ── After ExtractionAgent ─────────────────────────────────────────────
    extracted_claims:     Annotated[list[dict], operator.add]  # New claims found in this transcript
    extraction_errors:    Annotated[list[str],  operator.add]

    # ── After ResolutionAgent ─────────────────────────────────────────────
    resolutions:          Annotated[list[dict], operator.add]  # [{claim_id, new_status, reasoning, evidence_quote}]
    resolution_errors:    Annotated[list[str],  operator.add]

    # ── After AuditAgent ──────────────────────────────────────────────────
    audit_entries:        Annotated[list[dict], operator.add]
    run_summary:          Optional[dict]       # Stats for this run

    # ── Control flow ──────────────────────────────────────────────────────
    error:                Optional[str]        # Fatal error aborts graph
    agent_trace:          Annotated[list[str], operator.add]  # Human-readable step log
