"""
LangGraph pipeline graph.

Graph topology:
                    ┌──────────────┐
                    │  IngestAgent │  Parse PDF, snapshot open claims
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ ResolutionAgent│  Resolve OPEN claims against new transcript
                    └──────┬───────┘    (BEFORE extracting new claims — walk-forward)
                           │
                    ┌──────▼────────┐
                    │ExtractionAgent│  Extract NEW claims from this transcript
                    └──────┬────────┘
                           │
                    ┌──────▼──────┐
                    │   END        │  State is persisted by the caller
                    └─────────────┘

Why resolution BEFORE extraction?
  Walk-forward integrity. If we extracted new claims first, they'd be visible
  to the resolver in the same pass. That would let a claim made in transcript N
  be "resolved" by evidence also in transcript N — which is cheating.
  By resolving first, new claims from this transcript only become eligible
  for resolution when the NEXT transcript is processed.

Conditional routing:
  - If IngestAgent returns error → skip resolution + extraction, go to END
  - If no open claims → skip resolution (nothing to resolve), go to extraction
  - If no management speech → skip extraction, go to END

The graph is compiled once at startup, then invoked per transcript.
LangGraph handles the state merging (Annotated[list, operator.add] fields
accumulate across nodes automatically).
"""

from __future__ import annotations
import os
from functools import partial
from typing import Literal, Optional

from langgraph.graph import StateGraph, END
from sqlalchemy.ext.asyncio import AsyncSession

from .state import AgentState
from .ingest_agent import run_ingest_agent
from .extraction_agent import ExtractionAgent, run_extraction_agent
from .resolution_agent import ResolutionAgent, run_resolution_agent


# ── Routing functions ─────────────────────────────────────────────────────────

def route_after_ingest(state: AgentState) -> Literal["resolution", "extraction", "end"]:
    """Route after ingestion. Skip resolution if no open claims."""
    if state.get("error"):
        return "end"
    open_claims = state.get("open_claims") or []
    if len(open_claims) > 0:
        return "resolution"
    return "extraction"   # First transcript ever: no claims to resolve yet


def route_after_resolution(state: AgentState) -> Literal["extraction", "end"]:
    if state.get("error"):
        return "end"
    return "extraction"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(
    db: AsyncSession,
    api_key: Optional[str] = None,
) -> StateGraph:
    """
    Build and compile the LangGraph pipeline.
    db and agents are injected here so the graph nodes are pure async functions.
    """
    extraction_agent = ExtractionAgent(api_key=api_key)
    resolution_agent = ResolutionAgent(api_key=api_key)

    # Wrap agent functions with injected dependencies
    async def ingest_node(state: AgentState) -> dict:
        return await run_ingest_agent(state, db)

    async def resolution_node(state: AgentState) -> dict:
        return await run_resolution_agent(state, db, resolution_agent)

    async def extraction_node(state: AgentState) -> dict:
        return await run_extraction_agent(state, db, extraction_agent)

    # Build graph
    graph = StateGraph(AgentState)
    graph.add_node("ingest",      ingest_node)
    graph.add_node("resolution",  resolution_node)
    graph.add_node("extraction",  extraction_node)

    graph.set_entry_point("ingest")

    graph.add_conditional_edges(
        "ingest",
        route_after_ingest,
        {
            "resolution": "resolution",
            "extraction": "extraction",
            "end":        END,
        },
    )
    graph.add_conditional_edges(
        "resolution",
        route_after_resolution,
        {
            "extraction": "extraction",
            "end":        END,
        },
    )
    graph.add_edge("extraction", END)

    return graph.compile()
