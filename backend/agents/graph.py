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
import traceback
from typing import Literal, Optional

from langgraph.graph import StateGraph, END

from .state import AgentState
from .ingest_agent import run_ingest_agent
from .extraction_agent import ExtractionAgent, run_extraction_agent
from .resolution_agent import ResolutionAgent, run_resolution_agent


def route_after_ingest(state: AgentState) -> Literal["resolution", "extraction", "end"]:
    if state.get("error"):
        return "end"
    return "resolution" if state.get("open_claims") else "extraction"


def route_after_resolution(state: AgentState) -> Literal["extraction", "end"]:
    return "end" if state.get("error") else "extraction"


def build_graph(api_key: Optional[str] = None):
    """
    Compile graph once. Agents are stateless, safe to reuse across threads.
    Each node creates its own DB session inside the thread's event loop.
    """
    extraction_agent = ExtractionAgent()
    resolution_agent = ResolutionAgent()

    async def ingest_node(state: AgentState) -> dict:
        # Import here so session is created in the thread's event loop
        from ..db.session import AsyncSessionLocal
        try:
            async with AsyncSessionLocal() as db:
                result = await run_ingest_agent(state, db)
                await db.commit()
                return result
        except Exception as e:
            tb = traceback.format_exc()
            return {
                "error": f"IngestNode failed: {e}",
                "agent_trace": [f"[IngestNode] FATAL:\n{tb}"],
                "extracted_claims": [], "extraction_errors": [],
                "resolutions": [], "resolution_errors": [], "audit_entries": [],
            }

    async def resolution_node(state: AgentState) -> dict:
        from ..db.session import AsyncSessionLocal
        try:
            async with AsyncSessionLocal() as db:
                result = await run_resolution_agent(state, db, resolution_agent)
                await db.commit()
                return result
        except Exception as e:
            tb = traceback.format_exc()
            return {
                "agent_trace": [f"[ResolutionNode] FATAL:\n{tb}"],
                "resolutions": [], "resolution_errors": [str(e)], "audit_entries": [],
            }

    async def extraction_node(state: AgentState) -> dict:
        from ..db.session import AsyncSessionLocal
        try:
            async with AsyncSessionLocal() as db:
                result = await run_extraction_agent(state, db, extraction_agent)
                await db.commit()
                return result
        except Exception as e:
            tb = traceback.format_exc()
            return {
                "agent_trace": [f"[ExtractionNode] FATAL:\n{tb}"],
                "extracted_claims": [], "extraction_errors": [str(e)],
            }

    graph = StateGraph(AgentState)
    graph.add_node("ingest",     ingest_node)
    graph.add_node("resolution", resolution_node)
    graph.add_node("extraction", extraction_node)

    graph.set_entry_point("ingest")
    graph.add_conditional_edges("ingest", route_after_ingest,
        {"resolution": "resolution", "extraction": "extraction", "end": END})
    graph.add_conditional_edges("resolution", route_after_resolution,
        {"extraction": "extraction", "end": END})
    graph.add_edge("extraction", END)

    return graph.compile()
