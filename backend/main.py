"""
Main FastAPI application.

Architecture overview:
  POST /pipeline/run      → streams SSE, runs LangGraph walk-forward pipeline
  GET  /claims            → paginated, filterable claim list
  GET  /claims/:id        → full claim detail with evidence + audit trail
  GET  /transcripts       → all processed transcripts with claim counts
  GET  /transcripts/:id   → transcript detail with all its claims
  GET  /dashboard         → aggregated stats: resolution rates, by-type, timeline
  GET  /pipeline/status   → is pipeline running, how many done
  GET  /pipeline/runs     → history of processing runs
"""
from __future__ import annotations
import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

load_dotenv()

from .db.session import create_tables
from .api.claims import router as claims_router
from .api.dashboard import router as dashboard_router
from .api.pipeline import router as pipeline_router
from .api.transcripts import router as transcripts_router


logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    for attempt in range(1, 11):
        try:
            await create_tables()
            break
        except Exception as exc:
            if attempt == 10:
                raise
            wait = min(2 ** (attempt - 1), 16)
            logger.warning("DB not ready (attempt %d/10, retrying in %ds): %s", attempt, wait, exc)
            await asyncio.sleep(wait)
    yield


app = FastAPI(
    title="Rockwool Forward-Looking Claims Tracker",
    description=(
        "Multi-agent system that extracts, tracks, and resolves forward-looking "
        "claims from Rockwool earnings calls and analyst meetings. "
        "Processes transcripts in strict chronological (walk-forward) order."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routers
app.include_router(claims_router,      prefix="/api")
app.include_router(dashboard_router,   prefix="/api")
app.include_router(pipeline_router,    prefix="/api")
app.include_router(transcripts_router, prefix="/api")


# Health check
@app.get("/api/health")
async def health():
    from .utils.llm_config import current_provider_info
    return {"status": "ok", "llm": current_provider_info()}


# Serve frontend static files
frontend_dir = os.path.join(os.path.dirname(__file__), "../frontend")
if os.path.exists(os.path.join(frontend_dir, "index.html")):
    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(os.path.join(frontend_dir, "index.html"))

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        # Don't intercept /api routes
        if full_path.startswith("api/"):
            from fastapi import HTTPException
            raise HTTPException(status_code=404)
        return FileResponse(os.path.join(frontend_dir, "index.html"))