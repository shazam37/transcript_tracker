# Rockwool Forward-Looking Claims Tracker

Multi-agent system that extracts, tracks, and resolves forward-looking claims from Rockwool A/S earnings transcripts, processing them in strict chronological (walk-forward) order.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  FastAPI (backend/main.py)                                  │
│  ├── GET  /api/claims          paginated + filterable        │
│  ├── GET  /api/claims/:id      full detail + audit trail    │
│  ├── GET  /api/transcripts     all 31 transcripts           │
│  ├── GET  /api/dashboard       aggregated stats + charts    │
│  ├── GET  /api/pipeline/status                              │
│  └── POST /api/pipeline/run    SSE streaming pipeline       │
│                                                             │
│  LangGraph Pipeline (per transcript, walk-forward)          │
│  ┌──────────┐  ┌─────────────────┐  ┌──────────────────┐   │
│  │  Ingest  │→ │   Resolution    │→ │   Extraction     │   │
│  │  Agent   │  │   Agent         │  │   Agent          │   │
│  │          │  │ (resolves OPEN  │  │ (extracts NEW    │   │
│  │ Parse PDF│  │  claims FIRST)  │  │  claims SECOND)  │   │
│  └──────────┘  └─────────────────┘  └──────────────────┘   │
│       │                │                     │              │
│  PostgreSQL: transcripts, claims, evidence, audit_log       │
└─────────────────────────────────────────────────────────────┘
          │
    Frontend (frontend/index.html)
    Single-page app: Dashboard / Claims Explorer /
    Transcripts Timeline / Pipeline runner + live log
```

### Walk-Forward Enforcement

**The core invariant:** when processing transcript N, only transcripts 1..N-1 are used for resolution.

Enforced in `IngestAgent`: it snapshots all `OPEN` claims **before** `ExtractionAgent` adds new ones from this transcript. New claims from transcript N only become eligible for resolution when transcript N+1 is processed.

---

## Setup

```bash
# 1. Clone and enter directory
cd rockwool_tracker

# 2. Copy env
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY

# 3. Start Postgres
docker-compose up db -d

# 4. Install deps
pip install -r requirements.txt

# 5. Run the API
uvicorn backend.main:app --reload --port 8000

# 6. Open browser
open http://localhost:8000

# 7. Run pipeline (from UI or CLI)
python scripts/run_pipeline.py
```

Or with Docker (full stack):
```bash
docker-compose up --build
```

---

## Claim Taxonomy

| Type | Description | Falsifiability |
|------|-------------|----------------|
| `financial_guidance` | Revenue, margin, EBIT targets with numbers | High — verified against subsequent reported results |
| `operational_milestone` | Factory openings, capacity additions, launches | High — binary happened/didn't |
| `market_outlook` | Demand, volume, pricing expectations | Medium — verified directionally |
| `strategic_initiative` | M&A, restructuring, new markets | Medium — long horizon, sometimes multi-year |
| `esg_commitment` | Carbon, sustainability KPIs | Medium — annual reports needed |
| `macro_assumption` | External conditions cited as inputs | Low — not management commitments |
| `hedged_statement` | Strongly qualified ("if conditions allow") | Low — designed to be unfalsifiable |

### Hedge Levels
- **hard** — explicit commitment, numeric range, "we will"
- **soft** — "we expect", directional without tight bound
- **speculative** — "could", "might", "if X then Y"

### Resolution Statuses
- **materialized** — confirmed by later evidence, traceable to a quote
- **partially_materialized** — direction right, magnitude/timing off (5–20% miss)
- **not_materialized** — contradicted or timeframe passed without fulfillment
- **superseded** — management explicitly revised the claim (different from a miss)
- **unresolvable** — too qualitative to verify, or no data ever appeared
- **open** — still within timeframe, no resolution signal yet

---

## Discussion Points

### Taxonomy & Hedged Language
We track hedged_statement as its own type rather than discarding them. Rationale:
- They reveal what management *thinks* will happen even when not committing
- Pattern detection: claims that start speculative often harden in later calls
- Management confidence calibration: how often does "should" become "will"?

For unfalsifiable language like "we will continue investing in innovation" — we mark as `unresolvable` immediately at extraction time, keeping the OPEN list clean and actionable.

### Evaluation Approach
1. **Manual spot-check**: read the raw quote + resolution reasoning for 20 sampled claims — does the LLM's reasoning match what you'd conclude from the transcript?
2. **Coverage**: count claims per transcript against manual reading of the same transcript
3. **False positive rate**: how many "claims" are actually historical statements?
4. **Resolution accuracy**: compare LLM-resolved claims against analyst consensus (e.g., reported EBIT vs guided EBIT)

### Known Failure Modes
- **Long CEO speeches get chunked** — a claim that spans two chunks may be missed or duplicated
- **Multi-year claims are hard to resolve** — "by 2030" claims stay OPEN for years, polluting the signal
- **Implicit revisions**: when management drops a commitment without saying so, we may not mark it SUPERSEDED
- **Speaker attribution**: analyst calls sometimes have poor speaker headers in the FactSet PDFs, causing misattribution
- **Non-standard PDFs**: some transcripts (AGMs) have less structured speech patterns

### Scaling to 30k / 300k / 3M Documents

| Scale | Approach |
|-------|----------|
| Current (31 docs) | Batch all OPEN claims per transcript into LLM calls |
| 30k docs | Add vector DB (pgvector) — embed claim summaries, retrieve top-K relevant claims per new transcript before LLM resolution. Parallel processing per document. |
| 300k docs | Kafka/queue-based ingestion pipeline. Dedicated extraction workers. Claim deduplication (same claim repeated across transcripts). Company-level segmentation. |
| 3M docs | Hierarchical resolution: cheap small model does first pass (materialized/not), expensive model only on ambiguous cases. Streaming ingestion, incremental index updates. Distributed state store (not JSON). |

**Hard limit of current approach**: LLM context window. At 300 open claims, each resolution call sends all 300+ to Claude. Beyond ~500 claims, we need vector retrieval to pre-filter to the 20-30 most relevant before LLM scoring.

### What I'd Build Next
1. **Vector similarity for claim deduplication** — detect when management repeats the same claim across 4 quarters
2. **Web search integration** — auto-resolve claims using earnings press releases and annual report filings
3. **Confidence scoring** — instead of binary resolution, score 0-1 how well evidence matches
4. **Claim clustering** — group related claims (all "2024 revenue guidance" claims) into threads
5. **Alert system** — notify when a previously OPEN hard-commitment claim's timeframe has passed without resolution

---

## File Structure

```
rockwool_tracker/
├── backend/
│   ├── main.py                    FastAPI app, routes, lifespan
│   ├── agents/
│   │   ├── state.py               LangGraph TypedDict state
│   │   ├── ingest_agent.py        PDF parse + snapshot open claims
│   │   ├── extraction_agent.py    LLM claim extraction
│   │   ├── resolution_agent.py    LLM walk-forward resolution
│   │   ├── graph.py               LangGraph graph + routing
│   │   └── pipeline.py            Walk-forward SSE streaming runner
│   ├── api/
│   │   ├── claims.py              Claims CRUD + search
│   │   ├── dashboard.py           Stats aggregation
│   │   ├── pipeline.py            Pipeline run/status/history
│   │   └── transcripts.py        Transcript list + detail
│   ├── db/
│   │   ├── models.py              SQLAlchemy ORM + enums
│   │   └── session.py             Async engine + get_db
│   ├── ingestion/
│   │   └── parser.py              PyMuPDF PDF parser
│   └── schemas/
│       └── schemas.py             Pydantic v2 schemas
├── frontend/
│   └── index.html                 Single-page app (no build step)
├── scripts/
│   └── run_pipeline.py            CLI runner with rich output
├── data/raw_transcripts/          31 Rockwool PDFs
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```
