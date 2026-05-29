"""
ExtractionAgent: management speech blocks → structured Claim objects.

LLM Strategy:
  - Model: claude-sonnet-4-6 (accuracy > speed for this task) or GPT-OSS-120B
  - Temperature: 0 (reproducibility — same transcript should always produce same claims)
  - One API call per management speech block (not per transcript)
    WHY: shorter context = sharper focus = fewer hallucinated claims
         also allows per-block retry without reprocessing entire transcript
  - We extract from ALL management blocks including Q&A answers
    (management often makes commitments in response to analyst questions)
  - Structured output: we prompt for JSON and validate every field

Claim Taxonomy (explained in prompt, enforced at validation):
  financial_guidance    → hard numbers, revenue/margin/EBIT targets
  operational_milestone → factories, capacity, product launches  
  market_outlook        → demand, volume, pricing expectations
  strategic_initiative  → M&A, restructuring, new business areas
  esg_commitment        → carbon, sustainability, social KPIs
  macro_assumption      → external conditions cited as basis for guidance
  hedged_statement      → strongly qualified ("if X allows...") — low falsifiability

Hedge levels explain confidence calibration:
  hard        → "we will achieve", explicit numeric target
  soft        → "we expect", directional without tight bound
  speculative → "could", "might", "if conditions hold"

We track hedged_statement + speculative separately because:
  - They're still useful for detecting whether management over-hedges
  - They reveal what management THINKS will happen even if not committed
  - Pattern: claims that start as speculative often harden in later calls
"""

from __future__ import annotations
import asyncio
import json
import re
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage
from ..utils.llm_config import get_llm

from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import Claim, ClaimType, HedgeLevel, ResolutionStatus


# ─────────────────────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM = """You are a senior financial analyst specializing in forward-looking statement analysis for ROCKWOOL A/S (Danish building materials company listed on Copenhagen Stock Exchange: ROCK.B.DK).

TASK: Extract ALL forward-looking statements from the management speech excerpt provided.

WHAT COUNTS AS A FORWARD-LOOKING CLAIM:
✓ Any statement about what the company expects, plans, intends, or predicts will happen
✓ Guidance (revenue, margins, volumes, CapEx) for future periods
✓ Planned capacity expansions, factory openings, product launches
✓ Management's view on future market conditions (demand, pricing, competition)
✓ Strategic intentions (M&A, restructuring, entering new markets)
✓ ESG/sustainability targets and commitments
✓ External assumptions cited as basis for guidance (energy prices, construction activity)
✓ Implicit futures: "margins will recover", "the market should normalize"

WHAT TO EXCLUDE:
✗ Historical facts ("revenues grew 12% last year")
✗ Current state descriptions ("we currently have 46 factories")  
✗ General company descriptions
✗ Legal disclaimers

CLAIM TYPES — use EXACTLY these values:
  financial_guidance      Revenue, margin %, EBIT, EBITDA, CapEx, EPS guidance with numbers
  operational_milestone   Factory openings, capacity additions (tonnes/m²), product launches, headcount
  market_outlook          Future demand, volume trends, pricing power, geographic growth expectations  
  strategic_initiative    M&A intentions, restructuring, new market entry, divestments, partnerships
  esg_commitment          Carbon targets, Science Based Targets, water/waste/safety KPIs
  macro_assumption        Energy prices, construction rates, regulation, geopolitical conditions cited as inputs
  hedged_statement        Heavily qualified statements with "if", "provided", "assuming", "conditions permitting"

HEDGE LEVELS:
  hard        "we will", "we expect to achieve X%", explicit numeric range → strong commitment
  soft        "we expect", "we should see", directional without tight bound
  speculative "could", "might", "it's possible", "if X happens then Y"

TIMEFRAME: Extract exactly as stated. Examples:
  "FY2024", "H2 2023", "by year-end", "within 2 years", "Q1 2025", "long-term", "2025-2027", "next quarter"
  If no timeframe stated: "unspecified"

TIMEFRAME_DATE: Your best ISO date estimate for when this claim matures.
  "FY2024" → "2024-12-31"
  "Q2 2023" → "2023-06-30"  
  "by year-end" (said in Feb 2024) → "2024-12-31"
  "long-term" / "unspecified" → null

TAGS: 2-5 lowercase topic tags from: revenue, margin, ebit, volume, pricing, capacity, factory, 
  europe, north-america, asia, carbon, esg, safety, energy, construction, renovation, insulation,
  systems, guidance, outlook, capex, free-cash-flow, dividend, acquisition, restructuring

CRITICAL RULES:
1. raw_quote must be VERBATIM text from the transcript — no paraphrasing
2. claim_summary: one present-tense sentence summarizing what management expects/plans
3. Split compound sentences into separate claims if they make distinct claims
4. A claim about "revenue AND margins" = two claims
5. If no forward-looking claims exist, return empty array

Return ONLY valid JSON, zero prose:
{
  "claims": [
    {
      "raw_quote": "exact verbatim text from transcript",
      "claim_summary": "Management expects/plans/guides for X by Y",
      "claim_type": "financial_guidance",
      "hedge_level": "hard",
      "timeframe": "FY2024",
      "timeframe_date": "2024-12-31",
      "tags": ["revenue", "margin", "europe"]
    }
  ]
}"""


EXTRACTION_USER = """Transcript: {event_type} | Date: {transcript_date}
Speaker: {speaker_name} ({speaker_title})  
Section: {section}

--- SPEECH ---
{speech_text}
--- END ---

Extract forward-looking claims. Return JSON only."""


# ─────────────────────────────────────────────────────────────────────────────

VALID_CLAIM_TYPES  = {e.value for e in ClaimType}
VALID_HEDGE_LEVELS = {e.value for e in HedgeLevel}


def _validate(raw: dict) -> Optional[dict]:
    if not raw.get("raw_quote") or not raw.get("claim_summary"):
        return None
    ct = raw.get("claim_type", "")
    if ct not in VALID_CLAIM_TYPES:
        ct = ClaimType.HEDGED_STATEMENT.value
    hl = raw.get("hedge_level", "")
    if hl not in VALID_HEDGE_LEVELS:
        hl = HedgeLevel.SOFT.value
    tags = raw.get("tags", [])
    if not isinstance(tags, list): tags = []

    return {
        "raw_quote":      str(raw["raw_quote"])[:1500],
        "claim_summary":  str(raw["claim_summary"])[:600],
        "claim_type":     ct,
        "hedge_level":    hl,
        "timeframe":      str(raw.get("timeframe", "unspecified"))[:100],
        "timeframe_date": raw.get("timeframe_date"),
        "tags":           [str(t).lower()[:40] for t in tags[:8]],
    }


class ExtractionAgent:
    def __init__(self, api_key: Optional[str] = None):
        # api_key arg kept for backwards compat but provider is read from env
        self.llm = get_llm(temperature=0, max_tokens=4096)

    @staticmethod
    def _is_rate_limit(exc: Exception) -> bool:
        s = str(exc).lower()
        return "429" in str(exc) or "rate_limit" in s or "rate limit" in s or "too many" in s

    async def _call(self, block: dict, event_type: str, transcript_date: str) -> tuple[list[dict], Optional[str]]:
        """Single LLM call for one speech block. Returns (raw_claims, error)."""
        text = block["text"]
        words = text.split()

        # Chunk long blocks (>2500 words) with overlap
        chunks = []
        if len(words) > 2500:
            for i in range(0, len(words), 2300):
                chunks.append(" ".join(words[i:i + 2500]))
        else:
            chunks = [text]

        all_raw = []
        for chunk in chunks:
            messages = [
                SystemMessage(content=EXTRACTION_SYSTEM),
                HumanMessage(content=EXTRACTION_USER.format(
                    event_type=event_type,
                    transcript_date=transcript_date,
                    speaker_name=block["speaker_name"],
                    speaker_title=block["speaker_title"],
                    section=block["section"],
                    speech_text=chunk,
                )),
            ]
            last_err: Optional[str] = None
            for attempt in range(4):  # up to 4 attempts: 0, 1, 2, 3
                try:
                    resp = await self.llm.ainvoke(messages)
                    raw  = resp.content.strip()
                    raw  = re.sub(r"```json\s*", "", raw)
                    raw  = re.sub(r"```\s*", "", raw)
                    data = json.loads(raw)
                    all_raw.extend(data.get("claims", []))
                    last_err = None
                    break
                except json.JSONDecodeError as e:
                    last_err = f"JSON parse error on block [{block['speaker_name']}]: {e}"
                    break  # JSON errors are not transient — don't retry
                except Exception as e:
                    last_err = f"LLM error on block [{block['speaker_name']}]: {e}"
                    if self._is_rate_limit(e) and attempt < 3:
                        wait = 15 * (attempt + 1)  # 15s, 30s, 45s — only if still rate limited
                        await asyncio.sleep(wait)
                        continue
                    break
            if last_err:
                return [], last_err

        return all_raw, None

    async def extract_from_blocks(
        self,
        speech_blocks:  list[dict],
        event_type:     str,
        transcript_date: str,
    ) -> tuple[list[dict], list[str]]:
        """
        Extract claims from all management speech blocks.
        Returns (validated_claims, errors).
        """
        all_claims, errors = [], []

        # Process management blocks only (analysts don't make company claims)
        mgmt_blocks = [b for b in speech_blocks if b.get("is_management")]

        for block in mgmt_blocks:
            if len(block["text"].split()) < 25:
                continue
            await asyncio.sleep(1.5)  # ~30 req/min proactive throttle for Groq free tier
            raw_claims, err = await self._call(block, event_type, transcript_date)
            if err:
                errors.append(err)
                continue
            for raw in raw_claims:
                validated = _validate(raw)
                if validated:
                    validated["speaker_name"]  = block["speaker_name"]
                    validated["speaker_title"] = block["speaker_title"]
                    validated["section"]       = block["section"]
                    all_claims.append(validated)

        return all_claims, errors


async def run_extraction_agent(state: dict, db: AsyncSession, agent: ExtractionAgent) -> dict:
    """
    LangGraph node function.
    Reads:  speech_blocks, transcript_id, transcript_date, event_type, run_id
    Writes: extracted_claims, extraction_errors
    """
    updates: dict = {
        "extracted_claims": [],
        "extraction_errors": [],
        "agent_trace": [],
    }

    if state.get("error"):
        return updates

    try:
        speech_blocks    = state.get("speech_blocks", [])
        transcript_id    = state["transcript_id"]
        transcript_date  = state["transcript_date"]
        event_type       = state["event_type"]
        run_id           = state["run_id"]

        mgmt_count = sum(1 for b in speech_blocks if b.get("is_management"))
        updates["agent_trace"].append(
            f"[ExtractionAgent] {len(speech_blocks)} blocks total, {mgmt_count} management"
        )

        raw_claims, errors = await agent.extract_from_blocks(
            speech_blocks, event_type, transcript_date
        )
        updates["extraction_errors"] = errors

        # Persist to DB
        seq = 1
        for c in raw_claims:
            claim_row = Claim(
                transcript_id=transcript_id,
                speaker_name=c["speaker_name"],
                speaker_title=c["speaker_title"],
                raw_quote=c["raw_quote"],
                claim_summary=c["claim_summary"],
                claim_type=ClaimType(c["claim_type"]),
                hedge_level=HedgeLevel(c["hedge_level"]),
                timeframe=c["timeframe"],
                timeframe_date=c.get("timeframe_date"),
                tags=c["tags"],
                section=c["section"],
                resolution_status=ResolutionStatus.OPEN,
                extraction_run_id=run_id,
            )
            db.add(claim_row)
            await db.flush()

            c["db_id"] = claim_row.id
            updates["extracted_claims"].append(c)
            seq += 1

        updates["agent_trace"].append(
            f"[ExtractionAgent] Extracted {len(raw_claims)} claims "
            f"({len(errors)} errors)"
        )

    except Exception as e:
        updates["agent_trace"].append(f"[ExtractionAgent] ERROR: {e}")
        updates["extraction_errors"].append(str(e))

    return updates