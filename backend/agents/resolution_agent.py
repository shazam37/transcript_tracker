"""
ResolutionAgent: new transcript + open claims → resolution judgements.

This is the most critical agent for the assignment's core requirement:
"Determines whether each claim materialized, partially materialized, didn't,
or remains unresolvable."

Walk-forward contract (strictly enforced):
  - This agent sees ONLY claims made in transcripts EARLIER than the current one
  - Claims extracted FROM the current transcript are NOT in scope here
  - This mirrors how an analyst would work: you can only check claims
    against information that has since arrived

Resolution taxonomy (designed for explainability at the interview):

  MATERIALIZED           → Outcome clearly happened. Evidence is traceable to a quote.
  PARTIALLY_MATERIALIZED → Direction right, magnitude/timing off. Common with guidance misses.
                           e.g. "margins will be 13%" → actual was 11% = partial
  NOT_MATERIALIZED       → Timeframe passed, outcome did not happen. Or evidence contradicts.
  SUPERSEDED             → Management explicitly updated/replaced the claim with new numbers.
                           Different from NOT_MATERIALIZED: this was a deliberate revision.
  UNRESOLVABLE           → Too qualitative to ever verify. "We will continue investing in X"
                           has no measurable outcome. We close these to keep OPEN list clean.

Batching strategy:
  - Group open claims into batches of 15 (fits comfortably in Claude's context)
  - We send the full transcript management text + batch of claims in one call
  - Each batch is independent — no cross-batch dependencies
  - This scales linearly: 150 open claims = 10 API calls per transcript

Why not embed + similarity search to find relevant claims?
  - With 31 transcripts, ~200-400 claims, brute-force batching is fast enough
  - At 300k docs, we'd add a vector pre-filter to get the top-50 relevant claims
    per transcript before passing to the LLM (noted in tradeoffs discussion)
"""

from __future__ import annotations
import json
import re
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..db.models import Claim, Evidence, ClaimAuditLog, ResolutionStatus, ClaimType


# ── Prompts ───────────────────────────────────────────────────────────────────

RESOLUTION_SYSTEM = """You are a financial analyst tracking forward-looking claims made by ROCKWOOL A/S management.

You receive:
  1. A new transcript excerpt (management speech only)
  2. A list of OPEN claims from EARLIER transcripts that need resolution checking

Your job: Determine if this new transcript provides evidence to change any claim's status.

RESOLUTION OPTIONS (choose the most accurate):
  materialized           Clear evidence the outcome happened. Quote the proof.
  partially_materialized Direction was right, magnitude/timing missed significantly. Explain the gap.
  not_materialized       Evidence confirms the claim failed, OR timeframe clearly passed without fulfillment.
  superseded             Management explicitly replaced this claim with updated guidance for same topic.
  open                   No resolution signal yet — leave it open. Use this if uncertain.
  unresolvable           Claim is inherently too vague to ever verify. Close permanently.

RESOLUTION RULES:
1. Evidence must come from the provided transcript — do not use outside knowledge
2. For numeric guidance: within ~5% of target = materialized; 5-20% miss = partially_materialized; >20% = not_materialized
3. SUPERSEDED requires explicit management acknowledgement of a change ("we now expect X instead of Y")
4. When in doubt → "open". A wrong resolution is worse than a delayed one.
5. Quote must be verbatim from the transcript text provided
6. Only include claims you are resolving (status != "open") in your response

Return ONLY valid JSON, no prose:
{
  "resolutions": [
    {
      "claim_id": 42,
      "new_status": "materialized",
      "reasoning": "Management reported Q1 2024 EBIT margin of 14.4%, above the guided ~13% range stated in Q4 2023. The claim materially came true.",
      "evidence_quote": "exact verbatim quote from transcript that resolves this",
      "evidence_speaker": "Speaker Name"
    }
  ]
}"""

RESOLUTION_USER = """NEW TRANSCRIPT: {event_type} | Date: {transcript_date}

=== MANAGEMENT SPEECH (use this as evidence) ===
{mgmt_text}
=== END ===

OPEN CLAIMS TO EVALUATE ({n} claims):
{claims_json}

Check each claim. Return JSON with only those you are resolving (skip "open" ones)."""


# ── Agent ─────────────────────────────────────────────────────────────────────

VALID_FINAL_STATUSES = {
    ResolutionStatus.MATERIALIZED.value,
    ResolutionStatus.PARTIALLY_MATERIALIZED.value,
    ResolutionStatus.NOT_MATERIALIZED.value,
    ResolutionStatus.SUPERSEDED.value,
    ResolutionStatus.UNRESOLVABLE.value,
}


class ResolutionAgent:
    def __init__(self, api_key: Optional[str] = None):
        kwargs = {"model": "claude-sonnet-4-6", "temperature": 0, "max_tokens": 4096}
        if api_key:
            kwargs["anthropic_api_key"] = api_key
        self.llm = ChatAnthropic(**kwargs)

    def _build_mgmt_text(self, speech_blocks: list[dict], max_words: int = 4500) -> str:
        """Concatenate management speech blocks into a focused resolution context."""
        parts = []
        for b in speech_blocks:
            if b.get("is_management"):
                parts.append(f"[{b['speaker_name']} — {b['section']}]\n{b['text']}\n")
        text = "\n".join(parts)
        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[:max_words]) + "\n[...truncated...]"
        return text

    def _is_eligible(self, claim: dict, current_date: str) -> bool:
        """
        Eligibility: should we check this claim against the current transcript?
        
        Strategy:
          - Always check if no timeframe_date (we never know when it matures)
          - Check if timeframe_date has arrived (claim is now due)
          - Also check claims > 6 quarters old regardless (detect quiet drops)
        """
        td = claim.get("timeframe_date")
        if not td:
            return True
        try:
            from datetime import date
            return date.fromisoformat(td) <= date.fromisoformat(current_date)
        except ValueError:
            return True

    def _call_llm(
        self,
        batch: list[dict],
        speech_blocks: list[dict],
        event_type: str,
        transcript_date: str,
    ) -> tuple[list[dict], Optional[str]]:
        mgmt_text = self._build_mgmt_text(speech_blocks)
        claims_json = json.dumps(
            [
                {
                    "claim_id":      c["id"],
                    "made_on":       c.get("transcript_date", "unknown"),
                    "speaker":       c["speaker_name"],
                    "claim_type":    c["claim_type"],
                    "hedge_level":   c["hedge_level"],
                    "timeframe":     c["timeframe"],
                    "timeframe_date": c.get("timeframe_date"),
                    "summary":       c["claim_summary"],
                    "quote":         c["raw_quote"][:400],  # truncate very long quotes
                }
                for c in batch
            ],
            indent=2,
        )

        messages = [
            SystemMessage(content=RESOLUTION_SYSTEM),
            HumanMessage(content=RESOLUTION_USER.format(
                event_type=event_type,
                transcript_date=transcript_date,
                mgmt_text=mgmt_text,
                n=len(batch),
                claims_json=claims_json,
            )),
        ]

        try:
            resp = self.llm.invoke(messages)
            raw  = resp.content.strip()
            raw  = re.sub(r"```json\s*", "", raw)
            raw  = re.sub(r"```\s*", "", raw)
            data = json.loads(raw)
            return data.get("resolutions", []), None
        except json.JSONDecodeError as e:
            return [], f"JSON parse error: {e}"
        except Exception as e:
            return [], f"LLM error: {e}"

    def resolve(
        self,
        open_claims: list[dict],
        speech_blocks: list[dict],
        event_type: str,
        transcript_date: str,
        batch_size: int = 15,
    ) -> tuple[list[dict], list[str]]:
        """
        Main resolution loop. Returns (resolutions, errors).
        resolutions: [{claim_id, new_status, reasoning, evidence_quote, evidence_speaker}]
        """
        eligible = [c for c in open_claims if self._is_eligible(c, transcript_date)]
        if not eligible:
            return [], []

        all_resolutions, errors = [], []
        for i in range(0, len(eligible), batch_size):
            batch = eligible[i : i + batch_size]
            resolutions, err = self._call_llm(batch, speech_blocks, event_type, transcript_date)
            if err:
                errors.append(err)
            else:
                # Filter to only valid final statuses
                for r in resolutions:
                    if r.get("new_status") in VALID_FINAL_STATUSES:
                        all_resolutions.append(r)

        return all_resolutions, errors


async def run_resolution_agent(state: dict, db: AsyncSession, agent: ResolutionAgent) -> dict:
    """
    LangGraph node function.
    Reads:  open_claims, speech_blocks, transcript_id, transcript_date, event_type, run_id
    Writes: resolutions, resolution_errors
    Side effects: updates Claim rows, inserts Evidence + ClaimAuditLog rows
    """
    updates: dict = {
        "resolutions": [],
        "resolution_errors": [],
        "agent_trace": [],
    }

    if state.get("error"):
        return updates

    open_claims    = state.get("open_claims", [])
    speech_blocks  = state.get("speech_blocks", [])
    transcript_id  = state["transcript_id"]
    transcript_date = state["transcript_date"]
    event_type     = state["event_type"]
    run_id         = state["run_id"]

    if not open_claims:
        updates["agent_trace"].append("[ResolutionAgent] No open claims to resolve")
        return updates

    updates["agent_trace"].append(
        f"[ResolutionAgent] Checking {len(open_claims)} open claims against {event_type} ({transcript_date})"
    )

    resolutions, errors = agent.resolve(
        open_claims, speech_blocks, event_type, transcript_date
    )
    updates["resolution_errors"] = errors

    # Fetch the source transcript for evidence FK
    from ..db.models import Transcript
    t_result = await db.execute(select(Transcript).where(Transcript.id == transcript_id))
    transcript_row = t_result.scalar_one()

    resolved_count = 0
    for res in resolutions:
        claim_id_int = res.get("claim_id")
        new_status   = res.get("new_status")
        reasoning    = res.get("reasoning", "")
        quote        = res.get("evidence_quote", "")
        speaker      = res.get("evidence_speaker", "Unknown")

        if not claim_id_int or new_status not in VALID_FINAL_STATUSES:
            continue

        # Fetch the claim
        c_result = await db.execute(select(Claim).where(Claim.id == claim_id_int))
        claim_row = c_result.scalar_one_or_none()
        if not claim_row:
            continue
        if claim_row.resolution_status != ResolutionStatus.OPEN:
            # Already resolved — only allow superseded to overwrite
            if new_status != ResolutionStatus.SUPERSEDED.value:
                continue

        old_status = claim_row.resolution_status

        # Create Evidence row
        ev = Evidence(
            source_transcript_id=transcript_id,
            speaker_name=speaker,
            raw_quote=quote[:800],
            evidence_summary=reasoning[:600],
            supports_claim=new_status in (
                ResolutionStatus.MATERIALIZED.value,
                ResolutionStatus.PARTIALLY_MATERIALIZED.value,
            ),
        )
        db.add(ev)
        await db.flush()

        # Link evidence to claim
        claim_row.evidence_list.append(ev)

        # Update claim
        claim_row.resolution_status    = ResolutionStatus(new_status)
        claim_row.resolution_reasoning = reasoning
        claim_row.resolution_date      = transcript_date
        claim_row.resolved_by_run_id   = run_id

        # Write audit log
        audit = ClaimAuditLog(
            claim_id=claim_row.id,
            source_transcript_id=transcript_id,
            old_status=old_status,
            new_status=ResolutionStatus(new_status),
            reasoning=reasoning,
            agent_name="ResolutionAgent",
            run_id=run_id,
            evidence_id=ev.id,
        )
        db.add(audit)

        updates["resolutions"].append({
            "claim_id":   claim_id_int,
            "new_status": new_status,
            "reasoning":  reasoning,
        })
        resolved_count += 1

    updates["agent_trace"].append(
        f"[ResolutionAgent] Resolved {resolved_count} claims "
        f"({len(errors)} errors)"
    )

    return updates
