#!/usr/bin/env python3
"""
CLI runner — use this to execute the pipeline directly from terminal.

Usage:
  python scripts/run_pipeline.py                    # full run, resume=True
  python scripts/run_pipeline.py --no-resume        # start fresh
  python scripts/run_pipeline.py --limit 3          # first 3 transcripts only (testing)
  python scripts/run_pipeline.py --stats            # print stats only, no processing
"""
import asyncio
import os
import sys
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich import box

console = Console()


async def run_and_stream(resume: bool, limit: int | None):
    from rockwool_transcript_tracker.backend.agents.pipeline import run_pipeline_streaming, TRANSCRIPTS_DIR
    from rockwool_transcript_tracker.backend.db.session import create_tables

    await create_tables()

    claims_extracted = 0
    claims_resolved  = 0
    current_file     = ""

    console.print(Panel.fit(
        "[bold cyan]Rockwool Forward-Looking Claims Tracker[/bold cyan]\n"
        "Walk-forward multi-agent pipeline",
        border_style="cyan"
    ))

    async for raw in run_pipeline_streaming(
        transcripts_dir=TRANSCRIPTS_DIR,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        resume=resume,
    ):
        evt = json.loads(raw)
        kind = evt.get("event")

        if kind == "pipeline_start":
            console.print(f"[green]▶ Pipeline started[/green]")

        elif kind == "pipeline_info":
            total = evt.get("total_transcripts", "?")
            msg   = evt.get("message", "")
            console.print(f"  {msg or f'{total} transcripts to process'}")

        elif kind == "transcript_start":
            current_file = evt.get("filename", "")
            idx   = evt.get("index", "?")
            total = evt.get("total", "?")
            console.rule(f"[bold]{idx}/{total}[/bold] {current_file[:60]}")

        elif kind == "agent_trace":
            console.print(f"  [dim]{evt.get('message', '')}[/dim]")

        elif kind == "claim_extracted":
            claims_extracted += 1
            ctype  = evt.get("claim_type", "")
            spk    = evt.get("speaker", "")
            hedge  = evt.get("hedge_level", "")
            summ   = evt.get("summary", "")[:90]
            color  = {"hard": "green", "soft": "yellow", "speculative": "dim"}.get(hedge, "white")
            console.print(f"  [cyan]+[/cyan] [{color}]{ctype}[/{color}] [{spk}] {summ}")

        elif kind == "claim_resolved":
            claims_resolved += 1
            cid    = evt.get("claim_id", "?")
            status = evt.get("new_status", "?")
            reason = evt.get("reasoning", "")[:100]
            color  = {
                "materialized": "green",
                "partially_materialized": "yellow",
                "not_materialized": "red",
                "superseded": "magenta",
                "unresolvable": "dim",
            }.get(status, "white")
            console.print(f"  [bold {color}]✓ resolved #{cid} → {status}[/bold {color}]  {reason}")

        elif kind == "transcript_done":
            ex  = evt.get("claims_extracted", 0)
            res = evt.get("claims_resolved", 0)
            errs = evt.get("errors", [])
            console.print(
                f"  [green]Done[/green] — extracted: {ex}, resolved: {res}"
                + (f", errors: {len(errs)}" if errs else "")
            )

        elif kind == "transcript_error":
            console.print(f"  [red]ERROR:[/red] {evt.get('error', '')}")

        elif kind == "pipeline_done":
            console.print(f"\n[bold green]✓ Pipeline complete[/bold green] — "
                          f"total extracted: {claims_extracted}, "
                          f"total resolved: {claims_resolved}")

        elif kind == "error":
            console.print(f"[bold red]Fatal: {evt.get('message', '')}[/bold red]")
            break

        if limit and claims_extracted >= limit * 10:  # rough limit by claim volume
            break


async def print_stats():
    from rockwool_transcript_tracker.backend.db.session import AsyncSessionLocal
    from rockwool_transcript_tracker.backend.db.models import Claim, Transcript, ResolutionStatus
    from sqlalchemy import select, func

    async with AsyncSessionLocal() as db:
        total_claims = (await db.execute(select(func.count(Claim.id)))).scalar_one()
        total_t      = (await db.execute(select(func.count(Transcript.id)))).scalar_one()

        rows = (await db.execute(
            select(Claim.resolution_status, func.count(Claim.id))
            .group_by(Claim.resolution_status)
        )).fetchall()
        counts = {(r[0].value if hasattr(r[0], 'value') else r[0]): r[1] for r in rows}

    table = Table(title="Claim Resolution Summary", box=box.ROUNDED)
    table.add_column("Status", style="cyan", width=28)
    table.add_column("Count", justify="right")
    table.add_column("% of Total", justify="right")

    for status in ["open","materialized","partially_materialized","not_materialized","superseded","unresolvable"]:
        n   = counts.get(status, 0)
        pct = f"{100*n/total_claims:.1f}%" if total_claims else "0%"
        color = {"materialized":"green","not_materialized":"red",
                 "partially_materialized":"yellow","open":"blue",
                 "superseded":"magenta","unresolvable":"dim"}.get(status,"white")
        table.add_row(f"[{color}]{status}[/{color}]", str(n), pct)
    table.add_row("[bold]TOTAL[/bold]", str(total_claims), "100%")

    console.print(table)
    console.print(f"Transcripts processed: {total_t}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rockwool Claims Pipeline")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, reprocess all")
    parser.add_argument("--limit",     type=int, default=None, help="Process only N transcripts")
    parser.add_argument("--stats",     action="store_true", help="Print stats and exit")
    args = parser.parse_args()

    if args.stats:
        asyncio.run(print_stats())
    else:
        asyncio.run(run_and_stream(
            resume=not args.no_resume,
            limit=args.limit,
        ))
