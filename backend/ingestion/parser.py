"""
PDF ingestion: FactSet transcript PDFs → structured Python objects.

Key design decisions:
  - PyMuPDF (fitz) preserves paragraph structure better than pdfplumber for FactSet
  - We separate management_discussion from Q&A — claims from prepared remarks are
    stronger commitments than improvised Q&A answers
  - Speaker attribution is deterministic: FactSet uses consistent "Name\nTitle\n" 
    before each dotted separator
  - We identify management vs analyst speakers so the extraction agent only
    focuses on management speech (they're the ones making the claims)
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass, field

import fitz  # PyMuPDF


@dataclass
class SpeechBlock:
    speaker_name:  str
    speaker_title: str
    text:          str
    section:       str    # "management_discussion" | "qa"
    is_management: bool


@dataclass
class ParsedTranscript:
    filename:                str
    transcript_date:         str   # ISO YYYY-MM-DD
    event_type:              str
    fiscal_period:           str
    corporate_participants:  list[str]
    analyst_participants:    list[str]
    speech_blocks:           list[SpeechBlock]
    full_text:               str


# ── Boilerplate patterns to strip ────────────────────────────────────────────
_BOILERPLATE = re.compile(
    r"(1-877-FACTSET\s*www\.callstreet\.com|"
    r"Copyright © \d{4}[-–]\d{4} FactSet.*|"
    r"Total Pages: \d+|"
    r"Corrected Transcript|"
    r"Raw Transcript)",
    re.IGNORECASE
)
_SEPARATOR = re.compile(r"\.{15,}|_{15,}")


def _parse_filename(filename: str) -> dict:
    base = os.path.basename(filename).replace(".pdf", "")
    m = re.search(r"-(\d{4}-\d{2}-\d{2})-\d+$", base)
    transcript_date = m.group(1) if m else "unknown"
    b = base.lower()

    if   "q1" in b and "earnings" in b: ev, fp = "Q1 Earnings Call", f"Q1 {transcript_date[:4]}"
    elif "q2" in b and "earnings" in b: ev, fp = "Q2 Earnings Call", f"Q2 {transcript_date[:4]}"
    elif "q3" in b and "earnings" in b: ev, fp = "Q3 Earnings Call", f"Q3 {transcript_date[:4]}"
    elif "q4" in b and "earnings" in b:
        yr = str(int(transcript_date[:4]) - 1) if transcript_date != "unknown" else transcript_date[:4]
        ev, fp = "Q4 Earnings Call", f"FY {yr}"
    elif "annual general meeting" in b: ev, fp = "Annual General Meeting", f"AGM {transcript_date[:4]}"
    elif "extraordinary" in b:          ev, fp = "Extraordinary Shareholders Meeting", f"EGM {transcript_date[:4]}"
    elif "esg" in b and "analyst" in b: ev, fp = "ESG Analyst Meeting", f"ESG {transcript_date[:4]}"
    elif "analyst" in b:                ev, fp = "Analyst Meeting", f"Analyst Day {transcript_date[:4]}"
    else:                               ev, fp = "Meeting", transcript_date[:4]

    return {"transcript_date": transcript_date, "event_type": ev, "fiscal_period": fp}


def _extract_participants(text: str) -> tuple[list[str], list[str]]:
    corporate, analysts = [], []

    def extract_names(block: str) -> list[str]:
        names, lines = [], [l.strip() for l in block.split("\n") if l.strip()]
        i = 0
        while i < len(lines):
            line = lines[i]
            if _SEPARATOR.search(line) or _BOILERPLATE.search(line) or len(line) < 3:
                i += 1; continue
            if i + 1 < len(lines):
                nxt = lines[i + 1]
                if any(w in nxt.lower() for w in [
                    "analyst","director","president","ceo","cfo","svp","vp",
                    "officer","manager","head","investor","secretary","general",
                    "partner","research","group","senior","chief","executive",
                    "relations","communications","sustainability"
                ]):
                    names.append(f"{line} | {nxt}"); i += 2; continue
            names.append(line); i += 1
        return names

    corp = re.search(
        r"CORPORATE PARTICIPANTS(.*?)(?:OTHER PARTICIPANTS|MANAGEMENT DISCUSSION)",
        text, re.DOTALL | re.IGNORECASE
    )
    other = re.search(
        r"OTHER PARTICIPANTS(.*?)(?:MANAGEMENT DISCUSSION|$)",
        text, re.DOTALL | re.IGNORECASE
    )
    if corp:  corporate = extract_names(corp.group(1))
    if other: analysts  = extract_names(other.group(1))
    return corporate, analysts


def _parse_speech_blocks(
    text: str,
    corporate_participants: list[str]
) -> list[SpeechBlock]:
    """
    Parse speech blocks by splitting on FactSet separator lines,
    then identifying speaker headers (name + title pattern).
    """
    # Build set of management names for is_management detection
    mgmt_names = set()
    for p in corporate_participants:
        name = p.split("|")[0].strip().lower()
        mgmt_names.add(name)

    # Find section boundaries
    mgmt_m = re.search(r"MANAGEMENT DISCUSSION SECTION", text, re.IGNORECASE)
    qa_m   = re.search(r"QUESTION AND ANSWER SECTION",   text, re.IGNORECASE)
    sections = []
    if mgmt_m:
        end = qa_m.start() if qa_m else len(text)
        sections.append(("management_discussion", text[mgmt_m.end():end]))
    if qa_m:
        sections.append(("qa", text[qa_m.end():]))
    if not sections:
        sections = [("management_discussion", text)]

    blocks: list[SpeechBlock] = []
    TITLE_WORDS = {
        "analyst","director","president","ceo","cfo","svp","vp","officer",
        "manager","head","investor","secretary","general","partner","research",
        "group","senior","chief","executive","relations","communications",
        "sustainability","operator","participant"
    }

    for section_name, section_text in sections:
        segments = _SEPARATOR.split(section_text)
        for seg in segments:
            seg = seg.strip()
            if len(seg) < 40: continue
            lines = [_BOILERPLATE.sub("", l).strip() for l in seg.split("\n")]
            lines = [l for l in lines if l and not re.match(r"^\d+$", l)]
            if not lines: continue

            speaker_name, speaker_title, text_start = "Unknown", "", 0
            if len(lines) >= 2 and 4 < len(lines[0]) < 70:
                if any(w in lines[1].lower() for w in TITLE_WORDS):
                    speaker_name, speaker_title, text_start = lines[0], lines[1], 2
                elif 4 < len(lines[0]) < 50:
                    speaker_name, text_start = lines[0], 1

            speech_text = "\n".join(lines[text_start:]).strip()
            if len(speech_text.split()) < 20: continue

            is_mgmt = any(
                speaker_name.lower() in name or name in speaker_name.lower()
                for name in mgmt_names
            ) if speaker_name != "Unknown" else False

            blocks.append(SpeechBlock(
                speaker_name=speaker_name,
                speaker_title=speaker_title,
                text=speech_text,
                section=section_name,
                is_management=is_mgmt,
            ))
    return blocks


def parse_transcript(pdf_path: str) -> ParsedTranscript:
    filename = os.path.basename(pdf_path)
    meta     = _parse_filename(filename)

    doc       = fitz.open(pdf_path)
    full_text = "".join(page.get_text() for page in doc)
    doc.close()

    corporate, analysts = _extract_participants(full_text)
    speech_blocks       = _parse_speech_blocks(full_text, corporate)

    return ParsedTranscript(
        filename=filename,
        transcript_date=meta["transcript_date"],
        event_type=meta["event_type"],
        fiscal_period=meta["fiscal_period"],
        corporate_participants=corporate,
        analyst_participants=analysts,
        speech_blocks=speech_blocks,
        full_text=full_text,
    )


def get_sorted_transcript_files(directory: str) -> list[str]:
    """Return PDF filenames sorted chronologically by embedded date."""
    files = [f for f in os.listdir(directory) if f.endswith(".pdf")]
    def key(f):
        m = re.search(r"-(\d{4}-\d{2}-\d{2})-\d+\.pdf$", f)
        return m.group(1) if m else "9999-99-99"
    return sorted(files, key=key)
