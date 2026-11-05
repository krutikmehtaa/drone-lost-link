"""Build tiered incident files for the corpus from public occurrence sources.

collect_aaib pulls every UK AAIB record-only UAS list from gov.uk and keeps the
official occurrence text verbatim. asrs_other_models turns the ASRS rows that
prepare_asrs_sample screened out into a general lost-link file. Both write the
same columns as incidents/selected/asrs_dji_m30t.csv, plus model_group,
event_type and source_url. Run where gov.uk is reachable, for example Colab.
"""

from __future__ import annotations

import csv
import re
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

GOVUK = "https://www.gov.uk"
COLLECTION = "/government/collections/aaib-record-only-investigations"
FIELDS = [
    "record_id", "date", "model", "evidence_label", "screening_note",
    "synopsis", "narrative", "model_group", "event_type", "source_url",
]
EXACT_MODELS = {
    "DJI Matrice 30T": r"\b(?:matrice\s*)?m?30\s*t\b|\bmatrice\s*m30t\b",
    "DJI Matrice 30": r"\b(?:matrice\s*m?30|m30)\b(?!\s*t)(?!\d)",
    "DJI Matrice 4T": r"\b(?:matrice\s*4\s*t|m4t|m4td)\b",
    "DJI Matrice 4E": r"\b(?:matrice\s*4\s*e|m4e)\b",
    "Autel EVO Max 4N": r"\bevo\s*max\s*4n\b",
    "Autel EVO Max 4T": r"\bevo\s*max(?:\s*4t)?\b",
}
FILE_NAMES = {
    "DJI Matrice 30T": "aaib_dji_m30t.csv",
    "DJI Matrice 30": "aaib_dji_m30.csv",
    "DJI Matrice 4T": "aaib_dji_m4t.csv",
    "DJI Matrice 4E": "aaib_dji_m4e.csv",
    "Autel EVO Max 4N": "aaib_autel_evo_max_4n.csv",
    "Autel EVO Max 4T": "aaib_autel_evo_max_4t.csv",
}
RELATED = r"\bm\s*-?\s*(?:200|210|300|350)\b|matrice|mavic\s*(?:2|3)?\s*(?:enterprise|e\b|t\b|3t|3e)|\bm3t\b|\bm3e\b|\bautel\b|\bevo\b"
MODEL_AIRCRAFT = r"^(?:uas\s+)?ma\b|model aircraft|\bglider\b"
EVENT_PATTERNS = {
    "link_loss": r"lost (?:the )?(?:control |radio |data |c2 |communication )?(?:link|signal|contact|connection|communications?)"
                 r"|(?:link|signal|connection|contact|communications?) (?:was |had been )?(?:lost|dropped)"
                 r"|loss of (?:the )?(?:control |radio |data |c2 |communication )?(?:link|signal|connection|contact|communications?)"
                 r"|disconnect|datalink|data link|interference",
    "rth_problem": r"return(?:ed)?[ -]to[ -](?:home|base|launch|land)|\brth\b|return home|automatic(?:ally)? (?:return|land)",
    "unresponsive": r"(?:did not|didn't|stopped|failed to|not) respond|unresponsive|could not (?:be )?control|unable to (?:regain )?control"
                    r"|lost control|loss of control|control (?:was|of the ua was) lost|uncontrollabl|ignored",
    "flyaway": r"fl(?:y|ew)[ -]?away|flew off|not (?:been )?(?:found|recovered|located)",
    "uncommanded": r"uncommanded|without (?:any )?command|climbed uncontroll|unexpectedly",
    "gnss_loss": r"\bgps\b|\bgnss\b|satellite|\brtk signal\b|compass",
}


def _get(path_or_url: str) -> str:
    url = path_or_url if path_or_url.startswith("http") else GOVUK + path_or_url
    for attempt in range(3):
        response = requests.get(url, timeout=30, headers={"User-Agent": "course-research-corpus/1.0"})
        if response.status_code == 200:
            return response.text
        time.sleep(2 * (attempt + 1))
    response.raise_for_status()
    return response.text


def list_publications() -> list[str]:
    html = _get(COLLECTION)
    paths = re.findall(r'href="(/government/publications/aaib-record-only[a-z0-9-]*)"', html)
    return sorted(set(paths))


def html_document(publication: str, html: str | None = None) -> str | None:
    html = html or _get(publication)
    links = re.findall(rf'href="({re.escape(publication)}/[a-z0-9-]+)"', html)
    return links[0] if links else None


def _column(columns: list[str], *needles: str) -> str | None:
    for column in columns:
        if any(needle in column.lower() for needle in needles):
            return column
    return None


def occurrence_rows(html: str) -> list[dict[str, str]]:
    rows = []
    for table in pd.read_html(StringIO(html)):
        columns = [str(c) for c in table.columns]
        table.columns = columns
        text = _column(columns, "occurrence text")
        kind = _column(columns, "type")
        date = _column(columns, "date")
        place = _column(columns, "location")
        if not text or not kind:
            continue
        for _, row in table.fillna("").iterrows():
            rows.append({
                "date": str(row[date]).strip() if date else "",
                "aircraft": str(row[kind]).strip(),
                "location": str(row[place]).strip() if place else "",
                "text": re.sub(r"\s+", " ", str(row[text])).strip(),
            })
    return rows


def classify_model(aircraft: str) -> tuple[str | None, str]:
    name = aircraft.lower()
    if re.search(MODEL_AIRCRAFT, name):
        return None, "model_aircraft"
    for model, pattern in EXACT_MODELS.items():
        if re.search(pattern, name):
            return model, "exact"
    if re.search(RELATED, name):
        return None, "related"
    return None, "other"


def classify_event(text: str) -> str:
    found = [name for name, pattern in EVENT_PATTERNS.items() if re.search(pattern, text, re.IGNORECASE)]
    return ";".join(found)


def _record_id(date: str, location: str, seen: set[str]) -> tuple[str, str]:
    parsed = pd.to_datetime(date, dayfirst=True, errors="coerce")
    stamp = parsed.strftime("%Y%m%d") if not pd.isna(parsed) else "undated"
    place = "".join(word[:1].upper() + word[1:] for word in re.split(r"[^A-Za-z0-9]+", location.split(",")[0]) if word)
    base = f"AAIB-RO-{stamp}-{place or 'Unknown'}"
    record_id, n = base, 2
    while record_id in seen:
        record_id, n = f"{base}-{n}", n + 1
    seen.add(record_id)
    return record_id, parsed.strftime("%Y%m") if not pd.isna(parsed) else ""


def build_aaib_records(pages: dict[str, str]) -> list[dict[str, str]]:
    records, seen = [], set()
    for url, html in pages.items():
        for row in occurrence_rows(html):
            model, group = classify_model(row["aircraft"])
            event = classify_event(row["text"])
            if group == "model_aircraft" or (group != "exact" and not event):
                continue
            record_id, month = _record_id(row["date"], row["location"], seen)
            records.append({
                "record_id": record_id,
                "date": month,
                "model": model or row["aircraft"],
                "evidence_label": "official_record_only",
                "screening_note": f"AAIB record-only occurrence, reporter-supplied and not investigated. "
                                  f"Listed aircraft: {row['aircraft']}. Location: {row['location']}. "
                                  f"Event keywords: {event or 'none'}.",
                "synopsis": "",
                "narrative": row["text"],
                "model_group": group,
                "event_type": event or "other_occurrence",
                "source_url": url,
            })
    return records


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_tiers(records: list[dict[str, str]], output_dir: str | Path, prefix: str) -> dict[str, int]:
    output = Path(output_dir)
    counts = {}
    for model, name in FILE_NAMES.items():
        rows = [r for r in records if r["model_group"] == "exact" and r["model"] == model]
        if rows and prefix == "aaib":
            write_rows(output / name, rows)
            counts[name] = len(rows)
    for group in ("related", "other"):
        rows = [r for r in records if r["model_group"] == group]
        if rows:
            name = f"{prefix}_{group}_models.csv"
            write_rows(output / name, rows)
            counts[name] = len(rows)
    return counts


def collect_aaib(output_dir: str | Path, pause: float = 1.0) -> dict[str, int]:
    pages = {}
    for publication in list_publications():
        landing = _get(publication)
        document = html_document(publication, landing)
        if not document:
            continue
        pages[GOVUK + document] = _get(document)
        time.sleep(pause)
    records = build_aaib_records(pages)
    counts = write_tiers(records, output_dir, "aaib")
    counts["pages"] = len(pages)
    counts["records"] = len(records)
    return counts


def asrs_other_models(raw_export: str | Path, output_dir: str | Path, selected: set[str]) -> dict[str, int]:
    from prepare_asrs_sample import _read_asrs_export

    records = []
    for row in _read_asrs_export(raw_export):
        if row["record_id"] in selected:
            continue
        _, group = classify_model(row["reported_model"])
        records.append({
            "record_id": row["record_id"],
            "date": row["date"],
            "model": row["reported_model"],
            "evidence_label": "general_lost_link",
            "screening_note": "ASRS lost-link hit outside the three study models. Use for general patterns, never as evidence for a specific study model.",
            "synopsis": row["synopsis"],
            "narrative": row["narrative"],
            "model_group": "related" if group == "related" else "other",
            "event_type": classify_event(row["synopsis"] + " " + row["narrative"]) or "link_loss",
            "source_url": "https://asrs.arc.nasa.gov/search/database.html",
        })
    return write_tiers(records, output_dir, "asrs")
