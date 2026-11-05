"""Corpus ingestion and retrieval for the drone lost-link project.

This module never calls an LLM. It turns a frozen set of source files into
searchable passages that Nathan's workflow can cite by stable ID.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pypdf import PdfReader


REQUIRED_COLUMNS = {
    "source_id", "source_type", "title", "publisher", "model",
    "version_or_date", "status", "url", "retrieved_at", "local_path",
}
SOURCE_TYPES = {"faa_guidance", "manufacturer", "research", "ntsb", "asrs", "aaib", "forum"}
STATUSES = {"current", "historical", "unknown"}
INCIDENT_TYPES = {"ntsb", "asrs", "aaib", "forum"}
# These five record-only entries in the current corpus have a shared collection URL.
# Keep their direct publication pages here until the CSV supplies source_url values.
AAIB_RECORD_URLS = {
    "AAIB-RO-20240131-Derby": "https://www.gov.uk/government/publications/aaib-record-only-investigations-reviewed-february-march-2024/aaib-record-only-investigations-reviewed-february-march-2024",
    "AAIB-RO-20240304-EastGarston": "https://www.gov.uk/government/publications/aaib-record-only-investigations-reviewed-february-march-2024/aaib-record-only-investigations-reviewed-february-march-2024",
    "AAIB-RO-20240319-NorthGosforth": "https://www.gov.uk/government/publications/aaib-record-only-investigations-reviewed-february-march-2024/aaib-record-only-investigations-reviewed-february-march-2024",
    "AAIB-RO-20251016-Kettering": "https://www.gov.uk/government/publications/aaib-record-only-uas-investigations-reviewed-october-november-2025/aaib-record-only-uas-investigations-reviewed-october-november-2025",
    "AAIB-RO-20251201-Willington": "https://www.gov.uk/government/publications/aaib-record-only-uas-investigations-reviewed-december-2025-january-2026/aaib-record-only-uas-investigations-reviewed-december-2025-january-2026",
}
TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*", re.IGNORECASE)
BEHAVIOR_PATTERNS = {
    "link_loss_detection": r"lost[- ]link|loss of (?:control|c2|communication)|signal (?:loss|lost)",
    "return_to_home": r"return[- ]to[- ]home|\brth\b|return home",
    "hover_or_land": r"\bhover(?:ing)?\b|\bland(?:ing|ed)?\b",
    "reconnection": r"reconnect|re-establish|regain (?:the )?(?:link|signal|control)",
    "battery_or_home_point": r"\bbattery\b|home point|homepoint",
}
PHASE_PATTERNS = {
    "preflight": r"pre[- ]flight|before takeoff|flight planning",
    "takeoff": r"\btakeoff\b|\btake-off\b|launch",
    "in_flight": r"in[- ]flight|during flight|airborne|cruise|mission",
    "landing": r"\blanding\b|\bdescent\b|approach",
}


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _tags(text: str, patterns: dict[str, str]) -> list[str]:
    """Keyword tags aid search; they are not verified findings."""
    return [name for name, pattern in patterns.items() if re.search(pattern, text, re.IGNORECASE)]


def _split_text(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    if chunk_words <= 0 or not 0 <= overlap_words < chunk_words:
        raise ValueError("chunk_words must be positive and overlap_words must be smaller")
    words = _clean(text).split()
    if not words:
        return []
    pieces: list[str] = []
    starts: list[int] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        if pieces and end == len(words) and end - start < max(8, chunk_words // 3):
            # A short final window mostly repeats its neighbor. Extend that
            # neighbor instead, so no words are lost and no tiny citation remains.
            pieces[-1] = " ".join(words[starts[-1]:])
            break
        pieces.append(" ".join(words[start:end]))
        starts.append(start)
        if end == len(words):
            break
        start = end - overlap_words
    return pieces


def load_manifest(corpus_root: str | Path) -> list[dict[str, str]]:
    """Validate the manifest and source files before any indexing begins."""
    root = Path(corpus_root).resolve()
    manifest_path = root / "corpus_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest has no source rows")

    seen: set[str] = set()
    for line, row in enumerate(rows, start=2):
        source_id = row["source_id"].strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", source_id):
            raise ValueError(f"Line {line}: source_id must use letters, digits, _ or -")
        if source_id in seen:
            raise ValueError(f"Line {line}: duplicate source_id {source_id}")
        seen.add(source_id)
        if row["source_type"] not in SOURCE_TYPES:
            raise ValueError(f"Line {line}: invalid source_type {row['source_type']!r}")
        if row["status"] not in STATUSES:
            raise ValueError(f"Line {line}: invalid status {row['status']!r}")
        if not row["title"].strip() or not row["publisher"].strip():
            raise ValueError(f"Line {line}: title and publisher are required")
        if not row["retrieved_at"].strip():
            raise ValueError(f"Line {line}: retrieved_at is required")
        relative = Path(row["local_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Line {line}: local_path must stay inside corpus_root")
        source_path = (root / relative).resolve()
        if not source_path.is_relative_to(root) or not source_path.is_file():
            raise FileNotFoundError(f"Line {line}: source file is missing: {source_path}")
        if source_path.suffix.lower() not in {".pdf", ".csv", ".json", ".txt"}:
            raise ValueError(f"Line {line}: unsupported file type: {source_path.suffix}")
        row["file_hash"] = _digest(source_path.read_bytes())
        row["local_path"] = relative.as_posix()
    return rows


def _record_text(record: dict[str, Any]) -> str:
    candidates = ("narrative", "Narrative", "description", "Description", "report_text", "text")
    return _clean(" ".join(str(record[key]) for key in candidates if record.get(key)))


def _record_id(record: dict[str, Any], index: int) -> str:
    for key in ("record_id", "Record ID", "report_id", "Report ID", "NTSBNumber", "EventId", "id"):
        if record.get(key):
            return str(record[key]).strip()
    return f"row-{index}"


def _read_units(path: Path) -> list[tuple[str, str, dict[str, Any]]]:
    """Return passages with their locator and record fields, when present."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [(f"page {number}", page.extract_text() or "", {})
                for number, page in enumerate(reader.pages, start=1)]
    if suffix == ".txt":
        return [("text file", path.read_text(encoding="utf-8-sig"), {})]
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            records = list(csv.DictReader(handle))
    else:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        records = payload if isinstance(payload, list) else payload.get("records", [])
        if not isinstance(records, list):
            raise ValueError(f"JSON file needs a list or a records list: {path}")
    units = [(f"record {_record_id(record, index)}", _record_text(record), record)
             for index, record in enumerate(records, start=1)]
    if len({locator for locator, _, _ in units}) != len(units):
        raise ValueError(f"Duplicate record ID in {path}")
    return units


def build_index(corpus_root: str | Path, output_dir: str | Path,
                chunk_words: int = 180, overlap_words: int = 30) -> dict[str, Any]:
    """Build a reproducible, page-aware index from the frozen manifest."""
    root = Path(corpus_root).resolve()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sources = load_manifest(root)
    chunks: list[dict[str, Any]] = []
    warnings: list[str] = []
    for source in sources:
        path = root / source["local_path"]
        units = _read_units(path)
        kept = 0
        for locator, text, record in units:
            # Incident rows remain intact so a citation cannot mix two reports.
            pieces = [text] if record and source["source_type"] in INCIDENT_TYPES else _split_text(text, chunk_words, overlap_words)
            for piece in pieces:
                cleaned = _clean(piece)
                if len(_tokens(cleaned)) < 8:
                    continue
                kept += 1
                record_id = _record_id(record, 0) if record else None
                url = record.get("source_url") or AAIB_RECORD_URLS.get(record_id) or source["url"].strip()
                chunk = {
                    "chunk_id": f"{source['source_id']}_c{kept:04d}",
                    "source_id": source["source_id"],
                    "source_type": source["source_type"],
                    "title": source["title"],
                    "publisher": source["publisher"],
                    "model": record.get("model") or source["model"].strip() or None,
                    "version_or_date": source["version_or_date"].strip() or None,
                    "status": source["status"],
                    "url": url,
                    "locator": locator,
                    "text": cleaned,
                    "flight_phases": _tags(cleaned, PHASE_PATTERNS),
                    "lost_link_behaviors": _tags(cleaned, BEHAVIOR_PATTERNS),
                    "content_hash": _digest(cleaned.encode("utf-8")),
                }
                if source["source_type"] in INCIDENT_TYPES:
                    chunk.update({
                        "evidence_label": record.get("evidence_label") or
                            ("investigated" if source["source_type"] == "aaib" and path.suffix.lower() == ".pdf"
                             else "unlabeled"),
                        "model_group": record.get("model_group") or "exact",
                        "event_type": record.get("event_type") or "",
                    })
                if source["source_type"] == "aaib" and record and url == source["url"].strip():
                    raise ValueError(f"AAIB record needs a direct publication URL: {record_id}")
                chunks.append(chunk)
        if kept == 0:
            warnings.append(f"No searchable text extracted from {source['source_id']} ({path.name})")
    if not chunks:
        raise ValueError("No searchable passages were extracted. Check the files and manifest.")

    snapshot = _digest(json.dumps([(s["source_id"], s["file_hash"]) for s in sources], sort_keys=True).encode())[:16]
    index_path = output / "chunks.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    config = {
        "corpus_snapshot": snapshot,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_count": len(sources),
        "chunk_count": len(chunks),
        "retrieval_method": "BM25",
        "chunk_words": chunk_words,
        "overlap_words": overlap_words,
        "chunking": "page_window_or_whole_incident_record",
        "index_file": "chunks.jsonl",
        "warnings": warnings,
    }
    (output / "index_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


class CorpusSearch:
    """Read-only search and citation lookup over one frozen index."""

    def __init__(self, index_path: str | Path):
        path = Path(index_path)
        self.chunks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.chunks:
            raise ValueError("Index is empty")
        self.by_id = {item["chunk_id"]: item for item in self.chunks}
        if len(self.by_id) != len(self.chunks):
            raise ValueError("Index contains duplicate chunk IDs")
        self.terms = [Counter(_tokens(item["text"])) for item in self.chunks]
        self.lengths = [sum(term_counts.values()) for term_counts in self.terms]
        self.avg_length = sum(self.lengths) / len(self.lengths)
        self.df = Counter(term for terms in self.terms for term in terms)

    def document_search(self, query: str, filters: dict[str, str] | None = None,
                        top_k: int = 5) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        filters = filters or {}
        allowed = {"source_type", "model", "status", "flight_phase", "lost_link_behavior"}
        if set(filters) - allowed:
            raise ValueError(f"Unsupported filters: {sorted(set(filters) - allowed)}")
        query_terms = set(_tokens(query))
        results = []
        for index, chunk in enumerate(self.chunks):
            if not self._matches_filters(chunk, filters):
                continue
            score = 0.0
            for term in query_terms:
                frequency = self.terms[index].get(term, 0)
                if not frequency:
                    continue
                idf = math.log(1 + (len(self.chunks) - self.df[term] + 0.5) / (self.df[term] + 0.5))
                denominator = frequency + 1.2 * (0.25 + 0.75 * self.lengths[index] / self.avg_length)
                score += idf * (frequency * 2.2 / denominator)
            if score > 0:
                results.append((score, chunk))
        results.sort(key=lambda pair: (-pair[0], pair[1]["chunk_id"]))
        return [{"rank": rank, "score": round(score, 6), **chunk}
                for rank, (score, chunk) in enumerate(results[:top_k], start=1)]

    @staticmethod
    def _matches_filters(chunk: dict[str, Any], filters: dict[str, str]) -> bool:
        allowed = {"source_type", "model", "status", "flight_phase", "lost_link_behavior"}
        if set(filters) - allowed:
            raise ValueError(f"Unsupported filters: {sorted(set(filters) - allowed)}")
        if any(chunk.get(key) != value for key, value in filters.items()
               if key not in {"model", "flight_phase", "lost_link_behavior"}):
            return False
        target_model = filters.get("model")
        if target_model and chunk["model"] != target_model:
            # General guidance can accompany model-specific evidence.
            if not (chunk["model"] is None and chunk["source_type"] in {"faa_guidance", "research"}):
                return False
        if filters.get("flight_phase") and filters["flight_phase"] not in chunk["flight_phases"]:
            return False
        if filters.get("lost_link_behavior") and filters["lost_link_behavior"] not in chunk["lost_link_behaviors"]:
            return False
        return True

    def citation_lookup(self, chunk_id: str) -> dict[str, Any]:
        if chunk_id not in self.by_id:
            raise KeyError(f"Unknown chunk_id: {chunk_id}")
        chunk = self.by_id[chunk_id]
        return {key: chunk[key] for key in (
            "chunk_id", "source_id", "source_type", "title", "publisher",
            "model", "version_or_date", "status", "url", "locator", "text",
        )}
