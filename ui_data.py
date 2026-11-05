"""Read the frozen corpus and saved workflow runs for the demo interface."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


DEMO_QUESTIONS = (
    "Does the evidence show that the Autel EVO Max 4N returns home after losing its link?",
    "What is the lost-link failure rate per 1,000 flight hours for the DJI Matrice 4T?",
    "What can the DJI Matrice 30T be set to do when the remote controller signal is lost?",
)

MODE_LABELS = {"full": "Adaptive agent", "fixed": "Fixed workflow"}
STATUS_LABELS = {
    "completed": "Brief complete",
    "completed_with_gaps": "Brief with evidence gaps",
    "human_review_required": "Needs human review",
    "abstained": "No supported answer",
    "failed": "Run failed",
}
SOURCE_LABELS = {
    "faa_guidance": "FAA guidance",
    "manufacturer": "Manufacturer manual",
    "asrs": "ASRS report · unverified",
    "aaib": "AAIB investigation",
    "forum": "Forum account · unverified",
}


def load_corpus(project_root: Path) -> tuple[dict, dict[str, dict], str]:
    """Load only the index matching the saved corpus report."""
    processed = project_root / "processed"
    chunks_path = processed / "chunks.jsonl"
    report = json.loads((processed / "corpus_validation_report.json").read_text(encoding="utf-8"))
    index_hash = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
    if report.get("embedding_index_sha256") != index_hash:
        raise ValueError("The saved corpus report and passage index do not match.")
    chunks = {}
    with chunks_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunk = json.loads(line)
                chunks[chunk["chunk_id"]] = chunk
    return report, chunks, index_hash


def find_saved_run(project_root: Path, question: str, mode: str, snapshot: str, chunks_hash: str) -> dict | None:
    """Choose the newest complete run for this exact question and corpus."""
    trace_dir = project_root / "processed" / "traces"
    for path in sorted(trace_dir.glob("*.result.json"), reverse=True):
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (run.get("question") == question and run.get("mode") == mode
                and run.get("corpus_snapshot") == snapshot and run.get("chunks_sha256") == chunks_hash
                and isinstance(run.get("brief"), dict)):
            return run
    return None


def cited_passages(run: dict, chunks: dict[str, dict]) -> list[dict]:
    """Return each cited passage once, in the order it appears in the brief."""
    seen = set()
    passages = []
    for section in run.get("brief", {}).get("sections", []):
        for claim in section.get("claims", []):
            for chunk_id in claim.get("citations", []):
                if chunk_id in chunks and chunk_id not in seen:
                    passages.append(chunks[chunk_id])
                    seen.add(chunk_id)
    return passages


def brief_markdown(run: dict, chunks: dict[str, dict]) -> str:
    """Make a readable export with source locators and links."""
    lines = [f"# Drone lost-link evidence brief", "", f"Question: {run.get('question', '')}", "",
             f"Status: {STATUS_LABELS.get(run.get('status'), run.get('status', 'Unknown'))}", "",
             f"Corpus snapshot: {run.get('corpus_snapshot', '')}", "",
             "## Short answer", "", run.get("brief", {}).get("short_answer", "") or "No supported short answer.", ""]
    for section in run.get("brief", {}).get("sections", []):
        lines += [f"## {section.get('label') or section.get('section', 'Evidence')}", ""]
        for claim in section.get("claims", []):
            lines.append(f"- {claim.get('text', '')}")
            for chunk_id in claim.get("citations", []):
                chunk = chunks.get(chunk_id)
                if chunk:
                    lines.append(f"  - Source: {chunk.get('title', '')}, {chunk.get('locator', '')} — {chunk.get('url', '')}")
        lines.append("")
    if run.get("brief", {}).get("gaps"):
        lines += ["## Gaps and conflicts", ""]
        lines += [f"- {gap.get('text', '')}" for gap in run["brief"]["gaps"]]
    return "\n".join(lines) + "\n"
