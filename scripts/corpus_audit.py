"""Small, repeatable checks for the corpus handed to the evidence workflow."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from pipeline import INCIDENT_TYPES, load_manifest


EXCLUDED_LABELS = {"no_link_loss", "control_loss_unconfirmed_link"}
SCREENED_LABELS = {"narrative_explicit", "summary_only", "investigated", "unverified_forum"}


def incident_status(passage: dict) -> str:
    """Use the same basic scope rule as the notebook's incident search."""
    label = passage.get("evidence_label", "unlabeled")
    if label in EXCLUDED_LABELS:
        return "excluded"
    if passage.get("source_type") == "forum" and "link_loss" not in passage.get("event_type", ""):
        return "excluded"
    return "screened_in" if label in SCREENED_LABELS else "needs_review"


def _chunks(chunks_path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(chunks_path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def source_inventory(corpus_root: str | Path, chunks_path: str | Path) -> list[dict]:
    """Count incident records, rather than mistaking PDF passages for reports."""
    sources = load_manifest(corpus_root)
    by_source = defaultdict(list)
    for chunk in _chunks(chunks_path):
        by_source[chunk["source_id"]].append(chunk)
    inventory = []
    for source in sources:
        passages = by_source[source["source_id"]]
        if not passages:
            raise ValueError(f"No indexed passages for {source['source_id']}")
        by_model = defaultdict(list)
        for passage in passages:
            by_model[passage.get("model") or "general"].append(passage)
        for model, model_passages in sorted(by_model.items()):
            reports = {}
            if source["source_type"] in INCIDENT_TYPES:
                is_pdf = source["local_path"].lower().endswith(".pdf")
                for passage in model_passages:
                    report_key = "whole report" if is_pdf else passage["locator"]
                    reports.setdefault(report_key, passage)
            labels = Counter(passage.get("evidence_label", "unlabeled") for passage in reports.values())
            statuses = Counter(incident_status(passage) for passage in reports.values())
            inventory.append({
                "source_id": source["source_id"], "source_type": source["source_type"],
                "publisher": source["publisher"], "model": model,
                "file": source["local_path"], "passages": len(model_passages),
                "incident_reports": len(reports),
                "screened_in": statuses["screened_in"],
                "excluded": statuses["excluded"],
                "needs_review": statuses["needs_review"],
                "labels": dict(labels),
            })
    return inventory


def chunk_quality(chunks_path: str | Path) -> dict:
    """Flag repeated text and split reports for inspection; never discard evidence silently."""
    chunks = _chunks(chunks_path)
    by_locator = defaultdict(list)
    by_hash = defaultdict(list)
    for chunk in chunks:
        by_locator[(chunk["source_id"], chunk["locator"])].append(chunk)
        by_hash[chunk["content_hash"]].append(chunk["chunk_id"])
    split_reports = [f"{source} | {locator}" for (source, locator), items in by_locator.items()
                     if items[0]["source_type"] in INCIDENT_TYPES and locator.startswith("record ") and len(items) > 1]
    repeated_tails = [items[index]["chunk_id"]
                      for (source, locator), items in by_locator.items() if locator.startswith("page ")
                      for index in range(1, len(items))
                      if items[index - 1]["text"].endswith(items[index]["text"])]
    repeated = [ids for ids in by_hash.values() if len(ids) > 1]
    words = [len(chunk["text"].split()) for chunk in chunks]
    return {
        "passages": len(chunks), "shortest_words": min(words), "longest_words": max(words),
        "exact_duplicate_groups": repeated, "split_incident_records": split_reports,
        "repeated_page_tails": repeated_tails,
        "multi_passage_pages": sum(len(items) > 1 for (source, locator), items in by_locator.items()
                                   if locator.startswith("page ")),
    }
