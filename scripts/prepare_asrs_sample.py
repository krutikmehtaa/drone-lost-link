"""Screen an ASRS search export and prepare the model-specific incident sample.

The selected ACNs were reviewed against both the synopsis and the reporter's
narrative. The labels describe what the account says; they do not verify it.
"""

from __future__ import annotations

import csv
from pathlib import Path


REVIEWED = {
    "2185554": (
        "DJI Matrice 30T", "summary_only",
        "The synopsis says lost link; the narrative describes loss of control after liftoff without confirming link loss.",
    ),
    "2250946": (
        "DJI Matrice 30T", "narrative_explicit",
        "The reporter explicitly describes lost link and successful return to home; the cause is the reporter's account.",
    ),
    "2394566": (
        "DJI Matrice 30T", "summary_only",
        "The synopsis says lost link; the narrative describes unresponsive controls while the camera feed remained available.",
    ),
    "2202496": (
        "Autel EVO Max 4N", "summary_only",
        "The synopsis says lost link; the narrative describes loss of control, failed return to home, and an unknown error.",
    ),
}


def _read_asrs_export(path: str | Path) -> list[dict[str, str]]:
    """Read NASA's two-row CSV header without losing repeated column names."""
    with Path(path).open(encoding="utf-8-sig", newline="", errors="replace") as handle:
        rows = list(csv.reader(handle))
    header_index = next((i for i, row in enumerate(rows) if row and row[0] == "ACN"), None)
    if header_index is None:
        raise ValueError("ASRS export has no ACN header row")
    header = rows[header_index]
    required = {0: "ACN", 18: "Make Model Name", 120: "Narrative", 124: "Synopsis"}
    for index, name in required.items():
        if len(header) <= index or header[index] != name:
            raise ValueError(f"ASRS export column {index} should be {name!r}")
    records = []
    for row in rows[header_index + 1:]:
        if not row:
            continue
        if len(row) != len(header):
            raise ValueError(f"ASRS record has {len(row)} columns; expected {len(header)}")
        records.append({
            "record_id": row[0].strip(),
            "date": row[1].strip(),
            "reported_model": row[18].strip(),
            "synopsis": row[124].strip(),
            "narrative": row[120].strip(),
        })
    if len({record["record_id"] for record in records}) != len(records):
        raise ValueError("ASRS export contains duplicate ACNs")
    return records


def prepare_asrs_sample(raw_export: str | Path, output_dir: str | Path) -> dict[str, object]:
    """Write a screening log for every hit and two selected incident files."""
    records = _read_asrs_export(raw_export)
    by_acn = {record["record_id"]: record for record in records}
    missing = set(REVIEWED) - set(by_acn)
    if missing:
        raise ValueError(f"Reviewed ACNs are missing from export: {sorted(missing)}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    screening_path = output / "asrs_search_screening.csv"
    with screening_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "record_id", "date", "reported_model", "screening_decision",
            "evidence_label", "screening_note", "synopsis",
        ])
        writer.writeheader()
        for record in records:
            review = REVIEWED.get(record["record_id"])
            writer.writerow({
                "record_id": record["record_id"],
                "date": record["date"],
                "reported_model": record["reported_model"],
                "screening_decision": "selected" if review else "outside_named_models",
                "evidence_label": review[1] if review else "not_assessed",
                "screening_note": review[2] if review else "Model is outside the selected M30T and EVO Max 4N sample.",
                "synopsis": record["synopsis"],
            })

    selected_files = {}
    for model, filename in (
        ("DJI Matrice 30T", "asrs_dji_m30t.csv"),
        ("Autel EVO Max 4N", "asrs_autel_evo_max_4n.csv"),
    ):
        path = output / filename
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "record_id", "date", "model", "evidence_label", "screening_note",
                "synopsis", "narrative",
            ])
            writer.writeheader()
            for acn, (normalized_model, label, note) in REVIEWED.items():
                if normalized_model != model:
                    continue
                record = by_acn[acn]
                writer.writerow({
                    "record_id": acn,
                    "date": record["date"],
                    "model": model,
                    "evidence_label": label,
                    "screening_note": note,
                    "synopsis": record["synopsis"],
                    "narrative": record["narrative"],
                })
        selected_files[model] = str(path)

    return {
        "search_hits": len(records),
        "selected_accounts": len(REVIEWED),
        "explicit_in_narrative": sum(label == "narrative_explicit" for _, label, _ in REVIEWED.values()),
        "summary_only": sum(label == "summary_only" for _, label, _ in REVIEWED.values()),
        "screening_log": str(screening_path),
        "selected_files": selected_files,
    }
