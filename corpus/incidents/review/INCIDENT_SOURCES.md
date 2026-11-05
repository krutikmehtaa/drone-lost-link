# Incident sources and tiers (Oct 30, 2026)

## Tiers
Every incident file has `model_group`:
- `exact`: names a study model (M30T, M30, M4T, EVO Max). Cite as model evidence.
- `related`: other DJI enterprise or Autel aircraft (M300, M210, Mavic Enterprise, Evo II). Context only.
- `other`: any other drone. General lost-link patterns only, never model evidence.

`evidence_label` says how far to trust it:
- `official_record_only`, `narrative_explicit`, `summary_only`, `no_link_loss`, `control_loss_unconfirmed_link`: official database text.
- `general_lost_link`: official ASRS text, other models.
- `secondary_summary`: official occurrence known only through an aggregator; team paraphrase.
- `unverified_forum`: forum post, team paraphrase, not checked by anyone.

`event_type` lists what happened: link_loss, rth_problem, unresponsive, flyaway, uncommanded, gnss_loss, other_occurrence.

## Files in incidents/selected
| File | Rows | How made |
|---|---|---|
| asrs_dji_m30t.csv, asrs_autel_evo_max_4n.csv | 3, 1 | Krutik, unchanged |
| asrs_dji_m4t.csv | 1 | ASRS 2336078 |
| aaib_dji_m30t.csv, aaib_dji_m30.csv | 5, 1 now | Replace by running prepare_incident_tiers.collect_aaib |
| aaib_related_models.csv, aaib_other_models.csv | run script | collect_aaib |
| asrs_related_models.csv, asrs_other_models.csv | run script | asrs_other_models on asrs_lost_link_raw.csv |
| secondary_dji_m30t.csv | 1 | BFU M30T lost-link crash via ASN |
| forum_dji_m30t.csv, forum_dji_m4t.csv, forum_autel_evo_max_4t.csv | 2, 2, 3 | Team paraphrase of forum threads |
| incident_candidates_index.csv | 92 | Every official record found by hand, to check the script output |

## Running the script (Colab)
```
from prepare_incident_tiers import collect_aaib, asrs_other_models
collect_aaib("corpus/incidents/selected")
asrs_other_models("corpus/incidents/asrs_lost_link_raw.csv", "corpus/incidents/selected",
                  {"2185554", "2250946", "2394566", "2202496"})
```
Put prepare_asrs_sample.py in the same folder. Compare aaib counts with incident_candidates_index.csv.

## pipeline.py changes needed
1. Add `"aaib"`, `"secondary"`, `"forum"` to `SOURCE_TYPES`.
2. Keep these rows whole: change `{"ntsb", "asrs"}` to `{"ntsb", "asrs", "aaib", "secondary", "forum"}`.
3. Leave `model` blank in the manifest for related/other files so model filters never return them.

## Still to search by hand
- ASRS Database Online: search Make Model Name for "Matrice 30", "M30", "Matrice 4", "M4T", "EVO Max", "Autel" with no text filter. Also Flight Phase = Return to Home (UAS) and Anomaly = Loss Of Aircraft Control. Export CSV and run prepare_asrs_sample.
- ATSB National Aviation Occurrence Database (atsb.gov.au/avdata): searchable by aircraft and occurrence type, no bulk export.
- German BFU monthly bulletins: drone occurrences in German; confirm BFU24-0402-3X.
- CHIRP Drone/UAS FEEDBACK editions (chirp.co.uk): confidential reports with narratives; mostly smaller DJI models.
- AAIB record-only lists for 2020-2021 are covered by the script; none can involve the M30 (released 2022).

## Disclosures for the report
- AAIB: Crown copyright, Open Government Licence v3.0; reporter-supplied, accuracy not assured.
- ASRS: voluntary, unverified, cannot be used to infer frequency.
- Forum rows are paraphrased by the team and unverified; the original posts belong to their authors.
