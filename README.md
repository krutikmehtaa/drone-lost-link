# Drone Lost-Link Evidence Research

This project answers questions about drone control-link loss using a fixed collection of FAA guidance, manufacturer manuals, and incident accounts. It compares a fixed evidence workflow with an adaptive agent that can search again when the first passages are incomplete or conflicting. Both workflows draft cited claims, check them against the sources, and report gaps when the collection cannot support an answer.

The [Streamlit demo](https://drone-lost-link.streamlit.app/) shows the question, answer, cited passages, searches, agent decisions, and run details. This is an educational research prototype, not flight guidance.

## What you need

- Python 3.12 or another currently supported Python release
- The files in this repository, including `corpus/` and `processed/`
- Internet access for the first download of the public MiniLM embedding model
- A course API key **only** for new live questions or live notebook evaluation; saved examples work without one

Python package dependencies are listed in [requirements.txt](requirements.txt). The main packages are Streamlit, OpenAI, Pydantic, pandas, sentence-transformers, pypdf, and requests.

## Run the app locally

From a terminal, run:

```bash
git clone https://github.com/krutikmehtaa/drone-lost-link.git
cd drone-lost-link
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local address Streamlit prints, usually `http://localhost:8501`. Choose one of the three **Example Questions** in the sidebar to replay a saved run. These work without a key. Click a source in the answer to read its exact indexed passage; expand **Searches and passages**, **Agent decisions**, **Tool calls**, and **Run details** to inspect the workflow. **New question** starts another chat; **Recents** keeps up to four chats in the current browser session.

The app checks that `processed/chunks.jsonl` matches `processed/corpus_validation_report.json`. Keep these files together. A saved example is available only when its run matches this corpus snapshot. Browser-session chats are temporary; use the download buttons if you want to keep a brief or trace.

### Ask a new question

New questions run the adaptive workflow and need the course API key. Set `OPENAI_API_KEY` in your environment or in a local `.streamlit/secrets.toml` file:

```toml
OPENAI_API_KEY = "your-course-key"
```

Restart the app after setting the key. The app sends model calls through the course Portkey gateway and uses GPT-5 nano. It searches the relevant manual for a clearly manual-focused question; questions about guidance, incidents, or disagreements search across source tiers. Live runs have a 40-model-call ceiling and write traces to `processed/traces/` in the environment running the app. Hosted file storage may be temporary, so download any trace you need from the UI. The key file is ignored by Git; do not commit credentials.

## Reproduce the notebook workflow

The editable notebook is [notebooks/drone_lost_link.ipynb](notebooks/drone_lost_link.ipynb). It was built for **Google Colab** and starts by mounting Google Drive. To run it:

1. Put a copy of this project folder in your Google Drive, or add a Drive shortcut to the shared project folder. Keep `corpus/`, `scripts/`, `notebooks/`, and `processed/` inside the same project folder.
2. Open `notebooks/drone_lost_link.ipynb` in Colab. Its first code cell looks for `MyDrive/Project 1` or `MyDrive/Agentic Unlearning Group/Project 1`. If your folder is elsewhere, change `possible_roots` in that cell to its actual Drive location.
3. Run the cells in order. The notebook installs its Colab packages, screens the raw ASRS export, validates the manifest, builds the passage index and local MiniLM embeddings, checks retrieval, and defines the fixed and adaptive workflows.
4. In the workflow setup cell, set `RUN_LIVE = False` to exercise the notebook without model calls. This uses scripted outputs and **does not** reproduce the published model evaluation. For real model runs, add `OPENAI_API_KEY` to **Colab Secrets**, enable notebook access to it, and set `RUN_LIVE = True`. The key is read from Colab Secrets; do not paste it into a notebook cell.
5. In evaluation section **E8**, set `RUN_IDS` to a short list such as `['Q01', 'Q07']` for a trial, or leave it as `None` for all 50 labeled questions. Run through **E9** to score and inspect the results. A full live evaluation makes many model calls and may take substantial time and course credit.

The notebook writes the frozen corpus report, passages, and embeddings to `processed/`; labeled questions, run records, scores, and retrieval tests go to `processed/eval/`; detailed per-run traces go to `processed/traces/`. The repository includes the saved results used in the project report. Evaluation section E8 **reuses completed saved runs** for the same question IDs. To measure new workflow code rather than replay cached runs, work in a copy of the project and clear that copy's saved `runs_*.jsonl` files before E8. Because model runs are nondeterministic, a new evaluation can give different numbers.

Key saved files:

| File | Purpose |
| --- | --- |
| `corpus/corpus_manifest.csv` | Approved sources and their metadata |
| `processed/corpus_validation_report.json` | Corpus snapshot, source counts, and screening counts |
| `processed/chunks.jsonl` | Searchable passages with source and page locators |
| `processed/eval/eval_set.json` | The 50 labeled evaluation questions |
| `processed/eval/retrieval_summary.csv` | BM25, MiniLM, and hybrid search comparison |
| `processed/eval/scores_summary.csv` | Single-pass, fixed, and adaptive summary scores |
| `processed/eval/runs_*.jsonl` | Saved evaluation run records |
| `processed/traces/*.result.json` and matching `*.jsonl` | Full run result and step-by-step trace |

## Change the workflow or check the code

The notebook is the editable workflow source. The app imports `workflow_runtime.py`, which is generated from the notebook's workflow cells. After editing those cells, refresh the app module and run the local checks:

```bash
python scripts/export_workflow.py
python -m unittest discover -s tests
```

Then start the app and check a saved fixed/adaptive comparison and one live question before redeploying. Saved evaluation scores do not automatically update when workflow code changes; rerun the evaluation if you want new measured results.

## Deploy the demo

In Streamlit Community Cloud, connect this GitHub repository and choose `app.py` as the entrypoint. The root `requirements.txt` supplies dependencies. Saved examples need no hosted key. For live questions, add `OPENAI_API_KEY` in the app's secret settings. The first live run may download the public MiniLM model.

## Scope and source rights

The supported models are DJI Matrice 30T, DJI Matrice 4T, DJI Matrice 210 V1, and Autel EVO Max 4N. The app searches the frozen corpus rather than the live web. ASRS and forum records are unverified accounts; a missing incident in this collection does not show that no incident occurred. The repository's MIT license covers the team's code. FAA, AAIB, DJI, Autel, and other source documents in `corpus/` retain their own rights.
