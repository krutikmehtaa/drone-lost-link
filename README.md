# Drone Lost-Link Evidence Research

This project compares a fixed evidence workflow with an adaptive agent. A user asks a
question about a supported drone model. The workflow searches FAA guidance,
manufacturer manuals, and reported incidents; drafts cited claims; checks those
claims; and returns a brief with gaps or a human-review status when needed.

The Streamlit app provides the required input, tool-use, and output demonstration.
Its default mode replays real saved runs from the frozen corpus. A live mode can
run the same notebook workflow when a course API key is configured.

## Run the app locally

Use Python 3.12 or a currently supported Python release:

1. Install the packages in `requirements.txt`.
2. Run `python -m streamlit run app.py` from the project root.
3. Select an example question in the sidebar. Choose the fixed or adaptive
   workflow to compare saved runs for the same question and corpus snapshot.
4. Click a citation to read its exact indexed passage and source locator.

The app checks that `processed/chunks.jsonl` matches the frozen corpus report.
Recorded examples work without an API key. Only questions with a saved matching
run can be replayed; other questions require live mode.

## Optional live runs

Set `OPENAI_API_KEY` and `ENABLE_LIVE_RUNS=true` in your local environment or
in Streamlit's deployment secrets. The live workflow uses the course Portkey
gateway and GPT-5 nano, as configured in the notebook. Keep the key out of
GitHub. Live runs have a 40-model-call UI ceiling and save their traces under
`processed/traces/` in the running environment.

The app loads the local MiniLM search model when the first live run starts.
This may take time or require a model download. The saved-run interface remains
available when live mode is disabled.

## Project files

- `notebooks/drone_lost_link.ipynb`: the team's workflow and evaluation notebook.
- `scripts/`: corpus preparation, indexing, and retrieval.
- `corpus/`: frozen source collection and manifest.
- `processed/`: passage index, embeddings, evaluations, and saved traces.
- `app.py` and `ui_data.py`: the user interface and saved-run loader.
- `workflow_runtime.py`: importable copy of the notebook's workflow definitions.
- `scripts/export_workflow.py`: refreshes `workflow_runtime.py` after notebook
  workflow code changes, without editing the original notebook.
- `tests/test_ui_data.py`: checks snapshot matching and citation resolution.

## Rebuild the workflow module

After changing workflow sections in the notebook, run
`python scripts/export_workflow.py` from the project root. Then run the local
tests and check one fixed and one adaptive example before redeploying.

## Deploy

Connect this repository to Streamlit Community Cloud and select `app.py` as
the entrypoint. The root `requirements.txt` supplies dependencies. Add
secrets in Streamlit's app settings if live mode is needed. The default
recorded demo requires no hosted API key.

This is an educational research prototype. ASRS and forum records are
unverified accounts. A gap in this collection does not establish that an
incident never occurred, and the brief is not operational flight guidance.

The repository's MIT license covers the team's code. The FAA, AAIB, DJI,
Autel, and other source documents in `corpus/` retain their own rights;
their inclusion is for source review and does not relicense them.
