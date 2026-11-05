"""Interactive evidence brief for the drone lost-link project."""

from __future__ import annotations

import json
import os
import base64
from pathlib import Path

import streamlit as st

from ui_data import (
    DEMO_QUESTIONS, MODE_LABELS, SOURCE_LABELS, STATUS_LABELS,
    brief_markdown, cited_passages, find_saved_run, load_corpus,
)


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("DRONE_PROJECT_ROOT", APP_ROOT))
ASSET = APP_ROOT / "assets" / "drone-hero.png"

st.set_page_config(
    page_title="Lost Link | Evidence Research",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
html, body {font-family: 'DM Sans', Arial, sans-serif; color:#111817;}
[data-testid="stAppViewContainer"] {background:#fafaf7;}
[data-testid="stHeader"] {background:#fafaf7;}
[data-testid="stSidebar"] {background:#f3f4f0; border-right:1px solid #d7dbd7;}
[data-testid="stSidebar"] {overflow-x:hidden;}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {overflow-x:hidden; overflow-y:hidden;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.55rem;}
[data-testid="stSidebar"] [data-testid="stButton"] button {height:auto; min-height:3.7rem;
  padding:.42rem .55rem; line-height:1.25; font-size:.84rem; white-space:normal; text-align:left;}
[data-testid="stSidebar"] [data-testid="stButton"] button p {text-align:left;}
[data-testid="stSidebarCollapseButton"], [data-testid="collapsedControl"] {display:none;}
.block-container {padding-top:1.25rem; padding-bottom:3rem; max-width:1500px;}
h1,h2,h3 {letter-spacing:-.04em; color:#111817;}
h1 {font-size:clamp(2.8rem,5vw,5.4rem)!important; line-height:1.02!important; font-weight:600!important;}
h2 {font-size:1.55rem!important;}
hr {border-color:#d7dbd7;}
.eyebrow,.meta,.source-tag,.run-tag {font-family:'IBM Plex Mono',monospace; font-size:.7rem;
  letter-spacing:.09em; text-transform:uppercase; color:#42504d;}
.brand {font-family:'IBM Plex Mono',monospace; font-weight:600; letter-spacing:.08em;
  font-size:.9rem; border-bottom:1px solid #151d1a; padding:.45rem 0 1rem;}
.hero-copy {font-size:1.1rem; max-width:34rem; line-height:1.5; color:#44514c; margin-top:.5rem;}
.st-key-hero-visual {background-color:#fff; background-image:linear-gradient(45deg,#edf0eb 1px,transparent 1px),
  linear-gradient(-45deg,#edf0eb 1px,transparent 1px); background-size:42px 42px;
  border:1px solid #d7dbd7; padding:1rem; margin:.6rem 0 1.6rem;}
.hero-image {display:block; width:100%; height:auto;}
.status {font-family:'IBM Plex Mono',monospace; font-size:.72rem; letter-spacing:.04em;
  display:inline-block; padding:.32rem .6rem; border:1px solid #9ca9a3; margin:.35rem 0 .7rem;}
.status-review {background:#f7ead7; border-color:#bd9157;}
.status-gap {background:#e9eeeb; border-color:#9eada5;}
.status-done {background:#e2eee7; border-color:#729781;}
.status-failed {background:#fae9e7; border-color:#bf8580;}
.answer {font-size:1.12rem; line-height:1.6; border-left:3px solid #182721; padding:.1rem 0 .1rem 1rem;}
.note {color:#5e6964; font-size:.87rem; line-height:1.45;}
.step {border-left:1px solid #a8b7ae; padding:.2rem .15rem .75rem .9rem; margin-left:.2rem;}
.step-title {font-family:'IBM Plex Mono',monospace; font-size:.73rem; text-transform:uppercase; letter-spacing:.05em;}
[data-testid="stChatMessage"] {background:#fff; border:1px solid #e2e6e1; border-radius:0;}
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] {display:none;}
[data-testid="stButton"] button {border-radius:0; border:1px solid #89958d; font-weight:600;}
[data-testid="stButton"] button:hover {border-color:#111817; color:#111817;}
[data-testid="stChatInput"] textarea {border-radius:0;}
[data-testid="stMetric"] {background:#fff; border:1px solid #e1e5df; padding:.5rem .8rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def corpus_context(root: str):
    return load_corpus(Path(root))


@st.cache_data(show_spinner=False)
def saved_run(root: str, question: str, mode: str, snapshot: str, index_hash: str):
    return find_saved_run(Path(root), question, mode, snapshot, index_hash)


@st.cache_resource(show_spinner="Loading the search model…")
def runtime():
    import workflow_runtime
    return workflow_runtime


def app_secret(name: str) -> str:
    value = os.environ.get(name, "")
    if value:
        return value
    try:
        return str(st.secrets.get(name, ""))
    except (FileNotFoundError, KeyError):
        return ""


def status_class(status: str) -> str:
    if status == "human_review_required":
        return "status-review"
    if status == "completed_with_gaps":
        return "status-gap"
    if status == "completed":
        return "status-done"
    return "status-failed"


def show_brief(run: dict, chunks: dict[str, dict], run_index: int) -> None:
    status = run.get("status", "")
    label = STATUS_LABELS.get(status, status.replace("_", " ").title())
    st.markdown(f'<span class="status {status_class(status)}">{label}</span>', unsafe_allow_html=True)
    answer = run.get("brief", {}).get("short_answer") or "The available documents do not support a short answer."
    st.markdown('<div class="eyebrow">Short answer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer">{escape_html(answer)}</div>', unsafe_allow_html=True)
    st.write("")

    for section_index, section in enumerate(run.get("brief", {}).get("sections", [])):
        claims = section.get("claims", [])
        if not claims:
            continue
        st.subheader(section.get("label") or section.get("section", "Evidence"))
        for claim_index, claim in enumerate(claims):
            st.write(claim.get("text", ""))
            conditions = claim.get("conditions") or []
            if conditions:
                st.caption("Condition: " + "; ".join(conditions))
            verification = run.get("verifications", {}).get(claim.get("claim_id"), {})
            if verification:
                st.caption("Citation check: " + verification.get("status", "not checked").replace("_", " "))
            for cite_index, chunk_id in enumerate(claim.get("citations", [])):
                chunk = chunks.get(chunk_id)
                if chunk:
                    label = f"{SOURCE_LABELS.get(chunk.get('source_type'), chunk.get('source_type', 'Source'))} · {chunk.get('locator', '')}"
                    if st.button(label, key=f"cite-{run_index}-{section_index}-{claim_index}-{cite_index}"):
                        st.session_state.active_citation = chunk_id
            st.divider()

    gaps = run.get("brief", {}).get("gaps", [])
    if gaps:
        st.subheader("Gaps and conflicts")
        for gap in gaps:
            st.write(f"**{gap.get('kind', 'Gap').replace('_', ' ').title()}** · {gap.get('text', '')}")
    st.caption("This brief describes the saved evidence. Reported incidents are not verified findings or flight instructions.")


def escape_html(text: str) -> str:
    import html
    return html.escape(text).replace("\n", "<br>")


def show_activity(run: dict, chunks: dict[str, dict]) -> None:
    selected = st.session_state.get("active_citation")
    if selected in chunks:
        chunk = chunks[selected]
        st.markdown('<div class="eyebrow">Selected citation</div>', unsafe_allow_html=True)
        st.markdown(f"### {chunk.get('title', 'Source')}")
        st.caption(f"{SOURCE_LABELS.get(chunk.get('source_type'), chunk.get('source_type', 'Source'))} · "
                   f"{chunk.get('model') or 'General'} · {chunk.get('locator', '')}")
        st.write(chunk.get("text", ""))
        if chunk.get("url", "").startswith("https://"):
            st.link_button("Open original source ↗", chunk["url"])
        st.divider()

    st.markdown('<div class="eyebrow">Tool use</div>', unsafe_allow_html=True)
    searches = run.get("searches", [])
    decisions = run.get("decisions", [])
    verifications = run.get("verifications", {})
    st.caption(f"{len(searches)} searches · {len(decisions)} agent decisions · {len(verifications)} citation checks")
    with st.expander("Searches and passages", expanded=True):
        if not searches:
            st.write("No searches were needed for this question.")
        for item in searches:
            st.markdown(f'<div class="step"><div class="step-title">{escape_html(item.get("search_id", "Search"))} '
                        f'· {escape_html(item.get("section", ""))}</div></div>', unsafe_allow_html=True)
            st.write(item.get("query", ""))
            alternatives = item.get("alternate_queries") or []
            if alternatives:
                st.caption("Also searched: " + " · ".join(alternatives))
            if item.get("reason"):
                st.caption("Why: " + item["reason"])
            st.caption(f"{len(item.get('returned_ids', []))} found · {len(item.get('kept_ids', []))} kept")
    with st.expander("Agent decisions", expanded=bool(decisions)):
        if not decisions:
            st.write("The fixed workflow followed its preset steps.")
        for step in decisions:
            st.markdown(f"**{step.get('action', 'decision').replace('_', ' ').title()}** · "
                        f"{step.get('section', '')}")
            st.caption(step.get("reason") or step.get("continue_or_stop", ""))
    with st.expander("Source list"):
        passages = cited_passages(run, chunks)
        if not passages:
            st.write("No source passage was cited.")
        for passage in passages:
            st.write(f"**{passage.get('title', '')}** · {passage.get('locator', '')}")
            st.caption(SOURCE_LABELS.get(passage.get("source_type"), passage.get("source_type", "")))


try:
    report, chunks, chunks_hash = corpus_context(str(PROJECT_ROOT))
except (OSError, ValueError, json.JSONDecodeError) as exc:
    st.error(f"The saved corpus could not be loaded: {exc}")
    st.stop()

if "turns" not in st.session_state:
    st.session_state.turns = []
if "active_citation" not in st.session_state:
    st.session_state.active_citation = None

st.markdown('<div class="brand">◈ &nbsp; LOST LINK <span style="font-weight:400">/ EVIDENCE RESEARCH</span></div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="eyebrow">Saved example briefs</div>', unsafe_allow_html=True)
    st.caption("Open a saved answer and inspect its sources.")
    for index, example in enumerate(DEMO_QUESTIONS):
        if st.button(example, key=f"demo-{index}", use_container_width=True):
            st.session_state.pending_example = example
    st.divider()
    st.caption(f"Corpus snapshot · {report['corpus_snapshot']}")
    st.caption(f"{report['source_count']} sources · {report['chunk_count']} passages")
    st.caption("Educational research prototype. Verify source conditions before operational use.")
    st.caption("Drone and company names are used only for education and research, not marketing or monetization. No manufacturer endorses this project.")

if not st.session_state.turns:
    hero_left, hero_right = st.columns([1.05, 1], gap="large", vertical_alignment="center")
    with hero_left:
        st.markdown('<div class="eyebrow">Evidence-grounded drone research</div>', unsafe_allow_html=True)
        st.title("When the link goes quiet, what does the evidence say?")
        st.markdown('<div class="hero-copy">Ask how a documented drone model responds to a lost control link. '
                    'Read the answer, inspect each source, and see why the agent searched again or stopped.</div>',
                    unsafe_allow_html=True)
        st.write("")
        st.caption("Choose a saved question on the left, or ask your own below.")
    with hero_right:
        with st.container(key="hero-visual"):
            image_data = base64.b64encode(ASSET.read_bytes()).decode("ascii")
            st.markdown(f'<img class="hero-image" src="data:image/png;base64,{image_data}" alt="Monochrome illustration of a drone">',
                        unsafe_allow_html=True)
else:
    conversation, evidence = st.columns([1.65, 1], gap="large")
    with conversation:
        for run_index, turn in enumerate(st.session_state.turns):
            with st.chat_message("user"):
                st.write(turn["question"])
            with st.chat_message("assistant"):
                st.caption(f"{MODE_LABELS.get(turn['result'].get('mode'), turn['result'].get('mode'))} · "
                           f"{'Live run' if turn['live'] else 'Recorded run'}")
                show_brief(turn["result"], chunks, run_index)
        current = st.session_state.turns[-1]["result"]
        st.download_button("Download brief", brief_markdown(current, chunks),
                           file_name="drone-lost-link-brief.md", mime="text/markdown")
        if not st.session_state.turns[-1]["live"]:
            other_mode = "fixed" if current.get("mode") == "full" else "full"
            other = saved_run(str(PROJECT_ROOT), current["question"], other_mode,
                              report["corpus_snapshot"], chunks_hash)
            if other:
                with st.expander("Compare the fixed workflow and adaptive agent"):
                    st.caption("Saved runs for the same question and corpus snapshot.")
                    for name, item in (("Adaptive agent", current if current["mode"] == "full" else other),
                                       ("Fixed workflow", current if current["mode"] == "fixed" else other)):
                        st.markdown(f"**{name}** · {STATUS_LABELS.get(item.get('status'), item.get('status'))}")
                        st.caption(f"{item.get('brief_stats', {}).get('claims', 0)} claims · "
                                   f"{item.get('brief_stats', {}).get('gaps', 0)} gaps · "
                                   f"{item.get('cost', {}).get('searches', 0)} searches · "
                                   f"{item.get('cost', {}).get('model_calls', 0)} model calls")
    with evidence:
        st.subheader("Evidence trail")
        show_activity(st.session_state.turns[-1]["result"], chunks)
        run = st.session_state.turns[-1]["result"]
        with st.expander("Run details"):
            st.write(f"Stop reason: {run.get('stop_reason', '')}")
            st.write(f"Corpus snapshot: {run.get('corpus_snapshot', '')}")
            st.write(f"Model calls: {run.get('cost', {}).get('model_calls', 0)}")
            st.write(f"Searches: {run.get('cost', {}).get('searches', 0)}")
            st.write(f"Run time: {run.get('cost', {}).get('seconds', 0)} s")
            st.download_button("Download trace JSON", json.dumps(run, indent=2),
                               file_name="drone-lost-link-run.json", mime="application/json")

question = st.chat_input("Ask about a lost link, a model, or a reported incident")
example = st.session_state.pop("pending_example", None)
pending = example or question
if pending:
    pending = pending.strip()
    if len(pending) > 500:
        st.warning("Please keep the question under 500 characters.")
    elif example:
        result = saved_run(str(PROJECT_ROOT), pending, "full", report["corpus_snapshot"], chunks_hash)
        if result is None:
            st.warning("The saved example does not match the current corpus snapshot.")
        else:
            st.session_state.turns.append({"question": pending, "result": result, "live": False})
            st.session_state.active_citation = None
            st.rerun()
    elif not app_secret("OPENAI_API_KEY"):
        st.warning("Live questions need a course API key. Add OPENAI_API_KEY to your local Streamlit secrets, then try again.")
    else:
        with st.status("Searching and checking the evidence…", expanded=True):
            engine = runtime()
            engine.configure_live_client(app_secret("OPENAI_API_KEY"))
            result = engine.run_workflow(pending, config=engine.Config(mode="full", max_model_calls=40))
            st.write(f"Run finished: {STATUS_LABELS.get(result.get('status'), result.get('status'))}")
        st.session_state.turns.append({"question": pending, "result": result, "live": True})
        st.session_state.active_citation = None
        st.rerun()
