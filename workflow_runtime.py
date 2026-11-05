"""Workflow exported from notebooks/drone_lost_link.ipynb.

Run scripts/export_workflow.py after changing workflow code in the notebook.
The web app uses the same schemas and functions without running notebook demos.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("DRONE_PROJECT_ROOT", Path(__file__).resolve().parent))
OUTPUT_ROOT = PROJECT_ROOT / "processed"
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
_report = json.loads((OUTPUT_ROOT / "corpus_validation_report.json").read_text(encoding="utf-8"))
EMBEDDINGS_PATH = PROJECT_ROOT / _report["embedding_path"]
RUN_LIVE = False
llm_client = None


# Notebook cell 24
import json, re, time, uuid, hashlib

from dataclasses import dataclass

from datetime import datetime, timezone

from typing import Literal, Optional

import pandas as pd

from openai import OpenAI

from pydantic import BaseModel, Field, ValidationError, field_validator

from hybrid_retrieval import LocalEmbeddingClient, HybridSearch

MODEL = '@30875-fall26-openai/gpt-5-nano'

MODELS = ['DJI Matrice 30T', 'DJI Matrice 4T', 'Autel EVO Max 4N', 'DJI Matrice 210 V1']

CHUNKS_PATH = OUTPUT_ROOT / 'chunks.jsonl'

SEARCH = HybridSearch(CHUNKS_PATH, EMBEDDINGS_PATH, LocalEmbeddingClient())

CORPUS_REPORT = json.loads((OUTPUT_ROOT / 'corpus_validation_report.json').read_text())

CHUNKS_SHA256 = hashlib.sha256(CHUNKS_PATH.read_bytes()).hexdigest()

TRACE_DIR = OUTPUT_ROOT / 'traces'

@dataclass(frozen=True)
class Config:
    mode: str = 'full'                      # 'full' (agent) or 'fixed' (baseline)
    max_revisions: int = 2
    # ---- Evidence agent, per section ----
    agent_base_searches_per_section: int = 2    # searches the agent starts with
    agent_max_searches_per_section: int = 4     # hard cap: each search that adds a new passage earns one more, up to this
    agent_max_tool_calls_per_section: int = 7
    agent_max_turns_per_section: int = 8
    agent_results_shown: int = 10               # results an agent search shows; base searches keep only top_k
    # ---- Evidence agent, whole run ----
    agent_max_searches_per_run: int = 8
    agent_max_turns_per_run: int = 40
    max_reopens_per_run: int = 1
    # ---- Whole run ----
    max_searches: int = 40
    max_model_calls: int = 120              # worst-case ceiling; typical runs use far fewer
    top_k: int = 5
    max_claims_per_section: int = 5
    max_claim_words: int = 40
    max_gaps: int = 6
    max_short_answer_words: int = 120


# Notebook cell 26
FlightPhase = Literal['preflight', 'takeoff', 'in_flight', 'landing']

class SearchArgs(BaseModel):
    query: str = Field(min_length=3, max_length=200)
    alternate_queries: list[str] = Field(default_factory=list, max_length=2)   # other wordings of the same need
    section: str
    flight_phase: Optional[FlightPhase] = None
    follows_from: Optional[str] = None      # chunk_id of the passage this query was built from, if any
    reason: str = Field(min_length=10)      # the agent must say why it is searching

    @field_validator('alternate_queries')
    @classmethod
    def _phrasing_length(cls, values: list[str]) -> list[str]:
        for v in values:
            if not 3 <= len(v.strip()) <= 200:
                raise ValueError('each alternate query must be 3-200 characters')
        return values

class LookupArgs(BaseModel):
    chunk_id: str
    reason: str = Field(min_length=10)

class ExpandArgs(BaseModel):
    chunk_id: str
    direction: Literal['before', 'after', 'both']
    reason: str = Field(min_length=10)

class FinishArgs(BaseModel):
    coverage: Literal['sufficient', 'partial', 'insufficient']
    missing: list[str]
    conflicts_noticed: list[str]
    reason: str = Field(min_length=10)

class IntakeOut(BaseModel):
    intent: Literal['in_scope', 'out_of_scope', 'unsupported_model']
    models: list[Literal['DJI Matrice 30T', 'DJI Matrice 4T', 'Autel EVO Max 4N', 'DJI Matrice 210 V1']]   # must match MODELS
    focus: str
    flight_phase: Optional[FlightPhase]
    reason: str

class ClaimDraft(BaseModel):
    text: str
    citations: list[str]          # chunk_ids
    quote: str                    # verbatim span from a cited passage
    conditions: list[str]         # e.g. "if Signal Lost Action is set to RTH"
    assessment: Literal['supported', 'supported_with_conditions', 'conflicting', 'not_enough_evidence']

class SectionDraftOut(BaseModel):
    claims: list[ClaimDraft]
    abstain_reason: Optional[str]

class GapItem(BaseModel):
    kind: Literal['missing', 'unclear', 'conflict']
    text: str
    related_claim_ids: list[str]

class GapsOut(BaseModel):
    gaps: list[GapItem]

class Verification(BaseModel):
    claim_id: str
    status: Literal['supported', 'partial', 'not_supported']
    missing_conditions: list[str]
    rationale: str
    supporting_ids: list[str]

class VerificationOut(BaseModel):
    verifications: list[Verification]

class RevisionItem(BaseModel):
    claim_id: str
    action: Literal['rewrite', 'drop', 'to_gap', 'reopen_section']
    reason: str
    new_claim: Optional[ClaimDraft]

class RevisionOut(BaseModel):
    decisions: list[RevisionItem]

class ShortAnswerOut(BaseModel):
    short_answer: str

class Claim(ClaimDraft):
    claim_id: str
    section: str                  # 'faa', 'manual:<model>' or 'incident:<model>'
    tier: Literal['official', 'manufacturer', 'unverified']
    incident_label: Optional[str] = None   # 'narrative_explicit' or 'summary_only' for ASRS claims

def word_count(text: str) -> int:
    return len(text.split())

def length_problems(claims: list, cfg: Config) -> list[str]:
    """Empty list = within limits. Otherwise the problems are sent back to the model for one retry."""
    problems = []
    if len(claims) > cfg.max_claims_per_section:
        problems.append(f'{len(claims)} claims; keep only the {cfg.max_claims_per_section} most important')
    for c in claims:
        if word_count(c.text) > cfg.max_claim_words:
            problems.append(f'claim has {word_count(c.text)} words (limit {cfg.max_claim_words}): "{c.text[:60]}..."')
    return problems


# Notebook cell 28
REASONING_EFFORT = 'low'   # can change to medium if needed

class LLMError(Exception):
    """A model call failed twice. The calling stage fails closed."""

class BudgetExceeded(Exception):
    """The run used its model-call budget. The run ends as 'failed'."""

class LLM:
    """Real model calls through Portkey. Create one per run so the counters start at zero."""

    def __init__(self, client, max_calls: int):
        self.client = client
        self.max_calls = max_calls
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def _start_call(self):
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f'model-call budget of {self.max_calls} used')
        self.calls += 1

    def _add_usage(self, response):
        if response.usage:
            self.tokens_in += response.usage.input_tokens
            self.tokens_out += response.usage.output_tokens

    def structured(self, name: str, instructions: str, payload: dict, schema):
        """One structured-output call, retried once. Returns an instance of `schema`."""
        error = None
        for attempt in range(2):
            self._start_call()
            try:
                response = self.client.responses.parse(
                    model=MODEL, instructions=instructions, input=json.dumps(payload, default=str),
                    text_format=schema, reasoning={'effort': REASONING_EFFORT})
            except Exception as exc:
                error = exc
                continue
            self._add_usage(response)
            if response.output_parsed is not None:
                return response.output_parsed
            error = 'empty or refused reply'
        raise LLMError(f'{name} failed twice: {error}')

    def tool_turn(self, name: str, instructions: str, input_items, tools: list, previous_id=None) -> dict:
        """One agent step. Returns {'id', 'calls': [{'call_id', 'name', 'arguments'}], 'text'}."""
        error = None
        for attempt in range(2):
            self._start_call()
            try:
                request = dict(model=MODEL, instructions=instructions, input=input_items, tools=tools,
                               parallel_tool_calls=False, reasoning={'effort': REASONING_EFFORT})
                if previous_id:
                    request['previous_response_id'] = previous_id
                response = self.client.responses.create(**request)
            except Exception as exc:
                error = exc
                continue
            self._add_usage(response)
            calls = [{'call_id': item.call_id, 'name': item.name, 'arguments': item.arguments}
                     for item in response.output if item.type == 'function_call']
            return {'id': response.id, 'calls': calls, 'text': response.output_text or ''}
        raise LLMError(f'{name} failed twice: {error}')

class FakeLLM(LLM):
    """Same interface, no API. `handlers` maps a call name to a function that returns the scripted answer.
    For structured calls the function gets the payload and returns a dict matching the schema.
    For tool turns it gets the input items and returns a list of (tool_name, arguments_dict)."""

    def __init__(self, handlers: dict, max_calls: int = 70):
        super().__init__(client=None, max_calls=max_calls)
        self.handlers = handlers
        self.log = []                         # names of the calls made, for the offline checks

    def structured(self, name, instructions, payload, schema):
        error = None
        for attempt in range(2):
            self._start_call()
            self.log.append(name)
            try:
                return schema.model_validate(self.handlers[name](payload))
            except Exception as exc:          # a handler can raise to simulate a failed call
                error = exc
        raise LLMError(f'{name} failed twice: {error}')

    def tool_turn(self, name, instructions, input_items, tools, previous_id=None):
        self._start_call()
        self.log.append(name)
        calls = [{'call_id': f'call_{uuid.uuid4().hex[:6]}', 'name': tool, 'arguments': json.dumps(args)}
                 for tool, args in self.handlers[name](input_items)]
        return {'id': f'fake_{uuid.uuid4().hex[:6]}', 'calls': calls, 'text': ''}


# Notebook cell 30
@dataclass
class RunState:
    """Everything about ONE run. Created fresh by each run, never reused."""
    question: str
    config: Config
    llm: LLM
    run_id: str = ''
    sections: list = None        # planned section names, e.g. ['faa', 'manual:DJI Matrice 30T', ...]
    pools: dict = None           # section -> {chunk_id: passage dict} (evidence kept for that section)
    searches: list = None        # one dict per search
    decisions: list = None       # one dict per agent / revision decision
    events: list = None          # the full trace, also written to disk
    agent_searches: int = 0      # across the whole run
    reopens: int = 0
    intake: dict = None          # the intake decision: intent, models, focus
    assessments: dict = None     # section -> the evidence agent's final assessment
    drafts: dict = None          # section -> {'status', 'claims', 'abstain_reason'}
    gaps: list = None            # gap and conflict items for the brief
    removed_claims: list = None  # claims taken out of the brief by revision
    gate_results: dict = None    # claim_id -> deterministic citation checks
    verifications: dict = None   # claim_id -> verifier result
    first_draft_results: list = None   # verification results before any revision
    revisions_used: int = 0
    short_answer: str = ''

    def __post_init__(self):
        self.run_id = self.run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{self.config.mode}-{uuid.uuid4().hex[:4]}"
        self.sections, self.searches, self.decisions, self.events = [], [], [], []
        self.gaps, self.removed_claims, self.first_draft_results = [], [], []
        self.pools, self.assessments, self.drafts, self.gate_results, self.verifications = {}, {}, {}, {}, {}
        self.trace_path = TRACE_DIR / f'{self.run_id}.jsonl'

    def log(self, event_type: str, **data):
        """Add one event to the trace and append it to the run's JSONL file."""
        event = {'n': len(self.events) + 1, 'time': datetime.now(timezone.utc).isoformat(),
                 'type': event_type, **data}
        self.events.append(event)
        with self.trace_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(event, default=str) + '\n')

    def decide(self, section: str, action: str, reason: str, continue_or_stop: str, **details):
        """Record one decision (e.g. search, finish, rewrite, drop) and why it was made."""
        decision = {'decision_id': f'D{len(self.decisions) + 1:03d}', 'section': section, 'action': action,
                    'continue_or_stop': continue_or_stop, 'reason': reason, **details}
        self.decisions.append(decision)
        self.log('decision', **decision)

    def all_claims(self) -> list:
        """Every claim currently in the brief, in section order."""
        return [c for s in self.sections for c in self.drafts.get(s, {}).get('claims', [])]

def trace_table(state: RunState) -> pd.DataFrame:
    return pd.DataFrame([{'n': e['n'], 'type': e['type'], 'section': e.get('section', ''),
                          'details': json.dumps({k: v for k, v in e.items() if k not in ('n', 'time', 'type', 'section')},
                                                default=str)[:200]}
                         for e in state.events])

def decision_table(state: RunState) -> pd.DataFrame:
    return pd.DataFrame(state.decisions)


# Notebook cell 32
import csv

def incident_label(passage: dict) -> Optional[str]:
    if passage['source_type'] not in {'asrs', 'aaib', 'forum'}:
        return None
    return passage.get('evidence_label', 'unlabeled')

LOST_LINK_RE = re.compile(
    r"lost[- ]?link|signal (is )?(lost|loss)|loss of (the )?(remote controller|control|c2|communication)"
    r"|lost control|c2 link|failsafe|signal lost action|return[- ]to[- ]home|\brth\b", re.IGNORECASE)

def is_relevant(passage: dict, rank: int, section: str, agent_query: bool) -> bool:
    if section_kind(section) == 'incident':
        if passage['source_type'] == 'forum' and 'link_loss' not in passage.get('event_type', ''):
            return False
        return passage.get('evidence_label') not in {'no_link_loss', 'control_loss_unconfirmed_link'}
    return bool(LOST_LINK_RE.search(passage['text'])) or (agent_query and rank <= 2)

SECTION_SOURCE = {'faa': {'faa_guidance'}, 'manual': {'manufacturer'}, 'incident': {'asrs', 'aaib', 'forum'}}

SECTION_TIER = {'faa': 'official', 'manual': 'manufacturer', 'incident': 'unverified'}

BASE_QUERY = {
    'faa': 'lost link loss of control link communication remote pilot contingency procedures',
    'manual': 'remote controller signal lost failsafe return to home signal lost action hover land reconnect',
    'incident': 'lost link lost control signal lost return to home flyaway',
}

def plan_sections(models: list[str]) -> list[str]:
    return ['faa'] + [f'manual:{m}' for m in models] + [f'incident:{m}' for m in models]

def section_kind(section: str) -> str:
    return section.split(':')[0]

def section_model(section: str) -> Optional[str]:
    return section.split(':', 1)[1] if ':' in section else None

PHRASING_RRF_K = 60     # same constant as the hybrid search's own rank fusion

def phrasings_for(query: str, alternates: Optional[list[str]]) -> list[str]:
    """The main query plus any alternate wordings, without duplicates (ignoring case and spacing)."""
    out, seen = [], set()
    for q in [query] + list(alternates or []):
        key = ' '.join(q.lower().split())
        if key and key not in seen:
            seen.add(key)
            out.append(q.strip())
    return out

def run_one_phrasing(query: str, search_filters: list[dict]) -> list[dict]:
    """One wording across every filter, in the original order (source types sorted, first rank kept)."""
    hits, errors = [], []
    for filters in search_filters:
        for attempt in range(2):
            try:
                hits.extend(SEARCH.document_search(query, filters, top_k=10))
                break
            except Exception as exc:
                if attempt == 1:
                    errors.append(f'{filters}: {type(exc).__name__}: {exc}')
    if errors:
        raise RuntimeError('Search failed for a source: ' + '; '.join(errors))
    # One passage may be returned by more than one filter. Keep its first rank.
    hits = list({h['chunk_id']: h for h in reversed(hits)}.values())
    hits.reverse()
    return hits

def merge_phrasings(ranked_lists: list[list[dict]]) -> list[dict]:
    """Reciprocal rank fusion across wordings. A passage found by several wordings, or ranked high by one, rises.
    With one wording the order is unchanged."""
    if len(ranked_lists) == 1:
        return ranked_lists[0]
    score, first_hit = {}, {}
    for hits in ranked_lists:
        for rank, h in enumerate(hits, 1):
            score[h['chunk_id']] = score.get(h['chunk_id'], 0.0) + 1 / (PHRASING_RRF_K + rank)
            first_hit.setdefault(h['chunk_id'], h)
    order = sorted(score, key=lambda cid: (-score[cid], cid))
    return [first_hit[cid] for cid in order]

def search_section(state: RunState, section: str, query: str, origin: str,
                   reason: str = '', flight_phase: Optional[str] = None,
                   follows_from: Optional[str] = None, alternate_queries: Optional[list[str]] = None) -> dict:
    """Search one section. Relevant passages are added to that section's pool. Returns the search record.
    An agent search may send alternate wordings, which are searched too and merged with the main query.
    An agent search also records the results below the cutoff, which the agent may choose to keep."""
    if len(state.searches) >= state.config.max_searches:
        raise BudgetExceeded(f'search budget of {state.config.max_searches} used')
    if origin != 'agent':
        alternate_queries = None          # base searches stay single-wording in every mode
    phrasings = phrasings_for(query, alternate_queries)
    model = section_model(section)
    kind = section_kind(section)
    allowed_sources = SECTION_SOURCE[kind]

    # Search inside each incident source type before ranking. A global top-k search
    # can return only manuals and leave a real incident invisible after filtering.
    search_filters = []
    for source_type in sorted(allowed_sources):
        filters = {'source_type': source_type}
        if model:
            filters['model'] = model
        if flight_phase:
            filters['flight_phase'] = flight_phase
        search_filters.append(filters)
    if kind == 'incident' and model == 'DJI Matrice 210 V1':
        alias_filters = {'source_type': 'asrs', 'model': 'DJI Matrice 210'}
        if flight_phase:
            alias_filters['flight_phase'] = flight_phase
        search_filters.append(alias_filters)

    def in_section(h):
        return h['source_type'] in allowed_sources and (h['model'] == model or (
            model == 'DJI Matrice 210 V1' and h['source_id'] == 'asrs_m210_1918661' and h['model'] == 'DJI Matrice 210'))

    per_phrasing = [[h for h in run_one_phrasing(q, search_filters) if in_section(h)] for q in phrasings]
    hits = merge_phrasings(per_phrasing)
    filters = search_filters  # keep every actual filter in the trace
    error = None

    agent_query = origin == 'agent'
    pool = state.pools.setdefault(section, {})
    kept, below = [], []
    for rank, h in enumerate(hits, 1):
        relevant = is_relevant(h, rank, section, agent_query)
        if relevant and len(kept) < state.config.top_k:
            kept.append(h)
        elif (relevant or kind != 'incident') and h['chunk_id'] not in pool:
            # Incident reports screened out (e.g. no_link_loss) are never shown again,
            # and passages already in the pool aren't worth showing as candidates.
            below.append(h)
    if agent_query:
        below = below[: max(0, state.config.agent_results_shown - len(kept))]
    else:
        below = []          # base searches behave exactly as before

    new_ids = [h['chunk_id'] for h in kept if h['chunk_id'] not in pool]
    for h in kept:
        pool.setdefault(h['chunk_id'], {**h, 'incident_label': incident_label(h), 'added_by': 'search'})

    # For each wording: how many of the kept and of the new passages it found (shows which wording mattered).
    kept_ids, new_set = {h['chunk_id'] for h in kept}, set(new_ids)
    phrasing_stats = [{'query': q, 'returned': len(found),
                       'kept': len(kept_ids & {h['chunk_id'] for h in found}),
                       'new': len(new_set & {h['chunk_id'] for h in found})}
                      for q, found in zip(phrasings, per_phrasing)]

    record = {'search_id': f'S{len(state.searches) + 1:03d}', 'section': section, 'origin': origin,
              'query': query, 'alternate_queries': phrasings[1:], 'phrasing_stats': phrasing_stats,
              'filters': filters, 'backend_calls': len(search_filters) * len(phrasings), 'reason': reason,
              'follows_from': follows_from, 'returned_ids': [h['chunk_id'] for h in hits],
              'kept_ids': [h['chunk_id'] for h in kept], 'new_ids': new_ids,
              'below_cutoff_ids': [h['chunk_id'] for h in below], 'promoted_ids': [], 'context_ids': [],
              'error': error}
    state.searches.append(record)
    state.log('search', **record)
    return record

def citation_lookup(chunk_id: str) -> dict:
    try:
        return {'found': True, **SEARCH.citation_lookup(chunk_id)}
    except KeyError:
        return {'chunk_id': chunk_id, 'found': False}

CHUNK_ID_RE = re.compile(r'(.+)_c(\d{4})')

def is_whole_record(passage: dict) -> bool:
    """Incident CSV rows are indexed as one passage per report, so they have no neighbors."""
    return passage['locator'].startswith('record ')

def neighbor_ids(chunk_id: str, direction: str) -> list[str]:
    """The passages just before and/or after this one in the same source (reading order, across pages)."""
    m = CHUNK_ID_RE.fullmatch(chunk_id)
    if not m or chunk_id not in SEARCH.by_id or is_whole_record(SEARCH.by_id[chunk_id]):
        return []
    source, n = m.group(1), int(m.group(2))
    steps = {'before': [-1], 'after': [1], 'both': [-1, 1]}[direction]
    ids = [f'{source}_c{n + s:04d}' for s in steps if n + s >= 1]
    return [i for i in ids if i in SEARCH.by_id and not is_whole_record(SEARCH.by_id[i])]

def pool_section_of(state: RunState, chunk_id: str) -> Optional[str]:
    """The section whose evidence pool holds this passage, if any."""
    return next((s for s, pool in state.pools.items() if chunk_id in pool), None)

def owning_search(state: RunState, section: str, chunk_id: str) -> Optional[dict]:
    """The search record that put this passage into the section's pool."""
    return next((r for r in state.searches if r['section'] == section and chunk_id in r['kept_ids']), None)

def add_to_pool(state: RunState, section: str, chunk_id: str, record: dict, how: str) -> bool:
    """Add a passage the agent chose to keep ('promoted') or opened as a neighbor ('context').
    It is credited to the search record it came from, so rebuilding pools from kept_ids (score_brief) includes it.
    Returns False if the passage was already in the pool."""
    pool = state.pools.setdefault(section, {})
    if chunk_id in pool:
        return False
    passage = SEARCH.by_id[chunk_id]
    pool[chunk_id] = {**passage, 'incident_label': incident_label(passage), 'added_by': how}
    record['kept_ids'].append(chunk_id)
    record[f'{how}_ids'].append(chunk_id)
    state.log('pool_add', section=section, chunk_id=chunk_id, how=how, search_id=record['search_id'])
    return True


# Notebook cell 34
def agent_tools(sections: list[str]) -> list[dict]:
    """The four tools the evidence agent may call. `section` is limited to this run's planned sections."""
    nullable_str = {'anyOf': [{'type': 'string'}, {'type': 'null'}]}
    return [
        {'type': 'function', 'name': 'document_search', 'strict': True,
         'description': ('Search the approved collection within ONE planned section. The code applies that '
                         "section's source and model filters. Use a specific query and say why you need it. "
                         'alternate_queries: up to 2 other wordings of the same need, using terms other sources '
                         'might use (e.g. "Signal Loss Action", "C2 link lost"); all wordings are searched and '
                         'merged as ONE search. Use [] if one wording is enough. '
                         'If the query uses a term from a passage you already read, set follows_from to that '
                         "passage's chunk_id; otherwise null. The result lists the passages kept, the next "
                         'results below the cutoff, and what each wording found.'),
         'parameters': {'type': 'object', 'additionalProperties': False,
                        'required': ['query', 'alternate_queries', 'section', 'flight_phase', 'follows_from', 'reason'],
                        'properties': {
                            'query': {'type': 'string'},
                            'alternate_queries': {'type': 'array', 'items': {'type': 'string'}},
                            'section': {'type': 'string', 'enum': sections},
                            'flight_phase': {'anyOf': [{'type': 'string', 'enum': ['preflight', 'takeoff', 'in_flight', 'landing']},
                                                       {'type': 'null'}]},
                            'follows_from': nullable_str,
                            'reason': {'type': 'string'}}}},
        {'type': 'function', 'name': 'citation_lookup', 'strict': True,
         'description': ('Open the full text of a passage you have been shown. Opening a passage listed below the '
                         'cutoff of one of your searches also keeps it as evidence for that section.'),
         'parameters': {'type': 'object', 'additionalProperties': False, 'required': ['chunk_id', 'reason'],
                        'properties': {'chunk_id': {'type': 'string'}, 'reason': {'type': 'string'}}}},
        {'type': 'function', 'name': 'expand_context', 'strict': True,
         'description': ('Open the passage just before and/or after one already in the evidence, from the same '
                         'document, when it looks cut off (e.g. a list of conditions continues). New passages are '
                         'kept as evidence. Not available for incident records, which are whole reports.'),
         'parameters': {'type': 'object', 'additionalProperties': False, 'required': ['chunk_id', 'direction', 'reason'],
                        'properties': {'chunk_id': {'type': 'string'},
                                       'direction': {'type': 'string', 'enum': ['before', 'after', 'both']},
                                       'reason': {'type': 'string'}}}},
        {'type': 'function', 'name': 'finish_assessment', 'strict': True,
         'description': 'Stop gathering evidence for this section and say how well the evidence covers it.',
         'parameters': {'type': 'object', 'additionalProperties': False,
                        'required': ['coverage', 'missing', 'conflicts_noticed', 'reason'],
                        'properties': {
                            'coverage': {'type': 'string', 'enum': ['sufficient', 'partial', 'insufficient']},
                            'missing': {'type': 'array', 'items': {'type': 'string'}},
                            'conflicts_noticed': {'type': 'array', 'items': {'type': 'string'}},
                            'reason': {'type': 'string'}}}},
    ]

AGENT_TOOL_NAMES = ('document_search', 'citation_lookup', 'expand_context', 'finish_assessment')

@dataclass
class AgentLoop:
    """Counters for the agent's work on ONE section."""
    section: str
    searches_allowed: int = 2     # starts at Config.agent_base_searches_per_section and grows as searches are earned
    tool_calls: int = 0
    searches: int = 0
    turns: int = 0
    repeats: int = 0
    bad_finishes: int = 0
    seen_calls: set = None
    search_ids: set = None        # searches made in this loop (only these can earn more searches)

    def __post_init__(self):
        self.seen_calls, self.search_ids = set(), set()

def short_error(exc: ValidationError) -> str:
    return '; '.join(f"{e['loc'][0] if e['loc'] else 'arguments'}: {e['msg']}" for e in exc.errors())

def snippet(passage: dict, n: int = 300) -> dict:
    return {'chunk_id': passage['chunk_id'], 'locator': passage['locator'],
            'incident_label': passage.get('incident_label'), 'text': passage['text'][:n]}

def shown_to_agent(state: RunState, chunk_id: str) -> bool:
    """A passage the agent has been shown: in an evidence pool, or listed below the cutoff of an agent search."""
    return pool_section_of(state, chunk_id) is not None or any(
        chunk_id in r['below_cutoff_ids'] for r in state.searches if r['origin'] == 'agent')

def earn_search(state: RunState, loop: AgentLoop, record: dict) -> bool:
    """A search made in this loop that added a passage earns one more search (once per search, up to the hard cap)."""
    if record.get('earned') or record['search_id'] not in loop.search_ids:
        return False
    record['earned'] = True
    if loop.searches_allowed < state.config.agent_max_searches_per_section:
        loop.searches_allowed += 1
        return True
    return False

def searches_left(state: RunState, loop: AgentLoop) -> int:
    run_left = state.config.agent_max_searches_per_run - state.agent_searches
    return max(0, min(loop.searches_allowed - loop.searches, run_left))

def dispatch(state: RunState, loop: AgentLoop, call: dict):
    """Check and run one tool call from the agent.
    Returns (observation for the model, None) to continue, or (None, assessment) to stop the section."""
    cfg, name = state.config, call['name']

    def reject(error: str, detail: str):
        # The call is refused; the model sees why and can correct itself. It still counts toward the cap.
        loop.tool_calls += 1
        state.decide(loop.section, 'rejected_by_code', detail, 'continue', tool=name, error=error)
        return {'error': error, 'detail': detail}, None

    def stop(stop_reason: str, reason: str):
        # Code ends the section's loop (a cap was hit or the agent kept misbehaving).
        state.decide(loop.section, 'stopped_by_code', reason, 'stop', tool=name, stop_reason=stop_reason)
        coverage = 'partial' if state.pools.get(loop.section) else 'insufficient'
        return None, {'coverage': coverage, 'missing': [], 'conflicts_noticed': [],
                      'reason': reason, 'stop_reason': stop_reason}

    # 1. Only the four agent tools are allowed.
    if name not in AGENT_TOOL_NAMES:
        return reject('unknown_tool', f'{name!r} is not an allowed tool')
    try:
        args = json.loads(call['arguments'])
    except json.JSONDecodeError as exc:
        return reject('invalid_arguments', f'arguments are not valid JSON: {exc}')

    # 2. finish_assessment ends the loop (one chance to fix invalid arguments).
    if name == 'finish_assessment':
        try:
            a = FinishArgs.model_validate(args)
        except ValidationError as exc:
            loop.bad_finishes += 1
            if loop.bad_finishes > 1:
                return stop('invalid_finish', 'finish_assessment was invalid twice')
            return reject('invalid_arguments', short_error(exc))
        action = 'finish_insufficient' if a.coverage == 'insufficient' else 'finish_sufficient'
        state.decide(loop.section, action, a.reason, 'stop', tool=name, coverage=a.coverage,
                     missing=a.missing, conflicts_noticed=a.conflicts_noticed)
        return None, {**a.model_dump(), 'stop_reason': 'model_finished'}

    # 3. The same call twice is refused; a third time stops the loop. (follows_from doesn't make a search new,
    #    and neither does reordering or recasing the same set of wordings.)
    key_args = {k: v for k, v in args.items() if k not in ('reason', 'follows_from')}
    if name == 'document_search' and isinstance(key_args.get('query'), str):
        alternates = key_args.pop('alternate_queries', None)
        alternates = alternates if isinstance(alternates, list) else []
        key_args['query'] = sorted({' '.join(str(q).lower().split()) for q in [key_args['query']] + alternates})
    key = name + json.dumps(key_args, sort_keys=True, default=str)
    if key in loop.seen_calls:
        loop.repeats += 1
        if loop.repeats >= 2:
            return stop('repeated_call', 'the agent repeated an identical call again')
        return reject('repeated_call', 'identical call already made; change the query or call finish_assessment')

    # 4. Caps.
    if loop.tool_calls >= cfg.agent_max_tool_calls_per_section:
        return stop('tool_cap', f'{cfg.agent_max_tool_calls_per_section} tool calls used in this section')

    if name == 'document_search':
        hard = cfg.agent_max_searches_per_section
        if state.agent_searches >= cfg.agent_max_searches_per_run:
            return stop('run_search_cap', f'{cfg.agent_max_searches_per_run} agent searches used in this run')
        if loop.searches >= loop.searches_allowed:
            if loop.searches_allowed >= hard:
                return stop('search_cap', f'{hard} searches used in this section')
            return stop('earned_cap', f'{loop.searches} searches used; not enough of them added new passages '
                                      f'to earn another (each one that does earns one more, up to {hard})')
        try:
            a = SearchArgs.model_validate(args)
        except ValidationError as exc:
            return reject('invalid_arguments', short_error(exc))
        if a.section not in state.sections:
            return reject('section_not_allowed', f'{a.section!r} is not a planned section')
        if a.follows_from and not shown_to_agent(state, a.follows_from):
            return reject('unknown_follows_from', f'{a.follows_from} has not been shown to you in this run')
        loop.tool_calls += 1
        loop.searches += 1
        state.agent_searches += 1
        loop.seen_calls.add(key)
        record = search_section(state, a.section, a.query, 'agent', a.reason, a.flight_phase, a.follows_from,
                                alternate_queries=a.alternate_queries)
        loop.search_ids.add(record['search_id'])
        earned = earn_search(state, loop, record) if record['new_ids'] else False
        state.decide(loop.section, 'search', a.reason, 'continue', tool=name, searched_section=a.section,
                     query=a.query, alternate_queries=record['alternate_queries'],
                     follows_from=a.follows_from, search_id=record['search_id'],
                     new_passages=len(record['new_ids']), below_cutoff=len(record['below_cutoff_ids']),
                     earned_search=earned)
        if record['error']:
            return {'error': 'search_unavailable', 'detail': record['error']}, None
        pool = state.pools[a.section]
        return {'section': a.section, 'passages': [snippet(pool[i]) for i in record['kept_ids']],
                'new_to_pool': record['new_ids'],
                'below_cutoff': [snippet(SEARCH.by_id[i] | {'incident_label': incident_label(SEARCH.by_id[i])}, 200)
                                 for i in record['below_cutoff_ids']],
                'what_each_wording_found': record['phrasing_stats'],
                'searches_left': searches_left(state, loop),
                'note': ('This search added new passages and earned one more search.' if earned else
                         'This search added nothing new, so it earned no extra search.'
                         if not record['new_ids'] else 'No more searches can be earned in this section.')}, None

    if name == 'expand_context':
        try:
            a = ExpandArgs.model_validate(args)
        except ValidationError as exc:
            return reject('invalid_arguments', short_error(exc))
        section = pool_section_of(state, a.chunk_id)
        if section is None:
            return reject('not_in_pool', f'{a.chunk_id} is not in the evidence; only passages already kept can be expanded')
        if is_whole_record(state.pools[section][a.chunk_id]):
            return reject('whole_record', 'incident records are indexed as whole reports; there is nothing before or after')
        record = owning_search(state, section, a.chunk_id)
        if record is None:     # every pooled passage is credited to a search record; this would be a bug
            raise RuntimeError(f'{a.chunk_id} is pooled in {section} but no search record keeps it')
        loop.tool_calls += 1
        loop.seen_calls.add(key)
        neighbors = neighbor_ids(a.chunk_id, a.direction)
        added = [n for n in neighbors if add_to_pool(state, section, n, record, 'context')]
        state.decide(loop.section, 'expand', a.reason, 'continue', tool=name, chunk_id=a.chunk_id,
                     direction=a.direction, added_to_section=section, new_passages=len(added))
        return {'section': section, 'expanded': a.chunk_id,
                'neighbors': [{**snippet(state.pools[section][n], 1500), 'new_to_pool': n in added} for n in neighbors],
                'note': 'no neighboring passage in this document' if not neighbors else
                        f'{len(added)} new passage(s) kept as evidence for {section}'}, None

    # citation_lookup: a passage already in a pool, or one shown below the cutoff of an agent search (kept on opening).
    try:
        a = LookupArgs.model_validate(args)
    except ValidationError as exc:
        return reject('invalid_arguments', short_error(exc))
    if pool_section_of(state, a.chunk_id) is not None:
        loop.tool_calls += 1
        loop.seen_calls.add(key)
        state.decide(loop.section, 'lookup', a.reason, 'continue', tool=name, chunk_id=a.chunk_id)
        return citation_lookup(a.chunk_id), None
    source = next((r for r in reversed(state.searches)
                   if r['origin'] == 'agent' and a.chunk_id in r['below_cutoff_ids']), None)
    if source is None:
        return reject('not_in_pool', f'{a.chunk_id} has not been returned by a search in this run')
    loop.tool_calls += 1
    loop.seen_calls.add(key)
    add_to_pool(state, source['section'], a.chunk_id, source, 'promoted')
    earned = earn_search(state, loop, source)
    state.decide(loop.section, 'promote', a.reason, 'continue', tool=name, chunk_id=a.chunk_id,
                 kept_in_section=source['section'], search_id=source['search_id'], earned_search=earned)
    return {**citation_lookup(a.chunk_id), 'kept_in_section': source['section'],
            'searches_left': searches_left(state, loop)}, None

def call(name, **args):
    return {'call_id': 'x', 'name': name, 'arguments': json.dumps(args)}

def run_tests(loop, tests):
    rows = []
    for label, c in tests:
        observation, assessment = dispatch(demo, loop, c)
        rows.append({'test': label, 'result': observation.get('error', 'ok') if observation else f"stop: {assessment['stop_reason']}",
                     'searches allowed': loop.searches_allowed, 'tool calls': loop.tool_calls})
        last[label] = observation
    return rows


# Notebook cell 36
MODEL_PATTERNS = {
    'DJI Matrice 30T': r'matrice\s*30\s*t\b|\bm30t\b',
    'DJI Matrice 4T': r'matrice\s*4\s*t?\b|\bm4t\b',
    'Autel EVO Max 4N': r'evo\s*max\s*4n\b|\bmax\s*4n\b',
    'DJI Matrice 210 V1': r'matrice\s*(?:m)?210(?:\s*(?:v1|version\s*1))?\b|\bm210(?:\s*(?:v1|version\s*1))?\b',
}

OTHER_DRONES_RE = re.compile(
    r'mavic|phantom|\bmini\s*\d|inspire|skydio|parrot|anafi|avata|agras|'
    r'matrice\s*(300|350|200|600)|\bm(300|350|200)\b|evo\s*(ii|2|lite|nano)\b|autel\s*alpha', re.IGNORECASE)

INTAKE_INSTRUCTIONS = """You route questions for a drone lost-link evidence brief.
The collection holds FAA AC 107-2A, manuals for the DJI Matrice 30T, Matrice 4T, Autel EVO Max 4N and historical DJI Matrice 210 V1, plus AAIB, NASA ASRS and community accounts. AAIB record-only, ASRS and community accounts are not verified findings.
- in_scope: the question asks what guidance, manuals or incident reports say about lost link, loss of the
  control link or signal loss, and what the aircraft or pilot does (return to home, hover, land, reconnect).
- out_of_scope: anything else, such as buying advice or unrelated topics.
- unsupported_model: the question is only about drone models that are not in the collection.
models: the supported models the question names (empty list means all three).
focus: a short search phrase for the specific behavior asked about.
flight_phase: only if the question names one."""

def run_intake(state: RunState, models: Optional[list[str]] = None) -> dict:
    """Decide whether to answer, and for which models. Sets state.sections when the question is in scope."""
    q = state.question
    named = [m for m, pattern in MODEL_PATTERNS.items() if re.search(pattern, q, re.IGNORECASE)]
    others = sorted({m.group(0) for m in OTHER_DRONES_RE.finditer(q)})

    if others and not named and not models:
        # Only unsupported drones are named: abstain without calling the model.
        intake = {'intent': 'unsupported_model', 'models': [], 'focus': '', 'flight_phase': None,
                  'reason': f"not in the approved collection: {', '.join(others)}", 'decided_by': 'code'}
    else:
        out = state.llm.structured('intake', INTAKE_INSTRUCTIONS,
                                   {'question': q, 'models_named_in_question': named}, IntakeOut)
        intake = {**out.model_dump(), 'decided_by': 'model'}
        if intake['intent'] == 'unsupported_model' and named:
            intake['intent'] = 'in_scope'          # a supported model is named too, so answer for that one
        if intake['intent'] == 'in_scope':
            intake['models'] = models or named or out.models or MODELS
    intake['unsupported_named'] = others

    if intake['intent'] == 'in_scope':
        state.sections = plan_sections(intake['models'])
    state.intake = intake
    state.log('intake', **intake)
    return intake


# Notebook cell 38
AGENT_INSTRUCTIONS = """You check the evidence for ONE section of a drone lost-link evidence brief. Every other section
gets its own check, so judge only yours. What each section is for:
- faa: what FAA guidance tells the PILOT to do when the control link is lost. It does not describe any drone
  model's automatic behavior, so never search it for model settings or incidents.
- manual:<model>: what the manual says the aircraft does on lost link, the setting that controls it
  (e.g. Signal Lost Action / Signal Loss Action) and its conditions (Home Point recorded, GNSS, RTH altitude,
  flight mode, obstacle sensing).
- incident:<model>: what an UNVERIFIED reporter says happened.
Steps:
1. Read your passages.
2. Incident section only: compare each report with related_section_passages (the same model's manual). If a
   report describes something the manual says should not happen (e.g. return-to-home failed, the aircraft flew
   away), that is a conflict. Record it in conflicts_noticed, and call document_search in manual:<model> for the
   conditions that could explain it (e.g. when RTH cannot be triggered or can fail).
3. Read on: if a manual or FAA passage stops in the middle of something you need (a list of conditions, a
   sentence cut off at the end), call expand_context on it ('after' if it is cut off at the end, 'before' if it
   starts mid-thought). Incident records are whole reports and cannot be expanded.
4. Follow a term: if a passage names a mode, setting or condition your section does not explain yet (e.g. "RTH
   cannot be triggered in Attitude Mode"), call document_search for that term and set follows_from to that
   passage's chunk_id. Use follows_from = null for a search that does not build on a specific passage.
5. If something else specific is missing from your own section, call document_search with a targeted query.
   Sources use different words for the same thing (DJI: "Signal Lost Action"; Autel: "Signal Loss Action",
   "C2 link"; FAA: "loss of control link"; reporters: "flyaway", "lost control"). When the right term is
   uncertain, add up to 2 alternate_queries with other wordings; they are merged into ONE search. Check
   what_each_wording_found to learn which terms the source uses, and use those terms in later searches.
   Use alternate_queries = [] when one wording is enough.
6. Every search also lists results below_cutoff. If one of them states something your section needs, call
   citation_lookup on it to keep it. Do not keep passages that only repeat what you already have.
7. Use citation_lookup on a kept passage only when its snippet is cut off before something you need.
8. Call finish_assessment when the evidence is enough or another search is unlikely to help.
   coverage judges whether your passages cover your SECTION'S purpose above, not whether they answer the
   user's exact question. Use 'insufficient' only if your passages do not describe lost-link behavior at all.
   Example: FAA passages on what the pilot should do after losing the control link are 'sufficient' or
   'partial', even when the question asks about incidents. List the concrete things that are missing.
   conflicts_noticed holds only real disagreements between sources; if it is unclear whether sources
   disagree (e.g. it depends on an unknown setting), put that in missing instead.
Searches are earned: you start with 2. A search that adds a new passage (or one you keep from below its cutoff)
earns one more, up to 4. Each search result says how many you have left. Never invent passages or chunk_ids."""

def retrieve_base(state: RunState):
    """The same first search for every section, in every mode: one broad query plus the intake focus."""
    focus = state.intake.get('focus', '') if state.intake else ''
    for section in state.sections:
        search_section(state, section, f"{BASE_QUERY[section_kind(section)]} {focus}".strip(), origin='base')

def related_passages(state: RunState, section: str) -> list[dict]:
    """A manual section sees the same model's incident passages and vice versa, so the agent can spot conflicts."""
    kind, model = section_kind(section), section_model(section)
    if kind == 'faa':
        return []
    other = f"{'incident' if kind == 'manual' else 'manual'}:{model}"
    return [snippet(p) for p in list(state.pools.get(other, {}).values())[:3]]

def agent_turns_used(state: RunState) -> int:
    return sum(1 for e in state.events if e['type'] == 'agent_turn')

def run_evidence_agent(state: RunState, section: str, reopen_reason: Optional[str] = None) -> dict:
    """Plan -> act -> observe -> decide, for one section, until the agent finishes or the code stops it."""
    cfg = state.config
    loop = AgentLoop(section, searches_allowed=cfg.agent_base_searches_per_section)
    payload = {'section': section, 'planned_sections': state.sections,
               'passages': [snippet(p, 600) for p in state.pools.get(section, {}).values()],
               'related_section_passages': related_passages(state, section),
               'searches_left': searches_left(state, loop)}
    if section_kind(section) != 'faa':
        # The FAA section is judged only on its own purpose (pilot procedures), so it never sees the question:
        # a question about one model's incidents must not make it discard real FAA lost-link guidance.
        payload.update(question=state.question, focus=state.intake.get('focus', ''))
    if reopen_reason:
        payload['reopened_because'] = reopen_reason        # set when revision sends the agent back to search
    tools = agent_tools(state.sections)
    input_items, previous_id, assessment = json.dumps(payload), None, None

    def code_stop(stop_reason: str, reason: str) -> dict:
        state.decide(section, 'stopped_by_code', reason, 'stop', stop_reason=stop_reason)
        return {'coverage': 'partial' if state.pools.get(section) else 'insufficient', 'missing': [],
                'conflicts_noticed': [], 'reason': reason, 'stop_reason': stop_reason}

    while assessment is None:
        if loop.turns >= cfg.agent_max_turns_per_section:
            assessment = code_stop('turn_cap', f'{cfg.agent_max_turns_per_section} turns used')
            break
        if agent_turns_used(state) >= cfg.agent_max_turns_per_run:
            assessment = code_stop('run_turn_cap', f'{cfg.agent_max_turns_per_run} agent turns used in this run')
            break
        loop.turns += 1
        try:
            turn = state.llm.tool_turn('agent', AGENT_INSTRUCTIONS, input_items, tools, previous_id)
        except LLMError as exc:
            assessment = code_stop('llm_error', f'model call failed: {exc}')
            break
        previous_id = turn['id']
        state.log('agent_turn', section=section, turn=loop.turns, calls=[c['name'] for c in turn['calls']],
                  text=turn['text'][:300])
        if not turn['calls']:          # the model replied with text instead of a tool call
            input_items = [{'role': 'user', 'content': 'Reply with a tool call: document_search, citation_lookup, '
                                                       'expand_context or finish_assessment.'}]
            continue
        outputs = []
        for call in turn['calls']:
            observation, assessment = dispatch(state, loop, call)
            if assessment is not None:
                break
            outputs.append({'type': 'function_call_output', 'call_id': call['call_id'],
                            'output': json.dumps(observation, default=str)[:8000]})
        input_items = outputs

    assessment.update(section=section, turns=loop.turns, tool_calls=loop.tool_calls, searches=loop.searches,
                      searches_allowed=loop.searches_allowed)
    state.assessments[section] = assessment
    state.log('assessment', **assessment)
    return assessment

def scripted(*turns):
    """A fake agent that plays the given turns in order. Each turn is a function of the last observation
    (or a fixed list) returning a list of (tool, arguments)."""
    queue = list(turns)
    def play(input_items):
        turn = queue.pop(0)
        if callable(turn):
            last = json.loads(input_items[-1]['output']) if isinstance(input_items, list) and input_items else {}
            return turn(last)
        return turn
    return play

def searching_agent(section, queries):
    """A fake agent that searches the given queries in order, then keeps searching with new queries."""
    queries = iter(list(queries) + [f'additional lost link query {i}' for i in range(100)])
    return lambda input_items: [('document_search', {'query': next(queries), 'alternate_queries': [], 'section': section,
                                                     'flight_phase': None, 'follows_from': None,
                                                     'reason': 'keep looking for more evidence'})]

def search(query, section, reason, follows_from=None, alternates=()):
    return ('document_search', {'query': query, 'alternate_queries': list(alternates), 'section': section,
                                'flight_phase': None, 'follows_from': follows_from, 'reason': reason})

def keep_first_below_cutoff(last):
    below = last.get('below_cutoff', [])
    if not below:
        return [('finish_assessment', {'coverage': 'sufficient', 'missing': [], 'conflicts_noticed': [],
                                       'reason': 'nothing useful below the cutoff'})]
    return [('citation_lookup', {'chunk_id': below[0]['chunk_id'],
                                 'reason': 'ranked below the cutoff but states a condition for RTH'})]


# Notebook cell 40
DRAFT_INSTRUCTIONS = """You draft ONE section of a drone lost-link evidence brief from the supplied passages ONLY.
- One claim = one complete sentence of at most 40 words that makes sense on its own: say what triggers the
  behavior and what the aircraft or pilot does. Never a fragment (e.g. "Only then will the drone ..." or
  "If the connection is lost."). At most 5 claims. Keep the facts most directly about lost-link behavior;
  prefer conditions over background; merge closely related facts from the same passage.
- Every claim cites chunk_ids from the passages and copies a VERBATIM quote (an exact span of 5-30 words)
  from a cited passage.
- List every condition the quote states (e.g. "if the Signal Lost Action is set to RTH") in conditions,
  and keep those conditions in the claim text.
- Attribute every claim: FAA -> "AC 107-2A states ..."; manuals -> "The manual states ...";
  incident evidence -> name its source: "AAIB found ..." for a full investigation, "AAIB records ..." for a record-only entry, "The ASRS reporter says ...", or "A forum participant says ...". Record-only, ASRS and forum accounts are not verified findings. The ASRS Matrice 210 version is unknown; do not call it a V1 event.
- Use only what the passages say. Do not add people, roles, causes or details the passage does not state.
- Incident reports: a claim is "supported" when the report says it, even though the report itself is unverified.
  If a report is labeled summary_only, its narrative never says the link was lost: describe only what the
  reporter says happened (e.g. loss of control) and never state as fact that the link was lost.
- assessment: supported | supported_with_conditions | conflicting (contradicts another passage) |
  not_enough_evidence (the passages do not actually say this; the claim will be dropped).
- If no passage describes lost-link behavior, return no claims and give abstain_reason."""

def section_label(section: str) -> str:
    kind, model = section_kind(section), section_model(section)
    return {'faa': 'FAA guidance', 'manual': f'{model} manual', 'incident': f'{model} incident reports'}[kind]

def no_evidence(state: RunState, section: str, reason: str):
    """The section abstains: nothing is drafted, and the brief says so in its gaps."""
    state.drafts[section] = {'status': 'no_evidence', 'claims': [], 'abstain_reason': reason}
    state.gaps.append({'kind': 'missing', 'origin': 'section_abstained', 'related_claim_ids': [],
                       'text': f'No usable evidence found in the approved collection for {section_label(section)}.'})
    state.log('draft', section=section, status='no_evidence', reason=reason)

def make_claim(state: RunState, section: str, draft: ClaimDraft, claim_id: str) -> Claim:
    """Code, not the model, sets the section, source tier and incident label."""
    pool = state.pools.get(section, {})
    labels = [pool[c]['incident_label'] for c in draft.citations if c in pool and pool[c].get('incident_label')]
    cited = [pool[c] for c in draft.citations if c in pool]
    investigated = bool(cited) and all(p['source_id'] == 'aaib_m210_v1_poole' for p in cited)
    tier = 'official' if investigated else SECTION_TIER[section_kind(section)]
    return Claim(**draft.model_dump(), claim_id=claim_id, section=section,
                 tier=tier, incident_label=labels[0] if labels else None)

def draft_section(state: RunState, section: str):
    cfg = state.config
    pool = state.pools.get(section, {})
    assessment = state.assessments.get(section)
    if not pool:
        return no_evidence(state, section, 'no passage passed the relevance check')
    if assessment and assessment['coverage'] == 'insufficient':
        # The agent's judgment is recorded, but it cannot discard passages: the drafter and the verifier decide
        # what the passages support. (gpt-5-nano tended to call real evidence "insufficient" for the question.)
        state.decide(section, 'rejected_by_code', 'agent judged the evidence insufficient, but the section has '
                     'passages; drafting anyway and letting verification decide', 'continue',
                     agent_reason=assessment['reason'])

    payload = {'section': section_label(section), 'tier': SECTION_TIER[section_kind(section)],
               'passages': [{'chunk_id': p['chunk_id'], 'locator': p['locator'],
                             'incident_label': p.get('incident_label'), 'source_type': p['source_type'], 'source_id': p['source_id'], 'url': p['url'], 'text': p['text']} for p in pool.values()]}
    if section_kind(section) != 'faa':
        # As in the evidence agent: the FAA section is drafted from its own purpose (pilot procedures),
        # so a question about one model's incidents cannot make the drafter skip real FAA guidance.
        payload.update(question=state.question, focus=state.intake.get('focus', ''))
    try:
        out = state.llm.structured('draft', DRAFT_INSTRUCTIONS, payload, SectionDraftOut)
        problems = length_problems(out.claims, cfg)
        if problems:                                   # one retry with the problems as feedback
            state.log('draft_retry', section=section, problems=problems)
            out = state.llm.structured('draft', DRAFT_INSTRUCTIONS, {**payload, 'feedback': problems}, SectionDraftOut)
    except LLMError as exc:
        return no_evidence(state, section, f'drafting failed: {exc}')

    claims = []
    for d in out.claims:
        if d.assessment == 'not_enough_evidence':     # the drafter itself says the passages don't support it
            state.decide(section, 'drop', 'drafter marked it not_enough_evidence', 'stop', text=d.text)
        elif len(claims) == cfg.max_claims_per_section:
            state.decide(section, 'trimmed_by_code', 'over the claim limit after one retry', 'stop', text=d.text)
        else:
            claims.append(make_claim(state, section, d, f'C{len(state.all_claims()) + len(claims) + 1:03d}'))
    # A claim still over 40 words is kept here; the verification gate flags it and revision must shorten it.

    if not claims:
        return no_evidence(state, section, out.abstain_reason or 'the drafter found no claim the passages support')
    state.drafts[section] = {'status': 'drafted', 'claims': claims, 'abstain_reason': None}
    state.log('draft', section=section, status='drafted', claims=[c.model_dump() for c in claims])

def fake_draft(payload):
    """Quotes the first 12 words of up to 2 passages. Asks for 6 claims for the EVO manual to test the limit."""
    n = 6 if payload['section'] == 'Autel EVO Max 4N manual' else 2
    passages = (payload['passages'] * 6)[:n] if n == 6 else payload['passages'][:n]
    claims = [{'text': f"The source states that {' '.join(p['text'].split()[:12])}.", 'citations': [p['chunk_id']],
               'quote': ' '.join(p['text'].split()[:12]), 'conditions': [], 'assessment': 'supported'}
              for p in passages]
    if payload['section'] == 'FAA guidance':
        claims.append({'text': 'AC 107-2A states what the aircraft must do automatically.', 'citations': [], 'quote': '',
                       'conditions': [], 'assessment': 'not_enough_evidence'})
    return {'claims': claims, 'abstain_reason': None}


# Notebook cell 42
CONDITIONAL_RE = re.compile(r'\b(if|when|whenever|unless|only|provided|until)\b', re.IGNORECASE)

def normalize(text: str) -> str:
    """Make quote matching ignore PDF ligatures, curly quotes, spacing and case."""
    for a, b in (('ﬁ', 'fi'), ('ﬂ', 'fl'), ('’', "'"), ('‘', "'"), ('“', '"'), ('”', '"')):
        text = text.replace(a, b)
    return re.sub(r'\s+', ' ', text).strip().lower()

def clean_quote(quote: str) -> str:
    """The model often wraps a quote in quotation marks or ends it with a period; ignore those edges."""
    return normalize(quote).strip(' "\'.,;:…')

def gate(state: RunState, claim: Claim) -> dict:
    """Deterministic citation checks. No model call."""
    section, kind = claim.section, section_kind(claim.section)
    pool = state.pools.get(section, {})
    cited = [citation_lookup(c) for c in claim.citations]
    found = [c for c in cited if c['found']]
    checks = {
        'has_citation': len(claim.citations) > 0,
        'in_section_pool': all(c in pool for c in claim.citations),
        'resolves': len(found) == len(cited) > 0,
        'quote_in_source': len(claim.quote.split()) >= 3
                           and any(clean_quote(claim.quote) in normalize(c['text']) for c in found),
        'source_matches': len(found) > 0 and all(c['source_type'] in SECTION_SOURCE[kind]
                                                 and (c['model'] == section_model(section) or (section_model(section) == 'DJI Matrice 210 V1' and c['source_id'] == 'asrs_m210_1918661' and c['model'] == 'DJI Matrice 210')) for c in found),
        'attributed': ('report' in claim.text.lower()) if kind == 'incident'
                      else ('reporter' not in claim.text.lower()
                            and (kind != 'faa' or 'manual' not in claim.text.lower())),
        'incident_labeled': kind != 'incident' or claim.incident_label is not None,
        'conditions_kept': not CONDITIONAL_RE.search(claim.quote) or len(claim.conditions) > 0
                           or bool(CONDITIONAL_RE.search(claim.text)),
        'within_length': word_count(claim.text) <= state.config.max_claim_words,
    }
    failures = [name for name, ok in checks.items() if not ok]
    return {'claim_id': claim.claim_id, 'passed': not failures, 'failures': failures}

VERIFY_INSTRUCTIONS = """You verify claims in a drone lost-link evidence brief against the passages they cite.
For each claim, read ONLY the passages it cites and decide:
- supported: everything in the claim, including its conditions, is stated in the cited passages.
- partial: the gist is there, but the claim drops or misstates a condition, setting, model or level of certainty
  (for example it states a reporter's account as fact, or leaves out "if the Signal Lost Action is set to RTH").
- not_supported: the cited passages do not say this.
Put conditions the passage states but the claim leaves out in missing_conditions. supporting_ids must come from
the claim's own citations. Return one verification for every claim_id. Be strict and use no outside knowledge."""

def verify_claims(state: RunState, claims: list[Claim]):
    """Gate every claim, then one verifier call per section. A claim without a verdict counts as not supported."""
    for section in state.sections:
        group = [c for c in claims if c.section == section]
        if not group:
            continue
        for c in group:
            state.gate_results[c.claim_id] = gate(state, c)

        passages = {}
        for c in group:
            for chunk_id in c.citations:
                cit = citation_lookup(chunk_id)
                if cit['found']:
                    passages[chunk_id] = {'chunk_id': chunk_id, 'source': cit['title'], 'locator': cit['locator'],
                                          'source_type': cit['source_type'], 'evidence_label': cit.get('evidence_label'), 'text': cit['text']}
        payload = {'claims': [{'claim_id': c.claim_id, 'text': c.text, 'conditions': c.conditions,
                               'citations': c.citations} for c in group],
                   'passages': list(passages.values())}
        try:
            out = state.llm.structured('verify', VERIFY_INSTRUCTIONS, payload, VerificationOut)
            verdicts = {v.claim_id: v for v in out.verifications}
        except LLMError as exc:
            state.log('verify_error', section=section, error=str(exc))
            verdicts = {}
        for c in group:
            state.verifications[c.claim_id] = verdicts.get(c.claim_id) or Verification(
                claim_id=c.claim_id, status='not_supported', missing_conditions=[], supporting_ids=[],
                rationale='no verdict returned by the verifier')
        state.log('verify', section=section, results=[
            {'claim_id': c.claim_id, 'gate_failures': state.gate_results[c.claim_id]['failures'],
             'status': state.verifications[c.claim_id].status,
             'missing_conditions': state.verifications[c.claim_id].missing_conditions} for c in group])

def passes(state: RunState, claim: Claim) -> bool:
    """A claim passes only if every gate check passes AND the verifier says fully supported."""
    g, v = state.gate_results.get(claim.claim_id), state.verifications.get(claim.claim_id)
    return bool(g and g['passed'] and v and v.status == 'supported' and not v.missing_conditions)

def plain_span(text, n=10):
    """The first n-word span with no condition words, for claims that should pass."""
    words = text.split()
    return next(' '.join(words[i:i + n]) for i in range(len(words) - n + 1)
                if not CONDITIONAL_RE.search(' '.join(words[i:i + n])))

def claim(cid, section, text, cites, quote, conditions=()):
    return make_claim(demo, section, ClaimDraft(text=text, citations=cites, quote=quote,
                                                conditions=list(conditions), assessment='supported'), cid)


# Notebook cell 44
REVISE_INSTRUCTIONS = """You fix claims in a drone lost-link evidence brief that failed citation verification.
For EACH failing claim choose one action from allowed_actions:
- rewrite: give new_claim, using only the supplied passages. Fix every listed problem: copy a verbatim quote,
  cite the right chunk_ids, list and include the conditions, attribute the claim exactly as how_to_attribute
  says, and keep it to one sentence of at most 40 words.
- drop: the claim adds nothing the section needs.
- to_gap: the passages cannot support the point; reason is a gap statement for the brief (at most 40 words).
- reopen_section (only if allowed): the needed evidence is missing from the passages and one targeted search
  could find it; reason says what to search for.
Problem names: quote_in_source (quote is not verbatim), in_section_pool (cites a passage outside this section),
source_matches (wrong source or model), attributed (follow how_to_attribute: only incident claims mention a
reporter, and FAA claims never say "the manual"), conditions_kept (the quote has a condition the claim leaves
out), too_long (over 40 words).
Use only what the passages say; do not add people, roles or details they do not state.
If a claim already failed in an earlier round, do not resubmit the same quote: find the exact wording in the
passages, or choose drop or to_gap."""

HOW_TO_ATTRIBUTE = {
    'faa': 'Write it as what AC 107-2A (FAA guidance) states. Never call it a manual or mention a reporter.',
    'manual': 'Write it as what the manual states. Never mention a reporter.',
    'incident': 'Start with "The reporter states": this is an unverified personal account.',
}

def replace_claim(state: RunState, claim_id: str, new: Optional[Claim] = None):
    """Swap a claim for its rewrite, or remove it when `new` is None."""
    for draft in state.drafts.values():
        draft['claims'] = [new if c.claim_id == claim_id else c for c in draft['claims']
                           if c.claim_id != claim_id or new is not None]

def revise_section(state: RunState, section: str, failing: list[Claim], round_no: int, allow_reopen: bool) -> list[Claim]:
    """One revision call for one section's failing claims. Returns the rewritten claims, to be verified again."""
    actions = ['rewrite', 'drop', 'to_gap'] + (['reopen_section'] if allow_reopen else [])
    payload = {'section': section_label(section), 'tier': SECTION_TIER[section_kind(section)], 'round': round_no,
               'how_to_attribute': HOW_TO_ATTRIBUTE[section_kind(section)], 'allowed_actions': actions,
               'passages': [{'chunk_id': p['chunk_id'], 'locator': p['locator'], 'incident_label': p.get('incident_label'),
                             'text': p['text']} for p in state.pools.get(section, {}).values()],
               'failing_claims': [{'claim_id': c.claim_id, 'text': c.text, 'citations': c.citations, 'quote': c.quote,
                                   'conditions': c.conditions,
                                   'problems': ['too_long' if f == 'within_length' else f
                                                for f in state.gate_results[c.claim_id]['failures']],
                                   'verifier': state.verifications[c.claim_id].status,
                                   'verifier_rationale': state.verifications[c.claim_id].rationale,
                                   'missing_conditions': state.verifications[c.claim_id].missing_conditions}
                                  for c in failing]}
    try:
        out = state.llm.structured('revise', REVISE_INSTRUCTIONS, payload, RevisionOut)
    except LLMError as exc:
        state.log('revise_error', section=section, error=str(exc))
        return []                                 # the claims stay failing; the next round or the end removes them
    choices = {d.claim_id: d for d in out.decisions}
    changed = []
    for c in failing:
        d = choices.get(c.claim_id)
        if d is None:
            state.decide(section, 'rejected_by_code', 'no decision returned for this claim', 'continue', claim_id=c.claim_id)
            continue
        action = d.action
        if action == 'reopen_section' and not allow_reopen:
            state.decide(section, 'rejected_by_code', 'reopen_section is not allowed here; converted to a gap',
                         'continue', claim_id=c.claim_id)
            action = 'to_gap'

        if action == 'reopen_section':
            allow_reopen = False
            state.reopens += 1
            state.decide(section, 'reopen_section', d.reason, 'continue', claim_id=c.claim_id, round=round_no)
            run_evidence_agent(state, section, reopen_reason=(
                f'Claim "{c.text}" failed verification ({state.verifications[c.claim_id].rationale}). '
                f'Look for: {d.reason}'))
            changed += revise_section(state, section, [c], round_no, allow_reopen=False)
        elif action == 'rewrite':
            if d.new_claim is None:
                state.decide(section, 'rejected_by_code', 'rewrite without a new claim', 'continue', claim_id=c.claim_id)
                continue
            new = make_claim(state, section, d.new_claim, c.claim_id)
            replace_claim(state, c.claim_id, new)
            changed.append(new)
            state.decide(section, 'rewrite', d.reason, 'continue', claim_id=c.claim_id, round=round_no, new_text=new.text)
        elif action == 'drop':
            replace_claim(state, c.claim_id)
            state.removed_claims.append(c)
            state.decide(section, 'drop', d.reason, 'stop', claim_id=c.claim_id, round=round_no)
        else:  # to_gap
            replace_claim(state, c.claim_id)
            state.removed_claims.append(c)
            state.gaps.append({'kind': 'missing', 'origin': 'revision', 'related_claim_ids': [],
                               'text': ' '.join((d.reason or f'Not supported: {c.text}').split()[:40])})
            state.decide(section, 'to_gap', d.reason, 'stop', claim_id=c.claim_id, round=round_no)
    return changed

def revise(state: RunState):
    """Up to max_revisions rounds. Claims still failing after the last round are removed for human review."""
    cfg = state.config
    for round_no in range(1, cfg.max_revisions + 1):
        failing = [c for c in state.all_claims() if not passes(state, c)]
        if not failing:
            return
        state.revisions_used = round_no
        state.log('revision_round', round=round_no, failing=[c.claim_id for c in failing])
        changed = []
        for section in state.sections:
            group = [c for c in failing if c.section == section]
            if group:
                allow_reopen = cfg.mode == 'full' and state.reopens < cfg.max_reopens_per_run
                changed += revise_section(state, section, group, round_no, allow_reopen)
        verify_claims(state, changed)
    for c in [c for c in state.all_claims() if not passes(state, c)]:
        replace_claim(state, c.claim_id)
        state.removed_claims.append(c)
        state.decide(c.section, 'drop', f'still failing after {cfg.max_revisions} revision rounds', 'stop',
                     claim_id=c.claim_id, human_review=True)

def scenario(mode, revise_handler, verify_status='supported'):
    """One good claim and one with a made-up quote, drafted in the EVO manual section, then verified and revised."""
    s = RunState('demo', Config(mode=mode), FakeLLM({
        'revise': revise_handler,
        'verify': lambda p: {'verifications': [{'claim_id': c['claim_id'], 'status': verify_status,
                                                'missing_conditions': [], 'rationale': 'scripted',
                                                'supporting_ids': c['citations']} for c in p['claims']]},
        'agent': scripted([('finish_assessment', {'coverage': 'sufficient', 'missing': [], 'conflicts_noticed': [],
                                                  'reason': 'found the RTH conditions on a second look'})])}))
    s.intake = {'focus': 'lost-link behavior'}
    s.sections = [evo_manual]
    retrieve_base(s)
    p = next(iter(s.pools[evo_manual].values()))
    good = plain_span(p['text'])
    s.drafts[evo_manual] = {'status': 'drafted', 'claims': [
        make_claim(s, evo_manual, ClaimDraft(text=f'The manual states that {good}.', citations=[p['chunk_id']],
                                             quote=good, conditions=[], assessment='supported'), 'C001'),
        make_claim(s, evo_manual, ClaimDraft(text='The manual states the drone always lands at once.',
                                             citations=[p['chunk_id']], quote='the drone always lands at once',
                                             conditions=[], assessment='supported'), 'C002')]}
    verify_claims(s, s.all_claims())
    revise(s)
    s.trace_path.unlink()
    return s

def fix_quote(action='rewrite'):
    """A scripted reviser that rewrites each failing claim with a verbatim quote."""
    def handler(p):
        quote = plain_span(p['passages'][0]['text'])
        return {'decisions': [{'claim_id': c['claim_id'], 'action': action, 'reason': 'use the exact wording',
                               'new_claim': {'text': f'The manual states that {quote}.', 'citations': [p['passages'][0]['chunk_id']],
                                             'quote': quote, 'conditions': [], 'assessment': 'supported'}}
                              for c in p['failing_claims']]}
    return handler

def reopen_then_rewrite():
    calls = iter(['reopen_section', 'rewrite'])
    def handler(p):
        action = next(calls)
        return {'decisions': [{**d, 'action': action, 'reason': 'conditions under which RTH cannot start'}
                              for d in fix_quote()(p)['decisions']]}
    return handler


# Notebook cell 46
SYNTHESIS_INSTRUCTIONS = """You list the gaps and conflicts in a drone lost-link evidence brief.
Use ONLY the verified claims and the evidence agent's notes supplied. Each item has kind = conflict | missing | unclear,
text of at most 40 words, and related_claim_ids taken from the supplied claim ids.
Most important first: conflicts (e.g. a manual's described behavior vs. an UNVERIFIED incident report), then
missing evidence, then unclear points. A conflict needs claims on BOTH sides in this brief, from two different
sections, and both claim ids go in related_claim_ids. Never list a possible or hypothetical conflict ("may",
"could"); if unsure, use unclear. Return at most `slots` items and do not repeat `already_listed`."""

SHORT_ANSWER_INSTRUCTIONS = """Write the short answer of a drone lost-link evidence brief from the verified claims only.
3 to 5 sentences, at most 120 words, in this order: FAA guidance, then the manuals, then incident reports
(say they are unverified accounts), then the main gap or conflict. Every fact must come from a claim; the gaps are
open questions, not facts. If a section has no claims, say there is no usable evidence for it. No citations."""

NO_EVIDENCE_ANSWER = ('No claim in this brief passed citation verification. The approved collection may not cover '
                      'this question, or its evidence was too thin to state safely. See the gaps for what is missing.')

GAP_PRIORITY = {'conflict': 0, 'missing': 1, 'unclear': 2}

def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]

def cap_gaps(state: RunState):
    """Keep at most max_gaps items: conflicts first, then missing, then unclear. Log each one trimmed."""
    state.gaps.sort(key=lambda g: GAP_PRIORITY[g['kind']])
    for g in state.gaps[state.config.max_gaps:]:
        state.decide('gaps', 'trimmed_by_code', 'over the gap limit', 'stop', text=g['text'])
    state.gaps = state.gaps[: state.config.max_gaps]

def synthesize_gaps(state: RunState):
    """One model call that turns the agent's notes and the verified claims into gap and conflict items."""
    cfg = state.config
    claims = state.all_claims()
    notes = {s: {'missing': a['missing'], 'conflicts_noticed': a['conflicts_noticed']}
             for s, a in state.assessments.items() if a.get('missing') or a.get('conflicts_noticed')}
    slots = cfg.max_gaps - len(state.gaps)
    if slots > 0 and (claims or notes):
        payload = {'question': state.question, 'slots': slots, 'already_listed': [g['text'] for g in state.gaps],
                   'claims': [{'claim_id': c.claim_id, 'section': section_label(c.section), 'tier': c.tier,
                               'text': c.text} for c in claims],
                   'agent_notes': notes}
        try:
            out = state.llm.structured('synthesis', SYNTHESIS_INSTRUCTIONS, payload, GapsOut)
            problems = ([f'{len(out.gaps)} items; keep only the {slots} most important'] if len(out.gaps) > slots else []) \
                + [f'over 40 words: "{g.text[:50]}..."' for g in out.gaps if word_count(g.text) > cfg.max_claim_words]
            if problems:                                  # one retry with the problems as feedback
                out = state.llm.structured('synthesis', SYNTHESIS_INSTRUCTIONS, {**payload, 'feedback': problems}, GapsOut)
            section_of = {c.claim_id: c.section for c in claims}
            agent_saw_conflict = any(a.get('conflicts_noticed') for a in state.assessments.values())
            for g in out.gaps:
                related = [i for i in g.related_claim_ids if i in section_of]
                kind = g.kind
                # A conflict must rest on claims from two sections, or on a conflict the evidence agent recorded.
                if kind == 'conflict' and len({section_of[i] for i in related}) < 2 and not agent_saw_conflict:
                    kind = 'unclear'
                    state.decide('gaps', 'rejected_by_code', 'conflict without claims on both sides; kept as unclear',
                                 'continue', text=g.text)
                state.gaps.append({'kind': kind, 'origin': 'synthesis',
                                   'text': ' '.join(g.text.split()[: cfg.max_claim_words]),
                                   'related_claim_ids': related})
        except LLMError as exc:
            state.log('synthesis_error', error=str(exc))
    cap_gaps(state)
    state.log('gaps', gaps=state.gaps)

def write_short_answer(state: RunState):
    """3-5 sentences, at most 120 words, from the claims left after verification and revision."""
    cfg = state.config
    claims = state.all_claims()
    if not claims:
        state.short_answer = NO_EVIDENCE_ANSWER
        return
    payload = {'question': state.question,
               'claims': [{'section': section_label(c.section), 'tier': c.tier, 'text': c.text} for c in claims],
               'gaps': [g['text'] for g in state.gaps]}

    def problems_of(text):
        n_sentences, n_words = len(split_sentences(text)), word_count(text)
        return ([f'{n_sentences} sentences; use 3 to 5'] if not 3 <= n_sentences <= 5 else []) + \
               ([f'{n_words} words; the limit is {cfg.max_short_answer_words}'] if n_words > cfg.max_short_answer_words else [])

    try:
        text = state.llm.structured('short_answer', SHORT_ANSWER_INSTRUCTIONS, payload, ShortAnswerOut).short_answer
        problems = problems_of(text)
        if problems:                                      # one retry with the problems as feedback
            text = state.llm.structured('short_answer', SHORT_ANSWER_INSTRUCTIONS,
                                        {**payload, 'feedback': problems}, ShortAnswerOut).short_answer
        if problems_of(text):                             # still too long: keep whole sentences that fit
            kept = []
            for sentence in split_sentences(text)[:5]:
                if word_count(' '.join(kept + [sentence])) > cfg.max_short_answer_words:
                    break
                kept.append(sentence)
            state.decide('short_answer', 'trimmed_by_code', 'short answer over the limits after one retry', 'stop')
            text = ' '.join(kept)
    except LLMError as exc:
        state.log('short_answer_error', error=str(exc))
        text = (f'This brief has {len(claims)} verified claim(s). See the sections for FAA guidance, the manuals '
                'and unverified incident reports. See the gaps for what the sources leave open.')
    state.short_answer = text
    state.log('short_answer', text=text)


# Notebook cell 48
def finish_run(state: RunState, started: float, status: Optional[str] = None, stop_reason: Optional[str] = None,
               error: Optional[str] = None) -> dict:
    """Decide the final status, package the run as one dict, log run_end and save it next to the trace."""
    claims = state.all_claims()
    drafted = [s for s in state.sections if state.drafts.get(s, {}).get('claims')]
    if status is None:
        if not drafted:
            status, stop_reason = 'abstained', 'no_usable_evidence'
        elif any(d.get('human_review') for d in state.decisions):
            status, stop_reason = 'human_review_required', 'claims still failing after the revision limit'
        elif state.gaps or state.removed_claims or len(drafted) < len(state.sections):
            status, stop_reason = 'completed_with_gaps', 'verified claims; gaps listed'
        else:
            status, stop_reason = 'completed', 'all claims verified'

    words = word_count(state.short_answer) + sum(word_count(c.text) for c in claims) + sum(word_count(g['text']) for g in state.gaps)
    result = {
        'run_id': state.run_id, 'mode': state.config.mode, 'question': state.question,
        'status': status, 'stop_reason': stop_reason, 'error': error, 'intake': state.intake,
        'brief': {
            'short_answer': state.short_answer,
            'sections': [{'section': s, 'label': section_label(s), **{k: v for k, v in state.drafts.get(s, {}).items() if k != 'claims'},
                          'claims': [c.model_dump() for c in state.drafts.get(s, {}).get('claims', [])]}
                         for s in state.sections],
            'gaps': state.gaps,
            'removed_claims': [c.model_dump() for c in state.removed_claims],
        },
        'brief_stats': {'words': words, 'claims': len(claims), 'gaps': len(state.gaps)},
        'searches': state.searches, 'decisions': state.decisions, 'assessments': state.assessments,
        'gate_results': {c.claim_id: state.gate_results.get(c.claim_id) for c in claims},
        'verifications': {c.claim_id: state.verifications[c.claim_id].model_dump() for c in claims if c.claim_id in state.verifications},
        'first_draft_results': state.first_draft_results,
        'revisions_used': state.revisions_used, 'reopens': state.reopens,
        'cost': {'model_calls': state.llm.calls, 'tokens_in': state.llm.tokens_in, 'tokens_out': state.llm.tokens_out,
                 'searches': len(state.searches), 'agent_searches': state.agent_searches,
                 'agent_tool_calls': sum(1 for d in state.decisions if 'tool' in d or d['action'] in
                                         ('search', 'lookup', 'finish_sufficient', 'finish_insufficient')),
                 'seconds': round(time.time() - started, 1)},
        'corpus_snapshot': CORPUS_REPORT['corpus_snapshot'], 'chunks_sha256': CHUNKS_SHA256,
        'retrieval': CORPUS_REPORT['retrieval_method'], 'model': MODEL, 'reasoning_effort': REASONING_EFFORT,
        'config': vars(state.config), 'trace_path': str(state.trace_path),
    }
    state.log('run_end', status=status, stop_reason=stop_reason, cost=result['cost'], brief_stats=result['brief_stats'])
    state.trace_path.with_suffix('.result.json').write_text(json.dumps(result, indent=1, default=str), encoding='utf-8')
    return result

def run_workflow(question: str, models: Optional[list[str]] = None, config: Config = Config(), llm=None) -> dict:
    """intake -> base search -> evidence agent (full mode only) -> draft -> verify -> revise -> gaps -> short answer."""
    if llm is None:
        if not RUN_LIVE:
            raise RuntimeError('RUN_LIVE is False: pass llm=FakeLLM(...) for an offline run, or set RUN_LIVE = True')
        llm = LLM(llm_client, config.max_model_calls)
    state = RunState(question, config, llm)
    started = time.time()
    state.log('run_start', question=question, mode=config.mode, config=vars(config), model=MODEL,
              corpus_snapshot=CORPUS_REPORT['corpus_snapshot'], chunks_sha256=CHUNKS_SHA256,
              retrieval=CORPUS_REPORT['retrieval_method'])
    try:
        intake = run_intake(state, models)
        if intake['intent'] != 'in_scope':
            return finish_run(state, started, 'abstained', f"question {intake['intent']}")
        for name in intake['unsupported_named']:
            state.gaps.append({'kind': 'missing', 'origin': 'intake', 'related_claim_ids': [],
                               'text': f'{name} is not in the approved collection, so the brief does not cover it.'})
        retrieve_base(state)
        if config.mode == 'full':                     # the only difference between the two modes
            for section in state.sections:
                run_evidence_agent(state, section)
        for section in state.sections:
            draft_section(state, section)
        verify_claims(state, state.all_claims())
        state.first_draft_results = [{'claim_id': c.claim_id, 'passes': passes(state, c),
                                      'gate_failures': state.gate_results[c.claim_id]['failures'],
                                      'verifier': state.verifications[c.claim_id].status} for c in state.all_claims()]
        revise(state)
        synthesize_gaps(state)
        write_short_answer(state)
        return finish_run(state, started)
    except BudgetExceeded as exc:
        return finish_run(state, started, 'failed', f'budget: {exc}')
    except Exception as exc:
        state.log('error', error=f'{type(exc).__name__}: {exc}')
        return finish_run(state, started, 'failed', 'error', error=f'{type(exc).__name__}: {exc}')

def show_brief(result: dict):
    """Print a run's brief in plain text."""
    print(f"[{result['status']}] {result['question']}\n  mode: {result['mode']} | stop: {result['stop_reason']}")
    if result['error']:
        print('  error:', result['error'])
    brief = result['brief']
    if brief['short_answer']:
        print('\nSHORT ANSWER:', brief['short_answer'])
    for s in brief['sections']:
        print(f"\n## {s['label']}" + (f"  (no usable evidence: {s['abstain_reason']})" if s.get('status') == 'no_evidence' else ''))
        for c in s['claims']:
            print(f"  - [{c['tier'].upper()}] {c['text']}  {c['citations']}"
                  + (f"  (report label: {c['incident_label']})" if c['incident_label'] else ''))
    print('\n## Gaps and conflicts')
    for g in brief['gaps']:
        print(f"  - ({g['kind']}) {g['text']}")
    cost = result['cost']
    print(f"\n{result['brief_stats']['claims']} claims, {result['brief_stats']['words']} words | "
          f"{cost['model_calls']} model calls, {cost['searches']} searches ({cost['agent_searches']} by the agent), "
          f"{result['revisions_used']} revision rounds, {cost['seconds']} s")

def offline_llm() -> FakeLLM:
    def quote_from(text, n=10):
        words = text.split()
        return next((' '.join(words[i:i + n]) for i in range(len(words) - n + 1)
                     if not CONDITIONAL_RE.search(' '.join(words[i:i + n]))), ' '.join(words[:n]))

    def finish(coverage, conflicts=()):
        return ('finish_assessment', {'coverage': coverage, 'missing': [], 'conflicts_noticed': list(conflicts),
                                      'reason': f'scripted: evidence is {coverage}'})

    def agent(items):
        if isinstance(items, str):                       # first turn: the section payload
            p = json.loads(items)
            if not p['passages']:
                return [('document_search', {'query': 'lost link signal lost', 'section': p['section'],
                                             'flight_phase': None, 'reason': 'no passages yet; one targeted search'})]
            conflicts = ['scripted: report and manual disagree'] if p['section'].startswith('incident') and p['related_section_passages'] else []
            return [finish('sufficient', conflicts)]
        observation = json.loads(items[0]['output'])     # after a search: stop either way
        return [finish('sufficient' if observation.get('passages') else 'insufficient')]

    def draft(p):
        lead = {'official': 'AC 107-2A states that', 'manufacturer': 'The manual states that',
                'unverified': 'The reporter states that'}[p['tier']]
        return {'claims': [{'text': f"{lead} {quote_from(x['text'])}.", 'citations': [x['chunk_id']],
                            'quote': quote_from(x['text']), 'conditions': [], 'assessment': 'supported'}
                           for x in p['passages'][:2]], 'abstain_reason': None}

    return FakeLLM({
        'intake': lambda p: {'intent': 'out_of_scope' if 'buy' in p['question'] else 'in_scope', 'models': [],
                             'focus': 'lost-link behavior', 'flight_phase': None, 'reason': 'scripted'},
        'agent': agent,
        'draft': draft,
        'verify': lambda p: {'verifications': [{'claim_id': c['claim_id'], 'status': 'supported', 'missing_conditions': [],
                                                'rationale': 'scripted', 'supporting_ids': c['citations']} for c in p['claims']]},
        'revise': lambda p: {'decisions': []},
        'synthesis': lambda p: {'gaps': [{'kind': 'conflict', 'text': c, 'related_claim_ids': []}
                                         for note in p['agent_notes'].values() for c in note['conflicts_noticed']]},
        'short_answer': lambda p: {'short_answer': 'FAA guidance is summarized first. The manual describes the '
                                                   'configured behavior. Incident reports are unverified accounts.'},
    })

DEMO_CONFLICT = 'What does the Autel EVO Max 4N do when the control link is lost, and do incident reports match?'

DEMO_NO_EVIDENCE = 'What lost-link incidents have been reported for the DJI Matrice 4T?'


TRACE_DIR.mkdir(parents=True, exist_ok=True)


def configure_live_client(api_key: str) -> None:
    """Use the course gateway without storing the key in the repository."""
    global RUN_LIVE, llm_client
    if not api_key or not api_key.strip():
        raise ValueError("A course API key is required for a live run.")
    llm_client = OpenAI(api_key=api_key.strip(), base_url="https://api.portkey.ai/v1", timeout=60)
    RUN_LIVE = True
