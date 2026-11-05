"""A directly quoted manual fact survives an unsupported added condition."""

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import workflow_runtime as workflow


class QuoteRevisionTests(unittest.TestCase):
    def test_partial_manual_claim_keeps_narrower_quoted_fact(self):
        section = "manual:DJI Matrice 30T"
        chunk_id = "dji_m30t_manual_v22_c0070"
        passage = workflow.SEARCH.by_id[chunk_id]
        quote = "The aircraft cannot avoid obstacles during failsafe RTH when there is insufficient light."
        draft = workflow.Claim(
            text=("If the remote controller signal is lost, the aircraft cannot avoid obstacles "
                  "during failsafe RTH when there is insufficient light."),
            citations=[chunk_id], quote=quote,
            conditions=["remote controller signal is lost", "insufficient light"],
            assessment="supported", claim_id="C001", section=section, tier="manufacturer",
        )
        state = SimpleNamespace(
            config=workflow.Config(),
            pools={section: {chunk_id: passage}},
            gate_results={"C001": {"failures": []}},
            verifications={"C001": SimpleNamespace(status="partial")},
        )
        narrower = workflow.quote_only_rewrite(state, draft)
        self.assertIsNotNone(narrower)
        self.assertNotIn("remote controller signal is lost", narrower.text)
        self.assertIn("cannot avoid obstacles", narrower.text)
        self.assertEqual(narrower.citations, [chunk_id])
        self.assertTrue(workflow.gate(state, narrower)["passed"])

    def test_revision_rechecks_narrower_claim_without_another_drafting_call(self):
        section = "manual:DJI Matrice 30T"
        chunk_id = "dji_m30t_manual_v22_c0070"
        quote = "The aircraft cannot avoid obstacles during failsafe RTH when there is insufficient light."
        claim = workflow.Claim(
            text=("If the remote controller signal is lost, the aircraft cannot avoid obstacles "
                  "during failsafe RTH when there is insufficient light."),
            citations=[chunk_id], quote=quote,
            conditions=["remote controller signal is lost", "insufficient light"],
            assessment="supported", claim_id="C001", section=section, tier="manufacturer",
        )

        def verify(payload):
            return {"verifications": [{
                "claim_id": item["claim_id"],
                "status": "partial" if "remote controller signal is lost" in item["text"] else "supported",
                "missing_conditions": ["extra loss condition"] if "remote controller signal is lost" in item["text"] else [],
                "rationale": "only the quoted low-light fact is supported",
                "supporting_ids": item["citations"],
            } for item in payload["claims"]]}

        llm = workflow.FakeLLM({"verify": verify})
        state = workflow.RunState("What happens in low light?", workflow.Config(), llm)
        state.sections = [section]
        state.pools[section] = {chunk_id: workflow.SEARCH.by_id[chunk_id]}
        state.drafts[section] = {"status": "drafted", "claims": [claim], "abstain_reason": None}
        with tempfile.TemporaryDirectory() as temp:
            state.trace_path = Path(temp) / "trace.jsonl"
            workflow.verify_claims(state, [claim])
            workflow.revise(state)
        self.assertEqual(len(state.all_claims()), 1)
        self.assertTrue(workflow.passes(state, state.all_claims()[0]))
        self.assertFalse(state.removed_claims)
        self.assertIn("rewrite_quote_only", [decision["action"] for decision in state.decisions])


    def test_repeat_of_verified_claim_is_dropped_in_code(self):
        section = "manual:DJI Matrice 30T"
        chunk_id = "dji_m30t_manual_v22_c0070"
        quote = "The aircraft cannot avoid obstacles during failsafe RTH when there is insufficient light."
        make = lambda cid, text, quote_: workflow.Claim(text=text, citations=[chunk_id], quote=quote_, conditions=[],
                                                        assessment="supported", claim_id=cid, section=section,
                                                        tier="manufacturer")
        good = make("C001", f"The manual states that {quote}", quote)
        repeat = make("C002", f"If the remote controller signal is lost, {quote[0].lower() + quote[1:]}",
                      "cannot bypass obstacles in low light")
        different = make("C003", "The manual states the RTH altitude is fixed.", "the RTH altitude is fixed")
        seen = []

        def revise(payload):
            seen.extend(c["claim_id"] for c in payload["failing_claims"])
            return {"decisions": [{"claim_id": "C003", "action": "drop", "reason": "not supported", "new_claim": None}]}

        state = workflow.RunState("What happens in low light?", workflow.Config(), workflow.FakeLLM({"revise": revise}))
        state.sections = [section]
        state.pools[section] = {chunk_id: workflow.SEARCH.by_id[chunk_id]}
        state.drafts[section] = {"status": "drafted", "claims": [good, repeat, different], "abstain_reason": None}
        for claim_, status in ((good, "supported"), (repeat, "partial"), (different, "not_supported")):
            state.gate_results[claim_.claim_id] = {"passed": claim_ is good, "failures": [] if claim_ is good else ["quote_in_source"]}
            state.verifications[claim_.claim_id] = workflow.Verification(
                claim_id=claim_.claim_id, status=status, missing_conditions=[], rationale="scripted",
                supporting_ids=[chunk_id])
        with tempfile.TemporaryDirectory() as temp:
            state.trace_path = Path(temp) / "trace.jsonl"
            workflow.revise(state)
        self.assertEqual(seen, ["C003"])                       # the repeat never reaches the model
        self.assertEqual([c.claim_id for c in state.all_claims()], ["C001"])
        self.assertIn("drop_repeat", [d["action"] for d in state.decisions])
        self.assertFalse(state.gaps)

    def test_sentence_id_quote_becomes_exact_text_and_passes_the_gate(self):
        section, chunk_id = "manual:DJI Matrice 30T", "dji_m30t_manual_v22_c0070"
        text = workflow.SEARCH.by_id[chunk_id]["text"]
        n = next(k for k, (a, b) in enumerate(workflow.passage_sentences(text), 1) if "cannot avoid obstacles" in text[a:b])
        self.assertIn(f"[{n}] The aircraft cannot avoid obstacles", workflow.numbered(text))
        state = SimpleNamespace(config=workflow.Config(), pools={section: {chunk_id: workflow.SEARCH.by_id[chunk_id]}})
        draft = workflow.ClaimDraft(text="The manual states the aircraft cannot avoid obstacles during failsafe RTH "
                                         "when there is insufficient light.", citations=[chunk_id],
                                    quote=f"{chunk_id}#{n}", conditions=["insufficient light"], assessment="supported")
        claim = workflow.make_claim(state, section, draft, "C001")
        self.assertEqual(claim.quote, "The aircraft cannot avoid obstacles during failsafe RTH when there is insufficient light.")
        self.assertTrue(workflow.gate(state, claim)["passed"])
        self.assertEqual(workflow.resolve_quote("the drone always lands", [chunk_id]), "the drone always lands")
        # The model often puts the sentence id in citations and leaves quote empty.
        misplaced = workflow.make_claim(state, section, draft.model_copy(update={"citations": [f"{chunk_id}#{n}"],
                                                                                  "quote": ""}), "C002")
        self.assertEqual((misplaced.citations, misplaced.quote), ([chunk_id], claim.quote))
        self.assertTrue(workflow.gate(state, misplaced)["passed"])

    def test_identically_worded_claims_are_kept_once(self):
        section = "manual:DJI Matrice 30T"
        text = "The aircraft cannot avoid obstacles during failsafe RTH when there is insufficient light."
        make = lambda cid, text_: workflow.Claim(text=text_, citations=["x"], quote="x", conditions=[],
                                                 assessment="supported", claim_id=cid, section=section,
                                                 tier="manufacturer")
        state = workflow.RunState("What happens in low light?", workflow.Config(), workflow.FakeLLM({}))
        state.sections = [section]
        state.drafts[section] = {"status": "drafted", "abstain_reason": None, "claims": [
            make("C001", text), make("C002", "The RTH altitude is related to the Home Point."), make("C003", text.lower())]}
        with tempfile.TemporaryDirectory() as temp:
            state.trace_path = Path(temp) / "trace.jsonl"
            workflow.drop_repeated_claims(state)
        self.assertEqual([c.claim_id for c in state.all_claims()], ["C001", "C002"])
        self.assertFalse(state.removed_claims or state.gaps)

    def test_echoed_conditions_do_not_fail_a_supported_claim(self):
        claim = workflow.Claim(text="When the remote controller signal is lost, it can be set to RTH, land, or hover.",
                               citations=["dji_m30t_manual_v22_c0032"], quote="can be set to RTH, land, or hover",
                               conditions=["Loss of remote controller signal"], assessment="supported",
                               claim_id="C001", section="manual:DJI Matrice 30T", tier="manufacturer")
        for missing, expected in ((["loss of remote controller signal"], True), (["Home Point recorded"], False)):
            state = SimpleNamespace(gate_results={"C001": {"passed": True, "failures": []}},
                                    verifications={"C001": workflow.Verification(
                                        claim_id="C001", status="supported", missing_conditions=missing,
                                        rationale="scripted", supporting_ids=[])})
            self.assertIs(workflow.passes(state, claim), expected, missing)


if __name__ == "__main__":
    unittest.main()
