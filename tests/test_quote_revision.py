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


if __name__ == "__main__":
    unittest.main()
