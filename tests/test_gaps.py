"""The gap step runs only when there is something real to report."""

import tempfile
import unittest
from pathlib import Path

import workflow_runtime as workflow


class GapStepTests(unittest.TestCase):
    def run_gaps(self, sections, notes):
        llm = workflow.FakeLLM({"synthesis": lambda payload: {"gaps": [
            {"kind": "missing", "text": "Invented doubt.", "related_claim_ids": []}]}})
        state = workflow.RunState("What happens in low light?", workflow.Config(), llm)
        state.sections = sections
        for index, section in enumerate(sections, 1):
            state.drafts[section] = {"status": "drafted", "abstain_reason": None, "claims": [workflow.Claim(
                text="The manual states RTH starts automatically.", citations=["x"], quote="x", conditions=[],
                assessment="supported", claim_id=f"C{index:03d}", section=section, tier="manufacturer")]}
        state.assessments = {sections[0]: {"missing": notes, "conflicts_noticed": []}}
        with tempfile.TemporaryDirectory() as temp:
            state.trace_path = Path(temp) / "trace.jsonl"
            workflow.synthesize_gaps(state)
        return llm.log.count("synthesis"), state.gaps

    def test_one_section_without_agent_notes_skips_the_model(self):
        calls, gaps = self.run_gaps(["manual:DJI Matrice 30T"], [])
        self.assertEqual((calls, gaps), (0, []))

    def test_agent_notes_or_two_sections_still_run_it(self):
        self.assertEqual(self.run_gaps(["manual:DJI Matrice 30T"], ["RTH altitude limits"])[0], 1)
        self.assertEqual(self.run_gaps(["manual:DJI Matrice 30T", "incident:DJI Matrice 30T"], [])[0], 1)


if __name__ == "__main__":
    unittest.main()
