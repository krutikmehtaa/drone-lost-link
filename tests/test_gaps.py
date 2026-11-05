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


    def conflict_kind(self, sections):
        llm = workflow.FakeLLM({"synthesis": lambda payload: {"gaps": [
            {"kind": "conflict", "text": "The two sources disagree.", "related_claim_ids": ["C001", "C002"]}]}})
        state = workflow.RunState("Do they agree?", workflow.Config(), llm)
        state.sections = sections
        for index, section in enumerate(sections, 1):
            state.drafts[section] = {"status": "drafted", "abstain_reason": None, "claims": [workflow.Claim(
                text="Some statement.", citations=["x"], quote="x", conditions=[], assessment="supported",
                claim_id=f"C{index:03d}", section=section, tier="manufacturer")]}
        with tempfile.TemporaryDirectory() as temp:
            state.trace_path = Path(temp) / "trace.jsonl"
            workflow.synthesize_gaps(state)
        return state.gaps[0]["kind"]

    def test_only_manual_against_incident_is_a_conflict(self):
        self.assertEqual(self.conflict_kind(["faa", "manual:DJI Matrice 4T"]), "unclear")
        self.assertEqual(self.conflict_kind(["manual:DJI Matrice 4T", "incident:DJI Matrice 4T"]), "conflict")

    def test_screened_out_report_is_named_and_leads_the_answer(self):
        manual, incident = "manual:DJI Matrice 4T", "incident:DJI Matrice 4T"
        llm = workflow.FakeLLM({"short_answer": lambda payload: {"short_answer":
                                "Failsafe RTH starts after signal loss. It continues if the link returns."}})
        state = workflow.RunState("The Matrice 4T ASRS report proves it lost its link. What should it have done?",
                                  workflow.Config(), llm)
        state.sections = [manual, incident]
        state.searches = [{"section": incident, "returned_ids": ["asrs_dji_m4t_selected_c0001"]}]
        state.drafts[manual] = {"status": "drafted", "abstain_reason": None, "claims": [workflow.Claim(
            text="Failsafe RTH starts after signal loss.", citations=["x"], quote="x", conditions=[],
            assessment="supported", claim_id="C001", section=manual, tier="manufacturer")]}
        with tempfile.TemporaryDirectory() as temp:
            state.trace_path = Path(temp) / "trace.jsonl"
            workflow.draft_section(state, incident)          # no passages kept, so no model call
            workflow.write_short_answer(state)
        note = "The DJI Matrice 4T incident report found in the collection does not describe a lost link"
        self.assertEqual(state.gaps[0]["origin"], "reports_screened_out")
        self.assertTrue(state.gaps[0]["text"].startswith(note))
        self.assertTrue(state.short_answer.startswith(note))


    def answer_sections(self, question):
        seen = []
        llm = workflow.FakeLLM({"short_answer": lambda payload: (seen.extend(c["section"] for c in payload["claims"]),
                                                                 {"short_answer": "One. Two."})[1]})
        state = workflow.RunState(question, workflow.Config(), llm)
        state.sections = ["faa", "manual:DJI Matrice 4T"]
        for index, section in enumerate(state.sections, 1):
            state.drafts[section] = {"status": "drafted", "abstain_reason": None, "claims": [workflow.Claim(
                text="Some statement.", citations=["x"], quote="x", conditions=[], assessment="supported",
                claim_id=f"C{index:03d}", section=section, tier="manufacturer")]}
        with tempfile.TemporaryDirectory() as temp:
            state.trace_path = Path(temp) / "trace.jsonl"
            workflow.write_short_answer(state)
        return seen

    def test_manual_question_is_answered_without_faa_claims(self):
        self.assertEqual(self.answer_sections("What should it have done according to the manual?"),
                         ["DJI Matrice 4T manual"])
        self.assertEqual(self.answer_sections("What do the FAA guidance and the manual say?"),
                         ["FAA guidance", "DJI Matrice 4T manual"])


if __name__ == "__main__":
    unittest.main()
