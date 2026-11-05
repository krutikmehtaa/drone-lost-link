"""Check answer wording and question routing without making model calls."""

import unittest

from ui_data import DEMO_QUESTIONS
from ui_presenter import answer_text, contextualize_question, review_note, source_scope


class PresenterTests(unittest.TestCase):
    def test_manual_question_uses_focused_sources(self):
        question = ("Under what conditions does the DJI Matrice 30T start Failsafe "
                    "Return to Home, and when can it not avoid obstacles?")
        self.assertEqual(source_scope(question), ("manual",))
        self.assertIsNone(source_scope("Do the DJI Matrice 30T manual and incident reports agree about RTH?"))

    def test_short_followup_keeps_one_model(self):
        previous = {"result": {"intake": {"models": ["DJI Matrice 30T"]}}}
        self.assertEqual(contextualize_question("What about obstacle avoidance?", previous),
                         "For the DJI Matrice 30T, what about obstacle avoidance?")
        self.assertEqual(contextualize_question("What about the Autel EVO Max 4N?", previous),
                         "What about the Autel EVO Max 4N?")

    def test_missing_rate_is_not_invented(self):
        run = {"question": "What is the lost-link failure rate per 1,000 flight hours?",
               "brief": {"short_answer": "FAA guidance says to regain control.",
                         "gaps": [{"text": "No numerical failure rate or flight-hour denominator is available."}]}}
        self.assertIn("do not establish", answer_text(run))
        self.assertIn("do not establish", answer_text({"question": DEMO_QUESTIONS[1]}, recorded=True))

    def test_manual_answer_keeps_a_first_sentence_without_the_word_manual(self):
        run = {"question": "What does the Matrice 30T manual say about altitude and obstacles during Failsafe RTH?",
               "brief": {"short_answer": "The RTH altitude must be higher than any obstacle in the area. "
                                         "The manual notes a default RTH altitude of 100 m.", "gaps": []}}
        self.assertTrue(answer_text(run).startswith("The RTH altitude must be higher"))

    def test_review_note_explains_dropped_claim(self):
        run = {"status": "human_review_required",
               "question": "When can it not avoid obstacles?",
               "decisions": [{"action": "drop", "reason": "still failing after 2 revision rounds"}],
               "brief": {"sections": [{"claims": [{"text": "RTH starts after six seconds."}]}]}}
        note = review_note(run)
        self.assertIn("left out", note)
        self.assertIn("obstacle-avoidance", note)


if __name__ == "__main__":
    unittest.main()
