"""Check that the demo only displays evidence from the current corpus."""

import os
import unittest
from pathlib import Path

from ui_data import DEMO_QUESTIONS, brief_markdown, cited_passages, find_saved_run, load_corpus


class SavedRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ.get("DRONE_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
        cls.report, cls.chunks, cls.index_hash = load_corpus(cls.root)

    def test_conflict_example_resolves_its_citations(self):
        run = find_saved_run(self.root, DEMO_QUESTIONS[0], "full",
                             self.report["corpus_snapshot"], self.index_hash)
        self.assertIsNotNone(run)
        self.assertEqual(run["status"], "human_review_required")
        passages = cited_passages(run, self.chunks)
        self.assertGreater(len(passages), 0)
        self.assertTrue(all(p["chunk_id"] in self.chunks for p in passages))
        self.assertIn("Source:", brief_markdown(run, self.chunks))

    def test_outdated_snapshot_is_not_presented(self):
        run = find_saved_run(self.root, DEMO_QUESTIONS[0], "full", "wrong-snapshot", self.index_hash)
        self.assertIsNone(run)

    def test_no_evidence_example_is_available_in_both_modes(self):
        for mode in ("fixed", "full"):
            with self.subTest(mode=mode):
                run = find_saved_run(self.root, DEMO_QUESTIONS[1], mode,
                                     self.report["corpus_snapshot"], self.index_hash)
                self.assertIsNotNone(run)
                self.assertEqual(run["status"], "completed_with_gaps")


if __name__ == "__main__":
    unittest.main()
