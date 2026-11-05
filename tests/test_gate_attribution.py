"""Incident claims pass the attribution check when they name their source, as the drafting rules ask."""

import unittest
from types import SimpleNamespace

import workflow_runtime as workflow


def incident_claim(section, chunk_id, text, quote):
    return workflow.Claim(text=text, citations=[chunk_id], quote=quote, conditions=[], assessment="supported",
                          claim_id="C001", section=section, tier="unverified", incident_label="investigated")


class GateAttributionTests(unittest.TestCase):
    def check(self, section, chunk_id, text, quote):
        state = SimpleNamespace(config=workflow.Config(),
                                pools={section: {chunk_id: workflow.SEARCH.by_id[chunk_id]}})
        return workflow.gate(state, incident_claim(section, chunk_id, text, quote))["failures"]

    def test_named_sources_pass(self):
        aaib = ("incident:DJI Matrice 210 V1", "aaib_m210_v1_poole_c0005")
        forum = ("incident:DJI Matrice 30T", "forum_dji_m30t_c0001")
        self.assertEqual(self.check(*aaib, "AAIB found that both controllers then lost communication with the aircraft.",
                                    "Both controllers then lost communication with the aircraft"), [])
        self.assertEqual(self.check(*forum, "A forum participant says the aircraft entered return to home without warning.",
                                    "Without warning the aircraft entered return to home"), [])

    def test_unattributed_incident_claim_fails(self):
        failures = self.check("incident:DJI Matrice 210 V1", "aaib_m210_v1_poole_c0005",
                              "Both controllers then lost communication with the aircraft.",
                              "Both controllers then lost communication with the aircraft")
        self.assertEqual(failures, ["attributed"])


if __name__ == "__main__":
    unittest.main()
