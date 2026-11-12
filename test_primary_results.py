"""Dataset boundaries and immutable corrected prompt evidence for the paper."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import primary_results as primary


class PrimaryResultsTests(unittest.TestCase):
    def test_selected_corrected_evidence_and_recovery(self):
        manifest = primary.validate()
        self.assertEqual(manifest['total_slots'], 1200)
        self.assertEqual(manifest['followup_groq_responses'], 21)
        self.assertEqual(len(primary.LLM_PAIRS), 3)

    def test_prompt_or_raw_change_is_rejected(self):
        real_rows = primary.rows
        def tampered(path):
            data = real_rows(path)
            if str(path).endswith('raw_groq.jsonl'):
                data[0]['prompt'] += ' changed'
            return data
        with patch.object(primary, 'rows', side_effect=tampered):
            with self.assertRaises(AssertionError):
                primary.validate()



if __name__ == '__main__':
    unittest.main()
