"""Offline checks for frozen plans, slot retention, resumption, and safe errors."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_llm

import collect_primary as c


class CollectionTests(unittest.TestCase):
    def test_outbound_generation_settings_and_gemini_thought_filter(self):
        from unittest.mock import MagicMock
        response = MagicMock()
        response.read.return_value = json.dumps({'choices': [{'message': {'content': 'OK'}, 'finish_reason': 'stop'}]}).encode()
        with patch.dict('os.environ', {'GROQ_API_KEY': 'test', 'GEMINI_API_KEY': 'test'}):
            with patch('urllib.request.urlopen', return_value=response) as send:
                run_llm.call_compat('groq', 'test', 'hello', 1., 4000, reasoning_effort='medium')
                payload = json.loads(send.call_args.args[0].data)
                self.assertEqual(payload['max_completion_tokens'], 4000)
                self.assertNotIn('max_tokens', payload)
                self.assertEqual(payload['reasoning_effort'], 'medium')
            response.read.return_value = json.dumps({'candidates': [{'content': {'parts': [
                {'text': 'private thought', 'thought': True}, {'text': 'OK'}]}, 'finishReason': 'STOP'}]}).encode()
            with patch('urllib.request.urlopen', return_value=response) as send:
                result = run_llm.call_gemini('test', 'hello', None, 4000, thinking_level='medium')
                config = json.loads(send.call_args.args[0].data)['generationConfig']
                self.assertEqual(config['thinkingConfig'], {'thinkingLevel': 'MEDIUM'})
                self.assertNotIn('temperature', config)
                self.assertEqual(result['content'], 'OK')

    def invoke(self, directory, extra=()):
        args = ['collect_primary.py', '--mock', '--n', '1', '--sleep', '0',
                '--out-dir', str(directory), *extra]
        with patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()):
            c.main()

    def test_collection_resume_and_frozen_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with patch.object(c.subprocess, 'run') as grader:
                self.invoke(directory)
                self.assertEqual(grader.call_count, 3)
                plan = json.loads((directory/'plan.json').read_text())
                self.assertEqual(plan['total_slots'], 60)
                self.assertEqual(len({(r['key'], r['condition']) for r in plan['slots']}), 20)
                paths = list(directory.glob('*/raw.jsonl'))
                original = {p: p.read_bytes() for p in paths}
                with patch.object(c, 'request', wraps=c.request) as request:
                    self.invoke(directory)
                    self.assertEqual(request.call_count, 3)  # preflights only
                self.assertTrue(all(p.read_bytes() == original[p] for p in paths))
                with self.assertRaisesRegex(SystemExit, 'Frozen plan differs'):
                    self.invoke(directory, ['--max-tokens', '3999'])
                for path in paths:
                    rows = c.load_rows(path)
                    self.assertEqual(len(rows), 20)
                    self.assertTrue(all(r['error'] is None and r['transport_attempts'] for r in rows))

    def test_failed_slots_retained_and_not_retried_on_resume(self):
        real = c.request
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            def fail_study(config, prompt, mock=False):
                if prompt == 'Reply with OK.':
                    return real(config, prompt, mock)
                raise c.RateLimited(0)
            with patch.object(c, 'request', side_effect=fail_study), self.assertRaises(SystemExit):
                self.invoke(directory)
            for path in directory.glob('*/raw.jsonl'):
                row, = c.load_rows(path)
                self.assertEqual(row['error'], 'rate limited')
                self.assertEqual(len(row['transport_attempts']), 3)
            with patch.object(c.subprocess, 'run'):
                self.invoke(directory)
            for path in directory.glob('*/raw.jsonl'):
                rows = c.load_rows(path)
                self.assertEqual(len(rows), 20)
                self.assertEqual(sum(bool(r['error']) for r in rows), 1)

    def test_preflight_errors_do_not_write_study_records_or_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(c, 'request', side_effect=ValueError('SECRET_TEST_VALUE')):
                with self.assertRaises(SystemExit) as caught:
                    self.invoke(tmp)
            self.assertNotIn('SECRET_TEST_VALUE', str(caught.exception))
            self.assertNotIn('SECRET_TEST_VALUE', (Path(tmp)/'preflight.jsonl').read_text())
            self.assertFalse(list(Path(tmp).glob('*/raw.jsonl')))


if __name__ == '__main__':
    unittest.main()
