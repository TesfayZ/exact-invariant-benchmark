"""Offline checks for transport-only recovery and immutable original evidence."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import recover_primary as recovery


class RecoveryTests(unittest.TestCase):
    def fixture(self, root):
        source = root/'original'
        (source/'groq').mkdir(parents=True)
        config = dict(provider='groq', model='test-model', temperature=1.,
                      max_tokens=4000, reasoning_effort='medium')
        slots = [dict(key='test', condition='A', sample=i, prompt=f'prompt {i}',
                      prompt_sha256=hashlib.sha256(f'prompt {i}'.encode()).hexdigest())
                 for i in range(2)]
        plan = dict(schema='corrected-primary-v1', mock=False, source_sha256={},
                    providers=[config], slots=slots)
        (source/'plan.json').write_text(json.dumps(plan))
        original = [dict(s, **config, error='rate limited' if i else None,
                         output=None if i else 'original answer',
                         plan_sha256=recovery.sha(source/'plan.json'))
                    for i, s in enumerate(slots)]
        path = source/'groq/raw.jsonl'
        path.write_text(''.join(json.dumps(r)+'\n' for r in original))
        args = SimpleNamespace(source=str(source), out_dir=str(root/'recovery'),
                               sleep=0., cooldown=0., plan_only=False)
        return args, path, path.read_bytes()

    def test_only_failed_slots_recovered_and_empty_answer_never_resampled(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, path, before = self.fixture(Path(tmp))
            with patch.dict('os.environ', {'GROQ_API_KEY': 'test'}), \
                    patch.object(recovery, 'call_compat', return_value={'content': ''}) as request, \
                    patch.object(recovery.subprocess, 'run'), contextlib.redirect_stdout(io.StringIO()):
                recovery.run(args)
                recovery.run(args)
            self.assertEqual(request.call_count, 1)
            self.assertEqual(request.call_args.args[2], 'prompt 1')
            self.assertEqual(path.read_bytes(), before)
            combined = recovery.rows(Path(args.out_dir)/'combined_raw.jsonl')
            self.assertEqual([r['output'] for r in combined], ['original answer', ''])

    def test_failed_attempts_append_and_resume_honors_full_retry_after(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, path, before = self.fixture(Path(tmp))
            with patch.dict('os.environ', {'GROQ_API_KEY': 'test'}), \
                    patch.object(recovery, 'call_compat', side_effect=recovery.RateLimited(7200)), \
                    patch.object(recovery, 'wait_until') as wait, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(SystemExit, 'Recovery paused'):
                    recovery.run(args)
            log = recovery.rows(Path(args.out_dir)/'attempts.jsonl')
            self.assertEqual(len(log), 3)
            self.assertTrue(all(r['next_attempt_at']-r['t'] >= 7200 for r in log))
            with patch.dict('os.environ', {'GROQ_API_KEY': 'test'}), \
                    patch.object(recovery, 'call_compat', return_value={'content': 'answer'}), \
                    patch.object(recovery, 'wait_until') as resumed_wait, \
                    patch.object(recovery.subprocess, 'run'), contextlib.redirect_stdout(io.StringIO()):
                recovery.run(args)
            self.assertEqual(resumed_wait.call_args.args[0], log[-1]['next_attempt_at'])
            self.assertEqual(len(recovery.rows(Path(args.out_dir)/'attempts.jsonl')), 4)
            self.assertEqual(path.read_bytes(), before)

    def test_plan_only_makes_no_calls_and_source_changes_block_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, path, before = self.fixture(Path(tmp))
            args.plan_only = True
            with patch.object(recovery, 'call_compat') as request, contextlib.redirect_stdout(io.StringIO()):
                recovery.run(args)
                request.assert_not_called()
                path.write_bytes(before.replace(b'original answer', b'changed answer'))
                with self.assertRaisesRegex(SystemExit, 'Frozen recovery plan differs'):
                    recovery.run(args)

    def test_changed_failed_prompt_is_rejected_before_any_api_call(self):
        for update_hash in [False, True]:
            with self.subTest(update_hash=update_hash), tempfile.TemporaryDirectory() as tmp:
                args, path, _ = self.fixture(Path(tmp))
                records = recovery.rows(path)
                records[1]['prompt'] = 'historical or modified prompt'
                if update_hash:
                    records[1]['prompt_sha256'] = hashlib.sha256(
                        records[1]['prompt'].encode()).hexdigest()
                path.write_text(''.join(json.dumps(r)+'\n' for r in records))
                with patch.object(recovery, 'call_compat') as request:
                    with self.assertRaisesRegex(SystemExit, 'differs from frozen plan'):
                        recovery.run(args)
                    request.assert_not_called()


if __name__ == '__main__':
    unittest.main()
