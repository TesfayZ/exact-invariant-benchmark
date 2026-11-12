"""Freeze and collect the corrected primary suite; no credentials are stored.

python3 collect_primary.py --plan-only
python3 collect_primary.py
Resume with the same command. Completed slots, including failures, are immutable.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import signal
import subprocess
import sys
import time
import threading
import urllib.error

import systems
import prompts
from run_llm import COMPAT, RateLimited, call_compat, call_gemini, call_mock

ROOT = Path(__file__).resolve().parent
SOURCES = ['collect_primary.py', 'run_llm.py', 'prompts.py', 'systems.py', 'data.py',
           'grade_outputs.py', 'grade.py']


def sha(value):
    return hashlib.sha256(value).hexdigest()


def request(config, prompt, mock=False):
    if mock:
        return call_mock('mock', prompt, 1., config['max_tokens'])
    if config['provider'] == 'gemini':
        return call_gemini(config['model'], prompt, None, config['max_tokens'],
                           thinking_level='medium')
    return call_compat(config['provider'], config['model'], prompt, 1.,
                       config['max_tokens'], reasoning_effort='medium')


def load_rows(path):
    if not path.exists():
        return []
    # Fail on partial/corrupt records rather than silently losing an attempt.
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', default='results/recollection/primary-corrected-v1')
    p.add_argument('--n', type=int, default=20)
    p.add_argument('--max-tokens', type=int, default=4000)
    p.add_argument('--sleep', type=float, default=30.)
    p.add_argument('--max-attempts', type=int, default=3)
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--mock', action='store_true', help='Offline end-to-end verification only')
    a = p.parse_args()
    stopping = threading.Event()
    def stop(signum, frame):
        stopping.set()
        print('Stopping after in-flight requests; saved slots will be skipped on resume.', flush=True)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    def wait(seconds):
        if stopping.wait(seconds):
            raise RuntimeError('Collection interrupted')
    if a.n < 1 or a.max_tokens < 1 or a.max_attempts < 1 or a.sleep < 0:
        p.error('Counts must be positive and sleep nonnegative')
    directory = Path(a.out_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # Exclusive lock prevents two terminals from collecting the same slots.
    import fcntl
    with (directory/'collection.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another collector is using this directory')
        return run(a, directory, wait)


def run(a, directory, wait):
    slots = []
    for key in systems.CANONICAL + systems.PERTURBED:
        system = systems.REGISTRY[key]()
        for condition in ['A', 'B']:
            for sample in range(a.n):
                prompt = prompts.build(system, condition, seed=sample)
                slots.append(dict(key=key, system=system.name, condition=condition,
                                  sample=sample, prompt=prompt, prompt_sha256=sha(prompt.encode())))
    # Interleave systems and conditions to reduce confounding by collection time.
    random.Random(20261029).shuffle(slots)
    providers = []
    for provider, model in [('groq', 'openai/gpt-oss-120b'),
                            ('cerebras', 'gpt-oss-120b'),
                            ('gemini', 'gemini-3.5-flash-lite')]:
        providers.append(dict(provider=provider, model=model, max_tokens=a.max_tokens,
                              temperature=None if provider == 'gemini' else 1.,
                              reasoning_effort=None if provider == 'gemini' else 'medium',
                              thinking_level='medium' if provider == 'gemini' else None,
                              endpoint=('https://generativelanguage.googleapis.com/v1beta' if provider == 'gemini'
                                        else COMPAT[provider][0]),
                              stop=None, tools='not supplied; no tool execution',
                              unspecified_defaults=['top_p', 'sampling_seed', 'frequency_penalty', 'presence_penalty']
                              + (['temperature', 'top_k'] if provider == 'gemini' else [])))
    plan = dict(schema='corrected-primary-v1', mock=a.mock, providers=providers, slots=slots,
                slots_per_provider=len(slots), total_slots=len(slots)*3, noise=0.,
                conditions=['A', 'B'], samples_per_cell=a.n, seconds_between_slots=a.sleep,
                max_attempts=a.max_attempts, shuffle_seed=20261029,
                retry_policy='Transport only; retain empty/refused/invalid/truncated responses and exhausted errors; never replace completed slots',
                source_sha256={name: sha((ROOT/name).read_bytes()) for name in SOURCES},
                backend_sha256={name: sha((Path(systems._REPO)/name).read_bytes())
                                for name in ['ideal_diff.py', 'deflation.py', 'envs.py', 'bugs.py']},
                environment=dict(python=platform.python_version(), packages={
                    name: importlib.metadata.version(name) for name in ['numpy', 'sympy', 'scipy']}))
    manifest = directory/'plan.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != plan:
            raise SystemExit('Frozen plan differs (including source/environment). Choose a new directory; do not mix protocols.')
    else:
        manifest.write_text(json.dumps(plan, indent=2)+'\n')
    plan_hash = sha(manifest.read_bytes())
    print(f'Frozen plan: {manifest}\n{plan["total_slots"]} slots; {len(slots)} per provider; cap {a.max_tokens}; up to {a.max_attempts} transport attempts/slot', flush=True)
    if a.plan_only:
        return
    if not a.mock:
        missing = [key for key in ['GROQ_API_KEY', 'CEREBRAS_API_KEY', 'GEMINI_API_KEY'] if not os.environ.get(key)]
        if missing:
            raise SystemExit('Export these keys in this terminal: '+', '.join(missing))
    # Test exact request settings on every provider before starting study slots.
    # Diagnostics are separate from the study and never included in its scores.
    for config in providers:
        passed = False
        for attempt in range(a.max_attempts):
            try:
                result = request(config, 'Reply with OK.', a.mock)
                record = dict(provider=config['provider'], plan_sha256=plan_hash,
                              t=time.time(), status='accepted', usage=result.get('usage'),
                              response_model=result.get('response_model'), finish_reason=result.get('finish_reason'))
                passed = True
            except Exception as exc:
                status = getattr(exc, 'status', getattr(exc, 'code', None))
                record = dict(provider=config['provider'], t=time.time(), status='failed',
                              error_type=type(exc).__name__, http_status=status)
            with (directory/'preflight.jsonl').open('a') as out:
                out.write(json.dumps(record)+'\n')
            if passed:
                break
            if isinstance(status, int) and status not in [408, 429, 500, 502, 503, 504]:
                break
            if attempt+1 < a.max_attempts:
                wait(max(a.sleep, 60.) if not a.mock else 0.)
        if not passed:
            raise SystemExit(f'{config["provider"]}: preflight failed ({record["error_type"]}, HTTP {record["http_status"]}). No study slots started. Check account quota/access; settings were not changed.')
        print(config['provider']+': preflight accepted', flush=True)

    def collect(config):
        provider = config['provider']
        folder = directory/provider
        folder.mkdir(exist_ok=True)
        raw = folder/'raw.jsonl'
        rows = load_rows(raw)
        expected = {(r['key'], r['condition'], r['sample']): r for r in slots}
        done = set()
        for r in rows:
            identity = (r['key'], r['condition'], r['sample'])
            if identity in done or identity not in expected or r['plan_sha256'] != plan_hash or r['prompt_sha256'] != expected[identity]['prompt_sha256']:
                raise RuntimeError(provider+': duplicate or mismatched saved slot')
            done.add(identity)
        with raw.open('a') as out:
            for slot in slots:
                identity = (slot['key'], slot['condition'], slot['sample'])
                if identity in done:
                    continue
                wait(a.sleep)
                attempts = []
                result = {}; error = None; pause = False
                for attempt in range(a.max_attempts):
                    start = time.time()
                    try:
                        result = request(config, slot['prompt'], a.mock)
                        attempts.append(dict(attempt=attempt+1, start=start, end=time.time(), status='completed'))
                        error = None
                        break
                    except Exception as exc:
                        status = getattr(exc, 'status', getattr(exc, 'code', None))
                        limited = isinstance(exc, RateLimited) or status == 429
                        error = 'rate limited' if limited else (f'HTTP {status}' if isinstance(status, int) else type(exc).__name__)
                        attempts.append(dict(attempt=attempt+1, start=start, end=time.time(),
                                             status='transport_error', http_status=status, error_type=type(exc).__name__))
                        pause = status in [400, 401, 402, 403, 404, 405, 410, 413, 422]
                        if pause:
                            break
                        if attempt+1 < a.max_attempts:
                            delay = max(60., min(float(getattr(exc, 'wait', 60.)), 300.)) if limited else 2**attempt
                            try:
                                wait(delay if not a.mock else 0.)
                            except RuntimeError:
                                break  # Persist attempts already made before stopping.
                record = dict(slot, **config, noise=0., output=result.get('content'), error=error,
                              usage=result.get('usage', {}), finish_reason=result.get('finish_reason'),
                              response_model=result.get('response_model'), system_fingerprint=result.get('system_fingerprint'),
                              prompt_feedback=result.get('prompt_feedback'), transport_attempts=attempts,
                              max_attempts=a.max_attempts, plan_sha256=plan_hash,
                              collection_schema='corrected-primary-v1', suite='primary', t=time.time())
                out.write(json.dumps(record)+'\n'); out.flush(); os.fsync(out.fileno())
                done.add(identity)
                print(f'{provider}: {len(done)}/{len(slots)} {slot["key"]}/{slot["condition"]}/{slot["sample"]}: {error or "saved"}', flush=True)
                # Stop after an exhausted transport failure to avoid consuming an
                # entire sweep during a quota outage. This failed slot stays saved.
                if error:
                    raise RuntimeError(provider+': paused after '+error+'; resolve access/quota, then resume the same command. Failed slot retained.')
        subprocess.run([sys.executable, str(ROOT/'grade_outputs.py'), '--raw', str(raw),
                        '--out', str(folder/'graded.jsonl')], cwd=ROOT, check=True)
        print(provider+': collection and grading complete', flush=True)

    failures = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [(c['provider'], pool.submit(collect, c)) for c in providers]
        for provider, job in jobs:
            try:
                job.result()
            except Exception as exc:
                # Only our own errors are printed; unexpected exception messages
                # could contain request details. Credentials never enter records.
                failures.append(provider)
                print(provider+': collection incomplete ('+type(exc).__name__+'). Check saved transport records; rerun the same command to resume.', flush=True)
    if failures:
        raise SystemExit('Incomplete providers: '+', '.join(failures))
    print('All primary collections complete. Original results and paper unchanged.', flush=True)


if __name__ == '__main__':
    main()
