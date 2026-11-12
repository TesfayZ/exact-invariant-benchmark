"""Separate transport-only follow-up; never modify the frozen primary run.

Preview: python3 recover_primary.py --plan-only
Run/resume: python3 recover_primary.py
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from run_llm import RateLimited, call_compat

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def identity(row):
    return row['key'], row['condition'], row['sample']


def append(path, record):
    with path.open('a') as out:
        out.write(json.dumps(record)+'\n')
        out.flush()
        os.fsync(out.fileno())


def wait_until(deadline):
    while time.time() < deadline:
        time.sleep(max(0., min(60., deadline-time.time())))


def run(a):
    source = Path(a.source).resolve()
    dest = Path(a.out_dir).resolve()
    if dest == source or source in dest.parents or dest in source.parents:
        raise SystemExit('Recovery directory must be separate from the original run.')
    plan_path = source/'plan.json'
    plan = json.loads(plan_path.read_text())
    if plan.get('mock') or plan.get('schema') != 'corrected-primary-v1':
        raise SystemExit('Expected a real corrected-primary-v1 run.')
    for name, digest in plan['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise SystemExit('Original collection source changed: '+name)
    original_path = source/'groq/raw.jsonl'
    original = rows(original_path)
    expected = {identity(r): r for r in plan['slots']}
    if len(original) != len(expected) or {identity(r) for r in original} != set(expected):
        raise SystemExit('Original Groq collection must contain every planned slot exactly once.')
    config = next(c for c in plan['providers'] if c['provider'] == 'groq')
    for r in original:
        if (r['plan_sha256'] != sha(plan_path)
                or r['prompt_sha256'] != expected[identity(r)]['prompt_sha256']
                or hashlib.sha256(r['prompt'].encode()).hexdigest() != r['prompt_sha256']
                or any(r[k] != v for k, v in config.items())):
            raise SystemExit('Original Groq record differs from frozen plan.')
    targets = [r for r in original if r.get('error')]
    frozen = dict(schema='primary-transport-recovery-v1', source=str(source),
                  source_plan_sha256=sha(plan_path), source_raw_sha256=sha(original_path),
                  runner_sha256=sha(Path(__file__)), provider=config,
                  slots=[expected[identity(r)] for r in targets],
                  seconds_between_requests=a.sleep, minimum_rate_limit_wait=a.cooldown,
                  policy='Transport-only follow-up. Retain every returned answer. '
                  'Append all attempts; preserve original failed slots. '
                  'Stop after three consecutive transport errors per invocation; '
                  'resume unresolved slots with persisted cooldowns.')
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/'collection.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another recovery collector is using this directory.')
        manifest = dest/'plan.json'
        if manifest.exists():
            if json.loads(manifest.read_text()) != frozen:
                raise SystemExit('Frozen recovery plan differs; use the same settings and source.')
        else:
            manifest.write_text(json.dumps(frozen, indent=2)+'\n')
        log = dest/'attempts.jsonl'
        history = rows(log)
        wanted = {identity(r) for r in targets}
        completed = {}
        for r in history:
            ident = identity(r)
            if ident not in wanted or ident in completed or r['recovery_plan_sha256'] != sha(manifest):
                raise SystemExit('Duplicate or mismatched recovery record.')
            if not r.get('error'):
                completed[ident] = r
        print(f'{len(targets)} original Groq transport failures; '
              f'{len(completed)} recovered; {len(targets)-len(completed)} remaining.', flush=True)
        if a.plan_only:
            print('Plan saved; no API calls made. Recovery is a separate follow-up.')
            return
        if len(completed) < len(targets) and not os.environ.get('GROQ_API_KEY'):
            raise SystemExit('Export GROQ_API_KEY in this terminal; no other provider keys needed.')
        deadline = max((r.get('next_attempt_at', 0.) for r in history), default=0.)
        consecutive_errors = 0
        for old in targets:
            ident = identity(old)
            if ident in completed:
                continue
            while ident not in completed:
                delay = max(0., deadline-time.time())
                if delay > 1:
                    print(f'Waiting {math.ceil(delay)} seconds before next request.', flush=True)
                wait_until(deadline)
                start = time.time()
                result = {}
                error = None
                status = None
                retry_after = None
                try:
                    result = call_compat('groq', config['model'], old['prompt'],
                                         config['temperature'], config['max_tokens'],
                                         reasoning_effort=config['reasoning_effort'])
                except Exception as exc:
                    status = getattr(exc, 'status', getattr(exc, 'code', None))
                    error = 'rate limited' if isinstance(exc, RateLimited) or status == 429 else (
                        f'HTTP {status}' if isinstance(status, int) else type(exc).__name__)
                    if isinstance(exc, RateLimited):
                        retry_after = float(exc.wait)
                    consecutive_errors += 1
                else:
                    consecutive_errors = 0
                end = time.time()
                cooldown = (max(a.cooldown * 2**min(consecutive_errors-1, 4),
                                retry_after if retry_after is not None and math.isfinite(retry_after) else 0.)
                            if error else a.sleep)
                deadline = end+cooldown
                record = dict(old, output=result.get('content'), error=error,
                              usage=result.get('usage', {}), finish_reason=result.get('finish_reason'),
                              response_model=result.get('response_model'),
                              system_fingerprint=result.get('system_fingerprint'),
                              transport_attempts=[dict(attempt=1, start=start, end=end,
                                  status='transport_error' if error else 'completed',
                                  http_status=status, retry_after_seconds=retry_after)],
                              recovery_plan_sha256=sha(manifest),
                              collection_schema='primary-transport-recovery-v1',
                              next_attempt_at=deadline, t=end)
                append(log, record)
                if error:
                    print(f'{ident}: {error}; attempt retained.', flush=True)
                    if consecutive_errors >= 3 or status in [400, 401, 402, 403, 404, 405, 410, 413, 422]:
                        raise SystemExit('Recovery paused. Resolve quota/access, then rerun the same command. '
                                         'The next request will respect the saved cooldown.')
                else:
                    completed[ident] = record
                    print(f'{len(completed)}/{len(targets)} recovered: {ident}', flush=True)
        # A derived dataset for the follow-up only; original evidence stays intact.
        combined = [completed.get(identity(r), r) for r in original]
        raw = dest/'combined_raw.jsonl'
        raw.write_text(''.join(json.dumps(r)+'\n' for r in combined))
        subprocess.run([sys.executable, str(ROOT/'grade_outputs.py'), '--raw', str(raw),
                        '--out', str(dest/'combined_graded.jsonl')], cwd=ROOT, check=True)
        print('Follow-up complete. Combined raw and graded files saved in '+str(dest))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='results/recollection/primary-corrected-v1')
    p.add_argument('--out-dir', default='results/recollection/primary-corrected-v1-groq-recovery')
    p.add_argument('--sleep', type=float, default=120.)
    p.add_argument('--cooldown', type=float, default=900.)
    p.add_argument('--plan-only', action='store_true')
    a = p.parse_args()
    if not all(math.isfinite(v) and v >= 0 for v in [a.sleep, a.cooldown]):
        p.error('Waits must be finite and nonnegative.')
    try:
        run(a)
    except KeyboardInterrupt:
        raise SystemExit('Stopped. Rerun the same command to resume saved recovery attempts.')


if __name__ == '__main__':
    main()
