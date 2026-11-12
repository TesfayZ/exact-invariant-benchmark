"""Select completed corrected responses for the paper; no API calls.

The frozen collection and follow-up are read-only inputs. Run once locally;
offline supplementary reproduction uses the selected results/primary files.
"""
import json
from pathlib import Path
from primary_results import digest, rows, validate


def main():
    source = Path('results/recollection/primary-corrected-v1')
    followup = Path('results/recollection/primary-corrected-v1-groq-recovery')
    dest = Path('results/primary')
    plan = json.loads((source/'plan.json').read_text())
    recovery = json.loads((followup/'plan.json').read_text())
    assert recovery['source_plan_sha256'] == digest(source/'plan.json')
    assert recovery['source_raw_sha256'] == digest(source/'groq/raw.jsonl')
    original = rows(source/'groq/raw.jsonl')
    key = lambda r: (r['key'], r['condition'], r['sample'])
    failed = {key(r): r for r in original if r.get('error')}
    attempts = rows(followup/'attempts.jsonl')
    recovered = {}
    for r in attempts:
        assert key(r) in failed and key(r) not in recovered
        assert r['recovery_plan_sha256'] == digest(followup/'plan.json')
        assert r['prompt'] == failed[key(r)]['prompt']
        if not r.get('error'):
            recovered[key(r)] = r
    assert len(failed) == len(recovered) == 21
    combined = [recovered.get(key(r), r) for r in original]
    assert combined == rows(followup/'combined_raw.jsonl')
    content = {'plan.json': (source/'plan.json').read_bytes()}
    for provider in ('groq', 'cerebras', 'gemini'):
        data = combined if provider == 'groq' else rows(source/provider/'raw.jsonl')
        content[f'raw_{provider}.jsonl'] = ''.join(json.dumps(r)+'\n' for r in data).encode()
    audit = [dict(original_failure=old, recovery_attempts=[r for r in attempts if key(r) == ident])
             for ident, old in failed.items()]
    content['groq_transport_followup.jsonl'] = ''.join(json.dumps(r)+'\n' for r in audit).encode()
    manifest = dict(schema='corrected-paper-primary-v1', slots_per_provider=400, total_slots=1200,
                    plan_sha256=digest(source/'plan.json'),
                    raw_sha256={p: __import__('hashlib').sha256(content[f'raw_{p}.jsonl']).hexdigest()
                                for p in ('groq', 'cerebras', 'gemini')},
                    initial_groq_responses=379, initial_groq_transport_failures=21,
                    followup_groq_responses=21,
                    followup_policy='Same exact prompts and generation settings; transport-only retries. '
                    '120 seconds between requests, minimum 900-second error cooldown; '
                    'honor longer Retry-After. Never resample a returned answer.',
                    followup_audit_sha256=__import__('hashlib').sha256(content['groq_transport_followup.jsonl']).hexdigest())
    content['manifest.json'] = (json.dumps(manifest, indent=2)+'\n').encode()
    dest.mkdir(exist_ok=True)
    for name, data in content.items():
        path = dest/name
        if path.exists() and path.read_bytes() != data:
            raise SystemExit('Refusing to overwrite changed primary evidence: '+str(path))
        path.write_bytes(data)
    validate()
    hashes = {f"{r['key']}|{r['condition']}|{r['sample']}": r['prompt_sha256'] for r in plan['slots']}
    Path('prompt_hashes_AB.json').write_text(json.dumps(hashes, indent=2, sort_keys=True)+'\n')
    print('Selected corrected primary dataset: 400 per provider, 1,200 total.')


if __name__ == '__main__':
    main()
