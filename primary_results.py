"""Explicit paper dataset: 400 corrected responses per LLM, plus numerical data."""
import hashlib
import json
from pathlib import Path

PROVIDERS = ('groq', 'cerebras', 'gemini')
LLM_PAIRS = [(f'primary/raw_{p}', f'primary/graded_{p}') for p in PROVIDERS]
PRIMARY_PAIRS = LLM_PAIRS + [('raw_sindy15', 'graded_sindy15')]


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate():
    """Fail on added/omitted slots, changed prompts/settings, or altered raw evidence."""
    root = Path('results/primary')
    plan = json.loads((root/'plan.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    expected = {(r['key'], r['condition'], r['sample']): r for r in plan['slots']}
    assert len(expected) == 400 and plan['samples_per_cell'] == 20
    assert digest(root/'plan.json') == manifest['plan_sha256']
    audit_path = root/'groq_transport_followup.jsonl'
    assert digest(audit_path) == manifest['followup_audit_sha256']
    audit = rows(audit_path)
    assert len(audit) == manifest['followup_groq_responses'] == 21
    recovered = {}
    for entry in audit:
        old = entry['original_failure']
        assert old['error'] and old['output'] is None
        attempts = entry['recovery_attempts']
        assert all(a['prompt'] == old['prompt'] for a in attempts)
        delivered = [a for a in attempts if not a.get('error')]
        assert len(delivered) == 1
        key = (old['key'], old['condition'], old['sample'])
        assert key not in recovered
        recovered[key] = delivered[0]
    for config in plan['providers']:
        provider = config['provider']
        path = root/f'raw_{provider}.jsonl'
        data = rows(path)
        assert len(data) == 400
        identities = [(r['key'], r['condition'], r['sample']) for r in data]
        assert len(set(identities)) == 400 and set(identities) == set(expected)
        assert digest(path) == manifest['raw_sha256'][provider]
        for r, identity in zip(data, identities):
            assert not r.get('error'), (provider, identity, r.get('error'))
            assert r['prompt'] == expected[identity]['prompt']
            assert hashlib.sha256(r['prompt'].encode()).hexdigest() == r['prompt_sha256'] == expected[identity]['prompt_sha256']
            assert all(r[k] == v for k, v in config.items())
            if provider == 'groq' and identity in recovered:
                assert r == recovered[identity]
    return manifest


if __name__ == '__main__':
    validate()
    print('Corrected primary evidence verified: 400 per provider, 1,200 total.')
