"""
Usage:  python run_llm.py --provider ollama --model qwen2.5:7b --n 5                    # free, local
        python run_llm.py --provider gemini --model <model-id> --n 5 --sleep 5            # free; check model list
        python run_llm.py --model claude-sonnet-5-5 --provider anthropic --n 20
        python run_llm.py --provider mock --model mock --n 3          # offline test
        python run_llm.py --provider groq --model openai/gpt-oss-120b --n 20 \
            --systems twisted_cubic --conditions B --max-tokens 16000 \
            --out /tmp/fresh_groq_mt16000.jsonl
Env:    ANTHROPIC_API_KEY / OPENAI_API_KEY as needed.
"""
import argparse, json, os, sys, time, random, zlib
import systems, prompts


def call_anthropic(model, prompt, temperature, max_tokens=4000, stop=None):
    import anthropic
    c = anthropic.Anthropic()
    kw = dict(model=model, max_tokens=max_tokens, temperature=temperature,
              messages=[{"role": "user", "content": prompt}])
    if stop: kw["stop_sequences"] = list(stop)
    r = c.messages.create(**kw)
    content = "".join(b.text for b in r.content if getattr(b, "text", None))
    usage = {"input_tokens": getattr(r.usage, "input_tokens", None),
             "output_tokens": getattr(r.usage, "output_tokens", None)}
    return {"content": content, "usage": usage, "finish_reason": getattr(r, "stop_reason", None)}


def call_openai(model, prompt, temperature, max_tokens=4000, stop=None):
    from openai import OpenAI
    kw = dict(model=model, temperature=temperature, max_tokens=max_tokens,
              messages=[{"role": "user", "content": prompt}])
    if stop: kw["stop"] = list(stop)
    r = OpenAI().chat.completions.create(**kw)
    usage = r.usage.model_dump() if getattr(r, "usage", None) else {}
    return {"content": r.choices[0].message.content, "usage": usage,
            "finish_reason": r.choices[0].finish_reason}


def call_gemini(model, prompt, temperature, max_tokens=4000, stop=None, thinking_budget=None, thinking_level=None):
    """Google AI Studio free tier. Key: https://aistudio.google.com/app/apikey -> env GEMINI_API_KEY.
    Plain REST, no SDK. Free-tier limits are per-project and unpublished; use --sleep to stay under them."""
    import urllib.request, urllib.error
    key = os.environ["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    gen = {"maxOutputTokens": max_tokens}
    if temperature is not None: gen["temperature"] = temperature
    if stop: gen["stopSequences"] = list(stop)
    if thinking_budget is not None: gen["thinkingConfig"] = {"thinkingBudget": int(thinking_budget)}
    if thinking_level is not None:
        if thinking_budget is not None: raise ValueError('Choose thinking level or budget, not both')
        gen["thinkingConfig"] = {"thinkingLevel": thinking_level.upper()}
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": gen}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json", "x-goog-api-key": key})
    j = json.load(urllib.request.urlopen(req, timeout=120))
    cand = next(iter(j.get("candidates", [])), {})
    content = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get('thought'))
    return {"content": content, "usage": j.get("usageMetadata", {}),
            "response_model":j.get('modelVersion'),
            "finish_reason": cand.get("finishReason"), 'prompt_feedback': j.get('promptFeedback')}


def call_ollama(model, prompt, temperature, max_tokens=4000, stop=None):
    """Local, free, offline. Install https://ollama.com, then `ollama pull qwen2.5:7b`."""
    import urllib.request
    opts = {"temperature": temperature, "num_predict": max_tokens}
    if stop: opts["stop"] = list(stop)
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": opts}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", body, {"Content-Type": "application/json"})
    j = json.load(urllib.request.urlopen(req, timeout=600))
    usage = {"eval_count": j.get("eval_count"), "prompt_eval_count": j.get("prompt_eval_count")}
    return {"content": j.get("response", ""), "usage": usage, "finish_reason": j.get("done_reason")}


# ---- OpenAI-compatible chat endpoints (one function covers many hosts) -------------------
COMPAT = {
    "groq":       ("https://api.groq.com/openai/v1",                          "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1",                            "OPENROUTER_API_KEY"),
    "nvidia":     ("https://integrate.api.nvidia.com/v1",                     "NVIDIA_API_KEY"),
    "cerebras":   ("https://api.cerebras.ai/v1",                              "CEREBRAS_API_KEY"),
    "deepseek":   ("https://api.deepseek.com",                                "DEEPSEEK_API_KEY"),
    "mistral":    ("https://api.mistral.ai/v1",                               "MISTRAL_API_KEY"),
    "gemini-oai": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
}

class RateLimited(Exception):
    def __init__(self, wait): self.wait = wait

class ProviderHTTPError(RuntimeError):
    """Safe provider error: retain status, never credentials or response bodies."""
    def __init__(self, status):
        self.status = int(status)
        super().__init__(f'HTTP {self.status}')

def call_compat(provider, model, prompt, temperature, max_tokens=4000, stop=None,
                base_url=None, key_env=None, reasoning_effort=None):
    import urllib.request, urllib.error
    base, env = COMPAT.get(provider, (base_url, key_env))
    base = base_url or base; env = key_env or env
    if not base or not env: raise ValueError("custom endpoint needs --base-url and --key-env")
    key = os.environ[env]
    payload = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
               "messages": [{"role": "user", "content": prompt}]}
    if provider in ('groq', 'cerebras'):
        payload['max_completion_tokens'] = payload.pop('max_tokens')
    if stop: payload["stop"] = list(stop)
    if reasoning_effort: payload["reasoning_effort"] = reasoning_effort   # gpt-oss: low | medium | high
    body = json.dumps(payload).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", body,
                             {"Content-Type": "application/json",
                              "Authorization": f"Bearer {key}",
                              "User-Agent": "Mozilla/5.0 (compatible; ExactVerification/1.0)"})
    try:
        j = json.load(urllib.request.urlopen(req, timeout=180))
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise RateLimited(float(e.headers.get("Retry-After", 30)))
        raise ProviderHTTPError(e.code)
    ch = j["choices"][0]
    return {"content": ch["message"].get("content"),
            "response_model":j.get('model'), "system_fingerprint":j.get('system_fingerprint'),
            "usage": j.get("usage", {}),
            "finish_reason": ch.get("finish_reason")}


def call_mock(model, prompt, temperature, max_tokens=4000, stop=None, **_):
    """Deterministic fake model for pipeline tests: mixes right, wrong, refuse.
    Seeded by a stable hash of the prompt (zlib.crc32), reproducible across processes."""
    rng = random.Random(zlib.crc32(prompt.encode("utf-8")))
    k = rng.random()
    if "generating set" in prompt:
        out = json.dumps({"polynomials": ["y - x**2"] if k < .5 else ["y - x**2", "z - x**3"], "refuse": False})
    elif k < .2:
        out = json.dumps({"polynomials": [], "refuse": True})
    elif k < .5:
        out = "Sure! " + json.dumps({"polynomials": ["w1**2 + 2*w2**2 + 3*w3**2"], "refuse": False})
    else:
        out = json.dumps({"polynomials": ["w1**2 + w2**2 + w3**2"], "refuse": False})
    return {"content": out, "usage": {}, "finish_reason": "stop"}


PROVIDERS = {"anthropic": call_anthropic, "openai": call_openai, "gemini": call_gemini,
             "ollama": call_ollama, "mock": call_mock, "custom": None}
for _p in COMPAT:
    PROVIDERS[_p] = None   # dispatched through call_compat


def _stop_list(s):
    return [x for x in (s or "").split(",") if x] or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=PROVIDERS)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4000,
                    help="completion budget per call; recorded per row and in the resumability key")
    ap.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high"],
                    help="OpenAI-compatible hosts serving gpt-oss. RECORD IT: host defaults differ and are unpublished.")
    ap.add_argument("--thinking-budget", type=int, default=None,
                    help="gemini only: thinkingConfig.thinkingBudget (0 disables thinking)")
    ap.add_argument("--stop", default=None, help="comma-separated stop sequences (optional)")
    ap.add_argument("--systems", default=",".join(systems.DEFAULT),
                    help="comma list of keys, or: all | perturbed | canonical")
    ap.add_argument("--conditions", default="A,B")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--base-url", default=None, help="for --provider custom (OpenAI-style /chat/completions)")
    ap.add_argument("--key-env", default=None, help="env var holding the key, for --provider custom")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between calls (free-tier rate limits)")
    ap.add_argument("--out", default="results/new_llm_raw.jsonl")
    ap.add_argument('--suite', choices=['primary','planted'], default='primary')
    ap.add_argument('--max-attempts', type=int, default=6,
                    help='Transport attempts; empty/refused/invalid completions are never retried')
    a = ap.parse_args()
    if a.max_attempts < 1: ap.error('--max-attempts must be positive')
    if a.suite == 'planted':
        if a.conditions != 'A': ap.error('The prospective planted suite currently supports --conditions A only')
        from fresh_systems import REGISTRY
        systems.REGISTRY.update(REGISTRY)
        if a.systems == ','.join(systems.DEFAULT) or a.systems == 'all':
            a.systems = ','.join(REGISTRY)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    stop_list = _stop_list(a.stop)

    # Resume only the same generation settings and exact prompt. Missing legacy
    # budgets stay unknown, rather than being silently treated as 4000 tokens.
    import hashlib
    prompt_hash = lambda text: hashlib.sha256(text.encode()).hexdigest()
    def _cfg(provider, mt, stop, eff, tb, temperature, base_url):
        endpoint = base_url or COMPAT.get(provider, (None, None))[0]
        return (provider or "", mt, tuple(stop) if stop else None, eff, tb, temperature, endpoint)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            d = json.loads(l)
            done.add(_cfg(d.get("provider"), d.get("max_tokens"), d.get("stop"), d.get("reasoning_effort"),
                          d.get("thinking_budget"), d.get("temperature"), d.get("base_url")) +
                     (d["model"], d["system"], d["condition"], d["noise"], d["sample"], prompt_hash(d.get("prompt", ""))))
    this_cfg = _cfg(a.provider, a.max_tokens, stop_list, a.reasoning_effort, a.thinking_budget,
                    a.temperature, a.base_url)

    # Fail fast on a missing key instead of writing one KeyError row per call (paper audit item 9).
    if a.provider in COMPAT or a.provider == "custom":
        _env = a.key_env or COMPAT.get(a.provider, (None, None))[1]
        if not _env or _env not in os.environ: raise SystemExit(f"environment variable {_env} is not set")
    elif a.provider == "gemini" and "GEMINI_API_KEY" not in os.environ:
        raise SystemExit("GEMINI_API_KEY is not set")

    if a.provider in COMPAT or a.provider == "custom":
        fn = lambda m, p, t, mt, st: call_compat(a.provider, m, p, t, max_tokens=mt, stop=st,
                                                 base_url=a.base_url, key_env=a.key_env,
                                                 reasoning_effort=a.reasoning_effort)
    elif a.provider == "gemini":
        fn = lambda m, p, t, mt, st: call_gemini(m, p, t, mt, st, thinking_budget=a.thinking_budget)
    else:
        fn = PROVIDERS[a.provider]

    with open(a.out, "a") as f:
        keys = {"all": list(systems.REGISTRY), "perturbed": systems.PERTURBED,
                "canonical": systems.CANONICAL}.get(a.systems, a.systems.split(","))
        for key in keys:
            s = systems.REGISTRY[key]()
            for cond in a.conditions.split(","):
                for i in range(a.n):
                    prompt = prompts.build(s, cond, seed=i, noise=a.noise)
                    if this_cfg + (a.model, s.name, cond, a.noise, i, prompt_hash(prompt)) in done: continue
                    if a.sleep: time.sleep(a.sleep)
                    out = None; err = None; usage = {}; finish = None; pack = {}; permanent_error = False
                    attempts = []
                    for attempt in range(a.max_attempts):
                        attempt_start=time.time()
                        try:
                            pack = fn(a.model, prompt, a.temperature, a.max_tokens, stop_list)
                            out = pack.get("content")
                            usage = pack.get("usage") or {}
                            finish = pack.get("finish_reason")
                            attempts.append(dict(attempt=attempt+1,status='completed',start=attempt_start,end=time.time()))
                            err = None
                            break
                        except RateLimited as e:
                            err = "rate limited"
                            attempts.append(dict(attempt=attempt+1,status='rate_limited',start=attempt_start,end=time.time()))
                            if attempt+1<a.max_attempts: time.sleep(min(e.wait, 60))
                        except Exception as e:
                            # Exception strings and provider response bodies can contain secrets.
                            status = getattr(e,'status',getattr(e,'code',None))
                            err = f'HTTP {status}' if isinstance(status,int) else type(e).__name__
                            attempts.append(dict(attempt=attempt+1,status='transport_error',error_type=type(e).__name__,http_status=status if isinstance(status,int) else None,start=attempt_start,end=time.time()))
                            permanent_error = status in (400,401,402,403,404,405,410,422)
                            if permanent_error: break
                            if attempt+1<a.max_attempts: time.sleep(min(2 ** attempt,60))
                    rec = dict(model=a.model, provider=a.provider, system=s.name, key=key,
                               condition=cond, noise=a.noise, sample=i, temperature=a.temperature,
                               max_tokens=a.max_tokens, stop=stop_list,
                               base_url=a.base_url or COMPAT.get(a.provider, (None, None))[0],
                               prompt_sha256=prompt_hash(prompt),
                               reasoning_effort=a.reasoning_effort, thinking_budget=a.thinking_budget,
                               prompt=prompt, output=out,
                               error=None if out is not None else err,
                               usage=usage, finish_reason=finish,
                               suite=a.suite, transport_attempts=attempts, max_attempts=a.max_attempts,
                               collection_schema='recorded-settings-v2',
                               response_model=pack.get('response_model'), system_fingerprint=pack.get('system_fingerprint'),
                               t=time.time())
                    f.write(json.dumps(rec, default=str) + "\n"); f.flush()
                    print(f'{key} {cond} {i+1}/{a.n}: '+(err if err else 'response saved'),flush=True)
                    if permanent_error:
                        raise SystemExit(f'Collection stopped after {err}. Check provider access, model ID and request settings before retrying; failed record retained.')
    print("done ->", a.out)


if __name__ == "__main__":
    main()
