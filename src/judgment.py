"""The model layer. One call per message: a structured differential -- most
likely, can't miss, whether a clinician must act -- that policy routes on.

Provider is OpenRouter, which speaks the OpenAI chat-completions API. To go
direct to Anthropic instead, swap _get_client() for the Anthropic SDK -- the
rest of this file and everything downstream is unchanged.

A parse failure is not an exception to handle. It is a designed route:
policy turns it into ASYNC_PHYSICIAN + MODEL_FAILURE.

A missing API key is the opposite: it is an exception, and it must not become
that route -- see env.py. Filed as a deferral it would let a keyless sweep
write a full set of plausible result files.

One more failure that is neither: `finish_reason: length`. The served model
spends reasoning tokens inside max_tokens, and a response that runs out of
budget before the JSON closes parses as nothing. That lands as MODEL_FAILURE
through the designed route, which is correct policy and misleading telemetry,
so judge() records the finish reason and the caller can tell truncation from
a refusal. Raising max_tokens changes cache_key() and re-fetches everything.

The served model thinks by default and does not accept a sampling
temperature, so neither is left to chance: REASONING_EFFORT is sent on every
call and sits in cache_key(), and nothing claims temperature 0. Run-to-run
variation is measured (run_eval.py --no-cache), not assumed away.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

from .env import load_env, require_key
from pydantic import BaseModel

from .schema import ClinicalJudgment, ScreenVerdict

load_env()   # before the reads below: DISPO_MODEL and DISPO_BASE_URL live in .env too

BASE_URL = os.getenv("DISPO_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_MODEL = os.getenv("DISPO_MODEL", "anthropic/claude-sonnet-5.5")
# Thinking tokens are billed as output and spent inside MAX_TOKENS.
REASONING_EFFORT = os.getenv("DISPO_REASONING_EFFORT", "low")
# The differential is several times longer than the old flat schema.
MAX_TOKENS = 4000
PROMPT_PATH = Path("prompts/judgment.txt")
# Urgency screen: one short verdict, so a smaller budget (thinking included)
SCREEN_PROMPT_PATH = Path("prompts/screen.txt")
SCREEN_MAX_TOKENS = 1500
# Its own file: each cached wrapper holds the whole store in memory and rewrites
# it on every live call, so two wrappers on one file overwrite each other.
SCREEN_CACHE_PATH = Path("results/screen_cache.json")
CACHE_PATH = Path("results/judgment_cache.json")

# OpenRouter pricing, $ per 1M tokens. Update if you change models.
PRICE_IN, PRICE_OUT = 2.00, 10.00

_client = None


def _get_client():
    """Raises RuntimeError without a key. Never called from inside judge()'s
    retry loop, so that error propagates instead of being filed as a cause."""
    global _client
    if _client is None:
        from openai import OpenAI
        # A per-request ceiling, not the SDK's ten minutes: a connection that dies
        # mid-call (the machine sleeps, the network drops) must surface as a
        # retryable timeout, not hang the sweep. ~3s is a normal call.
        _client = OpenAI(base_url=BASE_URL, api_key=require_key(), timeout=90.0)
    return _client


def build_prompt(message: str, prompt_path: Path = PROMPT_PATH) -> str:
    return prompt_path.read_text().replace("{{MESSAGE}}", message)


def _extract_json(text: str) -> dict | None:
    """The first complete JSON object in the text, and nothing after it.

    Not "first { to last }": a stray brace in prose after the object made that
    span invalid and filed a finished answer as MODEL_FAILURE."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def cost_usd(in_tokens: int, out_tokens: int) -> float:
    return round(in_tokens / 1e6 * PRICE_IN + out_tokens / 1e6 * PRICE_OUT, 6)


def classify_failure(exc: Exception) -> tuple[bool, str]:
    """(retry?, short cause).

    Transport noise -- throttling, 5xx, dropped connections, timeouts -- says
    nothing about whether the model can route the message, and letting it land
    as MODEL_FAILURE puts infrastructure flakiness into the deferral rate.
    A 4xx that is not 408/409/429 is a wrong key or a wrong model slug: fail
    fast and loud instead of sleeping three times first.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    name = type(exc).__name__
    if status is not None:
        if status in (408, 409, 429) or 500 <= status < 600:
            return True, f"http_{status}"
        return False, f"http_{status}"
    if any(k in name.lower() for k in ("timeout", "connection")):
        return True, name
    return False, name


def judge(message: str, model: str = DEFAULT_MODEL,
          prompt_path: Path = PROMPT_PATH,
          max_tokens: int = MAX_TOKENS,
          effort: str = REASONING_EFFORT,
          retries: int = 2,
          backoff: float = 1.0,
          schema: type[BaseModel] = ClinicalJudgment) -> tuple[BaseModel | None, dict]:
    """Return (judgment | None, telemetry). None means MODEL_FAILURE.

    Retries cover the transport only. A response that arrives and does not
    parse is the designed route and is never retried: a MODEL_FAILURE rate that
    depends on retry luck measures nothing.
    """
    t0 = time.time()
    meta = {"model": model, "model_served": None, "ok": False, "latency_s": None,
            "in_tokens": None, "out_tokens": None, "reasoning_tokens": None,
            "cost_usd": None, "raw": None, "effort": effort,
            "attempts": 0, "retry_causes": [], "finish_reason": None}
    prompt = build_prompt(message, prompt_path)
    client = _get_client()      # a missing key raises here, outside the loop

    resp = None
    for attempt in range(retries + 1):
        meta["attempts"] = attempt + 1
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"reasoning": {"effort": effort}},
            )
            break
        except Exception as e:
            retryable, cause = classify_failure(e)
            meta["retry_causes"].append(cause)
            meta["error"] = f"{type(e).__name__}: {e}"
            if not retryable or attempt == retries:
                meta["latency_s"] = round(time.time() - t0, 2)
                return None, meta
            time.sleep(backoff * (2 ** attempt) + random.uniform(0, 0.25))

    try:
        choice = resp.choices[0]
        raw = choice.message.content
        u = resp.usage
        meta.update(
            latency_s=round(time.time() - t0, 2),
            # OpenRouter exposes no dated slug, so record what was ACTUALLY
            # served. This is your reproducibility evidence -- log it, and if
            # it changes between your eval run and your presentation, you know.
            model_served=getattr(resp, "model", None),
            finish_reason=getattr(choice, "finish_reason", None),
            in_tokens=getattr(u, "prompt_tokens", None),
            out_tokens=getattr(u, "completion_tokens", None),
            reasoning_tokens=getattr(getattr(u, "completion_tokens_details", None),
                                     "reasoning_tokens", None),
            raw=raw,
        )
        if meta["in_tokens"] is not None:
            meta["cost_usd"] = cost_usd(meta["in_tokens"], meta["out_tokens"] or 0)
        data = _extract_json(raw or "")
        if data is None:
            return None, meta
        judgment = schema.model_validate(data)
        meta["ok"] = True
        return judgment, meta
    except Exception as e:
        meta["error"] = f"{type(e).__name__}: {e}"
        meta["latency_s"] = round(time.time() - t0, 2)
        return None, meta


def cache_key(message: str, model: str = DEFAULT_MODEL,
              prompt_path: Path = PROMPT_PATH, max_tokens: int = MAX_TOKENS,
              effort: str = REASONING_EFFORT) -> str:
    """Everything that changes the answer is in the key.

    Keying on the message alone means editing prompts/judgment.txt and
    re-running serves answers produced under the old prompt, with nothing on
    screen to say so.
    """
    h = hashlib.sha256()
    for part in (model, Path(prompt_path).read_text(), message,
                 str(max_tokens), effort):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:20]


def cached_judge(judge_fn=judge, path: Path = CACHE_PATH,
                 model: str = DEFAULT_MODEL, prompt_path: Path = PROMPT_PATH,
                 max_tokens: int = MAX_TOKENS, effort: str = REASONING_EFFORT,
                 schema: type[BaseModel] = ClinicalJudgment, **judge_kw):
    """Wrap a judge so identical calls are served from disk.

    Only successful judgments are stored, so a failure is retried on the next
    run rather than frozen in. The store is flushed after every live call,
    which makes an interrupted run resumable at no cost.

    The returned function carries a `.stats` dict: hits, misses, and any change
    in the served model between a cached entry and a live one.
    """
    store = json.loads(path.read_text()) if path.exists() else {}
    stats = {"hits": 0, "misses": 0, "served_drift": []}

    def _fn(message, **kw):
        key = cache_key(message, model, prompt_path, max_tokens, effort)
        hit = store.get(key)
        if hit:
            stats["hits"] += 1
            j = schema.model_validate(hit["judgment"]) if hit["judgment"] else None
            meta = dict(hit["meta"]); meta["cached"] = True
            return j, meta

        j, meta = judge_fn(message, model=model, prompt_path=prompt_path,
                           max_tokens=max_tokens, effort=effort, schema=schema,
                           **judge_kw, **kw)
        stats["misses"] += 1

        # the slug cannot be pinned, so a cache is the only place a change in
        # the served model becomes visible
        served = meta.get("model_served")
        for prior in {e["meta"].get("model_served") for e in store.values()}:
            if served and prior and served != prior:
                pair = f"{prior} -> {served}"
                if pair not in stats["served_drift"]:
                    stats["served_drift"].append(pair)

        if meta.get("ok"):
            store[key] = {"judgment": j.model_dump(mode="json") if j else None,
                          "meta": {k: v for k, v in meta.items() if k != "raw"}}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(store, indent=2))
        meta = dict(meta); meta["cached"] = False
        return j, meta

    _fn.stats = stats
    return _fn


def screen(message: str, judge_fn=None, **kw) -> tuple[ScreenVerdict | None, dict]:
    """The urgency screen: judge() with its own prompt, budget and schema."""
    kw.setdefault("prompt_path", SCREEN_PROMPT_PATH)
    kw.setdefault("max_tokens", SCREEN_MAX_TOKENS)
    return (judge_fn or judge)(message, schema=ScreenVerdict, **kw)


def cached_screen(screen_fn=judge, path: Path = SCREEN_CACHE_PATH, **kw):
    """cached_judge() set up for the screen. screen_fn takes judge()'s arguments."""
    return cached_judge(screen_fn, path=path, prompt_path=SCREEN_PROMPT_PATH,
                        max_tokens=SCREEN_MAX_TOKENS, schema=ScreenVerdict, **kw)


def judge_stub(message: str, **_) -> tuple[ClinicalJudgment | None, dict]:
    """Model stubbed out. Used by the degraded-mode ablation row."""
    return None, {"model": "STUB", "ok": False, "latency_s": 0.0}
