"""Model-layer tests. No network: the client is stubbed throughout."""
import json
import os

import pytest

from src import judgment as J
from src.schema import ClinicalJudgment, PhysicianAction

VALID = json.dumps({"cant_miss": [], "needs_physician_action": {"value": False},
                    "basis": "mild URI"})
J0 = ClinicalJudgment(needs_physician_action=PhysicianAction(value=False))


def _exc(status=None, name="APIError"):
    return type(name, (Exception,), {"status_code": status})()


class _Resp:
    """Minimal stand-in for an OpenAI chat completion."""
    def __init__(self, content, finish="stop", model="anthropic/claude-sonnet-5.5"):
        msg = type("M", (), {"content": content, "refusal": None})()
        self.choices = [type("C", (), {"message": msg, "finish_reason": finish})()]
        self.usage = type("U", (), {"prompt_tokens": 100, "completion_tokens": 50})()
        self.model = model


def _client(responses):
    """Return a fake client that yields `responses` in order; Exceptions raise."""
    seq = list(responses)
    calls = {"n": 0}

    class _Completions:
        def create(self, **kw):
            calls["n"] += 1
            item = seq.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    client = type("Client", (), {"chat": type("Chat", (), {"completions": _Completions()})()})()
    return client, calls


# --------------------------------------------------------------------------
# which failures are worth retrying
# --------------------------------------------------------------------------

@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503])
def test_transport_noise_is_retryable(status):
    assert J.classify_failure(_exc(status))[0] is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_config_errors_fail_fast(status):
    """A wrong key or model slug must not be slept on three times."""
    assert J.classify_failure(_exc(status))[0] is False


def test_connection_and_timeout_are_retryable():
    assert J.classify_failure(_exc(None, "APITimeoutError"))[0] is True
    assert J.classify_failure(_exc(None, "APIConnectionError"))[0] is True


# --------------------------------------------------------------------------
# retry behaviour
# --------------------------------------------------------------------------

def test_judge_retries_transport_then_succeeds(monkeypatch):
    client, calls = _client([_exc(429), _exc(503), _Resp(VALID)])
    monkeypatch.setattr(J, "_get_client", lambda: client)
    j, meta = J.judge("hi", retries=2, backoff=0)
    assert j.needs_physician_action.value is False and j.cant_miss == []
    assert calls["n"] == 3
    assert meta["attempts"] == 3
    assert meta["retry_causes"] == ["http_429", "http_503"]


def test_judge_gives_up_after_the_budget(monkeypatch):
    client, calls = _client([_exc(429), _exc(429), _exc(429)])
    monkeypatch.setattr(J, "_get_client", lambda: client)
    j, meta = J.judge("hi", retries=2, backoff=0)
    assert j is None and meta["ok"] is False
    assert calls["n"] == 3


def test_judge_does_not_retry_a_config_error(monkeypatch):
    client, calls = _client([_exc(401), _Resp(VALID)])
    monkeypatch.setattr(J, "_get_client", lambda: client)
    j, meta = J.judge("hi", retries=2, backoff=0)
    assert j is None
    assert calls["n"] == 1, "a 401 must fail on the first attempt"


def test_judge_does_not_retry_an_unparseable_response(monkeypatch):
    """The designed route. Retrying would make the MODEL_FAILURE rate a
    function of retry luck."""
    client, calls = _client([_Resp("I'm sorry, I can't help with that.")])
    monkeypatch.setattr(J, "_get_client", lambda: client)
    j, meta = J.judge("hi", retries=2, backoff=0)
    assert j is None and calls["n"] == 1


def test_judge_records_finish_reason(monkeypatch):
    client, _ = _client([_Resp(VALID, finish="length")])
    monkeypatch.setattr(J, "_get_client", lambda: client)
    _, meta = J.judge("hi", retries=0, backoff=0)
    assert meta["finish_reason"] == "length"


# --------------------------------------------------------------------------
# the cache key is the whole game
# --------------------------------------------------------------------------

def test_extract_json_ignores_text_after_the_object():
    assert J._extract_json('{"a": 1}\nnote: {not json}') == {"a": 1}
    assert J._extract_json('preamble {"a": {"b": 2}} trailing }') == {"a": {"b": 2}}
    assert J._extract_json("no json here") is None


def test_cache_key_changes_with_the_prompt(tmp_path):
    a = tmp_path / "a.txt"; a.write_text("route this: {{MESSAGE}}")
    b = tmp_path / "b.txt"; b.write_text("route this differently: {{MESSAGE}}")
    assert J.cache_key("msg", prompt_path=a) != J.cache_key("msg", prompt_path=b)


def test_cache_key_changes_with_model_and_tokens(tmp_path):
    p = tmp_path / "p.txt"; p.write_text("{{MESSAGE}}")
    base = J.cache_key("msg", model="m1", prompt_path=p, max_tokens=900)
    assert base != J.cache_key("msg", model="m2", prompt_path=p, max_tokens=900)
    assert base != J.cache_key("msg", model="m1", prompt_path=p, max_tokens=500)


def test_cache_key_changes_with_reasoning_effort(tmp_path):
    p = tmp_path / "p.txt"; p.write_text("{{MESSAGE}}")
    assert (J.cache_key("msg", prompt_path=p, effort="low")
            != J.cache_key("msg", prompt_path=p, effort="high"))


def test_judge_sends_effort_and_no_temperature(monkeypatch):
    """The served model rejects non-default sampling; effort is the control."""
    seen = {}
    client, _ = _client([_Resp(VALID)])
    orig = client.chat.completions.create

    def spy(**kw):
        seen.update(kw)
        return orig(**kw)

    client.chat.completions.create = spy
    monkeypatch.setattr(J, "_get_client", lambda: client)
    J.judge("msg", effort="low")
    assert "temperature" not in seen
    assert seen["extra_body"] == {"reasoning": {"effort": "low"}}


def test_cache_serves_a_hit_without_calling_through(tmp_path):
    p = tmp_path / "p.txt"; p.write_text("{{MESSAGE}}")
    calls = {"n": 0}

    def fake(message, **kw):
        calls["n"] += 1
        return J0, {"ok": True}

    fn = J.cached_judge(fake, path=tmp_path / "c.json", prompt_path=p)
    fn("hello"); fn("hello")
    assert calls["n"] == 1
    assert fn.stats == {"hits": 1, "misses": 1, "served_drift": []}


def test_failures_are_not_cached(tmp_path):
    """A failure must be retried on the next run, not frozen in."""
    p = tmp_path / "p.txt"; p.write_text("{{MESSAGE}}")
    calls = {"n": 0}

    def fake(message, **kw):
        calls["n"] += 1
        return None, {"ok": False, "error": "boom"}

    fn = J.cached_judge(fake, path=tmp_path / "c.json", prompt_path=p)
    fn("hello"); fn("hello")
    assert calls["n"] == 2


def test_cache_survives_a_restart(tmp_path):
    p = tmp_path / "p.txt"; p.write_text("{{MESSAGE}}")
    store = tmp_path / "c.json"
    calls = {"n": 0}

    def fake(message, **kw):
        calls["n"] += 1
        return J0, {"ok": True}

    J.cached_judge(fake, path=store, prompt_path=p)("hello")
    J.cached_judge(fake, path=store, prompt_path=p)("hello")   # fresh wrapper, same file
    assert calls["n"] == 1


def test_served_model_drift_is_reported(tmp_path):
    p = tmp_path / "p.txt"; p.write_text("{{MESSAGE}}")
    served = iter(["model-a", "model-b"])

    def fake(message, **kw):
        return (J0,
                {"ok": True, "model_served": next(served)})

    fn = J.cached_judge(fake, path=tmp_path / "c.json", prompt_path=p)
    fn("one"); fn("two")
    assert fn.stats["served_drift"] == ["model-a -> model-b"]


# --------------------------------------------------------------------------
# a missing key is an error, not a deferral
# --------------------------------------------------------------------------

def test_missing_key_aborts_instead_of_landing_as_model_failure(monkeypatch):
    """Once, a sweep with no key in the shell filed every cache miss as
    MODEL_FAILURE (retry_causes: ['KeyError']) and wrote seven plausible
    result files. The client is built outside the retry loop now, so the
    error propagates."""
    def no_key():
        raise RuntimeError("no API key")
    monkeypatch.setattr(J, "_get_client", no_key)
    with pytest.raises(RuntimeError):
        J.judge("hi", retries=2, backoff=0)


def test_require_key_names_the_fix(monkeypatch):
    from src import env as E
    for var in E.KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError, match=r"\.env"):
        E.require_key()
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert E.require_key() == "sk-or-test"


def test_load_env_does_not_clobber_the_shell(tmp_path, monkeypatch):
    from src import env as E
    f = tmp_path / ".env"
    f.write_text("# comment\nOPENROUTER_API_KEY='from-file'\nDISPO_MODEL=m-file\n\nnot a pair\n")
    monkeypatch.setenv("DISPO_MODEL", "m-shell")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    E.load_env(f)
    assert E.api_key() == "from-file"            # quotes stripped
    assert os.environ["DISPO_MODEL"] == "m-shell"  # exported value wins
    E.load_env(tmp_path / "absent")              # missing file is not an error


def test_truncation_is_reported_as_length_not_as_a_cause(monkeypatch):
    """finish_reason=length with empty content is what a too-small budget does
    to a model that thinks inside max_tokens. It must land as MODEL_FAILURE
    (policy) with the finish reason on the telemetry and no retry cause, so it
    reads as a budget."""
    client, calls = _client([_Resp(None, finish="length")])
    monkeypatch.setattr(J, "_get_client", lambda: client)
    j, meta = J.judge("hi", retries=2, backoff=0)
    assert j is None and meta["ok"] is False
    assert meta["finish_reason"] == "length"
    assert meta["retry_causes"] == [] and calls["n"] == 1
