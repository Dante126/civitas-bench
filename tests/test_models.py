"""The model registry (bench/models.py): round-trips, validation, API keys in ./.env, run overrides (a registry
provider reaches the runs) and the preflight against a fake chat server. No test calls a real model."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from civitas.agents.llm_client import ChatClient, ProviderConfig
from civitas.bench import models as M
from civitas.config import DEFAULT_PROVIDERS, apply_cli_overrides, load_config, provider_config
from civitas.fleet.config import fleet_agent_config

ROOT = Path(__file__).resolve().parents[1]
KEY = "CIVITAS_FAKE_PREFLIGHT_API_KEY"


def test_the_shipped_registry():
    reg = M.load_registry()
    assert {"trinity-large-thinking", "trinity-mini"} <= set(reg)
    late = load_config(ROOT / "configs/fleet_smoke_late.toml")["fleet"]
    for e in reg.values():
        assert e.provider == "arcee" and e.temperature == late["temperature"] and e.max_tokens == late["max_tokens"]
        assert M.validate(e.key, {k: v for k, v in e.to_dict().items() if k != "key"}) == []
    text = M.REGISTRY.read_text()
    assert text.startswith("# Model registry") and text == M.render(reg, M.registry_providers())   # canonical
    assert M.is_registry(load_config(M.REGISTRY))


def test_round_trip_and_canonical_rewrite(tmp_path):
    p = tmp_path / "models.toml"
    assert M.load_registry(p) == {} and M.registry_providers(p) == {}
    p.write_text('[providers.together]\nbase_url = "https://api.together.xyz/v1"\napi_key_env = "TOGETHER_API_KEY"\n')
    e = M.save_entry("gpt-5.1", {"provider": "openai", "id": "gpt-5.1-2025", "temperature": 1, "max_tokens": 9000,
                                 "extra_body": {"reasoning": {"effort": "low"}, "top_p": 0.9},
                                 "notes": 'says "hi"\nand more'}, p)
    assert e.served_id == "gpt-5.1-2025" and e.temperature == 1.0
    M.save_entry("aaa", {"provider": "together", "id": "", "temperature": None, "extra_body": {}}, p)   # unset fields
    reg = M.load_registry(p)
    assert list(reg) == ["aaa", "gpt-5.1"] and reg["gpt-5.1"] == e
    assert reg["aaa"].to_dict() == {"key": "aaa", "provider": "together"} and reg["aaa"].served_id == "aaa"
    text = p.read_text()
    assert text.startswith(M.HEADER) and '[models."gpt-5.1"]' in text and "[models.aaa]" in text
    assert text.index("[providers.together]") < text.index("[models.aaa]") < text.index('[models."gpt-5.1"]')
    assert 'extra_body = {reasoning = {effort = "low"}, top_p = 0.9}' in text
    assert M.registry_providers(p)["together"]["api_key_env"] == "TOGETHER_API_KEY"   # kept on rewrite
    M.save_entry("aaa", {"provider": "arcee", "max_tokens": 100}, p)                  # replaces the whole entry
    assert M.load_registry(p)["aaa"].to_dict() == {"key": "aaa", "provider": "arcee", "max_tokens": 100}
    assert M.delete_entry("aaa", p) and not M.delete_entry("aaa", p) and list(M.load_registry(p)) == ["gpt-5.1"]


def test_validation(tmp_path):
    p = tmp_path / "models.toml"
    p.write_text('[providers.together]\nbase_url = "https://x/v1"\napi_key_env = "TOGETHER_API_KEY"\n')
    assert M.validate("m", {"provider": "together"}, p) == []
    assert M.validate("m", {"provider": "openai"}, p) == [] and M.validate("m", {"provider": "anthropic"}, p) == []
    errs = M.validate("bad key", {"provider": "nowhere", "max_steps": 30, "turn_tokens": 1, "colour": "red",
                                  "temperature": "hot", "max_tokens": 1.5, "extra_body": [1], "id": "a b"}, p)
    text = "\n".join(errs)
    for want in ("bad model key", "unknown provider 'nowhere'", "'max_steps' is an environment setting",
                 "'turn_tokens' is an environment setting", "unknown key 'colour'", "temperature must be",
                 "max_tokens must be", "extra_body must be a table", "id must be"):
        assert want in text, (want, errs)
    for k in M.ENV_KEYS:
        assert any(k in x and "environment" in x for x in M.validate("m", {k: 1}, p))
    assert M.validate("m", {"extra_body": {"tools": []}}, p) and M.validate("m", {"temperature": True}, p)
    with pytest.raises(M.RegistryError) as e:
        M.save_entry("m", {"max_steps": 5}, p)
    assert e.value.errors and not M.load_registry(p)


def test_set_api_key(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# keys\nARCEE_API_KEY=a1\nexport FOO_API_KEY='old'\nOTHER=1\nFOO_API_KEY=dup\n")
    monkeypatch.delenv("FOO_API_KEY", raising=False)
    monkeypatch.delenv("BAR_API_KEY", raising=False)
    M.set_api_key("FOO_API_KEY", "sk-new", env)
    M.set_api_key("BAR_API_KEY", " sk-bar ", env)
    assert env.read_text() == "# keys\nARCEE_API_KEY=a1\nFOO_API_KEY=sk-new\nOTHER=1\nBAR_API_KEY=sk-bar\n"
    assert os.environ["FOO_API_KEY"] == "sk-new" and os.environ["BAR_API_KEY"] == "sk-bar"
    for name, value in (("PATH", "x"), ("foo_api_key", "x"), ("FOO_API_KEY", "a b"), ("FOO_API_KEY", "x\nY=1"),
                        ("FOO_API_KEY", "")):
        with pytest.raises(ValueError):
            M.set_api_key(name, value, env)
    fresh = tmp_path / "new.env"
    M.set_api_key("BAR_API_KEY", "k", fresh)
    assert fresh.read_text() == "BAR_API_KEY=k\n" and (fresh.stat().st_mode & 0o777) == 0o600
    st = M.key_status(M.ModelEntry("x", provider="arcee"))
    assert st["provider"] == "arcee" and st["env"] == "ARCEE_API_KEY"
    assert M.key_status("ollama") == {"provider": "ollama", "env": "", "set": True}


def test_built_in_providers():
    assert DEFAULT_PROVIDERS["openai"]["max_tokens_field"] == "max_completion_tokens"
    assert DEFAULT_PROVIDERS["anthropic"]["api_key_env"] == "ANTHROPIC_API_KEY"
    a = provider_config({"providers": {}}, "anthropic")
    assert a.base_url.rstrip("/") + "/chat/completions" == "https://api.anthropic.com/v1/chat/completions"
    assert provider_config({"providers": {}}, "openai").max_tokens_field == "max_completion_tokens"
    assert provider_config({"providers": {}}, "arcee").max_tokens_field == "max_tokens"


def test_eval_overrides_reach_the_runs(tmp_path):
    p = tmp_path / "models.toml"
    p.write_text('[providers.together]\nbase_url = "https://api.together.xyz/v1"\napi_key_env = "TOGETHER_API_KEY"\n'
                 'max_tokens_field = "max_completion_tokens"\n')
    e = M.save_entry("llama", {"provider": "together", "id": "meta-llama/Llama-3-70b", "temperature": 0.7,
                               "max_tokens": 8000, "extra_body": {"top_p": 0.9}}, p)
    agents, sets = M.eval_overrides(e, path=p)
    assert agents == [] and sets[:2] == ["fleet.model=meta-llama/Llama-3-70b", "fleet.provider=together"]
    assert "fleet.temperature=0.7" in sets and "fleet.max_tokens=8000" in sets and 'fleet.extra_body={"top_p":0.9}' in sets
    assert "providers.together.base_url=https://api.together.xyz/v1" in sets
    cfg = load_config(ROOT / "configs/fleet_smoke_late.toml")
    cfg["agents"]["police"].update(model="other", temperature=0.1)
    agents, sets = M.eval_overrides(e, cfg, path=p)
    assert agents == ["police=fleet:together/meta-llama/Llama-3-70b"] and "agents.police.temperature=0.7" in sets
    out = apply_cli_overrides(cfg, agents, sets)
    for role in ("mayor", "police"):
        fc = fleet_agent_config(out, role)
        assert (fc.provider, fc.model, fc.temperature, fc.max_tokens, fc.extra_body) == \
            ("together", "meta-llama/Llama-3-70b", 0.7, 8000, {"top_p": 0.9})
    pc = provider_config(out, "together")             # the run config never had [providers.together]
    assert pc.base_url == "https://api.together.xyz/v1" and pc.api_key_env == "TOGETHER_API_KEY"
    assert pc.max_tokens_field == "max_completion_tokens"


# ------------------------------------------------------------------ preflight on a fake chat server
class Fake(BaseHTTPRequestHandler):
    mode = "tool"
    bodies: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Fake.bodies.append(body)
        if Fake.mode == "401":
            out = b'{"error": "invalid api key"}'
            self.send_response(401)
        else:
            msg = {"role": "assistant", "content": "Sure, ending the turn."}
            if Fake.mode == "tool":
                msg = {"role": "assistant", "content": "", "tool_calls": [{"id": "c0", "type": "function", "function": {
                    "name": "end_turn", "arguments": json.dumps({"report": "preflight"})}}]}
            out = json.dumps({"choices": [{"message": msg, "finish_reason": "tool_calls" if Fake.mode == "tool"
                                           else "stop"}], "usage": {"prompt_tokens": 300, "completion_tokens": 9}}).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture()
def fake(tmp_path, monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    p = tmp_path / "models.toml"
    p.write_text(f'[providers.fake]\nbase_url = "http://127.0.0.1:{srv.server_address[1]}/v1"\n'
                 f'api_key_env = "{KEY}"\nstream = false\nmax_tokens_field = "max_completion_tokens"\n')
    M.save_entry("fake-model", {"provider": "fake", "id": "served-fake", "extra_body": {"top_p": 0.5}}, p)
    monkeypatch.setenv(KEY, "sk-fake")
    Fake.bodies.clear()
    yield p
    srv.shutdown()


def test_preflight_passes_on_a_tool_call(fake):
    Fake.mode = "tool"
    out = M.preflight(M.load_registry(fake)["fake-model"], fake)
    assert out["ok"] and out["error"] is None and out["served_id"] == "served-fake" and out["provider"] == "fake"
    assert [s["name"] for s in out["steps"]] == ["key", "request", "tool_call"] and all(s["ok"] for s in out["steps"])
    assert "end_turn" in out["steps"][-1]["detail"]
    body = Fake.bodies[-1]
    assert body["model"] == "served-fake" and body["max_completion_tokens"] == M.PREFLIGHT_TOKENS
    assert "max_tokens" not in body and body["top_p"] == 0.5
    assert [t["function"]["name"] for t in body["tools"]] == ["end_turn", "inspect"]
    assert all(t["function"]["parameters"]["type"] == "object" for t in body["tools"])


def test_preflight_fails_without_a_tool_call(fake):
    Fake.mode = "text"
    out = M.preflight(M.load_registry(fake)["fake-model"], fake)
    assert not out["ok"] and out["steps"][-1]["name"] == "tool_call" and not out["steps"][-1]["ok"]
    assert "no tool call" in out["error"] and "ending the turn" in out["error"]


def test_preflight_failures_are_steps_not_exceptions(fake, monkeypatch):
    Fake.mode = "401"
    out = M.preflight(M.load_registry(fake)["fake-model"], fake)
    assert not out["ok"] and out["steps"][-1] == {"name": "request", "ok": False, "detail": out["error"]}
    assert "401" in out["error"] and len(Fake.bodies) == 1        # no retry on an auth error
    monkeypatch.delenv(KEY)
    out = M.preflight(M.load_registry(fake)["fake-model"], fake)
    assert not out["ok"] and out["steps"] == [{"name": "key", "ok": False, "detail": out["error"]}] and KEY in out["error"]
    out = M.preflight(M.ModelEntry("x", provider="nowhere"), fake)
    assert not out["ok"] and out["steps"][0]["name"] == "key" and "unknown provider" in out["error"]
    monkeypatch.setenv(KEY, "sk")
    dead = ProviderConfig(base_url="http://127.0.0.1:9/v1", api_key_env=KEY, stream=False, max_retries=0)
    out = M.preflight(M.ModelEntry("x", provider="fake"), fake, provider=dead)
    assert not out["ok"] and out["steps"][-1]["name"] == "request"


def test_chat_client_token_field(fake):
    Fake.mode = "tool"
    pc = M.provider_of(M.load_registry(fake)["fake-model"], fake)
    ChatClient(ProviderConfig(**{**pc.__dict__, "max_tokens_field": "max_tokens"})).chat("m", [], max_tokens=7)
    assert Fake.bodies[-1]["max_tokens"] == 7 and "max_completion_tokens" not in Fake.bodies[-1]


def test_temperature_default_sends_none(fake):
    """`temperature = "default"`: no temperature key in any request (OpenAI's reasoning models refuse one), through
    the registry, the run overrides, the fleet config, the client and the preflight."""
    Fake.mode = "tool"
    assert M.validate("o", {"provider": "openai", "temperature": "default"}) == []
    assert M.validate("o", {"provider": "openai", "temperature": "warm"})
    e = M.save_entry("reasoner", {"provider": "fake", "id": "o-mini", "temperature": "default"}, fake)
    assert e.temperature == "default" and 'temperature = "default"' in fake.read_text()
    assert M.load_registry(fake)["reasoner"].to_dict()["temperature"] == "default"
    out = M.preflight(M.load_registry(fake)["reasoner"], fake)
    assert out["ok"] and "temperature" not in Fake.bodies[-1]
    out = M.preflight(M.load_registry(fake)["fake-model"], fake)      # unset: the suite's 0.4, as before
    assert out["ok"] and Fake.bodies[-1]["temperature"] == 0.4
    _, sets = M.eval_overrides(e, path=fake)
    assert "fleet.temperature=default" in sets
    cfg = apply_cli_overrides(load_config(ROOT / "configs/fleet_smoke_late.toml"), [], sets)
    assert fleet_agent_config(cfg, "mayor").temperature is None
    with pytest.raises(ValueError, match="temperature"):
        fleet_agent_config(apply_cli_overrides(cfg, [], ["fleet.temperature=warm"]), "mayor")
    pc = M.provider_of(e, fake)
    ChatClient(pc).chat("m", [], temperature=None, max_tokens=5)
    assert "temperature" not in Fake.bodies[-1]
    ChatClient(pc).chat("m", [], temperature=0.3, max_tokens=5)
    assert Fake.bodies[-1]["temperature"] == 0.3


def test_non_bmp_text_round_trips(tmp_path):
    """An emoji in notes or extra_body: JSON's ensure_ascii writes a surrogate pair of escapes, which TOML refuses."""
    p = tmp_path / "models.toml"
    e = M.save_entry("rocket", {"provider": "arcee", "notes": "fast 🚀 «ok»\ttab\x7fdel",
                                "extra_body": {"tag 🚀": "🚀"}}, p)
    text = p.read_text(encoding="utf-8")
    assert "🚀" in text and "\\ud83d" not in text and "\x7f" not in text
    back = M.load_registry(p)["rocket"]
    assert back == e and back.notes == "fast 🚀 «ok»\ttab\x7fdel" and back.extra_body == {"tag 🚀": "🚀"}
