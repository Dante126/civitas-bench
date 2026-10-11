"""Environment versions (bench/envs.py): the probe is deterministic (twice, and in a fresh process with another hash
seed), a balance, briefing or model-only change moves the part it touches while model keys move nothing, the canned
model is never a real client, one process per seed, the cache, freeze / check / list / diff, and tagging that never
fails a freeze. A tiny suite (10 months, 6 of them warm-up) keeps it fast; no model is
called and no tag is written to the repository."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from civitas.bench import envs as V
from civitas.bench import evals as E

ROOT = Path(__file__).resolve().parents[1]
SETS = ["game.months=10", "game.warmup_months=6"]


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    d = tmp_path_factory.mktemp("envs")
    p = d / "tiny.toml"
    p.write_text(f'name = "tiny"\n[eval]\nenv = "tiny-v1"\nbaseline_model = "base"\n[[runs]]\n'
                 f'config = "{ROOT}/configs/fleet_smoke_late.toml"\nseeds = [0, 1]\n')
    return p


@pytest.fixture(scope="module")
def probed(suite):
    return V.probe(suite, workers=0, cache=False, sets=SETS)


def test_the_probe_is_deterministic(suite, probed):
    p = probed
    assert len(p["fingerprint"]) == 16 and set(p["parts"]) == {"config", "prompts", "tools", "seeds"}
    key = f"{ROOT}/configs/fleet_smoke_late.toml:0"
    assert set(p["parts"]["seeds"]) == {key, key[:-1] + "1"} and set(p["parts"]["seeds"][key]) == set(V.SEED_PARTS)
    assert p["seeds"][key]["turns"] == 16 and p["seeds"][key]["months"] == 10   # 8 roles at month 6 and 9
    assert p["seeds"][key]["model_turns"] == 8 and len(p["parts"]["seeds"][key]["model"]) == 16  # one ToolAgent turn each
    assert p["code"]["commit"] and not p["cached"]
    again = V.probe(suite, workers=0, cache=False, sets=SETS)
    assert again["parts"] == p["parts"] and again["fingerprint"] == p["fingerprint"]
    code = (f"from civitas.bench import envs as V; print(V.probe({str(suite)!r}, workers=1, cache=False, "
            f"sets={SETS!r})['fingerprint'])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                         env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONHASHSEED": "12345"}, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == p["fingerprint"]          # a fresh process and a process pool


def test_model_keys_move_nothing(suite, probed):
    p = V.probe(suite, workers=0, cache=False, sets=[*SETS, "fleet.model=other", "fleet.temperature=0.9",
                                                    "fleet.max_tokens=99", 'fleet.extra_body={"top_p":0.5}',
                                                    "providers.arcee.timeout=5", "agents.police.model=x"])
    assert p["fingerprint"] == probed["fingerprint"]


def test_a_balance_override_moves_the_metrics(suite, probed):
    p = V.probe(suite, workers=2, cache=False, sets=[*SETS, "game.balance.TAXABLE_PER_RESIDENT=0"])
    d = V.diff(probed, p)
    assert "config" in d and "metrics" in d and "tools" not in d, d


def test_a_briefing_change_moves_the_briefings(suite, probed, monkeypatch):
    from civitas.fleet import scripted
    real = scripted.briefing
    monkeypatch.setattr(scripted, "briefing", lambda *a, **k: real(*a, **k) + "\n(reworded)")
    p = V.probe(suite, workers=0, cache=False, sets=SETS)
    assert V.diff(probed, p) == ["briefings"]


def test_a_model_only_change_moves_the_model_part(suite, probed, monkeypatch):
    """What only a model-backed agent meets (ToolAgent's nudges, the model's ToolBox flags) moves `model` alone."""
    from civitas.fleet import agent
    from civitas.fleet.tools import ToolBox
    monkeypatch.setattr(agent, "NUDGE", agent.NUDGE + " (reworded)")
    assert V.diff(probed, V.probe(suite, workers=0, cache=False, sets=SETS)) == ["model"]
    monkeypatch.undo()
    real = ToolBox._next_refresh
    monkeypatch.setattr(ToolBox, "_next_refresh", lambda self, used: real(self, used) + " (reworded)")
    assert V.diff(probed, V.probe(suite, workers=0, cache=False, sets=SETS)) == ["model"]


def test_the_token_budget_moves_the_prompts_and_the_model_part(suite, probed, monkeypatch):
    """civitas-v3's [fleet] turn_budget: the rule in the system prompt and what the canned turn meets (its usage
    counted, "budget left" in every result, the cut) move; the scripted game doesn't. Budget 0 moves only the config,
    and the canned model's usage is what lets the budget bind (a 60k budget cuts some canned turns, not all)."""
    assert V.diff(probed, V.probe(suite, workers=0, cache=False, sets=[*SETS, "fleet.turn_budget=0"])) == ["config"]
    cuts = []
    from civitas.fleet import agent
    real = agent.ToolAgent.act
    monkeypatch.setattr(agent.ToolAgent, "act", lambda self, obs: cuts.append(real(self, obs)) or cuts[-1])
    p = V.probe(suite, workers=0, cache=False, sets=[*SETS, "fleet.turn_budget=60000"])
    assert V.diff(probed, p) == ["config", "prompts", "model"]
    turns = [d.meta for d in cuts if d.meta.get("messages")]
    assert turns and all(m["turn_budget"] == 60000 and m["budget_used"] > 0 for m in turns)
    assert any(m["budget_cut"] for m in turns) and any(m["finish_reason"] == "end_turn" for m in turns)
    out = V._usage(V._reply("", "length", reasoning="x"), [{"role": "user", "content": "y" * 400}], [], 16000)
    assert out["usage"] == {"prompt_tokens": (len(V.canonical([{"role": "user", "content": "y" * 400}])) + 2) // 4,
                            "completion_tokens": 16000}


def test_the_canned_model_is_never_a_real_client(monkeypatch):
    import hashlib
    from civitas.agents.llm_client import ChatClient, LLMError
    from civitas.fleet.agent import ToolAgent
    monkeypatch.delenv(V.NO_MODEL["api_key_env"], raising=False)
    a = ToolAgent("police", provider=V.no_provider())
    with pytest.raises(LLMError, match="is not set"):                 # refused before any connection
        a.client.chat("m", [{"role": "user", "content": "hi"}])
    canned = V.CannedClient(a, list, hashlib.sha256())
    with pytest.raises(RuntimeError, match="isn't the agent's client"):  # not installed as the client: refuses
        canned.chat("m", [])
    a.client = canned
    assert not isinstance(canned, ChatClient)
    assert canned.chat("m", [])["tool_calls"] == [] and canned.chat("m", [])["finish_reason"] == "length"
    assert canned.chat("m", [])["content"] == V.CANNED_DONE           # an empty script: the turn ends


def test_one_process_per_seed():
    pids = V._isolated_map(V._pid, list(range(4)), 2)
    assert len(set(pids)) == 4 and os.getpid() not in pids


def test_code_state():
    st = V.code_state()
    info = V.code_info()
    assert isinstance(st, str) and st == V.state_of(info) == f"{info['commit']}:{info['diff_hash']}"
    assert V.state_of(None) == V.state_of({"commit": None}) == "nogit"


def test_the_cache(suite, tmp_path):
    a = V.probe(suite, workers=0, cache_dir=tmp_path, sets=SETS)
    files = list(tmp_path.glob("probe_*.json"))
    assert not a["cached"] and len(files) == 1
    b = V.probe(suite, workers=0, cache_dir=tmp_path, sets=SETS)
    assert b["cached"] and b["fingerprint"] == a["fingerprint"]
    c = V.probe(suite, workers=0, cache_dir=tmp_path, sets=[*SETS, "game.act_every=2"])     # other overrides
    assert not c["cached"] and c["fingerprint"] != a["fingerprint"] and len(list(tmp_path.glob("probe_*"))) == 2


def test_freeze_check_list_and_diff(suite, probed, tmp_path, monkeypatch):
    monkeypatch.setattr(V, "ENVS_DIR", tmp_path / "envs")
    assert V.list_envs() == [] and V.load_env("tiny-v1") is None
    s = E.load_suite(suite)
    assert s.env == "tiny-v1" and s.baseline_model == "base"
    c = V.check(s, probe_result=probed)
    assert c["env"] == "tiny-v1" and not c["frozen"] and c["match"] is None and c["fingerprint"] == probed["fingerprint"]
    rec = V.freeze("tiny-v1", s, tag=False, probe_result=probed)
    assert rec["fingerprint"] == probed["fingerprint"] and rec["untagged"] and "tag" not in rec
    assert json.loads((tmp_path / "envs/tiny-v1.json").read_text())["parts"] == probed["parts"]
    assert [e["name"] for e in V.list_envs()] == ["tiny-v1"] and V.load_env("tiny-v1")["suite"] == str(suite)
    c = V.check(s, probe_result=probed)
    assert c["frozen"] and c["match"] is True and c["diff"] == [] and "matches" in V.check_text(c)
    with pytest.raises(V.EnvError, match="exists already"):
        V.freeze("tiny-v1", s, tag=False, probe_result=probed)
    for bad in ("Civitas", "-v1", "a b", "", "../x"):
        with pytest.raises(V.EnvError, match="bad environment name"):
            V.freeze(bad, s, tag=False, probe_result=probed)
    changed = json.loads(json.dumps(probed))
    k = next(iter(changed["parts"]["seeds"]))
    changed["parts"]["seeds"][k]["metrics"] = "x"
    changed["parts"]["tools"] = "y"
    changed["fingerprint"] = "z"
    c = V.check(s, probe_result=changed)
    assert c["match"] is False and c["diff"] == ["tools", "metrics"] and "CHANGED (tools, metrics)" in V.check_text(c)
    del changed["parts"]["seeds"][k]
    assert V.diff(probed, changed) == ["tools", "seeds"]


def test_tagging_never_fails_a_freeze(probed, monkeypatch):
    calls = []

    class R:
        def __init__(self, rc, out=""):
            self.returncode, self.stdout, self.stderr = rc, out, ""

    def git(*args, **kw):
        calls.append(args)
        return R(1) if args[:2] == ("rev-parse", "-q") else R(0)

    monkeypatch.setattr(V, "_git", git)
    head = {"commit": probed["code"]["commit"], "dirty": False}
    clean = {**probed, "code": {**probed["code"], "dirty": False}}
    monkeypatch.setattr(E, "code_version", lambda *a, **k: {**head, "dirty": True})
    assert "uncommitted" in V._tag("t", clean)["untagged"] and not calls
    monkeypatch.setattr(E, "code_version", lambda *a, **k: head)
    assert V._tag("t", clean) == {"tag": "env/t"} and calls[-1] == ("tag", "env/t")
    assert "uncommitted" in V._tag("t", {**probed, "code": {**probed["code"], "dirty": True}})["untagged"]
    assert "HEAD is" in V._tag("t", {**clean, "code": {**clean["code"], "commit": "other"}})["untagged"]
    monkeypatch.setattr(V, "_git", lambda *a, **k: R(0))                 # the tag exists already
    assert "exists already" in V._tag("t", clean)["untagged"]
    monkeypatch.setattr(E, "code_version", lambda *a, **k: None)
    assert V._tag("t", clean) == {"untagged": "not a git checkout"}


def test_the_shipped_version_and_suite():
    s = E.load_suite(ROOT / E.DEFAULT_SUITE)
    assert s.env == "civitas-v2" and s.baseline_model == "trinity-large-thinking"
    assert E.load_suite(ROOT / "configs/eval_open.toml").env == "civitas-v2-open"
    v3, v3o = E.load_suite(ROOT / "configs/eval_v3.toml"), E.load_suite(ROOT / "configs/eval_v3_open.toml")
    assert (v3.env, v3o.env) == ("civitas-v3", "civitas-v3-open")          # v2 + the token budget (fleet/budget.py)
    assert v3.baseline_model == v3o.baseline_model == "trinity-large-thinking"
    assert [e.seeds for e in v3.entries] == [e.seeds for e in v3o.entries] == [e.seeds for e in s.entries]
    assert E.load_suite(ROOT / "configs/eval_singles.toml").env == "civitas-v1"      # v1's frozen suite stays
    assert E.load_suite(ROOT / "configs/eval_late.toml").env is None
    rec = V.load_env("civitas-v1")
    if rec is not None:                                    # frozen with the CLI at the end of the change
        assert len(rec["fingerprint"]) == 16 and set(rec["parts"]["seeds"]) == {
            f"configs/fleet_smoke_late.toml:{i}" for i in range(8)}
