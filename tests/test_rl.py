"""RL tasks (civitas/rl/): the bank, TaskEpisode's parity with ToolAgent, the counterfactual reward and the endpoint
runner. A small bank (one seed, months 24 and 48, a 60-month game) is built once per session; no model is called."""
import contextlib
import json
import multiprocessing as mp
import threading
from concurrent.futures import ProcessPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from civitas.agents.llm_client import ProviderConfig
from civitas.config import load_config
from civitas.fleet.agent import NUDGE, ToolAgent
from civitas.rl import bank as B
from civitas.rl import check as K
from civitas.rl import evaluate as E
from civitas.rl import reward as R
from civitas.rl.episode import TaskEpisode
from civitas.rl.policies import play, scripted_calls

ROOT = Path(__file__).resolve().parents[1]


def _cfg(path="configs/fleet_smoke_late.toml") -> dict:
    cfg = load_config(ROOT / path)
    cfg["game"].update(months=60, warmup_months=24)
    cfg["name"] = Path(path).stem + "_short"
    return cfg


@pytest.fixture(scope="session")
def bank(tmp_path_factory):
    root = tmp_path_factory.mktemp("tasks")
    B.build("t", [_cfg()], [100], [24, 48], "all", horizon_max=6, noop=True, workers=1, root=root, verbose=False)
    return B.TaskBank("t", root=root)


# ------------------------------------------------------------------------------------------------- seeds and bank
def test_seed_policy_keeps_eval_seeds_out():
    suites = B.eval_suite_seeds()
    assert {0, 4, 5, 11, 12, 16, 18, 20} <= suites | set(B.EVAL_SEEDS) and set(range(8)) <= set(B.EVAL_SEEDS)
    for s in [0, 4, 7, 20, 99]:
        with pytest.raises(B.BankError, match="reserved for evals"):
            B.check_seeds([100, s])
    train, test = set(B.TRAIN_SEEDS), set(B.TEST_SEEDS)
    assert not train & test and not (train | test) & (suites | set(B.EVAL_SEEDS))
    assert min(train | test) >= B.RESERVED_BELOW
    assert B.check_seeds(range(100, 103)) == [100, 101, 102]
    assert (B.split_of(100), B.split_of(299), B.split_of(300), B.split_of(339), B.split_of(340)) == \
        ("train", "train", "test", "test", "extra")


def test_bank_index_and_composition(bank):
    rows = bank.rows
    assert len(rows) == 16 and {r["role"] for r in rows} == set(B.TURN_ORDER)
    r = rows[0]
    assert r["task_id"] == "fleet_smoke_late_short/s100/m24/mayor" and r["split"] == "train" and r["style"] == "guided"
    for r in rows:
        assert len(r["baseline"]["city"]) == 6 and r["baseline"]["months"][0] == r["month"] + 1
        assert len(r["noop"]["city"]) == 6 and r["actionable"] in (True, False) and r["why"]
        assert r["handover"]["population"] > 0 and r["state_fp"]
    c = bank.manifest["composition"]
    assert c["tasks"] == 16 and c["by_split"] == {"train": 16} and "noop_headroom_6" in c
    assert all(t.actionable for t in bank.tasks("train")) and len(bank.tasks("train", actionable=None)) == 16
    assert len(bank.tasks(None, actionable=None, limit=5)) == 5


def test_bank_refuses_a_changed_snapshot_or_city(bank, tmp_path):
    t = bank.tasks(actionable=None)[3]
    row = {**t.row, "state_fp": "0" * 24, "snapshot_sha256": t.row["snapshot_sha256"]}
    with pytest.raises(B.BankError, match="rebuild the bank"):
        TaskEpisode(row=row, cfg=t.cfg, blob=t.blob()).prepare()
    bad = bytearray(t.blob())
    bad[-5] ^= 0xFF
    with pytest.raises(B.BankError, match="SHA-256"):
        B.load_state(bytes(bad), t.row["snapshot_sha256"])


def test_bank_refuses_a_region_or_a_non_fleet_line_up():
    cfg = _cfg()
    cfg["agents"]["police"] = {"kind": "scripted"}
    with pytest.raises(B.BankError, match="fleet agent"):
        B._check_config(cfg)
    cfg = _cfg()
    cfg["game"]["region"] = True
    with pytest.raises(B.BankError, match="single-city"):
        B._check_config(cfg)


# ------------------------------------------------------------------------------------------------------ reward
def test_baseline_scores_zero(bank):
    for t in bank.tasks(actionable=None)[::5]:
        rec = K.play_task(t, "baseline", 6)
        assert rec["series"]["city"] == t.baseline["city"] and rec["series"]["role"] == t.baseline["role"]


@pytest.mark.slow
def test_baseline_scores_zero_and_scripted_about_zero(bank):
    """The scripted fleet agent's own turn on the prepared state reproduces the bank's game exactly; the same decision
    through the model toolbox stays close."""
    for t in bank.tasks(actionable=None):
        rec = K.play_task(t, "baseline", 6)
        r = R.compute(t.row, rec["series"], rec["counts"])
        assert r["city"] == 0 and r["role"] == 0, t.task_id
        assert rec["series"]["city"] == t.baseline["city"] and rec["series"]["role"] == t.baseline["role"]
    near = [R.compute(t.row, K.play_task(t, "scripted", 6)["series"], {"failed": 0, "refused": 0, "invalid": 0,
                                                                       "ended": True})["city"]
            for t in bank.tasks(actionable=None)]
    assert sum(abs(x) < 1e-9 for x in near) >= len(near) // 2


def test_noop_scores_below_scripted_on_average(bank):
    tasks = bank.tasks()
    noop = [R.compute(t.row, K.play_task(t, "noop", 6)["series"], R.counts([], [], True))["reward"] for t in tasks]
    cached = [E.noop_reward(t.row, R.RewardConfig()) for t in tasks]
    assert noop == pytest.approx(cached)                   # the bank's noop branch is the noop episode's
    assert sum(noop) / len(noop) < -0.05                   # scripted (the baseline) scores 0 by construction


def test_reward_components_and_penalties():
    row = {"baseline": {"months": [1, 2, 3], "city": [10.0, 11.0, 12.0], "role": [50.0, 50.0, 50.0],
                        "population": [100, 110, 120]}}
    series = {"months": [1, 2, 3], "city": [10.0, 11.5, 12.5], "role": [50.0, 52.0, 54.0],
              "population": [100, 112, 125], "seconds": [0, 0, 0]}
    n = {"calls": 5, "actions": 2, "failed": 1, "refused": 2, "invalid": 1, "ended": False}
    r = R.compute(row, series, n, R.RewardConfig(horizon=2, w_role=0.5, p_failed=0.1, p_refused=0.05, p_invalid=0.02,
                                                 p_unended=0.2))
    assert r["city"] == pytest.approx(0.5) and r["role"] == pytest.approx(2.0) and r["population"] == 2
    assert r["penalty"] == pytest.approx(0.1 + 0.1 + 0.02 + 0.2)
    assert r["reward"] == pytest.approx(0.5 + 1.0 - 0.42)
    assert R.compute(row, series, n, R.RewardConfig(horizon=12))["horizon"] == 3       # capped at what was played
    trace = [{"result": {"ok": True}}, {"result": {"ok": False, "refused": "repeat"}}, {"result": {"ok": False}},
             {"result": {"ok": False}}]
    assert R.counts([{"ok": True}, {"ok": False}], trace, True) == {"calls": 4, "actions": 1, "failed": 1,
                                                                     "refused": 1, "invalid": 1, "ended": True}


def _degraded(task_id_and_root):
    tid, root = task_id_and_root
    t = B.TaskBank("t", root=root).get(tid)
    rec = K.play_task(t, "degraded", 6)
    return rec["series"]["city"], rec["series"]["role"], rec["counts"]


def test_same_actions_give_the_same_reward_in_another_process(bank):
    t = next(t for t in bank.tasks() if t.role == "planning")
    here = _degraded((t.task_id, str(bank.dir.parent)))
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as ex:
        there = ex.submit(_degraded, (t.task_id, str(bank.dir.parent))).result()
    assert list(here[:2]) == list(there[:2]) and here[2] == there[2]


# ----------------------------------------------------------------------------------------------------- episode
class ScriptClient:
    """A ChatClient stand-in replying from a list (no network)."""

    def __init__(self, replies):
        self.replies, self.n = list(replies), 0

    def chat(self, model, messages, *, temperature=0.4, max_tokens=1500, extra=None):
        self.n += 1
        content, calls = self.replies.pop(0) if self.replies else ("done", [])
        return {"content": content, "reasoning": "", "tool_calls": calls, "finish_reason":
                "tool_calls" if calls else "stop", "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                "latency_s": 0.0}


def _call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _replies(task):
    """A plain reply (the nudge), the scripted decision's calls two per reply, a bad call, end_turn twice (decided on
    an episode of its own)."""
    ep = TaskEpisode(task)
    ep.reset()
    calls, report = scripted_calls(ep)
    out = [("Let me look first.", [])]
    cs = [_call(f"c{i}", n, a) for i, (n, a) in enumerate(calls)] + [_call("bad", "no_such_tool", {})]
    out += [("", cs[i:i + 2]) for i in range(0, len(cs), 2)]
    return out + [("", [_call("e1", "end_turn", {"report": report})]), ("", [_call("e2", "end_turn", {"report": "x"})])]


def test_episode_matches_a_toolagent_turn(bank):
    for t in [x for x in bank.tasks() if x.role in ("mayor", "planning", "police")]:
        replies = _replies(t)
        ep = TaskEpisode(t)
        obs = ep.reset()
        for content, calls in replies:
            if ep.done:
                break
            ep.reply(content, calls)
        assert ep.done and ep.reason == "end_turn"
        mine = ep.finish(horizon=6)

        ep2 = TaskEpisode(t)
        env = ep2.prepare()
        a = ToolAgent(t.role, cfg=ep2.fleet_cfg, provider=ProviderConfig(base_url="http://127.0.0.1:9", api_key_env=""))
        a.bind(env)
        a.client = ScriptClient(replies)
        d = a.act(env.observe(t.role))
        assert a.system == obs["system_prompt"] == ep.messages[0]["content"]
        assert d.meta["messages"][1:] == ep.messages[1:], t.task_id
        assert ep.messages[2] == {"role": "assistant", "content": "Let me look first."} and \
            ep.messages[3] == {"role": "user", "content": NUDGE}
        assert [x["tool"] for x in d.meta["tool_trace"]] == [x["tool"] for x in ep.trace]
        theirs = ep2.finish(decision=d, horizon=6)
        assert {k: theirs[k] for k in ("reward", "city", "role", "penalty")} == \
            {k: mine[k] for k in ("reward", "city", "role", "penalty")}
        assert theirs["series"]["city"] == mine["series"]["city"]


def _usage(messages) -> dict:
    """A provider's usage, deterministic: the request's characters / 4 and 300 completion tokens a reply."""
    return {"prompt_tokens": len(json.dumps(messages)) // 4, "completion_tokens": 300}


class UsageScriptClient(ScriptClient):
    def chat(self, model, messages, **kw):
        return {**super().chat(model, messages, **kw), "usage": _usage(messages)}


def test_episode_matches_a_toolagent_turn_with_the_token_budget(bank):
    """civitas-v3's turn_budget (fleet/budget.py): the same messages (the rule, the briefing line, every "budget
    left" line), the same cut and report, the same reward, whether the budget cuts the turn or not."""
    from civitas.fleet.budget import CUT_REPORT
    for t in [x for x in bank.tasks() if x.role in ("mayor", "planning")]:
        replies = _replies(t)
        first = _usage(TaskEpisode(t).reset()["messages"])["prompt_tokens"]
        assert TaskEpisode(t).reset().get("turn_budget") is None                  # off unless asked (v2 banks)
        for limit, cut in ((4000 + 4 * (first + 300), True), (10 ** 6, False)):
            ep = TaskEpisode(t, turn_budget=limit)
            obs = ep.reset()
            assert obs["turn_budget"] == limit and f"{limit:,} tokens this turn" in obs["briefing"]
            for content, calls in replies:
                if ep.done:
                    break
                ep.reply(content, calls, usage=_usage(ep.messages))
            assert ep.done and ep.reason == ("turn_budget" if cut else "end_turn"), (t.task_id, limit)
            mine = ep.finish(horizon=6)

            ep2 = TaskEpisode(t, turn_budget=limit)
            env = ep2.prepare()
            a = ToolAgent(t.role, cfg=ep2.fleet_cfg,
                          provider=ProviderConfig(base_url="http://127.0.0.1:9", api_key_env=""))
            a.bind(env)
            a.client = UsageScriptClient(replies)
            d = a.act(env.observe(t.role))
            assert a.system == obs["system_prompt"] and f"{limit:,} tokens per turn" in a.system
            assert d.meta["messages"][1:] == ep.messages[1:], t.task_id
            assert d.meta["finish_reason"] == ep.reason and d.meta["budget_used"] == ep.budget_used
            assert d.meta["budget_cut"] == cut and (d.thinking == CUT_REPORT) == cut
            assert [x["tokens_left"] for x in d.meta["tool_trace"]] == [x["tokens_left"] for x in ep.trace]
            assert all("budget left: ~" in m["content"] for m in ep.messages if m["role"] == "tool"
                       and '"turn ended"' not in m["content"])
            theirs = ep2.finish(decision=d, horizon=6)
            assert {k: theirs[k] for k in ("reward", "city", "role", "penalty")} == \
                {k: mine[k] for k in ("reward", "city", "role", "penalty")}


def test_replay_of_a_conversation_checks_every_tool_message(bank):
    t = next(x for x in bank.tasks() if x.role == "planning")
    replies = _replies(t)
    ep = TaskEpisode(t)
    ep.reset()
    for content, calls in replies:
        if not ep.done:
            ep.reply(content, calls)
    convo = ep.messages
    ep2 = TaskEpisode(t)
    out = ep2.replay(convo, check=True)
    assert out and all(r["match"] for r in out) and ep2.done and ep2.messages == convo
    tampered = [dict(m) for m in convo]
    i = next(i for i, m in enumerate(tampered) if m["role"] == "tool")
    tampered[i]["content"] = '{"ok":true}'
    assert not all(r["match"] for r in TaskEpisode(t).replay(tampered, check=True))
    with pytest.raises(ValueError, match="briefing|user"):
        TaskEpisode(t).replay([convo[0], {"role": "user", "content": "another city"}], check=True)


def test_step_limits_mirror_toolagent(bank):
    t = next(x for x in bank.tasks() if x.role == "police")
    ep = TaskEpisode(t)
    ep.reset()
    for i in range(ep.max_steps):
        assert not ep.done
        res = json.loads(ep.step("find_sites", {"purpose": "station"} if i % 2 else {}))
    assert ep.done and ep.reason == "max_steps" and len(ep.trace) == ep.max_steps
    assert "steps_left" in res and res["steps_left"] == 0
    assert "tool-call limit" in json.loads(ep.step("end_turn", {"report": "late"}))["error"]
    r = ep.finish(horizon=6)
    assert r["counts"]["ended"] is False and r["penalty"] >= R.RewardConfig().p_unended
    ep = TaskEpisode(t)
    ep.reset()
    ep.reply("thinking", [])
    ep.reply("still thinking", [])
    assert ep.done and ep.reason == "no_tool_calls"


def test_policies_play_every_task(bank):
    for t in bank.tasks()[:6]:
        for p in ("noop", "degraded", "random", "oracle"):
            ep = TaskEpisode(t)
            ep.reset()
            play(ep, p)
            assert ep.done


# ------------------------------------------------------------------------------------------------------ runner
class FakeModel(BaseHTTPRequestHandler):
    """An OpenAI-compatible chat server: find_sites (planning) or city_overview first, then end_turn."""
    requests: list[dict] = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeModel.requests.append(body)
        step = sum(m["role"] == "assistant" for m in body["messages"])
        names = {t["function"]["name"] for t in body.get("tools") or []}
        first = ("find_sites", {"purpose": "zone_R"}) if "zone" in names else ("city_overview", {})
        name, args = first if step == 0 else ("end_turn", {"report": "looked around"})
        msg = {"role": "assistant", "content": "", "tool_calls": [_call(f"k{step}", name, args)]}
        out = json.dumps({"choices": [{"message": msg, "finish_reason": "tool_calls"}],
                          "usage": {"prompt_tokens": 100, "completion_tokens": 7}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@contextlib.contextmanager
def _server():
    FakeModel.requests = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeModel)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield srv
    finally:
        srv.shutdown()


def test_eval_runner_against_a_fake_server(bank, tmp_path):
    with _server() as srv:
        s = E.run("t", root=bank.dir.parent, base_url=f"http://127.0.0.1:{srv.server_port}/v1", model="fake-nano",
                  split="train", limit=4, n=2, stream=False, workers=2, out_dir=tmp_path, verbose=False)
    d = Path(s["dir"])
    rows = [json.loads(x) for x in (d / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 8 and all(r["reward"] is not None and not r["error"] for r in rows), rows
    assert all(r["counts"]["ended"] and r["counts"]["calls"] >= 2 and r["usage"]["completion_tokens"] == 7 *
               r["counts"]["calls"] for r in rows)        # the mayor's / planning's first end_turn may be refused
    assert all(r["counts"]["calls"] == 2 for r in rows if r["role"] not in ("mayor", "planning"))
    assert all(b["model"] == "fake-nano" and b["tools"] for b in FakeModel.requests)
    # look-ups then end_turn change nothing: the noop turn's reward, task by task (end_turn reminders aside)
    by = {t.task_id: t.row for t in bank.tasks("train")}
    for r in rows:
        if r["role"] not in ("mayor", "planning"):
            assert r["reward"] == pytest.approx(E.noop_reward(by[r["task_id"]], R.RewardConfig()))
    assert s["reward"]["n"] == 4 and s["tokens"]["completion"] == sum(7 * r["counts"]["calls"] for r in rows)
    assert s["within_task_std"] == 0
    assert (d / "summary.md").read_text().startswith("# RL eval: fake-nano")
    tr = [json.loads(x) for x in (d / "transcripts.jsonl").read_text().splitlines()]
    assert len(tr) == 8 and tr[0]["messages"][1]["role"] == "user"


@pytest.mark.slow
def test_check_report(bank):
    s = K.run("t", root=bank.dir.parent, n=3, policies=("baseline", "noop", "random"), horizons=(3, 6),
              determinism=1, workers=2, verbose=False)
    assert s["n_tasks"] == 3 and s["determinism"]["identical"] == s["determinism"]["replayed"] == 1
    assert s["by_policy"]["baseline"][6]["mean"] <= 0 and s["by_policy"]["random"][6]["penalty"] > 0
    assert 1 <= s["within_task"][6]["differ"] <= 3 and s["timing"]["noop"][6]["mean"] > 0
    assert sorted(bank.dir.glob("check_*.md"))


def test_cli_rl_show_and_info(bank, capsys):
    from civitas import cli
    t = bank.tasks()[1]
    cli.main(["rl", "show", "t", t.task_id, "--root", str(bank.dir.parent)])
    out = capsys.readouterr().out
    assert "USER (briefing)" in out and "end_turn" in out
    cli.main(["rl", "bank", "info", "t", "--root", str(bank.dir.parent)])
    assert "tasks 16" in capsys.readouterr().out
