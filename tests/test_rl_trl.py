"""TRL GRPO integration (civitas/rl/trl_env.py, train.py): the env classes driven by a stand-in for TRL's tool loop
(no trl needed), append-only messages, the reward and its logging; the chat-template and GRPO smoke tests run only
where transformers / trl are installed (the training venv, README "RL training")."""
import copy
import importlib.util
import inspect
import json
import os
from pathlib import Path

import pytest

from civitas.config import load_config
from civitas.rl import bank as B
from civitas.rl import reward as R
from civitas.rl import train as TR
from civitas.rl import trl_env as T
from civitas.rl.episode import TaskEpisode
from civitas.rl.policies import scripted_calls

ROOT = Path(__file__).resolve().parents[1]
HAS = lambda m: importlib.util.find_spec(m) is not None


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    cfg = load_config(ROOT / "configs/fleet_smoke_late.toml")
    cfg["game"].update(months=60, warmup_months=24)
    cfg["name"] = "fleet_smoke_late_short"
    root = tmp_path_factory.mktemp("tasks")
    B.build("t", [cfg], [100], [24, 48], "all", horizon_max=6, noop=False, workers=1, root=root, verbose=False)
    return B.TaskBank("t", root=root)


@pytest.fixture(scope="module")
def tasks(bank):
    return T.task_set(bank, "train", roles=["mayor", "planning", "police"])


class FakeTRL:
    """What GRPOTrainer does with an environment_factory dict (trl 1.15.0 `_generate_and_score_completions` and
    `_tool_call_loop`): one instance per rollout, reset(**row without `environment`), its string appended to the last
    user message, tools found on the class, called with the parsed arguments, `str(result)` as the tool message; a
    reply without tool calls ends the rollout, as do `max_iterations` replies. The model is a list of replies."""

    def __init__(self, factories: dict, max_iterations: int = 20):
        self.factories, self.max_iterations = factories, max_iterations

    def rollout(self, row: dict, replies: list[dict], snapshots: list | None = None):
        env = self.factories[row["environment"]]()
        prompt = copy.deepcopy(row["prompt"])
        obs = env.reset(**{k: v for k, v in row.items() if k != "environment"})
        if obs is not None:
            prompt = prompt[:-1] + [{**prompt[-1], "content": prompt[-1]["content"] + obs}]
        tools = {n: getattr(env, n) for n, _ in inspect.getmembers(type(env), predicate=inspect.isfunction)
                 if n not in ("reset", "get_reward") and not n.startswith("_")}
        convo, completion = list(prompt), []
        for i, reply in enumerate(replies):
            if snapshots is not None:
                snapshots.append(copy.deepcopy(convo))
            convo.append(reply)
            completion.append(reply)
            if not reply.get("tool_calls") or i >= self.max_iterations:
                break
            for c in reply["tool_calls"]:
                name = c["function"]["name"]
                try:
                    result = tools[name](**c["function"]["arguments"])
                except Exception as e:                                # TRL: {"error": str(e)}
                    result = {"error": str(e)}
                msg = {"role": "tool", "name": name, "content": result if isinstance(result, list) else str(result)}
                convo.append(msg)
                completion.append(msg)
        return env, prompt, convo, completion


def _calls(calls):
    return [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in calls]


def _scripted_replies(task, per_reply=2):
    """The scripted decision's calls, `per_reply` per reply, then end_turn (twice when the first is refused: the
    mayor's / planning's open-requests reminder) and a closing text reply; with the tool messages a TaskEpisode
    returns for the same calls (= ToolAgent's)."""
    ep = TaskEpisode(task)
    ep.reset()
    calls, report = scripted_calls(ep)
    want = [ep.step(n, a) for n, a in calls]
    ends = []
    while not ep.done and len(ends) < 2:
        ends.append(("end_turn", {"report": report}))
        want.append(ep.step(*ends[-1]))
    out = [{"role": "assistant", "content": "", "tool_calls": _calls(calls[i:i + per_reply])}
           for i in range(0, len(calls), per_reply)]
    out += [{"role": "assistant", "content": "", "tool_calls": _calls([e])} for e in ends]
    out.append({"role": "assistant", "content": "Turn ended."})
    return out, calls + ends, want, ep


# ---------------------------------------------------------------------------------------------------- classes
def test_env_classes_expose_exactly_the_toolbox_tools(tasks):
    fac = tasks.factories()
    assert set(fac) == {r["environment"] for r in tasks.rows} == set(tasks.schemas)
    for key, cls in fac.items():
        names = [s["function"]["name"] for s in tasks.schemas[key]]
        assert sorted(T.tool_names(cls)) == sorted(names) and "end_turn" in names
        assert key.startswith(cls()._key.split("_")[0])
    row = tasks.rows[0]
    assert row["prompt"][1] == {"role": "user", "content": ""} and row["prompt"][0]["role"] == "system"
    ep = TaskEpisode(tasks.bank.get(row["task_id"]))
    obs = ep.reset()
    assert row["prompt"][0]["content"] == obs["system_prompt"]
    env = fac[row["environment"]]()
    assert env.reset(**{k: v for k, v in row.items() if k != "environment"}) == obs["briefing"]


def test_task_set_cache_round_trip(tasks, tmp_path):
    p = tmp_path / "ts.json"
    tasks.save(p)
    again = T.TaskSet.load(p, tasks.bank)
    assert again.rows == tasks.rows and again.schemas == tasks.schemas
    other = copy.copy(tasks.bank)
    other.manifest = {**tasks.bank.manifest, "created": 0}
    assert T.TaskSet.load(p, other) is None and T.TaskSet.load(tmp_path / "none.json", tasks.bank) is None


def test_reset_refuses_rows_that_dont_fit(tasks):
    fac = tasks.factories()
    row = tasks.rows[0]
    with pytest.raises(ValueError, match="system prompt"):
        fac[row["environment"]]().reset(task_id=row["task_id"], prompt=[{"role": "system", "content": "x"}])
    other = next(r for r in tasks.rows if r["environment"] != row["environment"])
    with pytest.raises(ValueError, match="tools"):
        fac[other["environment"]]().reset(task_id=row["task_id"])


@pytest.mark.skipif(not HAS("transformers"), reason="transformers not installed")
def test_docstring_schemas_match_the_toolbox_names_and_required(tasks):
    from transformers.utils import get_json_schema
    for key, cls in tasks.factories().items():
        env = cls()
        for s in tasks.schemas[key]:
            g = get_json_schema(getattr(env, s["function"]["name"]))["function"]
            want = s["function"]["parameters"]
            assert set(g["parameters"]["properties"]) == set(want["properties"]), s["function"]["name"]
            assert set(g["parameters"].get("required", [])) == set(want.get("required", []))
            for k, p in want["properties"].items():
                assert g["parameters"]["properties"][k]["type"] == p["type"]
                assert g["parameters"]["properties"][k].get("enum") == p.get("enum")


# ----------------------------------------------------------------------------------------------------- loop
def test_a_trl_rollout_scores_as_the_same_calls_through_the_episode(tasks):
    loop = FakeTRL(tasks.factories())
    for row in tasks.rows[:4]:
        task = tasks.bank.get(row["task_id"])
        replies, calls, want, ep = _scripted_replies(task)
        env, prompt, convo, completion = loop.rollout(row, replies)
        # the tool messages are exactly what TaskEpisode (= ToolAgent) returns for the same calls
        got = [m["content"] for m in completion if m["role"] == "tool"]
        assert got == want, row["task_id"]
        assert env._episode.done and env._episode.reason == "end_turn"
        logged, extra = {}, {}
        rew = T.CivitasReward()
        r = rew(prompts=[prompt], completions=[completion], environments=[env],
                log_metric=lambda k, v: logged.__setitem__(k, v),
                log_extra=lambda k, v: extra.__setitem__(k, v))
        mine = ep.finish(reward_cfg=R.RewardConfig())
        assert r == [pytest.approx(mine["reward"])]
        assert logged["civitas/reward"] == pytest.approx(mine["reward"]) and logged["civitas/ended"] == 1.0
        assert logged["civitas/actions"] == mine["counts"]["actions"] and "civitas/top_ups" in logged
        assert extra["task_id"] == [row["task_id"]] and len(extra["city"]) == 1
        assert rew(environments=[env]) == r                         # scored once, cached


def test_messages_within_a_turn_are_append_only(tasks):
    """Every reply and tool result only appends: no earlier message (system prompt, briefing, past calls and
    results) changes, in TRL's conversation and in the episode's own transcript."""
    loop = FakeTRL(tasks.factories())
    for row in tasks.rows[:3]:
        replies = _scripted_replies(tasks.bank.get(row["task_id"]), per_reply=1)[0]
        snaps = []
        env, prompt, convo, _ = loop.rollout(row, replies, snaps)
        assert convo[:2] == prompt and prompt[1]["content"] == env._episode.briefing
        for a, b in zip(snaps, snaps[1:] + [convo]):
            assert b[:len(a)] == a
        ep_msgs = env._episode.messages
        assert ep_msgs[0]["content"] == prompt[0]["content"] and ep_msgs[1]["content"] == prompt[1]["content"]
        # the episode transcript holds the same tool results, in order, never rewritten
        assert [m["content"] for m in ep_msgs if m["role"] == "tool"] == \
            [m["content"] for m in convo if m["role"] == "tool"]


def test_calls_after_end_turn_and_bad_arguments_cost_what_toolagent_charges(tasks):
    row = next(r for r in tasks.rows if r["role"] == "police")
    loop = FakeTRL(tasks.factories())
    replies = [{"role": "assistant", "content": "", "tool_calls": _calls([("find_sites", {"purpose": "nope"}),
                                                                          ("build_police_station", {"x": "a"})])},
               {"role": "assistant", "content": "", "tool_calls": _calls([("end_turn", {"report": "r"})])},
               {"role": "assistant", "content": "", "tool_calls": _calls([("build_police_station", {"x": 1, "y": 1})])},
               {"role": "assistant", "content": "done"}]
    env, _, convo, completion = loop.rollout(row, replies)
    tools = [json.loads(m["content"]) for m in convo if m["role"] == "tool"]
    assert tools[-1]["error"].startswith("not run: the turn already ended")
    res = T.CivitasReward().score([env], [completion])[0]
    assert res["calls"] == 3 and res["ended"] and res["invalid"] >= 1 and res["actions"] == 0


def test_unended_and_malformed_rollouts_are_penalized(tasks):
    row = next(r for r in tasks.rows if r["role"] == "planning")
    loop = FakeTRL(tasks.factories())
    replies = [{"role": "assistant", "content": "", "tool_calls": _calls([("city_overview", {})])},
               {"role": "assistant", "content": '<tool_call>\n{"name": "zone", "arguments": {"x1": 1,\n</tool_call>'}]
    env, _, _, completion = loop.rollout(row, replies)
    cfg = T.EnvConfig(p_malformed=0.3)
    res = T.CivitasReward(cfg).score([env], [completion])[0]
    assert res["malformed"] == 1 and not res["ended"] and res["end_turn_first"] is False
    assert res["penalty"] == pytest.approx(cfg.reward.p_unended + 0.3)


def test_reward_workers_give_the_in_process_reward(tasks):
    loop = FakeTRL(tasks.factories())
    rows = [r for r in tasks.rows if r["role"] in ("mayor", "planning")][:2]
    here, there = [], []
    for out in (here, there):
        for row in rows:
            replies = _scripted_replies(tasks.bank.get(row["task_id"]))[0]
            out.append(loop.rollout(row, replies)[0])
    a = T.CivitasReward().score(here)
    rw = T.CivitasReward(T.EnvConfig(reward_workers=2))
    try:
        b = rw.score(there)
    finally:
        rw.close()
    assert a == b


def test_summarize_rates():
    res = [{**{k: 0 for k in T.COMPONENTS + T.COUNTS}, "reward": 1.0, "ended": True, "failed": 2},
           {**{k: 0 for k in T.COMPONENTS + T.COUNTS}, "reward": 0.0, "ended": False}]
    s = T.summarize(res)
    assert s["reward"] == 0.5 and s["ended"] == 0.5 and s["unended"] == 0.5 and s["any_failed"] == 0.5


@pytest.mark.skipif(not HAS("torch"), reason="torch not installed")
def test_drop_zero_std_masks_zero_advantage_rollouts():
    import torch
    out = {"advantages": torch.tensor([0.0, 0.0, 1.0, -1.0]), "completion_mask": torch.ones(4, 3, dtype=torch.long),
           "tool_mask": torch.tensor([[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0]]),
           "num_items_in_batch": torch.tensor(9.0)}
    assert T.drop_zero_std(out) == 2
    assert out["completion_mask"][:2].sum() == 0 and out["completion_mask"][2:].sum() == 6
    assert float(out["num_items_in_batch"]) == 4.0


@pytest.mark.skipif(not HAS("torch"), reason="torch not installed")
def test_torch_logprob_fallback_matches_log_softmax():
    import torch
    torch.manual_seed(0)
    h, w, b, t = torch.randn(5, 8, requires_grad=True), torch.randn(11, 8), torch.randn(11), torch.randint(0, 11, (5,))
    lp, ent, lssp, mean, top1 = TR._TorchLogProbs.apply(h, w, b, t, 0.7, 4, None, 1.0)
    z = (h @ w.T + b) / 0.7
    ref = torch.log_softmax(z, -1)
    assert torch.allclose(lp, ref.gather(-1, t[:, None])[:, 0], atol=1e-5)
    assert torch.allclose(ent, -(ref.exp() * ref).sum(-1), atol=1e-5)
    assert torch.allclose(lssp, (ref.exp() ** 2).sum(-1).log(), atol=1e-5) and top1.dtype == torch.bool
    lp.sum().backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()


# --------------------------------------------------------------------------------------------------- config
def test_train_config_maps_onto_trl_and_the_reward():
    cfg = TR.load(ROOT / "configs/rl_qwen3_4b.toml", ["grpo.max_steps=7", "data.roles=[\"police\"]"])
    assert cfg["grpo"]["max_steps"] == 7 and cfg["data"]["roles"] == ["police"]
    e = TR.env_config(cfg)
    assert e.reward == R.RewardConfig() and e.malformed_penalty == 0.02 and e.reward_workers == 8
    kw = TR.grpo_kwargs(cfg, "/tmp/x", vllm_url="http://127.0.0.1:8000", has_eval=False)
    assert kw["use_vllm"] and kw["vllm_mode"] == "server" and kw["eval_strategy"] == "no"
    assert kw["num_generations"] == 8 and kw["beta"] == 0.0 and kw["epsilon_high"] == 0.28
    assert kw["max_tool_calling_iterations"] == 20 and kw["loss_type"] == "dapo"
    with pytest.raises(ValueError, match="unknown keys"):
        TR.env_config(TR.load(ROOT / "configs/rl_qwen3_4b.toml", ["reward.w_city=1"]))
    if HAS("trl"):
        from trl import GRPOConfig
        GRPOConfig(**TR.grpo_kwargs(cfg, "/tmp/civitas-rl-test", has_eval=True), use_cpu=True)


def test_modal_plan_prints_without_modal(capsys):
    from civitas.rl import modal_app as M
    cfg = M.load_cfg("configs/rl_qwen3_4b.toml", [])
    p = M.plan(cfg)
    assert "--weight-transfer-config" in p["vllm_train"] and "--tool-call-parser hermes" in p["vllm_eval"]
    e = p["estimate"]
    assert e["steps"] == 300 and e["tasks_per_step"] == 8 and e["gpus"] == 2 and 50 < e["train_usd"] < 400
    lora = M.vllm_command(cfg, lora={"run-final": "/vol/train/run/final"})
    assert "--enable-lora" in lora and "run-final=/vol/train/run/final" in lora
    assert M.main_cli(["--print-config", "--set", "grpo.max_steps=30"]) == 0
    out = capsys.readouterr().out
    assert "30 steps x 8 tasks x 8 rollouts" in out and "modal run --detach" in out


def test_rl_configs_stay_out_of_the_launcher(tmp_path):
    from civitas.server import Launcher
    names = {c["file"] for c in Launcher(tmp_path, ROOT / "configs").configs()}
    assert "fleet_smoke.toml" in names and not {"rl_qwen3_4b.toml", "rl_smoke_local.toml"} & names


def test_rl_compare_pairs_two_evals(tmp_path, capsys):
    from civitas import cli
    from civitas.rl import evaluate as E
    for name, rewards in (("a", {"t1": [0.0, 0.2], "t2": [0.5], "t3": [0.1]}), ("b", {"t1": [0.3], "t2": [0.4],
                                                                                    "t4": [1.0]})):
        (tmp_path / name).mkdir()
        (tmp_path / name / "results.jsonl").write_text("".join(
            json.dumps({"task_id": t, "role": "planning" if t != "t2" else "police", "reward": r}) + "\n"
            for t, rs in rewards.items() for r in rs) + json.dumps({"task_id": "t9", "reward": None}) + "\n")
    c = E.compare(tmp_path / "a", tmp_path / "b")
    assert c["tasks"] == 2 and c["only_a"] == 1 and c["only_b"] == 1
    assert c["diff"]["mean"] == pytest.approx(((0.3 - 0.1) + (0.4 - 0.5)) / 2) and (c["wins"], c["losses"]) == (1, 1)
    assert set(c["by_role"]) == {"planning", "police"}
    cli.main(["rl", "compare", str(tmp_path / "a"), str(tmp_path / "b")])
    assert "2 common tasks" in capsys.readouterr().out


def test_latest_checkpoint(tmp_path):
    for s in (5, 20, 10):
        (tmp_path / f"checkpoint-{s}").mkdir()
        (tmp_path / f"checkpoint-{s}" / "trainer_state.json").write_text("{}")
    (tmp_path / "checkpoint-30").mkdir()                          # incomplete: no trainer_state.json
    assert TR.latest_checkpoint(tmp_path).endswith("checkpoint-20") and TR.latest_checkpoint(tmp_path / "x") is None


# ------------------------------------------------------------------------------------- opt-in: real templates
def _tokenizer(name="Qwen/Qwen3-4B-Instruct-2507"):
    if not HAS("transformers"):
        pytest.skip("transformers not installed")
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:
        pytest.skip(f"{name}'s tokenizer isn't in the Hugging Face cache")


def test_qwen_template_renders_a_turn_append_only(tasks):
    """With the target model's template (TRL's training copy of it), each message of a turn renders as a pure
    token suffix: what TRL's token concatenation (prompt + completion + tool suffix) relies on."""
    tok = _tokenizer()
    if not HAS("trl"):
        pytest.skip("trl not installed")
    from trl.chat_template_utils import get_training_chat_template, is_chat_template_prefix_preserving
    tmpl = None if is_chat_template_prefix_preserving(tok) else get_training_chat_template(tok)
    row = next(r for r in tasks.rows if r["role"] == "police")
    replies = _scripted_replies(tasks.bank.get(row["task_id"]), per_reply=1)[0]
    env, prompt, convo, _ = FakeTRL(tasks.factories()).rollout(row, replies)
    schemas = tasks.schemas[row["environment"]]
    render = lambda msgs, gen: tok.apply_chat_template(msgs, tools=schemas, chat_template=tmpl, tokenize=True,
                                                       add_generation_prompt=gen, return_dict=False)
    prev = render(convo[:2], True)
    assert '"name": "end_turn"' in tok.decode(prev)                 # the exact schemas are in the prompt
    for k in range(3, len(convo) + 1):
        if convo[k - 1]["role"] == "tool" and k < len(convo) and convo[k]["role"] == "tool":
            continue                                               # several results render as one user turn
        cur = render(convo[:k], convo[k - 1]["role"] == "tool")
        assert cur[:len(prev)] == prev, k
        prev = cur


class ScriptedGeneration:
    """Stands in for GRPOTrainer._generate_single_turn: per call, one scripted reply per row (completion token ids,
    no logprobs), so TRL's own tool loop (parse, dispatch to the env, tool suffix, tool mask) runs on known calls."""

    def __init__(self, tok, turns):
        self.tok, self.turns, self.calls = tok, turns, 0

    def __call__(self, prompt_ids, images, multimodal_fields, num_generations, has_tool_images=False):
        texts = self.turns[min(self.calls, len(self.turns) - 1)]
        self.calls += 1
        ids = [self.tok(texts[i % len(texts)], add_special_tokens=False)["input_ids"] for i in range(len(prompt_ids))]
        return ids, None


def _tool_call_text(name, args):
    return f'<tool_call>\n{json.dumps({"name": name, "arguments": args})}\n</tool_call><|im_end|>'


@pytest.mark.skipif(not HAS("trl"), reason="trl not installed")
def test_trl_tool_loop_runs_the_env_and_masks_tool_output(bank, tmp_path):
    """A real CivitasGRPOTrainer step (tiny random Qwen3 + the 2507 tokenizer) whose generations are scripted: rollout
    A calls find_sites then end_turn, rollout B answers in text. TRL's loop must run the env's tools, mask their tokens
    out of the loss, score A and B apart (a non-zero advantage) and take an optimizer step."""
    tok = _tokenizer()
    from transformers import AutoModelForCausalLM  # noqa: F401  (the tiny model must be cached)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download("trl-internal-testing/tiny-Qwen3ForCausalLM", local_files_only=True)
    except Exception:
        pytest.skip("trl-internal-testing/tiny-Qwen3ForCausalLM isn't in the Hugging Face cache")
    cfg = TR.load(ROOT / "configs/rl_smoke_local.toml", [
        'model.id="trl-internal-testing/tiny-Qwen3ForCausalLM"', 'data.bank="t"', 'data.roles=["police"]',
        "data.limit=1", "grpo.max_steps=1", "grpo.max_completion_length=4096", "grpo.max_tool_calling_iterations=4",
        "lora.r=2", "lora.lora_alpha=4", "civitas.drop_zero_std=true", "grpo.gradient_checkpointing=false"])
    trainer, ts, _ = TR.build(cfg, output_dir=tmp_path / "out", root=bank.dir.parent)
    trainer._generate_single_turn = ScriptedGeneration(tok, [
        [_tool_call_text("find_sites", {"purpose": "station"}), "I'll wait this turn.<|im_end|>"],
        [_tool_call_text("end_turn", {"report": "looked at sites"})],
        ["Turn ended.<|im_end|>"]])
    seen = {}
    inner = trainer._generate_and_score_completions

    def spy(inputs):
        out = inner(inputs)
        seen.update(out=out, envs=list(trainer.environments))
        return out

    trainer._generate_and_score_completions = spy
    trainer.train()
    out, envs = seen["out"], seen["envs"]
    a, b = envs
    assert [t["tool"] for t in a._episode.trace] == ["find_sites", "end_turn"] and a._episode.done
    assert b._episode.trace == [] and a._result["ended"] and not b._result["ended"]
    assert a._result["reward"] != b._result["reward"]
    assert (out["advantages"] != 0).all()                          # a real signal: nothing dropped
    tm, cm = out["tool_mask"], out["completion_mask"]
    ids = out["completion_ids"]
    row = 0 if a is envs[0] else 1
    masked = tok.decode(ids[row][(tm[row] == 0) & (cm[row] == 1)])
    trained = tok.decode(ids[row][(tm[row] == 1) & (cm[row] == 1)])
    assert "<tool_response>" in masked and '"steps_left"' in masked     # the env's results: out of the loss
    assert "find_sites" in trained and "end_turn" in trained and "<tool_response>" not in trained
    hist = trainer.state.log_history
    assert any(h.get("tools/call_frequency", 0) > 0 for h in hist)
    assert any("civitas/reward" in h for h in hist) and any("loss" in h for h in hist)
    lines = [json.loads(x) for x in (tmp_path / "out" / "metrics.jsonl").read_text().splitlines()]
    assert any(x.get("civitas/actions") == 0 and "civitas/ended" in x for x in lines)


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("CIVITAS_TRL_SMOKE") != "1" or not HAS("trl"),
                    reason="opt-in: CIVITAS_TRL_SMOKE=1 in a venv with torch, trl, peft, datasets (minutes, a 0.6B model)")
def test_grpo_smoke_runs_a_step(bank, tmp_path):
    cfg = TR.load(ROOT / "configs/rl_smoke_local.toml", ["data.bank=\"t\"", "grpo.max_steps=1"])
    res = TR.train(cfg, output_dir=tmp_path / "out", root=bank.dir.parent, resume=False)
    assert res["steps"] == 1 and (tmp_path / "out" / "final").is_dir()
    lines = [json.loads(x) for x in (tmp_path / "out" / "metrics.jsonl").read_text().splitlines()]
    assert any("civitas/reward" in x for x in lines) and any("loss" in x for x in lines)


# ------------------------------------------------------------------------------------------------------- sft
def test_sft_export_writes_tool_calling_transcripts(bank, tmp_path):
    from civitas.rl import sft as S
    out = tmp_path / "sft.jsonl"
    s = S.export(bank, out, roles=["mayor", "police"], policies=("oracle", "scripted"), min_reward=-1.0, workers=1,
                 verbose=False)
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert s["kept"] == len(rows) >= 1 and s["tasks"] >= len(rows)
    for r in rows:
        m = r["messages"]
        assert [x["role"] for x in m[:2]] == ["system", "user"] and m[2]["role"] == "assistant"
        calls = [c for x in m if x["role"] == "assistant" for c in x.get("tool_calls") or []]
        assert calls and all(isinstance(c["function"]["arguments"], str) for c in calls)
        assert calls[-1]["function"]["name"] == "end_turn" and r["counts"]["ended"] and not r["counts"]["failed"]
        assert [c["id"] for c in calls] == [x["tool_call_id"] for x in m if x["role"] == "tool"]
        assert {t["function"]["name"] for t in json.loads(r["tools"])} >= {c["function"]["name"] for c in calls}
        assert r["policy"] in ("oracle", "scripted") and r["reward"] >= -1.0
