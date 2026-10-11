"""TRL GRPO on bank tasks: the environment classes, the dataset and the reward for `trl.GRPOTrainer`.

Written against trl 1.15.0 (`GRPOTrainer(environment_factory=...)`, checked in its source):
- `environment_factory` is a callable, or a dict {name: callable} with an `environment` field on each dataset row
  selecting the env; TRL makes one instance per rollout (pooled, `reset(**row)` before each), and the tools are the
  public functions *of the class* (`inspect.getmembers(type(env), inspect.isfunction)` minus `reset` / `get_reward`
  and `_*`), called as `method(**arguments)` with the parsed JSON arguments; the result goes back as
  `{"role": "tool", "name": ..., "content": str(result)}`. A string from `reset()` is appended to the row's last
  user message. The loop stops when a reply has no tool call or after `max_tool_calling_iterations` replies;
- the prompt is rendered once, then grown by token concatenation (prompt ids + completion ids + the template's tool
  suffix ids, `_tool_call_loop`): append-only by construction, and tool tokens are masked out of the loss
  (`tool_mask`). `max_completion_length` counts every token after the prompt, tool results included;
- the schemas shown to the model come from `transformers.get_json_schema(method)` (docstring + type hints). That
  can't say everything a civitas schema says (nested arrays, per-department objects, exact optional fields), so
  `CivitasGRPOTrainer` swaps in the exact ToolBox schemas after `__init__` (`trainer._env_tools`, read only when
  rendering prompts: `install_schemas`); the generated methods still carry typed signatures and Google docstrings
  that `get_json_schema` turns into a close equivalent, in case a TRL version renders from the methods only.

What a rollout is: one department's turn of a bank task (episode.py:TaskEpisode), its tools the model toolbox's
(guards, refusals, hints, refreshed Next lines; ToolBox(branch_dezone=True, max_steps, next_line=True)), each tool
result the exact tool message content ToolAgent sends (`TaskEpisode.step`). Differences from ToolAgent's loop
(fleet/agent.py), all TRL's: no NUDGE / LENGTH_NUDGE (a reply without a call ends the rollout), no recovery of
calls written as text (a malformed <tool_call> is plain text to TRL: counted here as `malformed`, a p_invalid
penalty), and the model sees end_turn's result and replies once more (calls after end_turn are "not run" and cost
nothing). The system prompt is the dataset row's (built by the same TaskEpisode.reset when the dataset is made and
checked again at reset), the briefing comes from reset().

The reward (`CivitasReward`, a reward function reading the env instances TRL passes as `environments`) is
reward.py's counterfactual reward of the turn against the bank's cached scripted baseline, H months on, plus the
malformed penalty; each component and the turn's counts go to `log_metric` (`civitas/<name>`, averaged per step) and
per completion to `log_extra` (the completions table). Branches can be played in a process pool
(`reward_workers`): each finished episode's env is pickled to a worker, as bank snapshots are.

    from civitas.rl import trl_env as T
    tasks = T.task_set("rl-v2", split="train", roles=["planning"])
    trainer = T.trainer_class()(model="Qwen/Qwen3-4B-Instruct-2507", args=GRPOConfig(...),
                                train_dataset=tasks.dataset(), reward_funcs=[T.CivitasReward(tasks.cfg)],
                                environment_factory=tasks.factories(), peft_config=LoraConfig(...), task_set=tasks)

`trainer_class()` and `tasks.dataset()` import trl / datasets; everything else here runs without them (the tests
drive the env classes with a stand-in for TRL's loop).
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import pickle
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from . import bank as B
from . import reward as R

TRL_TESTED = "1.15.0"
VERBOSE = os.environ.get("CIVITAS_RL_VERBOSE") == "1"     # each reset, tool call and reward batch to stderr


def _say(msg: str) -> None:
    if VERBOSE:
        print(f"[civitas-rl {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)
EXCLUDED = ("reset", "get_reward")                   # TRL's non-tool members of an environment class
COMPONENTS = ("reward", "city", "role", "penalty", "population")
COUNTS = ("calls", "actions", "failed", "refused", "invalid", "malformed", "ended", "end_turn_first", "lookups",
          "messages", "money_asks", "asks_held", "overrides", "loans", "repaid", "take_backs", "top_ups")


@dataclass
class EnvConfig:
    """How a training rollout is played and scored (configs/rl_*.toml [reward] / [env])."""
    reward: R.RewardConfig = field(default_factory=R.RewardConfig)
    p_malformed: float | None = None     # per malformed <tool_call> left as text (default: reward.p_invalid)
    end_at_action_cap: bool = False      # end the turn at the action cap (ToolAgent doesn't)
    reward_workers: int = 0              # processes playing the branches (0: in the trainer's process)
    check_prompt: bool = True            # reset() checks the row's system prompt and the class's schemas

    @property
    def malformed_penalty(self) -> float:
        return self.reward.p_invalid if self.p_malformed is None else float(self.p_malformed)

    def as_dict(self) -> dict:
        return {"reward": self.reward.as_dict(), "p_malformed": self.malformed_penalty,
                "end_at_action_cap": self.end_at_action_cap, "reward_workers": self.reward_workers}


# ------------------------------------------------------------------------------------------------- turn stats
def turn_stats(executed: list[dict], trace: list[dict], messages: list[dict], ended: bool) -> dict:
    """What a turn did beyond reward.counts: look-ups, council messages, money asks to the mayor (collab.is_request
    and MONEY) and asks held back (refused "ask"), earned-confirm overrides, the mayor's loans / repayments /
    take-backs (negative allocate_budget) / top-ups, and whether the first call was end_turn (a noop turn)."""
    from ..bench.collab import MONEY, is_request
    from ..fleet.tools import ALL_ACTIONS
    ok = [r.get("action") or {} for r in executed if r.get("ok")]
    kind = lambda t: [a for a in ok if a.get("type") == t]
    alloc = kind("allocate_budget")
    amount = lambda a: float(a.get("amount") or 0) if isinstance(a.get("amount"), (int, float)) else 0.0
    return {"lookups": sum(1 for t in trace if t.get("tool") not in ALL_ACTIONS
                           and t.get("tool") not in ("send_message", "end_turn", "update_memory", "update_plan")),
            "messages": len(messages),
            "money_asks": sum(1 for m in messages if m.get("to") == "mayor" and is_request("mayor", m.get("text"))
                              and MONEY.search(str(m.get("text") or ""))),
            "asks_held": sum(1 for t in trace if (t.get("result") or {}).get("refused") == "ask"),
            "overrides": sum(1 for t in trace if "override" in (t.get("result") or {})),
            "loans": len(kind("take_loan")), "repaid": len(kind("repay_loan")),
            "take_backs": sum(1 for a in alloc if amount(a) < 0), "top_ups": sum(1 for a in alloc if amount(a) > 0),
            "end_turn_first": bool(trace and trace[0].get("tool") == "end_turn"), "ended": bool(ended)}


# ------------------------------------------------------------------------------------------- schema -> method
_TYPES = {"integer": int, "number": float, "string": str, "boolean": bool}


def _hint(schema: dict):
    """A Python type hint transformers.get_json_schema turns back into (about) this JSON schema."""
    t = schema.get("type")
    if t in _TYPES:
        return _TYPES[t]
    if t == "array":
        items = schema.get("items")
        return list[_hint(items)] if isinstance(items, dict) else list
    if t == "object":
        vals = [p.get("type") for p in (schema.get("properties") or {}).values()]
        if vals and len(set(vals)) == 1 and vals[0] in _TYPES:
            return dict[str, _TYPES[vals[0]]]
        return dict
    return str


def _line(text: str) -> str:
    """One docstring line: no newlines, and no section header transformers' docstring parser would split on."""
    out = " ".join(str(text or "").split())
    for h in ("Args:", "Returns:", "Raises:"):
        out = out.replace(h, h[:-1] + " -")
    return out.replace("(choices:", "(options:")


def tool_method(schema: dict):
    """A method for one tool schema: it accepts any keyword arguments and runs the call through the episode
    (ToolBox does the checking and reports what's wrong, as for ToolAgent); its signature, annotations and Google
    docstring describe the schema for transformers.get_json_schema."""
    fn = schema["function"]
    name = fn["name"]
    params = fn.get("parameters") or {}
    props, required = params.get("properties") or {}, set(params.get("required") or [])

    def call(self, **kwargs):
        return self._call(name, kwargs)

    sig, ann, args = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)], {}, []
    for k in sorted(props, key=lambda k: k not in required):            # required first, as Python wants
        p = props[k]
        ann[k] = _hint(p)
        sig.append(inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY, annotation=ann[k],
                                     **({} if k in required else {"default": None})))
        desc = _line(p.get("description") or k)
        if p.get("enum"):
            desc += f" (choices: {json.dumps(p['enum'])})"
        args.append(f"    {k}: {desc}")
    ann["return"] = str
    call.__name__ = call.__qualname__ = name
    call.__signature__ = inspect.Signature(sig, return_annotation=str)
    call.__annotations__ = ann
    call.__doc__ = _line(fn.get("description") or name) + ("\n\nArgs:\n" + "\n".join(args) if args else "") + \
        "\n\nReturns:\n    The tool result (JSON text)."
    return call


def schema_key(role: str, schemas: list[dict]) -> str:
    """The environment name for a role's tool set: the role and a digest of the exact schemas."""
    return f"{role}_{hashlib.sha256(json.dumps(schemas, sort_keys=True).encode()).hexdigest()[:8]}"


# -------------------------------------------------------------------------------------------- environments
class TaskToolEnv:
    """One rollout: a bank task's turn (see the module doc). Subclasses made by `env_class` add one public method
    per tool and set `_bank`, `_schemas`, `_key` and `_cfg`; nothing public here but `reset`."""
    _bank: B.TaskBank | None = None
    _schemas: list[dict] = []
    _key: str = ""
    _cfg: EnvConfig = EnvConfig()

    def __init__(self):
        self._ep = None
        self._result: dict | None = None
        self._malformed = 0

    def reset(self, task_id: str = "", prompt: list | None = None, **_row) -> str:
        """Open the turn of task `task_id` and return its briefing (TRL appends it to the row's empty user message).
        With `check_prompt`, the row's system prompt and this class's tools must be the episode's own."""
        from .episode import TaskEpisode
        if not task_id:
            raise ValueError("TaskToolEnv.reset needs the row's task_id")
        self._ep = TaskEpisode(self._bank.get(task_id), end_at_action_cap=self._cfg.end_at_action_cap)
        self._result, self._malformed = None, 0
        obs = self._ep.reset()
        _say(f"reset {task_id} ({len(obs['briefing'])} chars of briefing)")
        if self._cfg.check_prompt:
            if obs["tools"] != self._schemas:
                raise ValueError(f"task {task_id}: its tools aren't environment {self._key}'s (make the dataset and the "
                                 "factories from one task_set)")
            if prompt and prompt[0].get("role") == "system" and prompt[0].get("content") != obs["system_prompt"]:
                raise ValueError(f"task {task_id}: the row's system prompt isn't the episode's (stale dataset?)")
        return obs["briefing"]

    def _call(self, name: str, kwargs: dict) -> str:
        if self._ep is None:
            raise RuntimeError("a tool was called before reset()")
        if self._result is not None:
            return json.dumps({"ok": False, "error": "not run: the turn was already scored"})
        out = self._ep.step(name, kwargs)
        _say(f"{self._ep.row.get('task_id')}: {name} {json.dumps(kwargs)[:120]} -> {out[:100]}")
        return out

    # ---- read by the reward function
    @property
    def _episode(self):
        return self._ep

    def _note_completion(self, completion) -> None:
        """Count a malformed tool call left as text at the end of the rollout (TRL's parse fallback)."""
        last = completion[-1] if isinstance(completion, list) and completion else completion
        if isinstance(last, dict) and last.get("role") == "assistant" and not last.get("tool_calls") \
                and "<tool_call>" in str(last.get("content") or ""):
            self._malformed = 1

    def _turn(self) -> dict:
        """The played turn, picklable, for a branch in another process."""
        ep = self._ep
        return {"row": ep.row, "role": ep.role, "decision": ep.decision(), "trace": list(ep.trace),
                "ended": bool(ep.box.ended), "messages": list(ep.box.messages)}

    def _score(self, cfg: EnvConfig | None = None) -> dict:
        """The reward of the turn (played once; later calls return it)."""
        if self._result is None:
            if self._ep is None or self._ep.box is None:
                raise RuntimeError("no turn to score: reset() first")
            cfg = cfg or self._cfg
            t = self._turn()
            out = self._ep.finish(reward_cfg=cfg.reward)
            self._result = _with_stats(out, t, self._malformed, cfg)
        return self._result


def _with_stats(out: dict, turn: dict, malformed: int, cfg: EnvConfig) -> dict:
    stats = turn_stats(turn["decision"].executed, turn["trace"], turn["messages"], turn["ended"])
    pen = cfg.malformed_penalty * malformed
    res = {k: out[k] for k in COMPONENTS}
    res.update(reward=round(out["reward"] - pen, 6), penalty=round(out["penalty"] + pen, 6),
               **{k: out["counts"][k] for k in ("calls", "actions", "failed", "refused", "invalid")},
               malformed=malformed, **stats)
    res["horizon"] = out["horizon"]
    return res


def env_class(key: str, schemas: list[dict], bank: B.TaskBank, cfg: EnvConfig | None = None) -> type:
    """A TaskToolEnv subclass whose public methods are exactly the tools in `schemas`."""
    ns: dict[str, Any] = {"_bank": bank, "_schemas": schemas, "_key": key, "_cfg": cfg or EnvConfig(),
                          "__doc__": f"Civitas task turn, tool set {key} ({len(schemas)} tools)."}
    for s in schemas:
        name = s["function"]["name"]
        if name in EXCLUDED or name.startswith("_") or hasattr(TaskToolEnv, name):
            raise ValueError(f"tool {name!r} clashes with the environment class")
        ns[name] = tool_method(s)
    return type(f"Civitas_{key}", (TaskToolEnv,), ns)


def tool_names(cls: type) -> list[str]:
    """The tools TRL finds on an environment class (its own discovery rule)."""
    return [n for n, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
            if n not in EXCLUDED and not n.startswith("_")]


# -------------------------------------------------------------------------------------------------- dataset
@dataclass
class TaskSet:
    """Dataset rows for a bank selection and the tool sets they use."""
    bank: B.TaskBank
    rows: list[dict]
    schemas: dict[str, list[dict]]
    cfg: EnvConfig = field(default_factory=EnvConfig)
    _classes: dict = field(default_factory=dict, repr=False)

    def factories(self) -> dict[str, type]:
        """{environment name: class}: TRL's dict `environment_factory` (each row's `environment` picks one)."""
        if not self._classes:
            self._classes = {k: env_class(k, s, self.bank, self.cfg) for k, s in self.schemas.items()}
        return self._classes

    def dataset(self):
        """A datasets.Dataset of the rows (the tool schemas stay out: install_schemas renders them)."""
        from datasets import Dataset
        return Dataset.from_list(self.rows)

    def save(self, path) -> None:
        """Rows and schemas as JSON (a restarted trainer skips resetting every task again: `load`)."""
        from pathlib import Path
        Path(path).write_text(json.dumps({"bank": self.bank.name, "created": self.bank.manifest.get("created"),
                                          "rows": self.rows, "schemas": self.schemas}))

    @classmethod
    def load(cls, path, bank: B.TaskBank, cfg: EnvConfig | None = None) -> "TaskSet | None":
        """A saved TaskSet of this very bank (name and build time), else None."""
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        if d.get("bank") != bank.name or d.get("created") != bank.manifest.get("created"):
            return None
        return cls(bank, d["rows"], d["schemas"], cfg or EnvConfig())

    def composition(self) -> dict:
        from collections import Counter
        return {"rows": len(self.rows), "environments": {k: len(v) for k, v in self.schemas.items()},
                "by_role": dict(Counter(r["role"] for r in self.rows)),
                "by_style": dict(Counter(r["style"] for r in self.rows)),
                "by_month": dict(Counter(r["month"] for r in self.rows))}


def select(bank: B.TaskBank, split: str = "train", *, roles=None, styles=None, months=None, seeds=None,
           exclude_seeds=None, limit: int | None = None, sample_seed: int | None = None) -> list:
    """The actionable tasks of `split` (TaskBank.tasks' filters) with these seeds / without those (a dev set held out
    of the training seeds), `limit` of them spread evenly (or sampled with `sample_seed`)."""
    import numpy as np
    tasks = bank.tasks(split, actionable=True, roles=roles or None, styles=styles or None, months=months or None)
    keep, drop = set(seeds or ()), set(exclude_seeds or ())
    tasks = [t for t in tasks if (not keep or t.seed in keep) and t.seed not in drop]
    if limit and limit < len(tasks):
        if sample_seed is not None:
            idx = sorted(np.random.default_rng(sample_seed).choice(len(tasks), size=limit, replace=False).tolist())
        else:
            idx = sorted({int(i * len(tasks) / limit) for i in range(limit)})
        tasks = [tasks[i] for i in idx]
    return tasks


def task_set(bank, split: str = "train", *, roles=None, styles=None, months=None, seeds=None, exclude_seeds=None,
             limit: int | None = None, sample_seed: int | None = None, cfg: EnvConfig | None = None,
             root=B.TASKS_DIR, repeat: int = 1) -> TaskSet:
    """Rows for the actionable tasks of `split` (filters as `select`): each {"prompt": [system, empty user],
    "task_id", "environment", "role", "style", "month"}, the system prompt and the tool set from the task's own
    TaskEpisode.reset. `repeat` lists every task that many times (epochs over a small bank in one dataset pass)."""
    from .episode import TaskEpisode
    bank = bank if isinstance(bank, B.TaskBank) else B.TaskBank(bank, root=root)
    tasks = select(bank, split, roles=roles, styles=styles, months=months, seeds=seeds, exclude_seeds=exclude_seeds,
                   limit=limit, sample_seed=sample_seed)
    if not tasks:
        raise B.BankError(f"bank {bank.name} has no actionable {split} tasks for that selection")
    rows, schemas = [], {}
    for t in tasks:
        obs = TaskEpisode(t).reset()
        key = schema_key(t.role, obs["tools"])
        schemas.setdefault(key, obs["tools"])
        rows.append({"prompt": [{"role": "system", "content": obs["system_prompt"]}, {"role": "user", "content": ""}],
                     "task_id": t.task_id, "environment": key, "role": t.role, "style": t.style, "month": t.month})
    return TaskSet(bank, rows * max(1, int(repeat)), schemas, cfg or EnvConfig())


# --------------------------------------------------------------------------------------------------- reward
def _branch(blob: bytes) -> dict:
    """In a worker: play a finished turn's branch (env and agents pickled after the turn) and score it."""
    env, agents, turn, malformed, cfg = pickle.loads(blob)     # our own pickle, made by CivitasReward below
    B._balance(env)
    from .episode import TaskEpisode
    ep = TaskEpisode(row=turn["row"], cfg={}, blob=b"")
    ep.env, ep.agents = env, agents
    d = turn["decision"]
    d.meta = {"tool_trace": turn["trace"], "ended": turn["ended"]}
    out = ep.finish(decision=d, reward_cfg=cfg.reward)
    return _with_stats(out, turn, malformed, cfg)


class CivitasReward:
    """The reward function for GRPOTrainer: each completion's env (`environments`, passed by TRL) scored by
    `TaskToolEnv._score`, components logged through `log_metric` / `log_extra` (see the module doc)."""
    __name__ = "civitas"

    def __init__(self, cfg: EnvConfig | None = None, prefix: str = "civitas"):
        self.cfg = cfg or EnvConfig()
        self.prefix = prefix
        self._pool = None
        self._lock = threading.Lock()
        self.last: list[dict] = []

    def _executor(self):
        with self._lock:
            if self._pool is None:
                import multiprocessing as mp
                self._pool = ProcessPoolExecutor(self.cfg.reward_workers, mp_context=mp.get_context("spawn"))
            return self._pool

    def score(self, environments, completions=None) -> list[dict]:
        for i, env in enumerate(environments):
            if completions is not None:
                env._note_completion(completions[i])
        if self.cfg.reward_workers > 0:
            todo = [e for e in environments if e._result is None]
            blobs = []
            for e in todo:
                if e._ep is None or e._ep.box is None:
                    raise RuntimeError("no turn to score: reset() first")
                e._ep.finished = True
                blobs.append(pickle.dumps((e._ep.env, e._ep.agents, e._turn(), e._malformed, self.cfg),
                                          protocol=pickle.HIGHEST_PROTOCOL))
            for e, res in zip(todo, self._executor().map(_branch, blobs)):
                e._result = res
        return [e._score(self.cfg) for e in environments]

    def __call__(self, prompts=None, completions=None, environments=None, log_metric=None, log_extra=None,
                 **kwargs) -> list[float]:
        if environments is None:
            raise ValueError("CivitasReward needs the environments (GRPOTrainer with environment_factory)")
        t0 = time.perf_counter()
        res = self.score(environments, completions)
        _say(f"scored {len(res)} rollouts in {time.perf_counter() - t0:.1f} s: rewards "
             f"{[round(r['reward'], 3) for r in res]}")
        self.last = res
        if log_metric is not None:
            for k, v in summarize(res).items():
                log_metric(f"{self.prefix}/{k}", v)
        if log_extra is not None:
            log_extra("task_id", [e._ep.row.get("task_id") for e in environments])
            for k in ("city", "role", "penalty", "actions", "failed", "refused", "invalid", "ended"):
                log_extra(k, [r[k] for r in res])
        return [r["reward"] for r in res]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(cancel_futures=True)
            self._pool = None


def summarize(res: list[dict]) -> dict:
    """Batch means of the components and counts (booleans as rates: `ended` is the end_turn rate), plus the share of
    turns with any failed / refused / invalid call."""
    if not res:
        return {}
    n = len(res)
    out = {k: sum(float(r[k]) for r in res) / n for k in COMPONENTS + COUNTS}
    out["unended"] = 1.0 - out["ended"]
    for k in ("failed", "refused", "invalid"):
        out[f"any_{k}"] = sum(1 for r in res if r[k]) / n
    return out


# -------------------------------------------------------------------------------------------------- trainer
def install_schemas(trainer, tasks: TaskSet) -> None:
    """Render the exact ToolBox schemas: replace the trainer's per-environment tool lists (TRL reads them only when
    rendering a prompt; tools are dispatched from the env instances) with the schemas. Refuses a TRL without
    `_env_tools` (the trainer's layout changed: check this module against its source)."""
    if not isinstance(getattr(trainer, "_env_tools", None), dict):
        raise RuntimeError(f"this TRL's GRPOTrainer has no _env_tools (written against trl {TRL_TESTED}); the tool "
                           "schemas would come from the method docstrings: check civitas/rl/trl_env.py")
    missing = set(tasks.schemas) - set(trainer._env_tools)
    if missing:
        raise RuntimeError(f"environments {sorted(missing)} aren't the trainer's")
    for k, s in tasks.schemas.items():
        trainer._env_tools[k] = list(s)


def drop_zero_std(output: dict) -> int:
    """Drop the rollouts whose advantage is 0 (every completion of a zero-variance group: DAPO's dynamic sampling,
    without the resampling) from the loss: their completion mask and the batch's token count. Returns how many."""
    import torch
    adv = output["advantages"]
    keep = (adv.abs() > 1e-8).to(output["completion_mask"].dtype)
    dropped = int((keep == 0).sum().item())
    if dropped:
        output["completion_mask"] = output["completion_mask"] * keep.unsqueeze(1)
        mask = output["completion_mask"] * output["tool_mask"] if "tool_mask" in output else output["completion_mask"]
        total = mask.sum().to(torch.float32)
        try:
            from accelerate.utils import gather
            total = gather(total.reshape(1)).sum()
        except Exception:
            pass
        output["num_items_in_batch"] = total
    return dropped


def trainer_class():
    """CivitasGRPOTrainer (needs trl): GRPOTrainer that takes `task_set=`, renders its exact schemas
    (install_schemas), drops zero-advantage rollouts from the loss when `drop_zero_std` (default) and logs
    `civitas/dropped_zero_std`."""
    from trl import GRPOTrainer

    class CivitasGRPOTrainer(GRPOTrainer):
        def __init__(self, *args, task_set: TaskSet | None = None, drop_zero_std: bool = True, **kw):
            if task_set is not None:
                kw.setdefault("environment_factory", task_set.factories())
            super().__init__(*args, **kw)
            self.civitas_tasks, self.civitas_drop = task_set, drop_zero_std
            if task_set is not None:
                install_schemas(self, task_set)

        def _generate_and_score_completions(self, inputs):
            out = super()._generate_and_score_completions(inputs)
            if self.civitas_drop and self.model.training:
                n = drop_zero_std(out)
                mode = "train"
                self._metrics[mode]["civitas/dropped_zero_std"].append(n / max(1, len(out["advantages"])))
            return out

    return CivitasGRPOTrainer
