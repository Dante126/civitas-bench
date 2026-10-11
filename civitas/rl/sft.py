"""SFT warm-start data: scripted-policy turns played through TaskEpisode, exported as tool-calling chat transcripts.

    python -m civitas.rl.sft export --bank rl-train --out runs/_sft/rl-train.jsonl --roles planning --workers 4
    python -m civitas.rl.sft export ... --policies oracle,scripted --min-reward 0     # the better policy per task

Each line is one task's turn in OpenAI chat format, what a model would be trained to send: {"messages": [system,
user (briefing), assistant {tool_calls: [{id, type, function: {name, arguments (JSON text)}}]}, tool {tool_call_id,
content}, ..., assistant end_turn], "tools": "<the exact tool schemas, JSON text>", "task_id", "role", "style",
"policy", "reward", "counts"}. The tool messages are the episode's own (= ToolAgent's), so a model fine-tuned on
these sees exactly what it will see in GRPO and in `civitas rl eval`. `tools` is JSON text because the schemas differ
between rows (Arrow can't hold them as structs); TRL's SFTTrainer parses it.

Per task, each policy in `--policies` (rl/policies.py: scripted, oracle, ...) plays the turn and is scored by the
counterfactual reward; the best one is kept if its reward is at least `--min-reward` and it made no failed or invalid
call. Seeds in `--exclude-seeds` (the dev seeds) are left out. Use it only if the base model's tool calls are poor
(the baseline eval's failed / invalid calls per task): SFT on scripted turns pulls the policy toward the scripted
fleet, whose turns score ~0 by construction, and too much of it makes GRPO groups zero-variance.

`train_sft(cfg, data, out)` (needs trl, peft, datasets) is the Modal app's `sft` function: TRL SFTTrainer with
assistant-only loss and the [lora] adapter, settings from the config's [sft] table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import bank as B
from . import reward as R

DEFAULT_SFT = {"learning_rate": 1e-4, "num_train_epochs": 2, "per_device_train_batch_size": 1,
               "gradient_accumulation_steps": 16, "max_length": 32768, "assistant_only_loss": True,
               "gradient_checkpointing": True, "bf16": True, "logging_steps": 5, "save_strategy": "epoch",
               "lr_scheduler_type": "cosine", "warmup_ratio": 0.05, "report_to": "none"}


def transcript(task, policy: str = "scripted", cfg: R.RewardConfig | None = None) -> dict:
    """One task's turn played by `policy` through the episode: the chat transcript (module doc) and its reward."""
    from .episode import TaskEpisode
    from .policies import play
    ep = TaskEpisode(task)
    obs = ep.reset()
    play(ep, policy)
    res = ep.finish(reward_cfg=cfg)
    msgs = []
    for m in ep.messages:
        m = dict(m)
        if m["role"] == "assistant" and m.get("tool_calls"):
            m["content"] = m.get("content") or ""
        msgs.append(m)
    return {"messages": msgs, "tools": json.dumps(obs["tools"]), "task_id": task.task_id, "role": task.role,
            "style": task.style, "month": task.month, "policy": policy, "reward": res["reward"],
            "counts": res["counts"]}


def _job(job: dict) -> list[dict]:
    bank = B.TaskBank(job["bank"], root=job["root"])
    cfg = R.RewardConfig(**job["reward"])
    out = []
    for tid in job["tasks"]:
        best = None
        for p in job["policies"]:
            try:
                rec = transcript(bank.get(tid), p, cfg)
            except Exception as e:                    # a policy that fails on a task is skipped, not fatal
                print(f"warning: {tid} {p}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            c = rec["counts"]
            if c["failed"] or c["invalid"] or not c["ended"] or rec["reward"] < job["min_reward"]:
                continue
            if best is None or rec["reward"] > best["reward"]:
                best = rec
        if best is not None:
            out.append(best)
    return out


def export(bank, out, *, split: str = "train", roles=None, styles=None, exclude_seeds=None, limit=None,
           policies=("oracle", "scripted"), min_reward: float = 0.0, workers: int = 4, root=B.TASKS_DIR,
           reward_cfg: R.RewardConfig | None = None, verbose: bool = True) -> dict:
    """Write the transcripts JSONL (module doc); returns counts."""
    from ..bench.envs import _isolated_map
    from .trl_env import select
    b = bank if isinstance(bank, B.TaskBank) else B.TaskBank(bank, root=root)
    tasks = select(b, split, roles=roles, styles=styles, exclude_seeds=exclude_seeds, limit=limit)
    groups: dict[str, list[str]] = {}
    for t in tasks:
        groups.setdefault(f"{t.config}:{t.seed}", []).append(t.task_id)
    cfg = reward_cfg or R.RewardConfig()
    jobs = [{"bank": str(b.dir), "root": str(Path(b.dir).parent), "tasks": ids, "policies": list(policies),
             "min_reward": float(min_reward), "reward": cfg.as_dict()} for ids in groups.values()]
    t0 = time.time()
    res = _isolated_map(_job, jobs, workers) if workers > 1 and len(jobs) > 1 else [_job(j) for j in jobs]
    rows = [r for rs in res for r in rs]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":"), default=str) + "\n")
    from collections import Counter
    summary = {"tasks": len(tasks), "kept": len(rows), "by_policy": dict(Counter(r["policy"] for r in rows)),
               "by_role": dict(Counter(r["role"] for r in rows)),
               "mean_reward": round(sum(r["reward"] for r in rows) / len(rows), 4) if rows else None,
               "mean_calls": round(sum(r["counts"]["calls"] for r in rows) / len(rows), 2) if rows else None,
               "seconds": round(time.time() - t0, 1), "out": str(out)}
    if verbose:
        print(json.dumps(summary))
    return summary


def train_sft(cfg: dict, data: str, out: str) -> dict:
    """TRL SFTTrainer on the transcripts: the config's [model] and [lora], [sft] over DEFAULT_SFT."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    from .train import lora_config
    m = cfg["model"]
    ds = load_dataset("json", data_files=data, split="train").select_columns(["messages", "tools"])
    kw = {**DEFAULT_SFT, **(cfg.get("sft") or {})}
    args = SFTConfig(output_dir=out, model_init_kwargs={"dtype": m.get("dtype", "bfloat16")}, **kw)
    tok = AutoTokenizer.from_pretrained(m.get("tokenizer") or m["id"])
    trainer = SFTTrainer(model=m["id"], args=args, train_dataset=ds, processing_class=tok,
                         peft_config=lora_config(cfg))
    res = trainer.train()
    trainer.save_model(str(Path(out) / "final"))
    tok.save_pretrained(str(Path(out) / "final"))
    return {"output_dir": out, "rows": len(ds), "train_loss": res.training_loss, "steps": trainer.state.global_step}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m civitas.rl.sft", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="scripted-policy turns as chat transcripts (JSONL)")
    e.add_argument("--bank", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--split", default="train")
    e.add_argument("--roles", default=None)
    e.add_argument("--styles", default=None)
    e.add_argument("--exclude-seeds", default="", help="e.g. 290-299 (the dev seeds)")
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--policies", default="oracle,scripted")
    e.add_argument("--min-reward", type=float, default=0.0)
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--root", default=str(B.TASKS_DIR))
    a = p.parse_args(argv)
    from ..cli import _seeds
    export(a.bank, a.out, split=a.split, roles=a.roles.split(",") if a.roles else None,
           styles=a.styles.split(",") if a.styles else None,
           exclude_seeds=_seeds(a.exclude_seeds) if a.exclude_seeds else None, limit=a.limit,
           policies=tuple(a.policies.split(",")), min_reward=a.min_reward, workers=a.workers, root=a.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
