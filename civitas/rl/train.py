"""GRPO on bank tasks with TRL (trl_env.py): the trainer the Modal app runs, and the local smoke test.

    python -m civitas.rl.train -c configs/rl_qwen3_4b.toml --print-config        # the merged config, no trl needed
    python -m civitas.rl.train -c configs/rl_smoke_local.toml                    # 2 GRPO steps on a laptop
    python -m civitas.rl.train -c configs/rl_qwen3_4b.toml --vllm-url http://127.0.0.1:8000 --output-dir /vol/...

A config (configs/rl_*.toml) has [model], [data], [env], [reward], [lora], [grpo] (trl.GRPOConfig as is),
[civitas], [vllm], [eval], [modal] and [estimate]; `--set section.key=value` overrides a key (JSON values parsed).
The run writes to `output_dir` (default runs/_rl_train/<name>): TRL's checkpoints (resumed from the latest unless
--no-resume), `run.json` (config, versions, the dataset's composition), `metrics.jsonl` (every log line, train and
eval), and `final/` (the LoRA adapter or the full model, with the tokenizer).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from . import bank as B
from . import reward as R
from . import trl_env as T

OUT_DIR = Path("runs") / "_rl_train"
SECTIONS = ("model", "data", "env", "reward", "lora", "grpo", "civitas", "vllm", "eval", "sft", "modal", "estimate")


# --------------------------------------------------------------------------------------------------- config
def _value(v: str):
    try:
        return json.loads(v)
    except ValueError:
        return v


def load(path, overrides=()) -> dict:
    """The TOML config with `section.key=value` overrides applied."""
    cfg = tomllib.loads(Path(path).read_text())
    cfg.setdefault("name", Path(path).stem)
    for s in SECTIONS:
        cfg.setdefault(s, {})
    for o in overrides or ():
        if "=" not in o:
            raise ValueError(f"--set wants section.key=value, got {o!r}")
        k, v = o.split("=", 1)
        parts = k.strip().split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = _value(v)
    return cfg


def env_config(cfg: dict) -> T.EnvConfig:
    r = dict(cfg.get("reward") or {})
    pm = r.pop("p_malformed", None)
    fields = R.RewardConfig().as_dict()
    bad = sorted(set(r) - set(fields))
    if bad:
        raise ValueError(f"[reward]: unknown keys {bad}; known: {sorted(fields)} and p_malformed")
    e = cfg.get("env") or {}
    return T.EnvConfig(reward=R.RewardConfig(**r), p_malformed=pm, end_at_action_cap=bool(e.get("end_at_action_cap")),
                       reward_workers=int(e.get("reward_workers", 0)), check_prompt=bool(e.get("check_prompt", True)))


def data_sets(cfg: dict, root=B.TASKS_DIR, ecfg: T.EnvConfig | None = None, cache_dir=None) -> tuple:
    """(train TaskSet, dev TaskSet or None): the training split without the dev seeds, and up to `eval_limit` dev
    tasks (the periodic eval; the test split stays for `civitas rl eval`). With `cache_dir` the sets are saved there
    (task_set_<digest of [data]>.json) and reused by a restarted run of the same bank."""
    import hashlib
    d = cfg["data"]
    ecfg = ecfg or env_config(cfg)
    bank = B.TaskBank(d["bank"], root=root)
    digest = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]
    cache = (lambda which: Path(cache_dir) / f"task_set_{which}_{digest}.json") if cache_dir else None
    pick = dict(roles=d.get("roles") or None, styles=d.get("styles") or None, months=d.get("months") or None)
    dev = d.get("dev_seeds") or []

    def get(which, **kw):
        ts = T.TaskSet.load(cache(which), bank, ecfg) if cache else None
        if ts is None:
            ts = T.task_set(bank, d.get("split", "train"), cfg=ecfg, **pick, **kw)
            if cache:
                ts.save(cache(which))
        return ts

    train = get("train", exclude_seeds=dev, limit=d.get("limit") or None, sample_seed=d.get("sample_seed"),
                repeat=int(d.get("repeat", 1)))
    evl = None
    if dev and int(d.get("eval_limit", 0)) > 0:
        try:
            evl = get("dev", seeds=dev, limit=int(d["eval_limit"]))
        except B.BankError:
            print(f"warning: no dev tasks for seeds {dev[0]}-{dev[-1]} in bank {bank.name}: no periodic eval",
                  file=sys.stderr)
    return train, evl


def grpo_kwargs(cfg: dict, output_dir: str, *, vllm_url: str | None = None, has_eval: bool = True) -> dict:
    """trl.GRPOConfig keyword arguments: [grpo] as is, the output dir, server-mode vLLM when `vllm_url`, and no
    periodic eval without a dev set."""
    kw = copy.deepcopy(cfg.get("grpo") or {})
    kw["output_dir"] = output_dir
    if vllm_url:
        kw.update(use_vllm=True, vllm_mode="server", vllm_server_base_url=vllm_url)
        if (cfg.get("vllm") or {}).get("max_model_len"):
            kw.setdefault("vllm_max_model_length", int(cfg["vllm"]["max_model_len"]))
    if not has_eval:
        kw["eval_strategy"] = "no"
        for k in ("eval_steps", "num_generations_eval", "per_device_eval_batch_size"):
            kw.pop(k, None)
    return kw


def lora_config(cfg: dict):
    lc = dict(cfg.get("lora") or {})
    if not lc.pop("enabled", True):
        return None
    from peft import LoraConfig
    lc.setdefault("task_type", "CAUSAL_LM")
    return LoraConfig(**lc)


def latest_checkpoint(output_dir) -> str | None:
    """The newest checkpoint-<step> folder with a trainer_state.json (a complete save)."""
    p = Path(output_dir)
    best = None
    for c in p.glob("checkpoint-*") if p.is_dir() else []:
        m = re.fullmatch(r"checkpoint-(\d+)", c.name)
        if m and (c / "trainer_state.json").exists() and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), c)
    return str(best[1]) if best else None


def versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for m in ("torch", "transformers", "trl", "peft", "datasets", "accelerate", "vllm"):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            out[m] = None
    return out


# -------------------------------------------------------------------------------------------------- trainer
class _TorchLogProbs:
    """A plain-PyTorch stand-in for TRL's Triton ChunkedLogProbFunction (trl/kernels/chunked_logprob.py: same
    arguments, same five outputs), for machines without Triton (macOS). trl 1.15's GRPOTrainer always scores
    completions through that kernel and fails there (`'NoneType' object has no attribute 'apply'`). This one
    materializes the [tokens, vocab] logits: fine for a laptop smoke, not for training (Linux has the kernel)."""

    @staticmethod
    def apply(hidden, weight, bias, targets, temperature, chunk_size, final_logit_softcapping=None, logit_scale=1.0,
              outputs=("entropy", "log_sum_sq_probs", "mean_logits", "is_top1")):
        import torch
        z = torch.nn.functional.linear(hidden, weight, bias).float() * logit_scale
        if final_logit_softcapping is not None:
            z = final_logit_softcapping * torch.tanh(z / final_logit_softcapping)
        z = z / temperature
        logp = torch.log_softmax(z, dim=-1)
        out = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        p = logp.exp()
        entropy = -(p * logp).sum(-1) if "entropy" in outputs else None
        with torch.no_grad():
            lssp = torch.logsumexp(2 * logp, dim=-1) if "log_sum_sq_probs" in outputs else None
            mean = z.mean(-1) if "mean_logits" in outputs else None
            top1 = (z.argmax(-1) == targets) if "is_top1" in outputs else None
        return out, entropy, lssp, mean, top1


def patch_trl_for_cpu() -> bool:
    """Install _TorchLogProbs where TRL has no Triton kernel (macOS); True if it did."""
    from trl.trainer import utils as U
    if getattr(U, "_ChunkedLogProbFunction", 1) is None:
        U._ChunkedLogProbFunction = _TorchLogProbs
        return True
    return False


def _callback(output_dir: Path, on_save=None):
    from transformers import TrainerCallback

    class CivitasLog(TrainerCallback):
        """metrics.jsonl (every log line) and a hook after each checkpoint (the Modal volume's commit)."""

        def on_log(self, args, state, control, logs=None, **kw):
            if state.is_world_process_zero and logs:
                with open(output_dir / "metrics.jsonl", "a") as f:
                    f.write(json.dumps({"step": state.global_step, "time": round(time.time(), 1), **logs},
                                       default=float) + "\n")

        def on_save(self, args, state, control, **kw):
            if on_save is not None and state.is_world_process_zero:
                on_save(state.global_step)

    return CivitasLog()


def build(cfg: dict, *, output_dir: str | Path | None = None, vllm_url: str | None = None, root=B.TASKS_DIR,
          on_save=None):
    """(trainer, train TaskSet, dev TaskSet): a CivitasGRPOTrainer ready to train."""
    from transformers import AutoTokenizer
    from trl import GRPOConfig
    if patch_trl_for_cpu():
        print("note: no Triton here (macOS?): TRL's chunked log-prob kernel replaced by plain PyTorch "
              "(train.py:_TorchLogProbs; fine for a smoke, slow and memory-hungry for real training)", file=sys.stderr)
    out = Path(output_dir or OUT_DIR / cfg["name"])
    out.mkdir(parents=True, exist_ok=True)
    ecfg = env_config(cfg)
    train, dev = data_sets(cfg, root=root, ecfg=ecfg, cache_dir=out)
    m = cfg["model"]
    kw = grpo_kwargs(cfg, str(out), vllm_url=vllm_url, has_eval=dev is not None)
    mik = dict(kw.pop("model_init_kwargs", {}) or {})
    mik.setdefault("dtype", m.get("dtype", "bfloat16"))
    if m.get("attn_implementation"):
        mik.setdefault("attn_implementation", m["attn_implementation"])
    args = GRPOConfig(model_init_kwargs=mik, **kw)
    tok = AutoTokenizer.from_pretrained(m.get("tokenizer") or m["id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    if dev is not None:                    # the dev rows' tool sets get environments in the trainer too
        for k, s in dev.schemas.items():
            train.schemas.setdefault(k, s)
    reward = T.CivitasReward(ecfg)
    cls = T.trainer_class()
    trainer = cls(model=m["id"], args=args, reward_funcs=[reward], train_dataset=train.dataset(),
                  eval_dataset=dev.dataset() if dev is not None else None, processing_class=tok,
                  peft_config=lora_config(cfg), task_set=train,
                  drop_zero_std=bool((cfg.get("civitas") or {}).get("drop_zero_std", True)),
                  callbacks=[_callback(out, on_save)])
    (out / "run.json").write_text(json.dumps({"config": cfg, "versions": versions(), "train": train.composition(),
                                              "dev": dev.composition() if dev else None, "env": ecfg.as_dict(),
                                              "trl_tested": T.TRL_TESTED, "started": time.time()},
                                             indent=2, default=str))
    return trainer, train, dev


def train(cfg: dict, *, output_dir=None, vllm_url=None, resume: bool = True, root=B.TASKS_DIR, on_save=None) -> dict:
    """Build and train; saves `final/`; returns the last log line and where things are."""
    out = Path(output_dir or OUT_DIR / cfg["name"])
    trainer, tr, dev = build(cfg, output_dir=out, vllm_url=vllm_url, root=root, on_save=on_save)
    ckpt = latest_checkpoint(out) if resume else None
    if ckpt:
        print(f"resuming from {ckpt}", flush=True)
    print(f"GRPO {cfg['model']['id']}: {len(tr.rows)} train rows {tr.composition()['by_role']}, "
          f"dev {len(dev.rows) if dev else 0}, -> {out}", flush=True)
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=ckpt)
    trainer.save_model(str(out / "final"))
    trainer.processing_class.save_pretrained(str(out / "final"))
    for f in trainer.reward_funcs:
        if isinstance(f, T.CivitasReward):
            f.close()
    if on_save is not None:
        on_save(trainer.state.global_step)
    logs = [h for h in trainer.state.log_history if "loss" in h or "reward" in h]
    return {"output_dir": str(out), "final": str(out / "final"), "steps": trainer.state.global_step,
            "seconds": round(time.time() - t0, 1), "train_loss": getattr(result, "training_loss", None),
            "last_log": logs[-1] if logs else None}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m civitas.rl.train", description=__doc__.split("\n\n")[0])
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--set", action="append", default=[], help="section.key=value (JSON values parsed)")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--vllm-url", default=None, help="a running `vllm serve` for server mode, e.g. http://127.0.0.1:8000")
    p.add_argument("--root", default=str(B.TASKS_DIR), help="the folder holding the banks")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--print-config", action="store_true", help="print the merged config and GRPOConfig kwargs; exit")
    a = p.parse_args(argv)
    cfg = load(a.config, a.set)
    if a.print_config:
        print(json.dumps({"config": cfg, "grpo_kwargs": grpo_kwargs(cfg, a.output_dir or str(OUT_DIR / cfg["name"]),
                                                                    vllm_url=a.vllm_url)}, indent=2, default=str))
        return 0
    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
    res = train(cfg, output_dir=a.output_dir, vllm_url=a.vllm_url, resume=not a.no_resume, root=a.root)
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
