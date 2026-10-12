"""Modal app: GRPO-train a model on civitas RL tasks (TRL 1.15, vLLM server mode) and evaluate it. Not run by any test
or command here: every function below spends money on Modal. docs/RL.md "RL training" has the pipeline.

    python -m civitas.rl.modal_app --print-config [-c configs/rl_qwen3_4b.toml] [--set grpo.max_steps=50]
        # the plan, the image pins, the exact launch commands and the cost estimate: no Modal account needed

    modal volume put civitas-rl runs/_tasks/rl-train /tasks/rl-train          # upload a bank built locally (free)
    modal run civitas/rl/modal_app.py::check                                  # image + volume check (CPU, cents)
    modal run --detach civitas/rl/modal_app.py --action baseline              # the base model on the test split
    modal run --detach civitas/rl/modal_app.py --action sft                   # optional SFT warm start
    modal run --detach civitas/rl/modal_app.py --action train                 # GRPO (resumes after preemption)
    modal run --detach civitas/rl/modal_app.py --action evaluate --run qwen3-4b-planning-v1
    modal volume get civitas-rl /evals ./runs/_rl_evals_modal                  # results back, then compare locally

The config is read when the module is imported (it sets GPUs, timeouts and the image): CIVITAS_RL_CONFIG names it
(default configs/rl_qwen3_4b.toml), and `--set` style overrides go in CIVITAS_RL_SET (";"-separated), so
`CIVITAS_RL_CONFIG=configs/my.toml modal run --detach civitas/rl/modal_app.py --action train`.

Layout. One Volume (`[modal].volume`) at /vol: tasks/<bank>/ (banks), train/<run>/ (checkpoints, metrics.jsonl,
run.json, final/), sft/<run>/, evals/<name>_<stamp>/ (`civitas rl eval` output). A second Volume caches Hugging Face
downloads at /hf. The repo is mounted at /repo (code changes need no image rebuild) and is on PYTHONPATH.

- `train` (`[modal].train_gpu`, "H100:2"): GPU 0 runs `vllm serve` with NCCL weight transfer (TRL server mode: the
  LoRA is merged and the weights pushed after each step), GPU 1 the trainer (train.py). Preemptible like every Modal
  GPU function: `retries` restarts it, and train.py resumes from the newest complete checkpoint on the volume
  (`save_steps`; the volume is committed after each save). One call runs at most 24 h (Modal's limit); a longer run
  is the same command again (it resumes).
- `baseline` / `evaluate` (`[modal].eval_gpu`): `vllm serve` with the hermes tool parser (the same chat template as
  training) and `civitas rl eval` on the test split against it (rl/evaluate.py: the fleet's own ToolAgent loop), for
  evaluate with the trained LoRA served as `<run>` (--enable-lora) or a full fine-tune served directly. With
  `suite=...` also a guided eval suite (`civitas eval -s configs/eval_v2.toml`) on the served model.
- `sft` (`[sft]`, optional): TRL SFTTrainer on scripted-policy transcripts (rl/sft.py export), LoRA, one GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REMOTE_REPO = "/repo"


def _repo() -> Path:
    """The repo root: $CIVITAS_REPO (set in the image), else this file's checkout (Modal may run the file as a script
    from another folder in the container, where parents[2] isn't the repo)."""
    if os.environ.get("CIVITAS_REPO"):
        return Path(os.environ["CIVITAS_REPO"])
    p = Path(__file__).resolve()
    return p.parents[2] if len(p.parents) > 2 and (p.parents[2] / "civitas" / "rl").is_dir() else Path(REMOTE_REPO)


REPO = _repo()
CONFIG = os.environ.get("CIVITAS_RL_CONFIG", "configs/rl_qwen3_4b.toml")
VOL, HF = "/vol", "/hf"
IGNORE = ["runs", ".venv", ".venv-*", ".git", "**/__pycache__", ".claude", "*.pyc", ".DS_Store", ".pytest_cache"]


def load_cfg(path: str = CONFIG, overrides=None) -> dict:
    sys.path.insert(0, str(REPO)) if str(REPO) not in sys.path else None
    from civitas.rl.train import load
    sets = overrides if overrides is not None else [s for s in os.environ.get("CIVITAS_RL_SET", "").split(";") if s]
    p = Path(path)
    return load(p if p.is_absolute() or p.exists() else REPO / p, sets)


# ------------------------------------------------------------------------------------------------- the plan
def vllm_command(cfg: dict, *, model: str | None = None, for_training: bool = False, lora: dict | None = None,
                 served_name: str | None = None) -> list[str]:
    """`vllm serve` for training (NCCL weight transfer, TRL's required logprob flags) or for evals (an
    OpenAI-compatible endpoint with the tool parser; `lora` = {name: path} serves adapters)."""
    v, m = cfg.get("vllm") or {}, cfg["model"]
    cmd = ["vllm", "serve", model or m["id"], "--port", str(v.get("port", 8000)), "--host", "127.0.0.1",
           "--gpu-memory-utilization", str(v.get("gpu_memory_utilization", 0.9)),
           "--max-model-len", str(v.get("max_model_len", 32768)), "--dtype", m.get("dtype", "bfloat16"),
           *map(str, v.get("extra_args") or [])]
    if for_training:
        cmd += ["--weight-transfer-config", json.dumps({"backend": "nccl"}), "--logprobs-mode", "processed_logprobs",
                "--max-logprobs", "-1"]
    else:
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", v.get("tool_call_parser", "hermes")]
        if served_name:
            cmd += ["--served-model-name", served_name]
        if lora:
            r = int((cfg.get("lora") or {}).get("r", 32))
            cmd += ["--enable-lora", "--max-lora-rank", str(max(16, r)), "--lora-modules",
                    *[f"{k}={p}" for k, p in lora.items()]]
    return cmd


def estimate(cfg: dict) -> dict:
    """Wall time and cost of a training run from [estimate] and [grpo] (back-of-envelope; measure on step 1)."""
    g, e, mo = cfg["grpo"], cfg.get("estimate") or {}, cfg.get("modal") or {}
    steps = int(g.get("max_steps", 300))
    gpus = int(str(mo.get("train_gpu", "H100:2")).split(":")[-1]) if ":" in str(mo.get("train_gpu", "")) else 1
    evals = steps // int(g["eval_steps"]) if g.get("eval_strategy", "no") != "no" and g.get("eval_steps") else 0
    minutes = steps * float(e.get("step_minutes", 3.0)) + evals * float(e.get("eval_minutes", 3.0)) + \
        float(e.get("startup_minutes", 15))
    price = float(mo.get("gpu_hour_usd", 3.95))
    per_step = int(g.get("per_device_train_batch_size", 1)) * int(g.get("gradient_accumulation_steps", 1))
    out = {"steps": steps, "completions_per_step": per_step,
           "tasks_per_step": per_step // max(1, int(g.get("num_generations", 8))), "periodic_evals": evals,
           "hours": round(minutes / 60, 1), "gpus": gpus, "gpu_hour_usd": price,
           "train_usd": round(minutes / 60 * gpus * price, 0),
           "calls_of_24h": max(1, -(-int(minutes) // (24 * 60)))}
    ev = cfg.get("eval") or {}
    out["eval_usd_each"] = round(1.0 * price, 0)          # ~1 GPU-hour: image, model load, ~100-200 turns
    out["budget_usd"] = {"baseline": f"{out['eval_usd_each']:.0f}-{3 * out['eval_usd_each']:.0f}",
                         "sft (optional)": f"{2 * price:.0f}-{6 * price:.0f}",
                         "grpo": f"{out['train_usd']:.0f}",
                         "debug / smoke runs": f"{12 * price:.0f}-{40 * price:.0f}",
                         "final evaluate": f"{out['eval_usd_each']:.0f}-{3 * out['eval_usd_each']:.0f}"}
    out["eval_samples"] = ev.get("n", 2)
    return out


def plan(cfg: dict) -> dict:
    mo = cfg.get("modal") or {}
    run = cfg["name"]
    env = f"CIVITAS_RL_CONFIG={CONFIG} " if CONFIG != "configs/rl_qwen3_4b.toml" else ""
    return {
        "config": CONFIG, "run": run, "model": cfg["model"]["id"],
        "data": {k: cfg["data"].get(k) for k in ("bank", "split", "roles", "styles", "dev_seeds", "test_split")},
        "image": {"python": mo.get("python", "3.12"), "vllm": mo.get("vllm"), "trl": mo.get("trl"),
                  "peft": mo.get("peft"), "extra": mo.get("extra_pip"),
                  "note": "torch and transformers as vLLM resolves them; `modal run ...::check` prints the versions"},
        "gpus": {"train": mo.get("train_gpu", "H100:2"), "eval": mo.get("eval_gpu", "H100")},
        "volumes": {"data": f"{mo.get('volume', 'civitas-rl')} at {VOL}", "hf": f"{mo.get('hf_volume')} at {HF}"},
        "timeout_hours": mo.get("timeout_hours", 24), "retries": mo.get("retries", 10),
        "vllm_train": " ".join(shlex.quote(x) for x in vllm_command(cfg, for_training=True)),
        "vllm_eval": " ".join(shlex.quote(x) for x in vllm_command(cfg)),
        "commands": [
            f"civitas rl bank build --name {cfg['data']['bank']} -c configs/fleet_smoke_late.toml "
            f"-c configs/fleet_open_late.toml --seeds 100-299,300-339 --workers 4   # locally, free",
            f"modal volume create {mo.get('volume', 'civitas-rl')}; "
            f"modal volume put {mo.get('volume', 'civitas-rl')} runs/_tasks/{cfg['data']['bank']} "
            f"/tasks/{cfg['data']['bank']}",
            f"{env}modal run civitas/rl/modal_app.py::check",
            f"{env}modal run --detach civitas/rl/modal_app.py --action baseline",
            f"{env}modal run --detach civitas/rl/modal_app.py --action sft          # optional",
            f"{env}modal run --detach civitas/rl/modal_app.py --action train",
            f"{env}modal run --detach civitas/rl/modal_app.py --action evaluate --run {run}",
            f"modal volume get {mo.get('volume', 'civitas-rl')} /evals runs/_rl_evals_modal",
        ],
        "estimate": estimate(cfg),
    }


# ------------------------------------------------------------------------------------------ remote helpers
def _wait_http(url: str, proc: subprocess.Popen, timeout: float = 1800.0) -> None:
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm exited with {proc.returncode} before {url} answered")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(5)
    raise TimeoutError(f"{url} didn't answer in {timeout:.0f} s")


def _serve(cmd: list[str], port: int, gpu: str | None = None, env: dict | None = None) -> subprocess.Popen:
    e = {**os.environ, **(env or {})}
    if gpu is not None:
        e["CUDA_VISIBLE_DEVICES"] = gpu
    print("$", " ".join(shlex.quote(x) for x in cmd), flush=True)
    proc = subprocess.Popen(cmd, env=e)
    _wait_http(f"http://127.0.0.1:{port}/health", proc)
    return proc


def _rl_eval(cfg: dict, model: str, name: str, *, roles=None, out=f"{VOL}/evals", base_url=None) -> dict:
    from civitas.rl import evaluate as E
    from civitas.rl.train import env_config
    ev, d = cfg.get("eval") or {}, cfg["data"]
    port = (cfg.get("vllm") or {}).get("port", 8000)
    return E.run(d["bank"], base_url=base_url or f"http://127.0.0.1:{port}/v1", model=model,
                 split=d.get("test_split", "test"), n=int(ev.get("n", 2)), limit=int(ev.get("limit", 0)) or None,
                 roles=None if ev.get("all_roles") else (roles or d.get("roles") or None),
                 styles=d.get("styles") or None, temperature=ev.get("temperature"),
                 max_tokens=ev.get("max_tokens"), workers=int(ev.get("workers", 16)),
                 reward_cfg=env_config(cfg).reward, root=f"{VOL}/tasks", out_dir=out, name=name)


def _suite(suite: str, model: str, port: int) -> int:
    """A guided eval suite on the served model (civitas eval: parallel `civitas run`s against the endpoint)."""
    cmd = [sys.executable, "-m", "civitas.cli", "eval", "-s", suite, "--model", model, "--provider", "vllm",
           "--set", f"providers.vllm.base_url=http://127.0.0.1:{port}/v1", "--set", "providers.vllm.api_key_env=",
           "--runs-dir", f"{VOL}/runs"]
    print("$", " ".join(shlex.quote(x) for x in cmd), flush=True)
    return subprocess.call(cmd, cwd=REMOTE_REPO)


# ----------------------------------------------------------------------------------------------- the app
try:
    import modal
except ImportError:                                   # --print-config and the tests need no Modal
    modal = None

if modal is not None:
    _cfg = load_cfg()
    _mo = _cfg.get("modal") or {}
    app = modal.App(_mo.get("app", "civitas-rl"))
    vol = modal.Volume.from_name(_mo.get("volume", "civitas-rl"), create_if_missing=True)
    hf_vol = modal.Volume.from_name(_mo.get("hf_volume", "civitas-hf-cache"), create_if_missing=True)
    pins = [f"vllm=={_mo['vllm']}", f"trl=={_mo['trl']}", f"peft=={_mo['peft']}", *(_mo.get("extra_pip") or [])]
    image = (modal.Image.debian_slim(python_version=str(_mo.get("python", "3.12")))
             .apt_install("git")
             .uv_pip_install(*pins, "numpy", "hf_transfer")          # torch / transformers: as vLLM resolves them
             .env({"HF_HOME": HF, "HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_SERVER_DEV_MODE": "1",
                   "TRL_EXPERIMENTAL_SILENCE": "1", "PYTHONPATH": REMOTE_REPO, "CIVITAS_REPO": REMOTE_REPO,
                   "TOKENIZERS_PARALLELISM": "false"})
             .workdir(REMOTE_REPO)
             .add_local_dir(str(REPO), REMOTE_REPO, ignore=IGNORE))
    volumes = {VOL: vol, HF: hf_vol}
    secrets = [modal.Secret.from_name(n) for n in _mo.get("secrets", [])]
    retries = modal.Retries(max_retries=int(_mo.get("retries", 10)), initial_delay=1.0, backoff_coefficient=1.0)
    timeout = int(float(_mo.get("timeout_hours", 24)) * 3600)
    cpu, memory = float(_mo.get("cpu", 16)), int(_mo.get("memory_gb", 128)) * 1024

    @app.function(image=image, volumes=volumes, cpu=2, timeout=1800)
    def check(config: str = CONFIG) -> dict:
        """The image's versions, CUDA-free imports, and the bank on the volume (CPU only)."""
        from civitas.rl import bank as B
        from civitas.rl.train import versions
        cfg = load_cfg(config)
        out = {"versions": versions(), "banks": sorted(p.name for p in Path(f"{VOL}/tasks").glob("*"))}
        try:
            b = B.TaskBank(cfg["data"]["bank"], root=f"{VOL}/tasks")
            out["bank"] = b.manifest.get("composition")
            from civitas.rl.trl_env import task_set
            ts = task_set(b, cfg["data"].get("split", "train"), roles=cfg["data"].get("roles") or None, limit=4)
            out["task_set"] = ts.composition()
        except Exception as e:
            out["bank_error"] = f"{type(e).__name__}: {e}"
        print(json.dumps(out, indent=2, default=str))
        return out

    @app.function(image=image, volumes=volumes, gpu=_mo.get("eval_gpu", "H100"), cpu=cpu, memory=memory,
                  timeout=timeout, retries=modal.Retries(max_retries=2, initial_delay=1.0), secrets=secrets,
                  single_use_containers=True)
    def baseline(config: str = CONFIG, suite: str = "", name: str = "") -> dict:
        """The base model served by vLLM; `civitas rl eval` on the test split (and a guided eval suite if asked)."""
        cfg = load_cfg(config)
        port = int((cfg.get("vllm") or {}).get("port", 8000))
        proc = _serve(vllm_command(cfg), port)
        try:
            model = cfg["model"]["id"]
            s = _rl_eval(cfg, model, name or f"baseline_{cfg['name']}")
            if suite:
                s["suite_exit"] = _suite(suite, model, port)
        finally:
            proc.terminate()
            vol.commit()
        return {k: s.get(k) for k in ("reward", "vs_noop", "by_role", "per_task", "ended_share", "dir", "suite_exit")}

    @app.function(image=image, volumes=volumes, gpu=_mo.get("train_gpu", "H100:2"), cpu=cpu, memory=memory,
                  timeout=timeout, retries=retries, secrets=secrets, single_use_containers=True)
    def train(config: str = CONFIG, run: str = "") -> dict:
        """GRPO in TRL server mode: vLLM on GPU 0, the trainer on GPU 1; resumes from the volume's checkpoints."""
        from civitas.rl import train as TR
        cfg = load_cfg(config)
        out = f"{VOL}/train/{run or cfg['name']}"
        Path(out).mkdir(parents=True, exist_ok=True)
        port = int((cfg.get("vllm") or {}).get("port", 8000))
        proc = _serve(vllm_command(cfg, for_training=True), port, gpu="0")
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        try:
            res = TR.train(cfg, output_dir=out, vllm_url=f"http://127.0.0.1:{port}", resume=True,
                           root=f"{VOL}/tasks", on_save=lambda step: vol.commit())
        finally:
            proc.terminate()
            vol.commit()
        return res

    @app.function(image=image, volumes=volumes, gpu=_mo.get("eval_gpu", "H100"), cpu=cpu, memory=memory,
                  timeout=timeout, retries=modal.Retries(max_retries=2, initial_delay=1.0), secrets=secrets,
                  single_use_containers=True)
    def evaluate(config: str = CONFIG, run: str = "", checkpoint: str = "final", suite: str = "") -> dict:
        """A trained run's adapter (or full model) served by vLLM; `civitas rl eval` on the test split."""
        cfg = load_cfg(config)
        run = run or cfg["name"]
        path = Path(f"{VOL}/train/{run}") / checkpoint
        if not path.exists():
            raise FileNotFoundError(f"{path} isn't on the volume (runs: {sorted(p.name for p in Path(f'{VOL}/train').glob('*'))})")
        port = int((cfg.get("vllm") or {}).get("port", 8000))
        is_lora = (path / "adapter_config.json").exists()
        name = f"{run}-{checkpoint}"
        cmd = vllm_command(cfg, lora={name: str(path)}) if is_lora else \
            vllm_command(cfg, model=str(path), served_name=name)
        proc = _serve(cmd, port)
        try:
            s = _rl_eval(cfg, name, f"eval_{name}")
            if suite:
                s["suite_exit"] = _suite(suite, name, port)
        finally:
            proc.terminate()
            vol.commit()
        return {k: s.get(k) for k in ("reward", "vs_noop", "by_role", "per_task", "ended_share", "dir", "suite_exit")}

    @app.function(image=image, volumes=volumes, gpu=_mo.get("eval_gpu", "H100"), cpu=cpu, memory=memory,
                  timeout=timeout, retries=retries, secrets=secrets, single_use_containers=True)
    def sft(config: str = CONFIG, data: str = "", run: str = "") -> dict:
        """SFT warm start on exported transcripts (rl/sft.py): LoRA, assistant-only loss; the adapter to /vol/sft."""
        from civitas.rl import sft as S
        cfg = load_cfg(config)
        data = data or f"{VOL}/sft/{cfg['data']['bank']}.jsonl"
        if not Path(data).exists():                              # export on the container's CPUs
            S.export(cfg["data"]["bank"], data, split=cfg["data"].get("split", "train"),
                     roles=cfg["data"].get("roles") or None, exclude_seeds=cfg["data"].get("dev_seeds"),
                     root=f"{VOL}/tasks", workers=int(cpu) - 2)
            vol.commit()
        out = f"{VOL}/sft/{run or cfg['name']}"
        res = S.train_sft(cfg, data, out)
        vol.commit()
        return res

    @app.local_entrypoint()
    def main(action: str = "train", config: str = CONFIG, run: str = "", suite: str = "", checkpoint: str = "final"):
        """`modal run --detach civitas/rl/modal_app.py --action train|baseline|evaluate|sft`: spawns the call and
        prints its id (follow it with `modal app logs`); --detach keeps it running after this exits."""
        fn = {"train": lambda: train.spawn(config, run), "baseline": lambda: baseline.spawn(config, suite),
              "evaluate": lambda: evaluate.spawn(config, run, checkpoint, suite),
              "sft": lambda: sft.spawn(config, "", run)}.get(action)
        if fn is None:
            raise SystemExit(f"--action is one of train, baseline, evaluate, sft (got {action!r})")
        call = fn()
        print(f"spawned {action}: {call.object_id}  (modal app logs {_mo.get('app', 'civitas-rl')})")


# --------------------------------------------------------------------------------------------- dry run
def main_cli(argv=None) -> int:
    global CONFIG
    p = argparse.ArgumentParser(prog="python -m civitas.rl.modal_app",
                                description="the Modal plan without Modal: images, GPUs, commands, cost")
    p.add_argument("--print-config", action="store_true", help="print the plan and the cost estimate (default)")
    p.add_argument("-c", "--config", default=CONFIG)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    CONFIG = a.config
    cfg = load_cfg(a.config, a.set)
    pl = plan(cfg)
    if a.json:
        print(json.dumps({"plan": pl, "config": cfg}, indent=2, default=str))
        return 0
    e = pl["estimate"]
    print(f"civitas RL on Modal: {pl['run']} ({pl['model']}), config {pl['config']}")
    print(f"  data: {json.dumps(pl['data'])}")
    print(f"  image: python {pl['image']['python']}, vllm {pl['image']['vllm']}, trl {pl['image']['trl']}, "
          f"peft {pl['image']['peft']}, {', '.join(pl['image']['extra'] or [])} ({pl['image']['note']})")
    print(f"  GPUs: train {pl['gpus']['train']} (GPU 0 vLLM server, GPU 1 trainer), eval {pl['gpus']['eval']}; "
          f"timeout {pl['timeout_hours']} h per call, {pl['retries']} retries (preemption: resume from checkpoint)")
    print(f"  volumes: {pl['volumes']['data']}; {pl['volumes']['hf']}")
    print(f"  vLLM (train): {pl['vllm_train']}")
    print(f"  vLLM (eval):  {pl['vllm_eval']}")
    g = cfg["grpo"]
    print(f"  GRPO: {e['steps']} steps x {e['tasks_per_step']} tasks x {g.get('num_generations')} rollouts, lr "
          f"{g.get('learning_rate')}, beta {g.get('beta')}, eps {g.get('epsilon')}/{g.get('epsilon_high')}, "
          f"T {g.get('temperature')}, max_completion_length {g.get('max_completion_length')}, tool iterations "
          f"{g.get('max_tool_calling_iterations')}, LoRA r{cfg['lora'].get('r')}/a{cfg['lora'].get('lora_alpha')}"
          f"{'' if cfg['lora'].get('enabled', True) else ' (off: full fine-tune)'}")
    print(f"  estimate: ~{e['hours']} h on {e['gpus']} x H100 at ${e['gpu_hour_usd']}/GPU-h = ~${e['train_usd']:.0f} "
          f"for GRPO ({e['periodic_evals']} periodic dev evals; {e['calls_of_24h']} call(s) of <= 24 h)")
    print("  budget: " + ", ".join(f"{k} ${v}" for k, v in e["budget_usd"].items()))
    print("  commands:")
    for c in pl["commands"]:
        print(f"    {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
