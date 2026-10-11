"""Command line: civitas run | resume | yardstick | eval | env | models | bench | serve | leaderboard | prompt |
check-llm | rl"""
from __future__ import annotations

import argparse
import json
import sys
import time


def _seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _load(args):
    from .config import apply_cli_overrides, load_config, warmup_months
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args.agent, args.set)
    if getattr(args, "warmup", None) is not None:
        cfg["game"]["warmup_months"] = args.warmup
    try:
        warmup_months(cfg)
    except ValueError as e:
        sys.exit(f"civitas {args.cmd}: {e}")
    return cfg


def _hub(args):
    """With --serve: the viewer + live API in the background, and the hub remote agents talk through."""
    if not args.serve:
        return None
    from .agents.remote import LiveHub
    from .server import serve
    hub = LiveHub()
    serve(args.runs_dir, args.port, hub=hub, background=True)
    return hub


def _linger(args, out: dict) -> None:
    if args.serve and not args.exit:
        print(f"Run {'paused' if out.get('paused') else 'stopped' if out.get('stopped') else 'finished'}. Viewer still running; Ctrl+C to quit.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


def cmd_run(args):
    from .bench.runner import run_episode
    cfg = _load(args)
    hub = _hub(args)
    out = run_episode(cfg, args.seed, args.runs_dir, hub=hub, verbose=not args.quiet, run_id=args.run_id,
                      yardstick=not args.no_yardstick)
    _linger(args, out)


def cmd_resume(args):
    """Continue a paused supervised run with its (edited) playbooks."""
    from .bench.runner import resume_episode
    hub = _hub(args)
    try:
        out = resume_episode(args.run, args.config, runs_dir=args.runs_dir, agents=args.agent, sets=args.set,
                             hub=hub, verbose=not args.quiet, yardstick=not args.no_yardstick)
    except ValueError as e:
        sys.exit(f"civitas resume: {e}")
    _linger(args, out)


def cmd_yardstick(args):
    """The scripted yardstick of a warm-up run: its game from the same hand-over played by scripted agents (no model
    calls, no logs), written to the run's yardstick.json and printed beside what the run's agents did."""
    from .bench import yardstick as Y
    from .bench.session import run_dir
    d = run_dir(args.run, args.runs_dir)
    try:
        ys = Y.yardstick(d, write=not args.dry_run, verbose=not args.quiet)
    except (ValueError, Y.ModelAgentError) as e:
        sys.exit(f"civitas yardstick: {e}")
    if args.json:
        print(json.dumps(ys, indent=2))
        return
    print(Y.comparison(ys))
    if not args.dry_run:
        print(f"wrote {d / Y.FILE}")


def cmd_eval(args):
    """A model evaluation: a suite of late-start runs (bench/evals.py), its scorecard, or several evals compared."""
    from pathlib import Path
    from .bench import evals as E
    try:
        if args.compare:
            evs = [E.load(d) for d in args.compare]
            for ev in evs:
                if not ev.get("scorecard"):
                    E.finish(ev)
            print(E.compare(evs))
            return
        if args.report:
            ev = E.finish(E.load(args.report))
            print((Path(ev["dir"]) / E.CARD).read_text())
            print(f"wrote {Path(ev['dir']) / E.CARD}")
            return
        if args.resume:
            ev = E.resume(args.resume, workers=args.workers, verbose=not args.quiet)
            _eval_done(ev)
            return
        if args.leaderboard:
            lb = E.leaderboard(args.env, args.runs_dir)
            if args.json:
                print(json.dumps(lb, indent=2, default=str))
                return
            p = E.write_leaderboard(lb, args.runs_dir)
            print(E.leaderboard_markdown(lb))
            print(f"wrote {p}")
            return
        if args.adopt:
            if not args.suite or not args.name:
                sys.exit("civitas eval --adopt: give the suite with -s and name the eval with --name")
            ev = E.adopt(args.adopt, E.load_suite(args.suite), name=args.name, runs_dir=args.runs_dir)
            print((Path(ev["dir"]) / E.CARD).read_text())
            print(f"wrote {ev['dir']}")
            return
        if not args.suite and not args.model:
            sys.exit(f"civitas eval: give a model with --model (a key of configs/models.toml; the suite is "
                     f"{E.DEFAULT_SUITE} unless -s), a suite with -s, or --resume / --report / --compare / "
                     f"--leaderboard")
        ev = E.prepare(args.model, suite=args.suite, name=args.name, allow_drift=args.allow_drift,
                       workers=args.workers, runs_dir=args.runs_dir, provider=args.provider, agents=args.agent,
                       sets=args.set, repeats=args.repeats, stamp=args.stamp, dry_run=args.dry_run,
                       preflight=not args.no_preflight, preflight_result=_preflight_result(args.preflight_result))
        if args.dry_run:
            print(E.dry_run_text(ev, ev["errors"]))
            print()
            print("\n".join(E.prepare_text(ev)))
            if ev["errors"]:
                sys.exit(2)
            return
        if not args.quiet:
            print("\n".join(E.prepare_text(ev)))
        if ev["errors"]:
            sys.exit("civitas eval: refusing to start the eval:\n  - " + "\n  - ".join(ev["errors"]))
        ev = E.start(ev, verbose=not args.quiet)
        _eval_done(ev)
    except E.EvalError as e:
        sys.exit(f"civitas eval: {e}")
    except KeyboardInterrupt:
        sys.exit(130)


def _preflight_result(text: str | None) -> dict | None:
    """--preflight-result: a passed preflight as JSON (the viewer's, so the eval records it instead of 'skipped')."""
    if not text:
        return None
    try:
        pf = json.loads(text)
    except ValueError:
        sys.exit("civitas eval: --preflight-result must be a preflight record as JSON")
    if not isinstance(pf, dict) or pf.get("ok") is not True:
        sys.exit("civitas eval: --preflight-result must be a preflight that passed")
    return pf


def _eval_done(ev: dict) -> None:
    from pathlib import Path
    from .bench import evals as E
    a = (ev.get("scorecard") or {}).get("aggregate") or {}
    print(f"[eval {ev['name']}] {ev['status']}: {a.get('finished', 0)} of {a.get('runs', 0)} runs finished; "
          f"city vs scripted {E.ci_text(a.get('city_delta'))}, beats scripted "
          f"{a.get('beats_scripted', 0)} of {a.get('n_units', 0)} units")
    print(f"scorecard: {Path(ev['dir']) / E.CARD}")
    if ev["status"] != "finished":
        print(f"rerun what didn't finish: civitas eval --resume {ev['dir']}")


def cmd_env(args):
    """Environment versions (bench/envs.py): freeze the suite's probe as a named version, check the code against
    the suite's version, list the versions."""
    from datetime import datetime
    from .bench import envs as V
    from .bench import evals as E
    try:
        if args.action == "list":
            rows = V.list_envs()
            if args.json:
                print(json.dumps(rows, indent=2))
                return
            if not rows:
                print("no environment versions yet (civitas env freeze <name>)")
            for e in rows:
                when = datetime.fromtimestamp(e["created"]).strftime("%Y-%m-%d %H:%M") if e.get("created") else "?"
                print(f"{e['name']:<16} {e.get('fingerprint')}  {when}  commit {e.get('commit') or '?'}  "
                      f"{e.get('tag') or 'untagged'}  suite {e.get('suite')}")
            return
        suite = E.load_suite(E.suite_path(args.suite))
        if args.action == "freeze":
            if not args.name:
                sys.exit("civitas env freeze: name the version, e.g. civitas env freeze civitas-v2")
            pr = V.probe(suite, workers=args.workers, cache=not args.no_cache)
            rec = V.freeze(args.name, suite, tag=not args.no_tag, probe_result=pr)
            print(f"froze {rec['name']}: fingerprint {rec['fingerprint']} (suite {suite.file}, commit "
                  f"{rec.get('commit')}, probe {pr['seconds']} s{', cached' if pr.get('cached') else ''})")
            print(f"wrote {V.ENVS_DIR / (rec['name'] + '.json')}; "
                  + (f"tagged {rec['tag']}" if rec.get("tag") else f"untagged: {rec.get('untagged')}"))
            if suite.env != rec["name"]:
                print(f"note: {suite.file} names environment {suite.env!r}; set env = \"{rec['name']}\" in its "
                      f"[eval] to evaluate on this version")
            return
        c = V.check(suite, workers=args.workers, cache=not args.no_cache)
        if args.json:
            print(json.dumps(c, indent=2))
        else:
            pr = c["probe"]
            print(V.check_text(c) + f"  [probe {'cached' if pr.get('cached') else str(pr['seconds']) + ' s'}]")
        if c["env"] and c["match"] is False:
            sys.exit(1)
    except (E.EvalError, V.EnvError) as e:
        sys.exit(f"civitas env: {e}")


def cmd_models(args):
    """The model registry (configs/models.toml) with each provider's API key status."""
    from .bench import models as M
    try:
        reg = M.load_registry()
    except M.RegistryError as e:
        sys.exit(f"civitas models: {e}")
    rows = [{**e.to_dict(), "served_id": e.served_id, "key_status": M.key_status(e)} for e in reg.values()]
    if args.json:
        print(json.dumps({"models": rows, "providers": M.providers()}, indent=2))
        return
    if not rows:
        print(f"no models in {M.REGISTRY}")
    w = max([len(r["key"]) for r in rows] + [5])
    for r in rows:
        ks = r["key_status"]
        key = "no key needed" if not ks["env"] else f"${ks['env']} {'set' if ks['set'] else 'NOT SET'}"
        extra = "".join(f", {k} {r[k]}" for k in ("temperature", "max_tokens") if k in r)
        served = f" (served id {r['served_id']})" if r["served_id"] != r["key"] else ""
        print(f"{r['key']:<{w}}  {r['provider']}{served}{extra}; {key}" + (f"  # {r['notes']}" if r.get("notes")
                                                                            else ""))


def cmd_bench(args):
    from .bench.runner import run_benchmark
    cfg = _load(args)
    run_benchmark(cfg, _seeds(args.seeds), args.runs_dir, workers=args.workers, verbose=not args.quiet)


def cmd_serve(args):
    from .server import serve
    serve(args.runs_dir, args.port, host=args.host)


def cmd_leaderboard(args):
    from .bench.runner import leaderboard, print_table
    rows = leaderboard(args.runs_dir)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print_table(rows)


def cmd_prompt(args):
    """Print what an LLM agent would see on month 0 (system prompt + observation; for a fleet agent the
    system prompt, the first briefing and the tool names). With a warm-up (`--warmup N` or [game].warmup_months) the
    warm-up is played first, with no logs and no model, and the prompt is the role's first turn after it."""
    from .agents.prompts import system_prompt
    from .config import agent_spec, build_agents, game_config, scoring_config
    from .core.env import make_env
    cfg = _load(args)
    game = game_config(cfg, args.seed)
    roles = make_env(game, scoring_config(cfg, game)).roles  # a region when game.region: "north.mayor" etc.
    if args.role not in roles:
        sys.exit(f"unknown role {args.role!r}; this game's roles: {', '.join(roles)}")
    agent = None
    if game.warmup_months:
        from .bench.runner import warm_up
        env, agents = warm_up(cfg, args.seed, role=args.role)
        agent = agents[args.role]
        print(f"(after a warm-up of {game.warmup_months} months played by scripted agents: month {env.world.month}, "
              f"the roles before {args.role} in turn order played this month by the warm-up line-up)")
    else:
        env = make_env(game, scoring_config(cfg, game))
    if agent_spec(cfg, args.role).get("kind") == "fleet":
        from .fleet.briefing import briefing
        from .fleet.tools import ToolBox
        if agent is None:
            agent = build_agents(cfg, [args.role], game, env.scoring)[args.role]
            agent.bind(env)                                 # in a region: its city's env (agent.env)
        print("=" * 30, "SYSTEM", "=" * 30)
        print(agent.system)
        print("=" * 30, "USER (briefing)", "=" * 30)
        print(briefing(agent.env, agent.role, mode=agent.cfg.mode, mark=False, style=agent.cfg.briefing))
        print("=" * 30, "TOOLS", "=" * 30)
        print(", ".join(ToolBox(agent.env, agent.role).tools))
        return
    city, _, base = args.role.rpartition(".")
    print("=" * 30, "SYSTEM", "=" * 30)
    print(system_prompt(base, months=game.months, target_pop=env.scoring.target_population,
                        max_actions=game.max_actions_per_turn, city=city or None))
    print("=" * 30, "USER (observation)", "=" * 30)
    print(env.observe(args.role)["text"])


def cmd_check_llm(args):
    from .agents.llm_client import ChatClient
    from .config import provider_config
    cfg = _load(args)
    client = ChatClient(provider_config(cfg, args.provider))
    out = client.chat(args.model, [{"role": "user", "content": 'Reply with exactly: {"ok": true}'}], max_tokens=4000)
    if out.get("reasoning"):
        out["reasoning"] = out["reasoning"][:500] + ("..." if len(out["reasoning"]) > 500 else "")
    print(json.dumps(out, indent=2))


def load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE lines from ./.env into the environment (existing vars win)."""
    import os
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.removeprefix("export ").strip()
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))


def main(argv=None):
    load_dotenv()
    p = argparse.ArgumentParser(prog="civitas", description="Multi-agent city-building benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--config", "-c", default=None, help="TOML config (see configs/)")
        sp.add_argument("--agent", action="append", default=[],
                        help="override a role, e.g. planning=llm:arcee/trinity-mini or mayor=scripted")
        sp.add_argument("--set", action="append", default=[], help="override a value, e.g. game.months=24")
        sp.add_argument("--runs-dir", default="runs")

    def warmup(sp):
        sp.add_argument("--warmup", type=int, default=None, metavar="N",
                        help="play the first N months with scripted agents, then hand the city to the configured ones "
                             "(game.months stays the total; sets game.warmup_months)")

    def no_yardstick(sp):
        sp.add_argument("--no-yardstick", action="store_true",
                        help="a warm-up run: skip the scripted yardstick from the same start (yardstick.json)")

    sp = sub.add_parser("run", help="play one episode")
    common(sp)
    warmup(sp)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--serve", action="store_true", help="start the viewer + live API while running")
    sp.add_argument("--port", type=int, default=8765)
    sp.add_argument("--exit", action="store_true", help="with --serve: exit when the run ends")
    sp.add_argument("--quiet", action="store_true")
    sp.add_argument("--run-id", default=None, help="name of the run folder (default: timestamp_name_seed)")
    no_yardstick(sp)
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("resume", help="continue a paused supervised fleet run (see [fleet].session_months)")
    sp.add_argument("run", help="run id in --runs-dir (or the run's folder)")
    sp.add_argument("--config", "-c", default=None,
                    help="TOML merged over the run's config (default: the run's playbooks.toml); [game] can't change")
    sp.add_argument("--agent", action="append", default=[], help="override a role, as for run")
    sp.add_argument("--set", action="append", default=[], help="override a value, e.g. fleet.session_months=0")
    sp.add_argument("--runs-dir", default="runs")
    sp.add_argument("--serve", action="store_true", help="start the viewer + live API while running")
    sp.add_argument("--port", type=int, default=8765)
    sp.add_argument("--exit", action="store_true", help="with --serve: exit when the session ends")
    sp.add_argument("--quiet", action="store_true")
    no_yardstick(sp)
    sp.set_defaults(fn=cmd_resume)

    sp = sub.add_parser("yardstick", help="a warm-up run's scripted yardstick: the same game from its hand-over, "
                                          "played by scripted agents (no model calls)")
    sp.add_argument("run", help="run id in --runs-dir (or the run's folder)")
    sp.add_argument("--runs-dir", default="runs")
    sp.add_argument("--dry-run", action="store_true", help="print it, don't write yardstick.json")
    sp.add_argument("--json", action="store_true", help="print the whole record")
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(fn=cmd_yardstick)

    sp = sub.add_parser("eval", help="evaluate a model: a suite of late-start runs scored against the scripted "
                                     "yardstick, in one scorecard (see configs/eval_late.toml)")
    sp.add_argument("--suite", "-s", default=None, help="suite file (default with --model: "
                                                        "configs/eval_v2.toml)")
    sp.add_argument("--model", default=None, help="the model every fleet agent plays: a key of configs/models.toml "
                                                  "(its provider and settings), else a served id with --provider "
                                                  "(sets [fleet].model and each [agents.<role>] that names its own)")
    sp.add_argument("--provider", default=None, help="its provider ([providers.<name>], e.g. arcee)")
    sp.add_argument("--agent", action="append", default=[], help="override a role in every run, as for run")
    sp.add_argument("--set", action="append", default=[], help="override a value in every run, e.g. game.months=102")
    sp.add_argument("--workers", type=int, default=None, help="runs at once (default: the suite's [eval].workers)")
    sp.add_argument("--name", default=None, help="the eval's name (default: the model): runs/eval_<name>_<time>/")
    sp.add_argument("--repeats", type=int, default=None, help="play each config + seed this many times (default: "
                                                              "the suite's repeats, 1): measures the replay noise")
    sp.add_argument("--dry-run", action="store_true", help="print the plan, its estimated tokens and time, and "
                                                           "whether it would start; run nothing")
    sp.add_argument("--resume", metavar="EVAL_DIR", default=None, help="rerun an eval's unfinished or failed runs")
    sp.add_argument("--report", metavar="EVAL_DIR", default=None, help="rebuild an eval's scorecard from its runs")
    sp.add_argument("--compare", metavar="EVAL_DIR", nargs="+", default=None,
                    help="several evals side by side, each against the first paired by config + seed")
    sp.add_argument("--adopt", metavar="RUN_ID", nargs="+", default=None,
                    help="make an eval (with -s and --name) of runs that exist already, without rerunning them")
    sp.add_argument("--allow-drift", action="store_true",
                    help="start even when the code no longer matches the suite's environment version (the eval is "
                         "recorded as drifted and never ranked)")
    sp.add_argument("--no-preflight", action="store_true",
                    help="skip the one-request check that the model answers with a tool call")
    sp.add_argument("--preflight-result", default=None, metavar="JSON",
                    help="a preflight that already passed for this model, as JSON: recorded as the eval's and no "
                         "request is sent (the viewer's cached preflight)")
    sp.add_argument("--stamp", default=None, help="the eval folder's time stamp (runs/eval_<name>_<stamp>; the "
                                                  "viewer names the folder before it starts the eval)")
    sp.add_argument("--leaderboard", action="store_true",
                    help="rank the latest finished eval of each model on an environment version (writes "
                         "runs/leaderboard_<env>.md)")
    sp.add_argument("--env", default=None, help="with --leaderboard: the version (default: the default suite's)")
    sp.add_argument("--json", action="store_true", help="with --leaderboard: print the record")
    sp.add_argument("--runs-dir", default="runs")
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(fn=cmd_eval)

    sp = sub.add_parser("env", help="environment versions: freeze a named fingerprint of the eval suite's game, "
                                    "check the code against it, list them")
    sp.add_argument("action", choices=("freeze", "check", "list"))
    sp.add_argument("name", nargs="?", default=None, help="freeze: the new version's name, e.g. civitas-v2")
    sp.add_argument("--suite", "-s", default=None, help="the suite (default configs/eval_v2.toml)")
    sp.add_argument("--workers", type=int, default=8, help="seeds probed at once")
    sp.add_argument("--no-cache", action="store_true", help="probe again even when this code state was probed")
    sp.add_argument("--no-tag", action="store_true", help="freeze: don't tag HEAD env/<name>")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_env)

    sp = sub.add_parser("models", help="the model registry (configs/models.toml) and its API keys")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_models)

    sp = sub.add_parser("bench", help="run several seeds and aggregate scores")
    common(sp)
    warmup(sp)
    sp.add_argument("--seeds", default="0-4")
    sp.add_argument("--workers", type=int, default=1)
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(fn=cmd_bench)

    sp = sub.add_parser("serve", help="3D web interface: watch, replay and launch runs")
    sp.add_argument("--runs-dir", default="runs")
    sp.add_argument("--port", type=int, default=8765)
    sp.add_argument("--host", default="127.0.0.1")
    sp.set_defaults(fn=cmd_serve)

    sp = sub.add_parser("leaderboard", help="aggregate all runs by agent line-up")
    sp.add_argument("--runs-dir", default="runs")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_leaderboard)

    sp = sub.add_parser("prompt", help="print an agent's prompt for month 0 (or its first turn after a warm-up)")
    common(sp)
    warmup(sp)
    sp.add_argument("--role", default="planning")
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(fn=cmd_prompt)

    sp = sub.add_parser("check-llm", help="send one test request to a provider")
    common(sp)
    sp.add_argument("--provider", default="arcee")
    sp.add_argument("--model", default="trinity-large-thinking")
    sp.set_defaults(fn=cmd_check_llm)

    _rl_parsers(sub)

    args = p.parse_args(argv)
    args.fn(args)


def _reward_cfg(args):
    from .rl.reward import RewardConfig
    d = RewardConfig()
    return RewardConfig(**{k: (getattr(args, k) if getattr(args, k, None) is not None else v)
                           for k, v in d.as_dict().items()})


def cmd_rl_build(args):
    from .rl import bank as B
    try:
        B.build(args.name, args.config, _seeds(args.seeds), B.parse_months(args.months), args.roles,
                horizon_max=args.horizon_max, noop=not args.no_noop, workers=args.workers, root=args.root,
                force=args.force, verbose=not args.quiet)
    except (B.BankError, ValueError) as e:
        sys.exit(f"error: {e}")


def cmd_rl_info(args):
    from .rl import bank as B
    try:
        bank = B.TaskBank(args.name, root=args.root)
    except B.BankError as e:
        sys.exit(f"error: {e}")
    if args.json:
        print(json.dumps(bank.manifest.get("composition"), indent=2))
        return
    print(bank.info())
    if args.tasks:
        for t in bank.tasks(args.split, actionable=None):
            print(f"  {t.task_id}  {t.split:<5} {t.style:<6} {'actionable' if t.actionable else 'idle      '}  "
                  f"{t.why}")


def cmd_rl_show(args):
    """A task's system prompt, briefing and tool names (no model; the turn isn't played)."""
    from .rl import bank as B
    from .rl.episode import TaskEpisode
    try:
        ep = TaskEpisode(B.TaskBank(args.name, root=args.root).get(args.task))
    except B.BankError as e:
        sys.exit(f"error: {e}")
    obs = ep.reset()
    if args.json:
        print(json.dumps(obs, indent=2))
        return
    print("=" * 30, "SYSTEM", "=" * 30)
    print(obs["system_prompt"])
    print("=" * 30, "USER (briefing)", "=" * 30)
    print(obs["briefing"])
    print("=" * 30, "TOOLS", "=" * 30)
    print(", ".join(t["function"]["name"] for t in obs["tools"]))


def cmd_rl_check(args):
    from .rl import bank as B
    from .rl import check as K
    try:
        K.run(args.bank, split=args.split, n=args.n, horizons=tuple(B.parse_months(args.horizons)),
              workers=args.workers, determinism=args.determinism, root=args.root, cfg=_reward_cfg(args),
              policies=tuple(args.policies.split(",")) if args.policies else K.ALL, verbose=True)
    except B.BankError as e:
        sys.exit(f"error: {e}")


def cmd_rl_eval(args):
    from .rl import bank as B
    from .rl import evaluate as E
    try:
        E.run(args.bank, base_url=args.base_url, model=args.model, split=args.split, n=args.n, limit=args.limit,
              roles=args.roles.split(",") if args.roles else None, styles=args.styles.split(",") if args.styles
              else None, api_key_env=args.api_key_env, stream=not args.no_stream, timeout=args.timeout,
              temperature=args.temperature, max_tokens=args.max_tokens,
              extra_body=json.loads(args.extra_body) if args.extra_body else None, workers=args.workers,
              reward_cfg=_reward_cfg(args), root=args.root, out_dir=args.out_dir, name=args.name,
              transcripts=not args.no_transcripts, verbose=not args.quiet)
    except B.BankError as e:
        sys.exit(f"error: {e}")


def cmd_rl_compare(args):
    from .rl import bank as B
    from .rl import evaluate as E
    try:
        c = E.compare(args.a, args.b)
    except (B.BankError, OSError) as e:
        sys.exit(f"error: {e}")
    print(json.dumps(c, indent=2) if args.json else E.compare_text(c))


def _rl_parsers(sub):
    from .rl.bank import DEFAULT_MONTHS, HORIZON_MAX, TASKS_DIR
    rl = sub.add_parser("rl", help="RL tasks: one department's turn from a saved city, scored against the scripted "
                                   "fleet (bank, show, check, eval)")
    rsub = rl.add_subparsers(dest="rl_cmd", required=True)

    def reward(sp):
        sp.add_argument("--horizon", type=int, default=None, help="months after the turn the scores are read "
                                                                  "(default 6)")
        sp.add_argument("--w-role", dest="w_role", type=float, default=None,
                        help="weight of the role's own score difference")
        sp.add_argument("--p-failed", dest="p_failed", type=float, default=None)
        sp.add_argument("--p-refused", dest="p_refused", type=float, default=None)
        sp.add_argument("--p-invalid", dest="p_invalid", type=float, default=None)
        sp.add_argument("--p-unended", dest="p_unended", type=float, default=None)

    bank = rsub.add_parser("bank", help="build or describe a task bank (runs/_tasks/<name>/)")
    bsub = bank.add_subparsers(dest="bank_cmd", required=True)
    sp = bsub.add_parser("build", help="play the scripted warm-up on training / test seeds and snapshot the city")
    sp.add_argument("--name", required=True)
    sp.add_argument("--config", "-c", action="append", required=True,
                    help="a late-start fleet config, e.g. configs/fleet_smoke_late.toml (repeat for several)")
    sp.add_argument("--seeds", required=True, help="e.g. 100-299 (train) or 300-339 (test); seeds below 100 and an "
                                                   "eval suite's seeds are refused")
    sp.add_argument("--months", default=",".join(map(str, DEFAULT_MONTHS)), help="task months (acting months)")
    sp.add_argument("--roles", default="all", help="all, or a comma list")
    sp.add_argument("--horizon-max", type=int, default=HORIZON_MAX, help="months of baseline scores cached")
    sp.add_argument("--no-noop", action="store_true", help="skip the noop branches (the bank's headroom figures)")
    sp.add_argument("--workers", type=int, default=4)
    sp.add_argument("--root", default=str(TASKS_DIR))
    sp.add_argument("--force", action="store_true", help="rebuild an existing bank")
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(fn=cmd_rl_build)
    sp = bsub.add_parser("info", help="a bank's composition")
    sp.add_argument("name")
    sp.add_argument("--root", default=str(TASKS_DIR))
    sp.add_argument("--tasks", action="store_true", help="list the tasks")
    sp.add_argument("--split", default=None)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_rl_info)

    sp = rsub.add_parser("show", help="print a task's system prompt, briefing and tools")
    sp.add_argument("name", help="the bank")
    sp.add_argument("task", help="the task id, e.g. fleet_smoke_late/s100/m48/planning")
    sp.add_argument("--root", default=str(TASKS_DIR))
    sp.add_argument("--json", action="store_true", help="the reset() dict (with the tool schemas)")
    sp.set_defaults(fn=cmd_rl_show)

    sp = rsub.add_parser("check", help="validate the reward with hand-made policies (no model)")
    sp.add_argument("--bank", required=True)
    sp.add_argument("--split", default="train")
    sp.add_argument("--n", type=int, default=150, help="tasks (actionable, spread over the split)")
    sp.add_argument("--horizons", default="3,6,12")
    sp.add_argument("--policies", default=None, help="comma list (default: baseline and every policy)")
    sp.add_argument("--determinism", type=int, default=12, help="tasks replayed in fresh processes (0: none)")
    sp.add_argument("--workers", type=int, default=4)
    sp.add_argument("--root", default=str(TASKS_DIR))
    reward(sp)
    sp.set_defaults(fn=cmd_rl_check)

    sp = rsub.add_parser("eval", help="a model on an OpenAI-compatible endpoint over a bank split")
    sp.add_argument("--bank", required=True)
    sp.add_argument("--split", default="test")
    sp.add_argument("--base-url", required=True, help="e.g. https://<app>.modal.run/v1 or http://localhost:8000/v1")
    sp.add_argument("--model", required=True, help="the served model id")
    sp.add_argument("--api-key-env", default="", help="environment variable holding the API key (none by default)")
    sp.add_argument("--n", type=int, default=1, help="samples per task")
    sp.add_argument("--limit", type=int, default=None, help="tasks at most (spread over the split)")
    sp.add_argument("--roles", default=None)
    sp.add_argument("--styles", default=None, help="guided, open or both")
    sp.add_argument("--workers", type=int, default=4, help="turns played at once")
    sp.add_argument("--temperature", type=float, default=None)
    sp.add_argument("--max-tokens", type=int, default=None, help="per model call (default: the config's)")
    sp.add_argument("--extra-body", default=None, help="JSON merged into each request")
    sp.add_argument("--timeout", type=float, default=300.0)
    sp.add_argument("--no-stream", action="store_true")
    sp.add_argument("--name", default=None)
    sp.add_argument("--out-dir", default="runs/_rl_evals")
    sp.add_argument("--no-transcripts", action="store_true")
    sp.add_argument("--root", default=str(TASKS_DIR))
    sp.add_argument("--quiet", action="store_true")
    reward(sp)
    sp.set_defaults(fn=cmd_rl_eval)

    sp = rsub.add_parser("compare", help="paired comparison of two `civitas rl eval` folders (b - a per task)")
    sp.add_argument("a", help="the reference eval folder (e.g. the baseline's)")
    sp.add_argument("b", help="the eval folder compared to it (e.g. the trained model's)")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_rl_compare)


if __name__ == "__main__":
    sys.exit(main())
