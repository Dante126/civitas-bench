"""Command line: civitas run | resume | yardstick | eval | env | models | bench | serve | leaderboard | prompt |
check-llm"""
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
                       preflight=not args.no_preflight)
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
        print(briefing(agent.env, agent.role, mode=agent.cfg.mode, mark=False))
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
                                                        "configs/eval_singles.toml)")
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
    sp.add_argument("--suite", "-s", default=None, help="the suite (default configs/eval_singles.toml)")
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

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
