"""Run episodes, log replays + trajectories, and aggregate benchmarks."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..config import agent_spec, apply_cli_overrides, build_agents, game_config, scoring_config, warmup_config, \
    warmup_months
from ..core.env import make_env
from ..fleet.config import fleet_agent_config, fleet_agents, fleet_version, playbook_versions, session_months
from ..fleet.report import fleet_report, turn_row
from . import control as C
from . import live as L
from . import session as S
from . import yardstick as Y


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if hasattr(o, "item"):
        return o.item()
    return o


class RunLogger:
    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.frames = open(self.dir / "frames.jsonl", "a", buffering=1)
        self.traj = open(self.dir / "trajectories.jsonl", "a", buffering=1)

    def meta(self, meta: dict) -> None:
        (self.dir / "meta.json").write_text(json.dumps(_jsonable(meta), indent=2))

    def frame(self, frame: dict) -> None:
        self.frames.write(json.dumps(_jsonable(frame), separators=(",", ":")) + "\n")

    def turn(self, rec: dict) -> None:
        self.traj.write(json.dumps(_jsonable(rec)) + "\n")

    def summary(self, summary: dict) -> None:
        (self.dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))

    def close(self) -> None:
        self.frames.close()
        self.traj.close()


def agent_signature(cfg: dict, roles: list[str]) -> str:
    parts = []
    seen = []
    for r in roles:
        base = r.split(".", 1)[-1]
        if "." in r and base in seen and agent_spec(cfg, r) == agent_spec(cfg, f"north.{base}"):
            continue                      # region mode: list each role once when both cities use the same agent
        seen.append(base)
        a = agent_spec(cfg, r)
        k = a.get("kind", "scripted")
        r = base if "." in r and agent_spec(cfg, f"north.{base}") == agent_spec(cfg, f"south.{base}") else r
        parts.append(f"{r}={k}:{a['model']}" if k == "llm" else
                     f"{r}={k}:{fleet_agent_config(cfg, r).model}" if k == "fleet" else f"{r}={k}")
    warm = (cfg.get("game") or {}).get("warmup_months") or 0
    if warm:
        parts.append(f"warmup:{warm}")           # a warm-up run scores a game it played only the end of
    fleet = [r for r in roles if agent_spec(cfg, r).get("kind") == "fleet"]
    if fleet:
        parts.append(f"[{fleet_agent_config(cfg, fleet[0]).mode}]")
    return " ".join(parts)


def _fleet_settings(run: dict) -> None:
    """What the run's config says about its fleet: signature, system prompts, version, mode and sessions."""
    cfg, env, agents = run["cfg"], run["env"], run["agents"]
    run["sig"] = agent_signature(cfg, env.roles)
    run["prompts"] = {r: a.system for r, a in agents.items() if a.kind == "fleet"}   # logged turns refer to these
    run["fv"] = fleet_version(cfg) if run["prompts"] else None                    # only when a role is a fleet agent
    run["mode"] = S.fleet_mode(cfg, env.roles) if run["prompts"] else None
    months = session_months(cfg) if run["prompts"] else 0
    run["session_months"] = months if run["mode"] == "supervised" and run["sessions_on"] else 0
    if months and run["mode"] == "autonomous" and run["verbose"]:
        print(f"[{run['run_id']}] note: [fleet].session_months only pauses supervised runs; this one plays through")


def _open_session(run: dict, *, previous: dict | None = None, config_file=None) -> None:
    """Start a session record (meta.json `sessions`); `previous` = the paused run's prompts and signature."""
    if not run["session_months"] and not run["sessions"]:
        return                                   # a run without sessions keeps its meta as it always was
    k = len(run["sessions"])
    s = {"session": k + 1, "start_month": max(run["env"].world.month, _warm_months(run)),   # after any warm-up
         "end_month": None, "status": "running", "started": time.time(), "session_months": run["session_months"], "fleet_version": run["fv"],
         "playbooks": playbook_versions(run["cfg"], list(run["prompts"]))}
    if config_file:
        s["config_file"] = str(config_file)
    if previous is not None:
        changed = [r for r, p in run["prompts"].items() if p != previous["prompts"].get(r)]
        s["changed"] = changed
        if changed:
            s["fleet_prompts"] = {r: run["prompts"][r] for r in changed}
            run["prompt_ref"].update({r: f"sessions[{k}].fleet_prompts.{r}" for r in changed})
        if run["fv"] != run["sessions"][-1]["fleet_version"]:
            s["fleet_config"] = {"fleet": run["cfg"].get("fleet") or {}, "agents": fleet_agents(run["cfg"])}
        if run["sig"] != previous["sig"]:
            s["signature"] = run["sig"]
    run["sessions"].append(s)


def _warm_months(run: dict) -> int:
    return (run.get("warmup") or {}).get("months", 0)


def _save_sessions(run: dict) -> None:
    path = run["log"].dir / "meta.json"
    meta = json.loads(path.read_text())
    meta["sessions"] = run["sessions"]
    run["log"].meta(meta)


def _resume_cmd(run: dict) -> str:
    runs_dir = run["log"].dir.parent
    return f"civitas resume {run['run_id']}" + ("" if runs_dir == Path("runs") else f" --runs-dir {runs_dir}")


def _pause(run: dict) -> dict:
    """End the session at its boundary: checkpoint, editable playbooks, PAUSED marker, resume instructions."""
    env, d, s = run["env"], run["log"].dir, run["sessions"][-1]
    _close_live(run)
    s.update(status="paused", end_month=env.world.month, ended=time.time())
    _save_sessions(run)
    run["log"].close()
    keep = ("cfg", "seed", "run_id", "env", "agents", "stats", "city_scores", "fleet_rows", "wall_s", "sessions",
            "prompts", "prompt_ref", "sig", "mode", "settings", "log_observations", "warmup")
    ck = S.save_checkpoint(d, {**{k: run[k] for k in keep}, "lineup": S.lineup(run["cfg"], env.roles),
                               "month": env.world.month, "log_offsets": S.log_offsets(d), "saved": time.time()})
    cmd = _resume_cmd(run)
    pb = S.write_playbooks(d / S.PLAYBOOKS, run["cfg"], list(run["prompts"]), run_id=run["run_id"],
                           month=env.world.month, session=s["session"], resume=cmd)
    out = {"run_id": run["run_id"], "paused": True, "session": s["session"], "month": env.world.month,
           "checkpoint": str(ck), "checkpoint_bytes": ck.stat().st_size, "playbooks": str(pb), "resume": cmd}
    (d / S.PAUSED).write_text(json.dumps(out, indent=2))
    size = out["checkpoint_bytes"]
    print(f"[{run['run_id']}] paused at month {env.world.month} after session {s['session']} "
          f"(checkpoint {f'{size / 1e6:.1f} MB' if size >= 1e6 else f'{size / 1e3:.0f} KB'})\n"
          f"  edit the playbooks in {pb}, then resume with:\n    {cmd}\n"
          f"  (or pass -c edited.toml: merged over the run's config; [game] can't change)")
    return out


def _new_stats() -> dict:
    return {"turns": 0, "actions": 0, "invalid": 0, "messages": 0, "spent": 0.0, "parse_errors": 0, "reward": 0.0}


def lineups(cfg: dict, env, game, scoring, seed: int, hub=None) -> tuple[dict, dict | None]:
    """(the configured agents, the warm-up's or None without a warm-up), every one reset and bound to `env` as a run
    starts. The warm-up line-up is `config.warmup_config(cfg)`; a role it plays as configured (a scripted agent, the
    scripted fleet) keeps one agent for the whole game, so its state carries on."""
    agents = build_agents(cfg, env.roles, game, scoring, hub=hub)
    warm = None
    if game.warmup_months:
        wcfg = warmup_config(cfg)
        changed = [r for r in env.roles if agent_spec(wcfg, r) != agent_spec(cfg, r)]
        built = build_agents(wcfg, changed, game, scoring, hub=hub) if changed else {}
        warm = {r: built.get(r, agents[r]) for r in env.roles}
    for a in {id(a): a for a in [*agents.values(), *(warm or {}).values()]}.values():
        a.reset(seed)
        a.bind(env)
    return agents, warm


def settle_fleet(agents: dict, warm: dict | None = None) -> None:
    """At the hand-over: a fleet agent its city's FleetState never saw take a turn (a region's warm-up plays its roles
    as plain scripted agents, outside the fleet tools) counts as having taken the warm-up's last turn, so its first
    briefing's digest covers the months since then rather than every event and council message of the warm-up; a
    fleet mayor finds the plan the scripted fleet would have written (goals, notes, priorities, the warm-up mayor's
    working balances as budgets) instead of none; and a fleet agent taking over from the scripted fleet finds its
    memory empty: the scripted fleet writes there the last failure of each kind as "lessons" and this turn's requests
    as "open_tasks" (fleet/scripted.py: _memory), e.g. "build_park: no empty land tiles there", which the late smoke's
    planning read as a lesson at month 96, and "asked mayor: ...", requests the new agent never sent."""
    from ..fleet.memory import DEPARTMENTS, MEMORY_FIELDS
    from ..fleet.scripted import GOALS, PLAN_NOTES, PRIORITIES, ScriptedFleetAgent
    for role, a in agents.items():
        env = getattr(a, "env", None)
        fleet = getattr(env, "fleet", None)
        before = (warm or {}).get(role)
        if a.kind == "fleet" and fleet is not None and isinstance(before, ScriptedFleetAgent) and before is not a:
            fleet.memory[a.role] = {k: [] for k in MEMORY_FIELDS}
        if a.kind != "fleet" or fleet is None or a.role in fleet.last_turn_month or not env.world.month:
            continue
        k = max(1, env.cfg.act_every)
        prev = (env.world.month - 1) // k * k               # the warm-up's last acting month
        fleet.last_turn_month[a.role] = prev
        fleet.seen_messages[a.role] = sum(m["month"] < prev for m in env.council.visible_to(env._name(a.role), 0))
        targets = getattr((warm or {}).get(role), "targets", None)
        if a.role == "mayor" and targets and not fleet.plan_versions:
            depts = [d for d in DEPARTMENTS if d in env.roles]
            fleet.update_plan(prev, goals=GOALS, notes=PLAN_NOTES,
                              budgets={d: round(float(v), 2) for d, v in targets.items() if d in depts},
                              priorities={d: PRIORITIES[d] for d in depts if d in PRIORITIES})


def snapshot(env) -> dict:
    """The city score, scores and population after the months played so far (a region: its mean score, its total)."""
    envs = getattr(env, "envs", None) or {"": env}
    pop = sum(e.world.history[-1]["population"] if e.world.history else 0 for e in envs.values())
    return {"month": env.world.month, "city": env.last_scores.get("city"), "population": pop,
            "scores": dict(env.last_scores)}


def warm_up(cfg: dict, seed: int, role: str | None = None) -> tuple:
    """Play `cfg`'s warm-up without logging anything and hand over: (env, configured agents) at the first month the
    configured agents act at or after `game.warmup_months` (the env as a run's first post-warm-up turn finds it), with
    the roles before `role` in turn order played that month by the warm-up line-up (as `civitas prompt` shows it).
    Without a warm-up, the env at month 0."""
    warmup_months(cfg)
    game = game_config(cfg, seed)
    env = make_env(game, scoring_config(cfg, game))
    agents, warm = lineups(cfg, env, game, env.scoring, seed)
    while warm and env.world.month < game.warmup_months and not env.done:
        acting = [r for r in env.roles if env.acts_this_month(r)]
        early = {r: env.observe(r) for r in acting} if game.simultaneous else {}
        for r in acting:
            env.apply(r, warm[r].act(early.get(r) or env.observe(r)))
        env.end_month()
    if warm:
        settle_fleet(agents, warm)
        while role and not env.done and not env.acts_this_month(role):
            env.end_month()                                  # months nobody acts in
        if role and not env.done and not game.simultaneous:
            for r in env.roles[:env.roles.index(role)]:
                if env.acts_this_month(r):
                    env.apply(r, warm[r].act(env.observe(r)))
    return env, agents


def _hand_over(run: dict) -> None:
    """The warm-up is over: the configured agents (bound since the start) take the same env from here; the agent stats
    restart (the warm-up's are kept for the summary) and the score and population at hand-over are noted."""
    env, w = run["env"], run["warmup"]
    w["at_handover"] = snapshot(env)
    w["agent_stats"] = copy.deepcopy(run["stats"])
    for r in run["stats"]:
        run["stats"][r] = _new_stats()
    settle_fleet(run["agents"], run["warm_agents"])
    run["warm_agents"] = None
    if run["verbose"]:
        h = w["at_handover"]
        print(f"[{run['run_id']}] warm-up over at month {env.world.month}: city {h['city'] or 0:.1f}, population "
              f"{h['population']:,}; the configured agents take over")


def _warmup_summary(run: dict, final_city: float, final_pop: int) -> dict:
    """summary["warmup"]: the warm-up line-up, the city score and population at hand-over and at the end, the change
    (what the configured agents' months added) and their average city score."""
    w, env = run["warmup"], run["env"]
    out = {k: w[k] for k in ("months", "signature", "agents")}
    hand = w.get("at_handover")
    out["handed_over"] = hand is not None
    if hand is None:                                   # the game ended (bankrupt, recalled) during the warm-up
        return out
    after = run["city_scores"][w["months"]:]
    out.update(at_handover=hand, at_end={"month": env.world.month, "city": final_city, "population": final_pop},
               change={"city": round(final_city - (hand["city"] or 0.0), 3),
                       "population": final_pop - hand["population"]},
               avg_city_score=round(statistics.fmean(after), 3) if after else None,
               agent_stats=w.get("agent_stats"))
    return out


def run_episode(cfg: dict, seed: int, runs_dir: str | Path = "runs", *, hub=None,
                verbose: bool = True, log_observations: bool = True, run_id: str | None = None,
                sessions: bool = True, yardstick: bool = True, live: bool = True) -> dict:
    """Play a run to the end and return its summary. A supervised fleet run with `[fleet].session_months` stops
    at the first session boundary instead and returns `{"paused": True, ...}` (`sessions=False` plays through);
    `resume_episode` continues it. With `[game].warmup_months = N` the warm-up line-up (`config.warmup_config`) plays
    months 0..N-1 and the configured agents the rest of the same game (`_hand_over`); its sessions start at N, and at
    the end (unless `yardstick=False`) the scripted yardstick from the same start goes to yardstick.json
    (bench/yardstick.py). `live` writes the live feed (bench/live.py: live.jsonl, live_frame.json; off in bench)."""
    warm = warmup_months(cfg)
    game = game_config(cfg, seed)
    scoring = scoring_config(cfg, game)
    env = make_env(game, scoring)
    agents, warm_agents = lineups(cfg, env, game, scoring, seed, hub=hub)

    run_id = run_id or f"{time.strftime('%Y%m%d-%H%M%S')}_{cfg['name']}_s{seed}_{os.getpid() % 1000:03d}"
    log = RunLogger(Path(runs_dir) / run_id)
    run = {"cfg": cfg, "seed": seed, "run_id": run_id, "env": env, "agents": agents, "log": log, "hub": hub,
           "verbose": verbose, "log_observations": log_observations, "sessions_on": sessions, "sessions": [],
           "yardstick": yardstick, "live": L.LiveLog(log.dir, env) if live else None,
           "prompt_ref": {}, "wall_s": 0.0, "city_scores": [], "fleet_rows": [],
           "settings": {"game": dataclasses.asdict(game), "scoring": dataclasses.asdict(scoring)},
           "stats": {r: _new_stats() for r in env.roles}, "warm_agents": warm_agents,
           "warmup": {"months": warm, "signature": agent_signature(warmup_config(cfg), env.roles),
                      "agents": {r: warm_agents[r].describe() for r in env.roles}} if warm else None}
    _fleet_settings(run)
    _open_session(run)
    sig, prompts, fv = run["sig"], run["prompts"], run["fv"]
    log.meta({"run_id": run_id, "name": cfg["name"], "seed": seed, "width": game.width,
              "height": game.height, "months": game.months, "roles": env.roles,
              "agents": {r: agents[r].describe() for r in env.roles}, "signature": sig,
              "config": cfg, "started": time.time(), **env.meta(),
              **({"fleet_version": fv} if fv else {}),
              **({"fleet_prompts": prompts} if prompts else {}),
              **({"sessions": run["sessions"]} if run["sessions"] else {}),
              **({"warmup_months": warm, "warmup": run["warmup"]} if warm else {})})
    if hub is not None:
        hub.run_id = run_id
    log.frame(env.frame())
    if verbose:
        print(f"[{run_id}] {sig}")
        if warm:
            print(f"[{run_id}] warm-up: months 0-{warm - 1} played by {run['warmup']['signature']}")
    return _play(run)


def resume_episode(run: str | Path, config: str | Path | None = None, *, runs_dir: str | Path = "runs",
                   agents: list[str] | None = None, sets: list[str] | None = None, hub=None,
                   verbose: bool = True, yardstick: bool = True) -> dict:
    """Continue a paused run (a run id in `runs_dir`, or its folder) with the playbooks of `config`, else of the
    run's `playbooks.toml`, merged over its config (`session.overlay`), then `--agent` / `--set` overrides. The
    game, scoring, line-up and mode must not change (ValueError). Returns what `run_episode` does."""
    d = S.run_dir(run, runs_dir)
    state = S.load_checkpoint(d)
    source = Path(config) if config else (d / S.PLAYBOOKS if (d / S.PLAYBOOKS).exists() else None)
    if source and not source.is_file():
        raise ValueError(f"no config file {source}")
    cfg = S.overlay(state["cfg"], S.read_toml(source)) if source else copy.deepcopy(state["cfg"])
    cfg = apply_cli_overrides(cfg, agents, sets)
    for w in S.check_resume(state, cfg, str(source) if source else None):
        S.warn(w)
    env = state["env"]
    S.apply_balance(env)
    cut = S.rewind_logs(d, state["log_offsets"])
    if cut:
        S.warn(f"{', '.join(cut)}: dropped lines logged after the checkpoint (an interrupted session)")
    for marker in (S.PAUSED, "STOPPED"):
        (d / marker).unlink(missing_ok=True)
    run = {**{k: state[k] for k in ("seed", "run_id", "stats", "city_scores", "fleet_rows", "wall_s", "sessions",
                                    "prompt_ref", "settings", "log_observations")},
           "warmup": state.get("warmup"), "warm_agents": None,      # a pause comes after the warm-up
           "cfg": cfg, "env": env, "agents": S.restore_agents(state["agents"], cfg, env, hub),
           "log": RunLogger(d), "hub": hub, "verbose": verbose, "sessions_on": True, "yardstick": yardstick}
    run["live"] = L.LiveLog(d, env)
    _fleet_settings(run)
    _open_session(run, previous=state, config_file=source)
    _save_sessions(run)
    if hub is not None:
        hub.run_id = run["run_id"]
    if verbose:
        s = run["sessions"][-1]
        changed = ", ".join(s.get("changed") or []) or "none"
        print(f"[{run['run_id']}] resumed at month {env.world.month}, session {s['session']} "
              f"(new system prompts: {changed})")
    return _play(run)


def _prompt(run: dict, role: str, messages: list[dict]) -> list[dict]:
    """A fleet turn's logged prompt points at the meta.json entry holding the system prompt it was played with."""
    ref = run["prompt_ref"].get(role)
    if ref and messages and str(messages[0].get("content", "")).startswith("(system prompt:"):
        return [{"role": "system", "content": f"(system prompt: meta.json {ref})"}, *messages[1:]]
    return messages


def _versions(run: dict) -> list:
    """The fleet versions the sessions played, in order, a run of sessions with the same one listed once."""
    s = run["sessions"]
    return [x["fleet_version"] for i, x in enumerate(s) if i == 0 or x["fleet_version"] != s[i - 1]["fleet_version"]]


def _fleet_version(run: dict) -> str | None:
    """The run's fleet version; one whose sessions played several is a hash of their sequence."""
    seq = _versions(run)
    if len(seq) <= 1:
        return run["fv"]
    return hashlib.sha1("|".join(map(str, seq)).encode()).hexdigest()[:12]


def _live(run: dict, kind: str, role: str, **data) -> None:
    if run.get("live"):
        run["live"].emit(kind, role, **data)


def _live_turn(run: dict, role: str, agent, decision, results: list[dict]) -> None:
    """The turn's end on the live feed; the actions and messages first of an agent that didn't report its own (a
    fleet agent's ToolBox reports each call as it runs)."""
    if not run.get("live"):
        return
    fleet = getattr(agent, "kind", "") == "fleet"
    for r in results[len(decision.executed):]:
        a = dict(r.get("action") or {})
        L.emit_result(run["live"], role, str(a.pop("type", "?")), a, r)
    if not fleet:
        for m in decision.messages:
            L.emit_result(run["live"], role, "send_message", {"to": m.get("to", "all"), "text": m.get("text", "")},
                          {"ok": True}, action=False)
    done = [r for r in results if r.get("ok")]
    _live(run, "done", role, actions=len(done), failed=len(results) - len(done),
          spent=round(sum(r.get("cost") or 0 for r in done), 2),
          report=L._clip(str(decision.thinking or ""), L.TEXT_CHARS))


def _close_live(run: dict) -> None:
    if run.get("live"):
        run["live"].close()
        run["live"] = None


def _hold(run: dict, role: str | None) -> bool:
    """The viewer's Pause / Stop (bench/control.py) before `role`'s turn: wait while paused; True = stop now."""
    stop, held = C.wait(run["log"].dir, run["env"].world.month, role)
    run["held_s"] = run.get("held_s", 0.0) + held
    if held and run["verbose"]:
        print(f"[{run['run_id']}] resumed at month {run['env'].world.month} after {held:.0f} s paused")
    return stop


def _stopped(run: dict, t0: float) -> dict:
    """Stopped from the viewer: the logs so far stay, STOPPED marks the run, no summary."""
    run["wall_s"] += time.time() - t0 - run.pop("held_s", 0.0)
    if run["sessions"]:
        run["sessions"][-1].update(status="stopped", end_month=run["env"].world.month, ended=time.time())
        _save_sessions(run)
    run["log"].close()
    _close_live(run)
    C.mark_stopped(run["log"].dir)
    if run["verbose"]:
        print(f"[{run['run_id']}] stopped at month {run['env'].world.month}")
    return {"run_id": run["run_id"], "stopped": True, "month": run["env"].world.month}


def _play(run: dict) -> dict:
    """Months until the end of the game (-> summary) or of the session (-> `_pause`)."""
    cfg, env, stats, log, hub = run["cfg"], run["env"], run["stats"], run["log"], run["hub"]
    run_id, seed, game, verbose = run["run_id"], run["seed"], env.cfg, run["verbose"]
    log_observations, prompts, sig = run["log_observations"], run["prompts"], run["sig"]
    city_scores, fleet_rows, t0 = run["city_scores"], run["fleet_rows"], time.time()
    start = max(env.world.month, _warm_months(run))      # sessions count the months after the warm-up
    while not env.done:
        if run.get("warm_agents") and env.world.month >= run["warmup"]["months"]:
            _hand_over(run)
        warming = bool(run.get("warm_agents"))
        agents = run["warm_agents"] if warming else run["agents"]
        pending = []
        acting = [r for r in env.roles if env.acts_this_month(r)]
        if _hold(run, acting[0] if acting else None):
            return _stopped(run, t0)
        # simultaneous mode: everybody sees the same start-of-month state, then all decisions apply in turn order
        early = {r: env.observe(r) for r in acting} if game.simultaneous else {}
        for i, role in enumerate(acting):
            if i and _hold(run, role):
                return _stopped(run, t0)
            obs = early.get(role) or env.observe(role)
            _live(run, "turn", role, agent=agents[role].describe(), **({"warmup": True} if warming else {}))
            decision = agents[role].act(obs)
            results = env.apply(role, decision)
            _live_turn(run, role, agents[role], decision, results)
            st = stats[role]
            st["turns"] += 1
            st["actions"] += len(results)
            st["invalid"] += sum(1 for r in results if not r["ok"])
            st["messages"] += len(decision.messages)
            st["spent"] += sum(r.get("cost", 0) for r in results if r["ok"])
            st["parse_errors"] += 1 if decision.parse_error and decision.parse_error != "recovered_after_retry" else 0
            rec = {"run_id": run_id, "seed": seed, "month": obs["month"], "role": role,
                   "agent": agents[role].describe(), "thinking": decision.thinking,
                   "actions": [r["action"] for r in decision.executed] + decision.actions,
                   "messages": decision.messages, "notes": decision.notes,
                   "results": results, "parse_error": decision.parse_error}
            if run["sessions"]:
                rec["session"] = run["sessions"][-1]["session"]
            if warming:
                rec["warmup"] = True                   # played by the warm-up line-up; reports skip these
            if "messages" in decision.meta:
                rec["prompt"] = _prompt(run, role, decision.meta["messages"])
                rec["response"] = decision.meta.get("response")
                rec["reasoning"] = decision.meta.get("reasoning")
                rec["finish_reason"] = decision.meta.get("finish_reason")
                rec["usage"] = decision.meta.get("usage")
                rec["latency_s"] = decision.meta.get("latency_s")
                rec.update({k: decision.meta[k] for k in ("tool_trace", "steps", "ended", "finish_reasons",
                                                          "length_stops", "recovered_calls")
                            if k in decision.meta})
            elif log_observations:
                rec["observation_text"] = obs["text"]
            pending.append(rec)
            if agents[role].kind == "fleet" and not warming:
                fleet_rows.append(turn_row(rec))       # slim: whole records hold every turn's messages
        info = env.end_month()
        for rec in pending:
            rec["reward"] = info["rewards"][rec["role"]]
            rec["scores_after"] = info["scores"]
            log.turn(rec)
        for r in env.roles:
            stats[r]["reward"] += info["rewards"][r]
        city_scores.append(info["scores"]["city"])
        frame = env.frame(turns=info["turns"], scores=info["scores"])
        log.frame(frame)
        if run.get("live"):
            run["live"].month(env.world.month)
        if hub is not None:
            hub.latest_frame = frame
        m = info["metrics"]
        if verbose and (m["month"] % 12 == 0 or env.done):
            print(f"  month {m['month']:>3}: pop {m['population']:>6}  treasury {m['treasury']:>10,.0f}  "
                  f"happy {m['happiness']:.2f}  scores " + " ".join(f"{k}={v:.1f}" for k, v in info["scores"].items()))
        if run["session_months"] and not env.done and env.world.month - start >= run["session_months"]:
            run["wall_s"] += time.time() - t0 - run.pop("held_s", 0.0)
            return _pause(run)

    run["wall_s"] += time.time() - t0 - run.pop("held_s", 0.0)
    C.clear(log.dir)                                     # a pause asked for after the last turn
    _close_live(run)
    agents = run["agents"]
    fv = _fleet_version(run)
    final = env.final_metrics()
    if "cities" in env.meta():
        fm = final
        final = {"population": sum(m["population"] for m in fm.values()),
                 "happiness": round(sum(m["happiness"] for m in fm.values()) / len(fm), 4),
                 "treasury": sum(m["treasury"] for m in fm.values()), "debt": sum(m["debt"] for m in fm.values()),
                 "net_worth": sum(m["net_worth"] for m in fm.values()),
                 "unemployment": round(sum(m["unemployment"] for m in fm.values()) / len(fm), 4),
                 **{k: None for k in ("power_coverage", "water_coverage", "pollution_res", "land_value_dev")},
                 "developed_total": sum(m["developed_total"] for m in fm.values()),
                 "by_city": {c: {k: m[k] for k in ("population", "happiness", "treasury", "river_pollution")}
                             for c, m in fm.items()}}
    for r in env.roles:
        st = stats[r]
        st["invalid_rate"] = round(st["invalid"] / st["actions"], 4) if st["actions"] else 0.0
        st["spent"] = round(st["spent"], 2)
        st["reward"] = round(st["reward"], 4)
        if hasattr(agents[r], "usage"):
            st["llm_usage"] = agents[r].usage
    summary = {
        "run_id": run_id, "name": cfg["name"], "seed": seed, "signature": sig,
        "agents": {r: agents[r].describe() for r in env.roles},
        "months_played": env.world.month,
        "bankrupt": any(e.world.bankrupt for e in getattr(env, "envs", {"": env}).values()),
        "final_scores": info["scores"],
        "avg_city_score": round(statistics.fmean(city_scores), 3) if city_scores else 0.0,
        "final_metrics": {k: final.get(k) for k in ("population", "happiness", "treasury", "debt", "net_worth",
                                                   "unemployment", "power_coverage", "water_coverage",
                                                   "pollution_res", "land_value_dev", "developed_total", "by_city")
                          if k in final},
        "agent_stats": stats,
        "wall_time_s": round(run["wall_s"], 2),        # playing time; a paused run's sessions add up
        **({"fleet_version": fv} if fv else {}),
    }
    if run.get("warmup"):            # agent_stats cover the configured agents' turns; the warm-up's are in here
        summary["warmup"] = _warmup_summary(run, info["scores"]["city"], final["population"])
    if run["sessions"]:
        s = run["sessions"][-1]
        s.update(status="finished", end_month=env.world.month, ended=time.time())
        _save_sessions(run)
        summary["sessions"] = len(run["sessions"])
        if fv != run["fv"]:
            summary["fleet_versions"] = _versions(run)
        for name in (S.CHECKPOINT, S.PAUSED):
            (log.dir / name).unlink(missing_ok=True)
    if prompts:
        try:
            log.frames.flush()                       # the collaboration score reads frames.jsonl
            report, md = fleet_report(env, summary, fleet_rows, log.dir)
            (log.dir / "fleet_report.json").write_text(json.dumps(_jsonable(report), indent=2))
            (log.dir / "fleet_report.md").write_text(md)
            summary["fleet"] = {"report": "fleet_report.md", "hints": len(report["hints"])}
        except Exception as e:                       # a report bug must not cost the run its summary
            print(f"[{run_id}] warning: fleet report failed: {type(e).__name__}: {e}", file=sys.stderr)
            summary["fleet"] = {"error": f"{type(e).__name__}: {e}"}
    # played before summary.json marks the run finished (so the server never starts a second one), written after it
    ys = _yardstick(run, summary) if run.get("yardstick") and (summary.get("warmup") or {}).get("handed_over") else None
    log.summary(summary)
    if ys:
        (log.dir / Y.FILE).write_text(json.dumps(_jsonable(ys), indent=2))
    log.close()
    if verbose:
        print(f"[{run_id}] done: city {info['scores']['city']:.1f}  " +
              " ".join(f"{r}={info['scores'][r]:.1f}" for r in env.roles) + f"  ({summary['wall_time_s']}s)")
        w = summary.get("warmup") or {}
        if w.get("handed_over"):
            h, e, c = w["at_handover"], w["at_end"], w["change"]
            print(f"[{run_id}] after the warm-up (months {w['months']}-{e['month'] - 1}): city {h['city'] or 0:.1f} -> "
                  f"{e['city']:.1f} ({c['city']:+.1f}), population {h['population']:,} -> {e['population']:,} "
                  f"({c['population']:+,})" + (f"; {Y.line(ys)}" if ys else ""))
    return summary


def _yardstick(run: dict, summary: dict) -> dict | None:
    """The scripted yardstick of a finished warm-up run (bench/yardstick.py: the same game from the same hand-over,
    every agent scripted, no logs; seconds) and its `noop` reference (the run's model roles idle from the hand-over),
    for the run folder's yardstick.json. A failure only warns."""
    try:
        if run["verbose"]:
            print(f"[{run['run_id']}] scripted yardstick: the same game from month {run['warmup']['months']}, "
                  f"every agent scripted ...")
        ys = Y.play(run["cfg"], run["seed"])
        ys["run_id"] = run["run_id"]
        Y.compare_with_run(ys, summary)
        ys["created"] = time.time()
        if ys.get("matches_run") is False:
            print(f"[{run['run_id']}] warning: the yardstick's city at hand-over differs from the run's",
                  file=sys.stderr)
        try:                                         # the do-nothing reference from the same hand-over
            ys["noop"] = Y.play_noop(run["cfg"], run["seed"], ys, summary)
        except Exception as e:
            print(f"[{run['run_id']}] warning: noop reference failed: {type(e).__name__}: {e}", file=sys.stderr)
        return ys
    except Exception as e:                           # the yardstick is extra: never lose the run over it
        print(f"[{run['run_id']}] warning: yardstick failed: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def _run_one(args):
    cfg, seed, runs_dir, verbose = args
    return run_episode(cfg, seed, runs_dir, verbose=verbose, sessions=False, yardstick=False, live=False)


def run_benchmark(cfg: dict, seeds: list[int], runs_dir="runs", workers: int = 1, verbose=True) -> dict:
    if verbose and fleet_agents(cfg) and session_months(cfg):
        print("note: bench plays supervised runs through without pausing ([fleet].session_months ignored)")
    jobs = [(cfg, s, runs_dir, verbose) for s in seeds]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(_run_one, jobs))
    else:
        results = [_run_one(j) for j in jobs]
    agg = aggregate(results)
    out = Path(runs_dir) / f"bench_{cfg['name']}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"config": cfg["name"], "seeds": seeds, "aggregate": agg, "runs": results}, indent=2))
    print_table([agg])
    print(f"saved {out}")
    return agg


def _ms(xs):
    xs = list(xs)
    if not xs:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": round(statistics.fmean(xs), 2), "std": round(statistics.pstdev(xs), 2) if len(xs) > 1 else 0.0}


def aggregate(summaries: list[dict]) -> dict:
    s0 = summaries[0]
    roles = list(s0["agents"])
    return {
        "name": s0["name"], "signature": s0["signature"], "fleet_version": s0.get("fleet_version"),
        "n": len(summaries),
        "city": _ms(s["final_scores"]["city"] for s in summaries),
        "avg_city": _ms(s["avg_city_score"] for s in summaries),
        "roles": {r: _ms(s["final_scores"][r] for s in summaries) for r in roles},
        "invalid_rate": {r: _ms(s["agent_stats"][r]["invalid_rate"] for s in summaries) for r in roles},
        "population": _ms(s["final_metrics"]["population"] for s in summaries),
        "bankruptcies": sum(1 for s in summaries if s["bankrupt"]),
    }


def leaderboard(runs_dir="runs") -> list[dict]:
    """Finished runs grouped by agent line-up; fleet runs also by `fleet_version` (coaching and limits)."""
    groups: dict[tuple, list] = {}
    for p in Path(runs_dir).glob("*/summary.json"):
        try:
            s = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        # a region's signature names each role once when its cities share the agent, as a single city's does: the
        # agents' role names (north.mayor ...) keep the two apart
        groups.setdefault((s["signature"], s.get("fleet_version"), tuple(s.get("agents") or ())), []).append(s)
    rows = [aggregate(v) for v in groups.values()]
    rows.sort(key=lambda r: -r["city"]["mean"])
    return rows


def _clip(text: str, n: int, tail: int = 16) -> str:
    """Cut the middle of a long line-up so its end (a fleet's `[mode]`) stays visible."""
    return text if len(text) <= n else text[:n - tail - 3] + "..." + text[-tail:]


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("no runs found")
        return
    roles = list(rows[0]["roles"])
    fleet = any(r.get("fleet_version") for r in rows)
    fcol = lambda v: f"{v or '-':<13}" if fleet else ""
    hdr = (f"{'agents':<58} {fcol('fleet')}{'n':>3} {'city':>12} {'pop':>8} " + " ".join(f"{r:>12}" for r in roles)
           + "  bankrupt")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{_clip(r['signature'], 58):<58} {fcol(r.get('fleet_version'))}{r['n']:>3} "
              f"{r['city']['mean']:>6.1f}±{r['city']['std']:<5.1f} "
              f"{r['population']['mean']:>8.0f} " +
              " ".join(f"{r['roles'][k]['mean']:>6.1f}±{r['roles'][k]['std']:<5.1f}" for k in roles) +
              f"  {r['bankruptcies']}")
