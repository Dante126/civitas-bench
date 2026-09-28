"""Supervised sessions: pause a fleet run at a session boundary, let the coach edit the playbooks, resume it.

A supervised fleet run with `[fleet].session_months = N` stops every N months (`bench/runner.py:Episode`) and
leaves in `runs/<run_id>/`:

- `checkpoint.ckpt`: the whole game in one gzipped pickle (env with its world, RNG, council, memory, scores
  history and fleet state; every agent's state; the runner's stats) plus the offsets of the logs at the pause,
  after a header line with its HMAC-SHA256;
- `playbooks.toml`: the run's current playbooks, ready to edit;
- `PAUSED`: a marker with the month and the resume command.

`civitas resume <run_id>` merges `playbooks.toml` (or the file given with `-c`) over the run's config
(`overlay()`), checks that the game settings and the line-up are unchanged (`check_resume()`), rebuilds the fleet
agents from the merged config with their state carried over (`restore_agents()`: new system prompts, same
memory and usage) and plays the next session. Frames and trajectories append.

Unpickling runs whatever the pickle says, so a checkpoint is signed: `save_checkpoint` writes an HMAC-SHA256 of the
payload keyed by a per-installation secret (`key_path()`: `$CIVITAS_CHECKPOINT_KEY_DIR`, else
`$XDG_CONFIG_HOME/civitas`, else `~/.config/civitas`, file `checkpoint.key`, created on the first save with mode
0600), and `load_checkpoint` checks it with `hmac.compare_digest` before unpickling. A checkpoint with no
signature, a wrong one (changed, or made on another machine or by another user) or no key to check it against is
refused: a checkpoint only resumes where it was made. It never holds API keys either: agents' chat clients (which
read the key from the environment) are dropped and rebuilt on resume.
"""
from __future__ import annotations

import copy
import dataclasses
import gzip
import hashlib
import hmac
import json
import os
import pickle
import secrets
import sys
from pathlib import Path

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from ..config import agent_spec, build_agents, game_config, scoring_config
from ..fleet.config import fleet_agent_config, session_months

CHECKPOINT = "checkpoint.ckpt"
MAGIC = b"CIVITAS-CHECKPOINT hmac-sha256 "        # header line: MAGIC + 64 hex digits of the MAC + "\n"
MAC_HEX = 64
KEY_DIR_ENV = "CIVITAS_CHECKPOINT_KEY_DIR"          # tests point this at a temp dir
KEY_FILE = "checkpoint.key"
PLAYBOOKS = "playbooks.toml"
PAUSED = "PAUSED"
LOGS = ("frames.jsonl", "trajectories.jsonl")
FORMAT = 1
TRANSIENT = ("client", "hub")                      # never pickled: rebuilt (client) or handed in (hub) on resume
REBUILT = ("cfg", "provider", "client", "system", "env")   # a fleet agent's attributes that come from the config
MERGED_PER_ENTRY = ("agents", "providers")


# ------------------------------------------------------------------ config
def read_toml(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def overlay(base: dict, over: dict) -> dict:
    """`over` merged over a run's config: top-level tables key by key, and each `[agents.<role>]` /
    `[providers.<name>]` entry key by key too (so `[agents.mayor] playbook = ...` keeps `kind`); other values,
    tables inside them included, replace the old ones. The run keeps its name."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k == "name":
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            if k in MERGED_PER_ENTRY:
                for e, spec in v.items():
                    old = out[k].get(e)
                    out[k][e] = ({**old, **copy.deepcopy(spec)} if isinstance(old, dict) and isinstance(spec, dict)
                                 else copy.deepcopy(spec))
            else:
                out[k].update(copy.deepcopy(v))
        else:
            out[k] = copy.deepcopy(v)
    return out


def lineup(cfg: dict, roles) -> dict:
    return {r: agent_spec(cfg, r).get("kind", "scripted") for r in roles}


def fleet_mode(cfg: dict, roles) -> str | None:
    fleet = [r for r, k in lineup(cfg, roles).items() if k == "fleet"]
    return fleet_agent_config(cfg, fleet[0]).mode if fleet else None


def _diff(old: dict, new: dict) -> list[str]:
    return [f"{k}: {old.get(k)!r} -> {new.get(k)!r}" for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]


def check_resume(state: dict, cfg: dict, source: str | None = None) -> list[str]:
    """Refuse (ValueError) a config that changes the game, the scoring, the line-up or the fleet mode of the
    paused run; return warnings for changed non-fleet agents, which keep their checkpointed settings."""
    seed, roles = state["seed"], state["env"].roles
    where = f" in {source}" if source else ""
    game, was = game_config(cfg, seed), state["settings"]
    problems = [f"[game] {d}" for d in _diff(was["game"], dataclasses.asdict(game))]
    problems += [f"[scoring] {d}" for d in _diff(was["scoring"], dataclasses.asdict(scoring_config(cfg, game)))]
    problems += [f"[agents] {d}" for d in _diff(state["lineup"], lineup(cfg, roles))]
    mode = fleet_mode(cfg, roles)
    if mode != state["mode"]:
        problems.append(f"[fleet] mode: {state['mode']!r} -> {mode!r}")
    if problems:
        raise ValueError(f"run {state['run_id']}: the config{where} changes what a resumed run keeps; only playbooks "
                         "and other fleet settings may change between sessions:\n  " + "\n  ".join(problems))
    session_months(cfg)                                          # a bad value fails here, not at the next pause
    old = state["cfg"]
    return [f"agents.{r}: settings changed, but only fleet agents are rebuilt on resume; {r} keeps its own"
            for r, k in state["lineup"].items() if k != "fleet" and agent_spec(old, r) != agent_spec(cfg, r)]


# ------------------------------------------------------------------ agents
def strip_agent(agent):
    """A shallow copy without its chat client or live hub (the rest pickles along with the env it may share)."""
    a = copy.copy(agent)
    for k in TRANSIENT:
        if k in a.__dict__:
            a.__dict__[k] = None
    return a


def _refill(agent, hub) -> None:
    if "client" in agent.__dict__ and agent.client is None and getattr(agent, "provider", None) is not None:
        from ..agents.llm_client import ChatClient
        agent.client = ChatClient(agent.provider)
    if "hub" in agent.__dict__:
        agent.hub = hub


def restore_agents(saved: dict, cfg: dict, env, hub=None) -> dict:
    """Fleet agents are rebuilt from `cfg` (new playbooks, models, limits; the system prompt is rebuilt by
    `bind`) and keep everything else they had (usage and any turn state); other agents come back as they were."""
    out = {}
    for role, old in saved.items():
        if old.kind == "fleet":
            new = build_agents(cfg, [role], env.cfg, env.scoring, hub=hub)[role]
            new.__dict__.update({k: v for k, v in old.__dict__.items() if k not in REBUILT})
            _refill(new, hub)
            new.bind(env)
        else:
            new = old
            _refill(new, hub)
        out[role] = new
    return out


# -------------------------------------------------------------- checkpoint
def key_path() -> Path:
    """Where this installation's checkpoint key lives: outside the repo and `runs/`."""
    d = os.environ.get(KEY_DIR_ENV)
    if not d:
        d = os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config"),
                         "civitas")
    return Path(d) / KEY_FILE


def _key(create: bool) -> bytes | None:
    """The secret the checkpoints are signed with; made (mode 0600) on the first save, never on a load."""
    p = key_path()
    if not p.exists():
        if not create:
            return None
        p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:                      # another process made it first
            pass
        else:
            with os.fdopen(fd, "w") as f:
                f.write(secrets.token_hex(32) + "\n")
    if os.name == "posix" and p.stat().st_mode & 0o077:
        raise ValueError(f"checkpoint key {p} can be read or written by other users: run `chmod 600 {p}`")
    key = p.read_bytes().strip()
    if len(key) < 32:
        raise ValueError(f"checkpoint key {p} is damaged (too short); delete it to make a new one, which makes "
                         "existing checkpoints unresumable")
    return key


def _mac(key: bytes, payload: bytes) -> bytes:
    return hmac.new(key, payload, hashlib.sha256).hexdigest().encode()


def split_checkpoint(data: bytes) -> tuple[bytes | None, bytes]:
    """(MAC, payload) of a checkpoint file's bytes; MAC None when the header is missing or malformed."""
    n = len(MAGIC)
    if not data.startswith(MAGIC) or data[n + MAC_HEX:n + MAC_HEX + 1] != b"\n":
        return None, data
    return data[n:n + MAC_HEX], data[n + MAC_HEX + 1:]


def save_checkpoint(run_dir: Path, state: dict) -> Path:
    """Sign and write the state atomically (a crash mid-write keeps the previous checkpoint)."""
    path = run_dir / CHECKPOINT
    tmp = path.with_name(path.name + ".tmp")
    state = {**state, "format": FORMAT, "agents": {r: strip_agent(a) for r, a in state["agents"].items()}}
    payload = gzip.compress(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL), compresslevel=6)
    tmp.write_bytes(MAGIC + _mac(_key(create=True), payload) + b"\n" + payload)
    os.replace(tmp, path)
    return path


def run_dir(run: str | Path, runs_dir: str | Path = "runs") -> Path:
    # Any folder is accepted (the CLI is local); what makes loading its checkpoint safe is the MAC check below.
    p = Path(run)
    return p if p.is_dir() and (p / "meta.json").exists() else Path(runs_dir) / str(run)


def load_checkpoint(d: Path) -> dict:
    """The paused run's state, unpickled only once its MAC checks out against this installation's key."""
    if not (d / "meta.json").exists():
        raise ValueError(f"no run at {d}")
    if (d / "summary.json").exists():
        raise ValueError(f"run {d.name} has finished; there is nothing to resume")
    path = d / CHECKPOINT
    if not path.exists():
        raise ValueError(f"run {d.name} has no checkpoint: only a supervised fleet run with [fleet].session_months "
                         "pauses (or it is still playing its session)")
    mac, payload = split_checkpoint(path.read_bytes())
    where = "a checkpoint can only be resumed on the machine and account that made it; refusing to unpickle it"
    if mac is None:
        raise ValueError(f"{path} has no signature, so it wasn't written by this version of civitas: {where}")
    key = _key(create=False)
    if key is None:
        raise ValueError(f"{path}: there is no checkpoint key at {key_path()}, so this installation never made a "
                         f"checkpoint and didn't make this one: {where}")
    if not hmac.compare_digest(mac, _mac(key, payload)):
        raise ValueError(f"{path}: the signature doesn't match the key at {key_path()}: the file was changed, or it "
                         f"was made with another key (another machine or account): {where}")
    state = pickle.loads(gzip.decompress(payload))            # signed by us, see the module docstring
    if state.get("format") != FORMAT:
        raise ValueError(f"{path}: checkpoint format {state.get('format')!r}, this version reads {FORMAT}")
    return state


def log_offsets(d: Path) -> dict:
    return {name: (d / name).stat().st_size if (d / name).exists() else 0 for name in LOGS}


def rewind_logs(d: Path, offsets: dict) -> list[str]:
    """Cut the logs back to where the checkpoint left them: lines from an interrupted later session go."""
    cut = []
    for name, size in offsets.items():
        p = d / name
        if p.exists() and p.stat().st_size > size:
            with open(p, "r+b") as f:
                f.truncate(size)
            cut.append(name)
    return cut


def apply_balance(env) -> None:
    """Balance overrides are process-global catalog state set when an env is built; an unpickled env wasn't."""
    from ..sim import catalog as C
    envs = getattr(env, "envs", None) or {"": env}
    for e in envs.values():
        C.apply_overrides(e.cfg.balance)


# ---------------------------------------------------------------- playbooks
def toml_string(s) -> str:
    """A TOML string that reads back as exactly `s`: multi-line literal when it can be, else escaped."""
    s = str(s or "")
    bad = lambda c: (ord(c) < 32 and c not in "\n\t") or ord(c) == 127
    if "\n" not in s:
        return json.dumps(s, ensure_ascii=False).replace("\x7f", "\\u007f")
    if "'''" not in s and not s.endswith("'") and not any(bad(c) for c in s):
        return "'''\n" + s + "'''"
    esc = s.replace("\\", "\\\\").replace('"', '\\"')
    return '"""\n' + "".join(f"\\u{ord(c):04x}" if bad(c) else c for c in esc) + '"""'


def write_playbooks(path: Path, cfg: dict, roles, *, run_id: str, month: int, session: int, resume: str) -> Path:
    """The playbooks the run plays with, as a config overlay for `civitas resume`."""
    fleet = cfg.get("fleet") or {}
    agents = cfg.get("agents") or {}
    L = [f"# Playbooks of run {run_id}, paused at month {month} after session {session}.",
         f"# Edit them, then resume with:  {resume}",
         "# On resume this file (or the one given with -c) is merged over the run's config: playbooks and other",
         "# fleet settings (model, limits, session_months) may change; [game], [scoring], the line-up and the mode",
         "# may not.",
         "",
         "[fleet]",
         f"session_months = {session_months(cfg)}  # months per session; 0 plays to the end without pausing",
         f"playbook = {toml_string(fleet.get('playbook', ''))}"]
    for r in roles:
        L += ["", f"[agents.{r}]", f"playbook = {toml_string((agents.get(r) or {}).get('playbook', ''))}"]
    path.write_text("\n".join(L) + "\n")
    return path


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)
