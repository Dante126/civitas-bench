"""The model registry (`configs/models.toml`): the models `civitas eval --model <key>` and the viewer's Evals page can
evaluate, each with its provider and sampling settings, plus providers that aren't built in.

```toml
[models.trinity-large-thinking]
provider = "arcee"
temperature = 0.4
max_tokens = 16000

[models.gpt-5]
provider = "openai"
id = "gpt-5"                     # the served id when it differs from the key
temperature = "default"          # send no temperature (OpenAI's reasoning models take only their own)

[providers.together]             # a provider not built in (config.DEFAULT_PROVIDERS)
base_url = "https://api.together.xyz/v1"
api_key_env = "TOGETHER_API_KEY"
```

An entry holds only model keys (`MODEL_KEYS`). The compute budget (`ENV_KEYS`: steps, tokens per turn, cadence,
action cap, playbooks, mode) belongs to the environment, equal for every model, and is refused in an entry. The file
is machine-written (`save_entry` / `delete_entry` rewrite it canonically: the header comment, providers then models,
sorted, values in TOML written here since the standard library has no writer) and may be edited by hand.

`eval_overrides(entry)` turns an entry into `civitas run` arguments (`--set fleet.model=<served id>` ..., and a
registry provider's fields as `--set providers.<name>.<field>=...`, since a run config has no such table);
`preflight(entry)` sends one real request with two fleet tool schemas and passes only when a tool call comes back.
API keys live in `./.env` (`set_api_key`), never in the registry.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from ..agents.llm_client import DEFAULT_TEMPERATURE, ChatClient, ProviderConfig, temperature_of

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs/models.toml"
MODEL_KEYS = ("provider", "id", "temperature", "max_tokens", "extra_body", "notes")
ENV_KEYS = ("max_steps", "turn_tokens", "turn_budget", "budget_reserve", "act_every", "max_actions_per_turn",
            "playbook", "mode")   # refused
PROVIDER_KEYS = tuple(f.name for f in dataclasses.fields(ProviderConfig))
KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
ENV_VAR_RE = re.compile(r"^[A-Z][A-Z0-9_]*_API_KEY$")
BARE = re.compile(r"^[A-Za-z0-9_-]+$")
PREFLIGHT_TOKENS = 4000          # room for a reasoning model's thinking before its tool call
PREFLIGHT_TOOLS = ("end_turn", "inspect")
HEADER = """\
# Model registry: the models `civitas eval --model <key>` and the viewer's Evals page evaluate.
# Machine-written (the viewer rewrites it: entries sorted, comments other than this header dropped); hand edits are
# fine. Model keys only: provider, id (the served id when it differs from the key), temperature (a number, or
# "default" to send none: the provider's own, which OpenAI's reasoning models require), max_tokens,
# extra_body (an inline table sent with every request), notes. The compute budget (max_steps, turn_tokens, act_every,
# ...) is the environment's, the same for every model, and is refused here. A provider that isn't built in (arcee,
# openrouter, ollama, openai, anthropic) gets a [providers.<name>] table (base_url, api_key_env). API keys go in ./.env.
"""


class RegistryError(ValueError):
    """An entry the registry refuses, with every reason."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class ModelEntry:
    key: str
    provider: str = "arcee"
    id: str | None = None               # served id; None = key
    temperature: float | str | None = None    # None = the suite config's; "default" = send none (the provider's)
    max_tokens: int | None = None
    extra_body: dict = field(default_factory=dict)
    notes: str = ""

    @property
    def served_id(self) -> str:
        return self.id or self.key

    def to_dict(self) -> dict:
        """The MODEL_KEYS that are set, plus "key"."""
        out = {"key": self.key, "provider": self.provider}
        if self.id:
            out["id"] = self.id
        if self.temperature is not None:
            out["temperature"] = self.temperature
        if self.max_tokens is not None:
            out["max_tokens"] = self.max_tokens
        if self.extra_body:
            out["extra_body"] = dict(self.extra_body)
        if self.notes:
            out["notes"] = self.notes
        return out


# ---------------------------------------------------------------------------------------------------- read
def _path(path) -> Path:
    return Path(path) if path else REGISTRY


def _data(path=None) -> dict:
    p = _path(path)
    try:
        return tomllib.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as e:
        raise RegistryError([f"{p}: {e}"]) from None


def is_registry(data: dict) -> bool:
    """A TOML file read as the registry ([models.*] tables and no [game]): the launcher should leave it out."""
    return isinstance(data.get("models"), dict) and not data.get("game")


def _clean(fields: dict) -> dict:
    """Unset values dropped: None, "" and an empty extra_body (a form sends every field)."""
    return {k: v for k, v in fields.items() if v is not None and v != "" and not (k == "extra_body" and v == {})}


def _is_default(v) -> bool:
    return isinstance(v, str) and v.strip().lower() == DEFAULT_TEMPERATURE


def _temperature(v) -> float | str | None:
    """An entry's temperature: None (unset), "default" (send none) or a float."""
    return None if v is None else DEFAULT_TEMPERATURE if _is_default(v) else float(v)


def _entry(key: str, fields: dict) -> ModelEntry:
    f = _clean(fields)
    return ModelEntry(key=key, provider=str(f.get("provider", "arcee")), id=f.get("id"),
                      temperature=_temperature(f.get("temperature")),
                      max_tokens=None if f.get("max_tokens") is None else int(f["max_tokens"]),
                      extra_body=dict(f.get("extra_body") or {}), notes=str(f.get("notes") or ""))


def load_registry(path=None) -> dict[str, ModelEntry]:
    """key -> ModelEntry, sorted by key; a missing file is an empty registry. Entries are read as written (an entry
    `validate` would refuse still loads; `save_entry` is where the registry refuses)."""
    models = _data(path).get("models") or {}
    out = {}
    for key in sorted(models):
        v = models[key]
        if isinstance(v, dict):
            try:
                out[key] = _entry(key, {k: x for k, x in v.items() if k in MODEL_KEYS})
            except (TypeError, ValueError):
                continue
    return out


def registry_providers(path=None) -> dict[str, dict]:
    """The registry file's [providers.X] tables."""
    provs = _data(path).get("providers") or {}
    return {k: dict(v) for k, v in sorted(provs.items()) if isinstance(v, dict)}


def builtin_providers() -> dict[str, dict]:
    from ..config import DEFAULT_PROVIDERS
    return {k: dict(v) for k, v in DEFAULT_PROVIDERS.items()}


def providers(path=None) -> dict[str, dict]:
    """Every provider an entry may name: the built-ins, then the registry's own (which may override a built-in's
    fields)."""
    out = builtin_providers()
    for k, v in registry_providers(path).items():
        out[k] = {**out.get(k, {}), **v}
    return out


# ---------------------------------------------------------------------------------------------------- validate
def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate(key: str, fields: dict, path=None) -> list[str]:
    """Why the registry refuses this entry (empty: it takes it): a bad key, an unknown provider, an environment key,
    an unknown key, a value of the wrong type."""
    errs = []
    if not isinstance(key, str) or not KEY_RE.match(key):
        errs.append(f"bad model key {key!r}: letters, digits, '.', '_' and '-' only, starting with a letter or digit "
                    f"(at most 64; the served id goes in 'id')")
    if not isinstance(fields, dict):
        return errs + [f"the entry must be a table of settings, got {type(fields).__name__}"]
    f = _clean(fields)
    f.pop("key", None)
    for k in f:
        if k in ENV_KEYS:
            errs.append(f"{k!r} is an environment setting (the compute budget, the same for every model): it belongs "
                        f"to the suite's configs, not a model entry")
        elif k not in MODEL_KEYS:
            errs.append(f"unknown key {k!r} (a model entry takes {', '.join(MODEL_KEYS)})")
    prov = f.get("provider", "arcee")
    if not isinstance(prov, str):
        errs.append(f"provider must be a name, got {prov!r}")
    elif prov not in providers(path):
        errs.append(f"unknown provider {prov!r}: one of {', '.join(providers(path))}, or add a [providers.{prov}] "
                    f"table (base_url, api_key_env) to {_path(path).name}")
    if "id" in f and (not isinstance(f["id"], str) or not f["id"].strip() or any(c.isspace() for c in f["id"])):
        errs.append(f"id must be the served model id (text without spaces), got {f['id']!r}")
    if "temperature" in f and not (_is_default(f["temperature"]) or
                                   (_num(f["temperature"]) and 0 <= f["temperature"] <= 2)):
        errs.append(f"temperature must be a number from 0 to 2, or \"default\" to send none (the provider's own), "
                    f"got {f['temperature']!r}")
    if "max_tokens" in f and not (isinstance(f["max_tokens"], int) and not isinstance(f["max_tokens"], bool)
                                  and 1 <= f["max_tokens"] <= 1_000_000):
        errs.append(f"max_tokens must be a whole number from 1 to 1,000,000, got {f['max_tokens']!r}")
    if "extra_body" in f:
        eb = f["extra_body"]
        if not isinstance(eb, dict):
            errs.append(f"extra_body must be a table (JSON object) sent with every request, got {eb!r}")
        else:
            try:
                _toml(eb)
                json.dumps(eb, allow_nan=False)
            except (TypeError, ValueError) as e:
                errs.append(f"extra_body: {e}")
            bad = [k for k in eb if k in ("model", "messages", "tools", "stream")]
            if bad:
                errs.append(f"extra_body can't set {', '.join(bad)} (the game sends those)")
    if "notes" in f and not isinstance(f["notes"], str):
        errs.append(f"notes must be text, got {f['notes']!r}")
    return errs


# ---------------------------------------------------------------------------------------------------- write
def _str(s: str) -> str:
    """A TOML basic string. JSON's escapes are TOML's, but JSON writes a character beyond U+FFFF as a surrogate pair
    of \\u escapes, which TOML refuses: so non-ASCII stays as it is (control characters are still escaped, and DEL,
    which JSON leaves raw and TOML doesn't take)."""
    return json.dumps(s, ensure_ascii=False).replace("\x7f", "\\u007f")


def _key(k: str) -> str:
    return k if BARE.match(k) else _str(k)


def _toml(v) -> str:
    """A TOML value: strings as basic strings (JSON's escapes are TOML's), tables inline."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError(f"{v} can't be written")
        return repr(v)
    if isinstance(v, str):
        return _str(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_key(str(k))} = {_toml(x)}" for k, x in v.items()) + "}" if v else "{}"
    raise TypeError(f"{type(v).__name__} {v!r} can't be written to TOML")


def render(models: dict[str, ModelEntry], provs: dict[str, dict]) -> str:
    """The registry file, canonically: header, providers, then models, each sorted by name."""
    L = [HEADER.rstrip("\n")]
    for name in sorted(provs):
        L += ["", f"[providers.{_key(name)}]"]
        L += [f"{_key(k)} = {_toml(v)}" for k, v in provs[name].items() if v is not None]
    for key in sorted(models):
        L += ["", f"[models.{_key(key)}]"]
        d = models[key].to_dict()
        L += [f"{k} = {_toml(d[k])}" for k in MODEL_KEYS if k in d]
    return "\n".join(L) + "\n"


def _write(path, models: dict[str, ModelEntry], provs: dict[str, dict]) -> None:
    p = _path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = render(models, provs)
    tomllib.loads(text)                               # never write a file the registry can't read back
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


def save_entry(key: str, fields: dict, path=None) -> ModelEntry:
    """Add or replace the entry `key` with `fields` (the whole entry: keys left out are unset), after `validate`
    (RegistryError with its errors); the file is rewritten canonically."""
    errs = validate(key, fields, path)
    if errs:
        raise RegistryError(errs)
    f = {k: v for k, v in _clean(fields).items() if k in MODEL_KEYS}
    entry = _entry(key, f)
    models = load_registry(path)
    models[key] = entry
    _write(path, models, registry_providers(path))
    return entry


def delete_entry(key: str, path=None) -> bool:
    """Remove the entry `key`; False when there was none."""
    models = load_registry(path)
    if key not in models:
        return False
    del models[key]
    _write(path, models, registry_providers(path))
    return True


# ---------------------------------------------------------------------------------------------------- providers, keys
def provider_spec(name: str, path=None) -> dict:
    spec = providers(path).get(name)
    if spec is None:
        raise ValueError(f"unknown provider {name!r}; add [providers.{name}] to {_path(path).name}")
    return spec


def provider_of(entry: ModelEntry, path=None) -> ProviderConfig:
    """The entry's provider: the built-ins and the registry's [providers.X] tables."""
    spec = provider_spec(entry.provider, path)
    return ProviderConfig(**{k: v for k, v in spec.items() if k in PROVIDER_KEYS})


def key_status(entry_or_provider, path=None) -> dict:
    """{"provider", "env": the API key's env var ("" when none is needed), "set": bool} for an entry, a provider name
    or a ProviderConfig (provider None)."""
    if isinstance(entry_or_provider, ProviderConfig):
        name, env = None, entry_or_provider.api_key_env
    else:
        name = entry_or_provider.provider if isinstance(entry_or_provider, ModelEntry) else str(entry_or_provider)
        env = (providers(path).get(name) or {}).get("api_key_env", "")
    env = env or ""
    return {"provider": name, "env": env, "set": True if not env else bool(os.environ.get(env))}


def set_api_key(env_var: str, value: str, dotenv=None) -> None:
    """Write `env_var=value` into ./.env (replacing its line, `export` form too; the file is created owner-only) and
    set it in this process's environment. ValueError for a name that isn't `..._API_KEY` or a value with a line
    break, quote or space."""
    if not isinstance(env_var, str) or not ENV_VAR_RE.match(env_var):
        raise ValueError(f"bad key name {env_var!r}: it must look like NAME_API_KEY")
    if not isinstance(value, str) or not value.strip() or re.search(r"[\s\"'\\]", value.strip()):
        raise ValueError("the key must be one word of text (no spaces, quotes or line breaks)")
    value = value.strip()
    p = Path(dotenv) if dotenv else Path(".env")
    lines = p.read_text().splitlines() if p.exists() else []
    pat = re.compile(rf"^\s*(export\s+)?{re.escape(env_var)}\s*=")
    out, done = [], False
    for line in lines:
        if pat.match(line):
            if not done:
                out.append(f"{env_var}={value}")
                done = True
            continue
        out.append(line)
    if not done:
        out.append(f"{env_var}={value}")
    new = not p.exists()
    tmp = p.with_name(p.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write("\n".join(out) + "\n")
    if not new:
        try:
            os.chmod(tmp, p.stat().st_mode & 0o777)
        except OSError:
            pass
    os.replace(tmp, p)
    os.environ[env_var] = value


# ---------------------------------------------------------------------------------------------------- runs
def _set_value(v) -> str:
    return json.dumps(v, separators=(",", ":")) if isinstance(v, (dict, list)) else str(v)


def eval_overrides(entry: ModelEntry, cfg: dict | None = None, path=None) -> tuple[list[str], list[str]]:
    """(--agent, --set) arguments for `civitas run` that put every fleet (and llm) agent on this entry:
    `fleet.model=<served id>`, `fleet.provider=`, and `fleet.temperature=` / `fleet.max_tokens=` /
    `fleet.extra_body=<json>` when the entry sets them; a registry provider's fields as `providers.<name>.<field>=`
    (the run config has no such table). With `cfg` (a run config) also the `--agent` of each `[agents.<role>]` that
    names its own model (evals.model_overrides) and `agents.<role>.<key>=` for a fleet role that sets its own
    temperature / max_tokens / extra_body, so the entry's reach it too."""
    from .evals import model_overrides
    sets = [f"fleet.model={entry.served_id}", f"fleet.provider={entry.provider}"]
    own = {"temperature": entry.temperature, "max_tokens": entry.max_tokens,
           "extra_body": entry.extra_body if entry.extra_body else None}
    sets += [f"fleet.{k}={_set_value(v)}" for k, v in own.items() if v is not None]
    reg = registry_providers(path).get(entry.provider)
    if reg:
        sets += [f"providers.{entry.provider}.{k}={_set_value(v)}" for k, v in reg.items()
                 if k in PROVIDER_KEYS and v is not None]
    agents = []
    if cfg is not None:
        agents, _ = model_overrides(cfg, entry.served_id, entry.provider)
        for role, spec in (cfg.get("agents") or {}).items():
            if isinstance(spec, dict) and spec.get("kind") == "fleet" and spec.get("model") != "scripted":
                sets += [f"agents.{role}.{k}={_set_value(v)}" for k, v in own.items() if v is not None and k in spec]
    return agents, sets


def _tool_schemas() -> list[dict]:
    """The fleet's own schemas of PREFLIGHT_TOOLS (a department's ToolBox on a fresh city)."""
    from ..core.env import GameConfig, make_env
    from ..fleet.tools import ToolBox
    env = make_env(GameConfig(months=12))
    box = ToolBox(env, "planning")
    return [box.tools[n].schema() for n in PREFLIGHT_TOOLS]


PREFLIGHT_SYSTEM = ("You are a department of a simulated city, playing through tools. Every turn ends with a call "
                    "of the end_turn tool.")
PREFLIGHT_USER = ("This is a connection test. Don't inspect anything: call the end_turn tool now, with the report "
                  "\"preflight\".")


def preflight(entry: ModelEntry, path=None, *, provider: ProviderConfig | None = None, max_tokens: int | None = None,
              timeout: float = 120.0) -> dict:
    """One real request to the entry's model with two fleet tool schemas (end_turn, inspect) and a message asking for
    end_turn; ok only when a tool call comes back. Catches a wrong key, a wrong id and a model without tool calls
    before any run spends tokens. Never raises: a failure is a failed step. `provider` overrides the registry's (a
    model from a run config's [providers] table)."""
    t0 = time.time()
    steps: list[dict] = []
    out = {"ok": False, "model": entry.key, "served_id": entry.served_id, "provider": entry.provider,
           "seconds": 0.0, "steps": steps, "error": None}

    def fail(name: str, detail: str) -> dict:
        steps.append({"name": name, "ok": False, "detail": detail})
        out.update(error=detail, seconds=round(time.time() - t0, 2))
        return out

    try:
        pc = provider or provider_of(entry, path)
    except ValueError as e:
        return fail("key", str(e))
    pc = dataclasses.replace(pc, max_retries=min(pc.max_retries, 1), timeout=min(pc.timeout, timeout),
                             deadline=min(pc.deadline, 2 * timeout))
    if pc.api_key_env and not os.environ.get(pc.api_key_env):
        return fail("key", f"${pc.api_key_env} isn't set (paste the key, or put {pc.api_key_env}=... in ./.env)")
    steps.append({"name": "key", "ok": True, "detail": f"${pc.api_key_env}" if pc.api_key_env else "no key needed"})
    try:
        tools = _tool_schemas()
        res = ChatClient(pc).chat(entry.served_id,
                                  [{"role": "system", "content": PREFLIGHT_SYSTEM},
                                   {"role": "user", "content": PREFLIGHT_USER}],
                                  temperature=0.4 if entry.temperature is None else temperature_of(entry.temperature),
                                  max_tokens=max_tokens or PREFLIGHT_TOKENS,
                                  extra={**(entry.extra_body or {}), "tools": tools})
    except Exception as e:                              # a failed step, never an exception
        return fail("request", f"{type(e).__name__}: {str(e)[:400]}")
    u = res.get("usage") or {}
    steps.append({"name": "request", "ok": True,
                  "detail": f"{res.get('latency_s', 0):.1f} s, finish {res.get('finish_reason')}, "
                            f"{int(u.get('prompt_tokens') or 0)} + {int(u.get('completion_tokens') or 0)} tokens"})
    calls = res.get("tool_calls") or []
    if not calls:
        said = (res.get("content") or "").strip()[:200]
        return fail("tool_call", "no tool call came back" + (f"; it said: {said!r}" if said else "")
                    + (" (cut at max_tokens)" if res.get("finish_reason") == "length" else ""))
    names = [((c or {}).get("function") or {}).get("name") or "?" for c in calls]
    steps.append({"name": "tool_call", "ok": True, "detail": "called " + ", ".join(names)})
    out.update(ok=True, seconds=round(time.time() - t0, 2))
    return out
