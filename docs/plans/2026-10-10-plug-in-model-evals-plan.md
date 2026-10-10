# Plug-in model evals: implementation plan

Design: `docs/plans/2026-10-10-plug-in-model-evals-design.md`. Two parallel tracks share the Python contract below:
track A (backend: registry, environment versions, eval integration, CLI) and track B (server endpoints and the
viewer's Evals page). Track B codes against the contract; where track A isn't merged yet it may stub, but the final
code must call these exact names.

## Contract

### `civitas/bench/models.py` (track A)

```python
REGISTRY: Path                      # ROOT / "configs/models.toml"
MODEL_KEYS = ("provider", "id", "temperature", "max_tokens", "extra_body", "notes")
ENV_KEYS = ("max_steps", "turn_tokens", "act_every", "max_actions_per_turn", "playbook", "mode")   # refused

@dataclass
class ModelEntry:
    key: str
    provider: str = "arcee"
    id: str | None = None               # served id; None = key
    temperature: float | None = None    # None = the suite config's
    max_tokens: int | None = None
    extra_body: dict = field(default_factory=dict)
    notes: str = ""
    @property
    def served_id(self) -> str: ...
    def to_dict(self) -> dict: ...      # MODEL_KEYS that are set, plus "key"

def load_registry(path=None) -> dict[str, ModelEntry]          # missing file -> {}
def registry_providers(path=None) -> dict[str, dict]            # [providers.X] tables of the registry file
def validate(key: str, fields: dict, path=None) -> list[str]    # errors: bad key chars, unknown provider, ENV_KEYS,
                                                                #   unknown keys, wrong types
def save_entry(key: str, fields: dict, path=None) -> ModelEntry # validate, then rewrite the whole file canonically
def delete_entry(key: str, path=None) -> bool
def provider_of(entry: ModelEntry, path=None) -> ProviderConfig # built-ins + registry providers
def key_status(entry_or_provider) -> dict                       # {"provider", "env": "ARCEE_API_KEY", "set": bool}
def set_api_key(env_var: str, value: str, dotenv=None) -> None  # write/replace the line in ./.env, set os.environ;
                                                                #   env_var must match ^[A-Z][A-Z0-9_]*_API_KEY$
def eval_overrides(entry: ModelEntry) -> tuple[list[str], list[str]]   # (--agent, --set) for `civitas run`
def preflight(entry: ModelEntry, path=None) -> dict
    # {"ok": bool, "model": key, "served_id", "provider", "seconds": float,
    #  "steps": [{"name": "key"|"request"|"tool_call", "ok": bool, "detail": str}], "error": str | None}
```

Built-in providers move to one place (`config.DEFAULT_PROVIDERS` gains `openai` and `anthropic`;
`ProviderConfig.max_tokens_field`, default `"max_tokens"`, `"max_completion_tokens"` for openai).

### `civitas/bench/envs.py` (track A)

```python
ENVS_DIR: Path                      # ROOT / "configs/envs"
def probe(suite, *, workers: int = 8, cache: bool = True) -> dict
    # {"fingerprint": str (16 hex), "parts": {"config": h, "prompts": h, "tools": h,
    #   "seeds": {"<config>:<seed>": {"briefings": h, "results": h, "metrics": h}}},
    #  "code": {"commit", "dirty", "diff_hash"}, "seconds": float}
def freeze(name: str, suite, *, tag: bool = True, probe_result=None) -> dict   # writes ENVS_DIR/<name>.json;
                                                                               #   refuses an existing name
def load_env(name: str) -> dict | None
def list_envs() -> list[dict]       # [{"name", "fingerprint", "created", "commit", "suite"}], newest first
def diff(frozen: dict, probed: dict) -> list[str]   # part names that differ, e.g. ["briefings", "metrics"]
def check(suite, *, probe_result=None) -> dict
    # {"env": name | None, "frozen": bool, "match": bool | None, "diff": [...], "fingerprint", "probe": {...}}
```

### `civitas/bench/evals.py` additions (track A)

```python
DEFAULT_SUITE = "configs/eval_singles.toml"
# Suite gains: env (str | None), baseline_model (str | None) from [eval]
def prepare(model: str, *, suite=None, name=None, allow_drift=False, workers=None, runs_dir="runs") -> dict
    # registry lookup (else the old --model/--provider path), plan(), check(), preflight, env check;
    # returns the ev with ev["model_entry"], ev["preflight"], ev["env"] = {"name","fingerprint","match","diff",
    # "drift": bool}, and ev["errors"] (empty = may start)
def progress(eval_dir) -> dict
    # {"status", "total", "finished", "running", "failed", "pending", "tokens": int, "eta_s": float | None,
    #  "runs": [{"run_id", "seed", "status", "month", "months"}]}
def leaderboard(env: str | None = None, runs_dir="runs") -> dict
    # {"env", "envs": [names with evals], "baseline": model | None,
    #  "rows": [{"model", "eval_dir", "created", "n_units", "vs_yardstick": {"mean","lo","hi"},
    #            "score": {"mean","lo","hi"}, "departments": {role: mean diff}, "tokens", "cost",
    #            "verdict": text | None, "paired": {...} | None}], "skipped": [{"eval_dir", "reason"}]}
def leaderboard_markdown(lb: dict) -> str
```

`civitas eval --model K` without `-s` uses `DEFAULT_SUITE`; `--allow-drift`; `civitas eval --leaderboard [--env]`;
`civitas env freeze NAME [-s]`, `civitas env check [-s]`, `civitas env list`; `civitas models` lists the registry.

### Server endpoints (track B)

```
GET  /api/models                  {"models": [entry + {"key_status", "latest": {eval_dir, score, env} | null}],
                                   "providers": [{"name", "env", "set", "builtin"}], "can_edit": bool}
POST /api/models                  {"key", "fields"} -> save_entry; errors 400 with {"errors": [...]}
POST /api/models/delete           {"key"}
POST /api/models/preflight        {"key"} -> preflight()
POST /api/keys                    {"env", "value"} -> set_api_key (only when launching is allowed); never echoes
GET  /api/env                     check() for the default suite, probing in a background thread:
                                  {"computing": true} until done; cached per code state; ?refresh=1
GET  /api/envs                    list_envs()
POST /api/env/freeze              {"name"} -> freeze()
POST /api/evals/start             {"model", "suite"?, "allow_drift"?} -> prepare(); errors -> 400; else launches
                                  `python -m civitas.cli eval --model K [-s S] [--allow-drift] --name ...` as a
                                  background subprocess (logs in runs/_logs/), returns {"eval_dir"}
GET  /api/evals/progress?dir=D    progress()
POST /api/evals/stop              {"dir"} -> terminate the eval process (and its runs)
GET  /api/evals/leaderboard?env=  leaderboard()
GET  /api/evals/scorecard?dir=D   the eval's scorecard.md text
```

Every POST that writes requires the launcher (allow_launch) and a localhost client.
