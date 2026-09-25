# Agent Fleets Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or subagent-driven-development) to implement
> this plan task-by-task.

**Goal:** Replace the one-shot JSON LLM agent with a fleet of tool-using agents: a manager (mayor) that plans and
allocates budgets, and seven departments that act through tools, with persistent memory, coach playbooks, and an
end-of-run report.

**Architecture:** A new `civitas/fleet/` package. `ToolAgent` implements the existing `Agent` interface, so the
runner loop is unchanged apart from a `bind(env)` hook: during `act()` it runs a native tool-calling loop against
the streaming chat client; action tools execute immediately through `apply_action`, and the already-executed
results are returned in `Decision.executed` so `env.apply` logs them as today. Memory (`FleetState`) lives on the
env. The mayor is already first in the turn order, so "manager phase, then departments" needs no scheduling change.

**Tech Stack:** Python 3.12 stdlib + numpy, pytest, OpenAI-compatible chat API with native `tools` (verified on
Arcee trinity-large-thinking).

**Design doc:** `docs/plans/2026-09-25-agent-fleets-design.md` — read it first.

**Conventions:** see `CLAUDE.md`. Git branch `agent-fleets` (already created). Commit after each task with a
message ending in the Co-Authored-By line from the session instructions. Checkpoint command after each task:
`source .venv/bin/activate && pytest -q -m "not slow"` (currently 151 passed).

**Scope limits (v1):** single-city only (region mode raises a clear error for fleet agents); no scripted fleet; no
server/supervised scheduling (mode is recorded as metadata; supervised = resume later).

---

### Task 1: Chat client returns tool calls (streamed and plain)

**Files:** Modify `civitas/agents/llm_client.py`; Test `tests/test_agents.py`.

**Step 1: failing tests** (append to `tests/test_agents.py`):

```python
class _ToolCallLLM(BaseHTTPRequestHandler):
    """Streams a tool call split across chunks, as OpenAI-style servers do."""
    stream = True

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        assert body["tools"][0]["function"]["name"] == "get_budget"
        chunks = [{"choices": [{"delta": {"content": "checking", "tool_calls": [
                      {"index": 0, "id": "call_1", "type": "function", "function": {"name": "get_budget", "arguments": ""}}]}}]},
                  {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"dep'}}]}}]},
                  {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'artment":"health"}'}}]},
                                "finish_reason": "tool_calls"}]}]
        if type(self).stream:
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            for c in chunks:
                self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            out = json.dumps({"choices": [{"message": {"content": "checking", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_budget", "arguments": '{"department":"health"}'}}]},
                "finish_reason": "tool_calls"}], "usage": {}}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)


def test_chat_client_returns_tool_calls_streamed_and_plain():
    from civitas.agents.llm_client import ChatClient
    tools = [{"type": "function", "function": {"name": "get_budget", "parameters": {"type": "object", "properties": {}}}}]
    for stream in (True, False):
        _ToolCallLLM.stream = stream
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _ToolCallLLM)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        c = ChatClient(ProviderConfig(base_url=f"http://127.0.0.1:{srv.server_port}/v1", api_key_env="", stream=stream))
        out = c.chat("fake", [{"role": "user", "content": "hi"}], extra={"tools": tools})
        assert out["finish_reason"] == "tool_calls"
        assert out["tool_calls"] == [{"id": "call_1", "type": "function",
                                      "function": {"name": "get_budget", "arguments": '{"department":"health"}'}}]
        srv.shutdown()
```

**Step 2:** run `pytest tests/test_agents.py -q -k tool_calls` → FAIL (`KeyError: 'tool_calls'`).

**Step 3: implement.**
- In `_read_events`: keep `calls: dict[int, dict]`. For each `delta.tool_calls` item, `slot = calls.setdefault(tc.get("index", 0), {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})`; set `id` / `type` when present; append `function.name` and `function.arguments` fragments. Emit `message["tool_calls"] = [calls[i] for i in sorted(calls)]` (omit the key when empty). A stream with tool calls and `finish_reason` `tool_calls` counts as finished.
- In `ChatClient.chat` return dict add `"tool_calls": msg.get("tool_calls") or []`.
- Nothing else changes (tools are passed via `extra`).

**Step 4:** tests pass; quick suite passes. **Step 5:** commit `LLM client: return tool calls (streamed and plain)`.

---

### Task 2: Executed results in decisions, `bind(env)` hook

**Files:** Modify `civitas/core/decision.py`, `civitas/core/env.py` (`apply`), `civitas/agents/base.py`,
`civitas/bench/runner.py`; Test `tests/test_fleet.py` (new).

**Step 1: failing tests** (`tests/test_fleet.py`):

```python
"""Agent fleets: tool-using manager + departments."""
from civitas.core.actions import apply_action
from civitas.core.decision import Decision
from civitas.core.env import CivitasEnv, GameConfig


def _env(**kw):
    return CivitasEnv(GameConfig(seed=0, months=12, **kw))


def test_env_apply_logs_already_executed_results():
    env = _env()
    env.world.treasury = 50000
    res = apply_action(env.world, "mayor", {"type": "allocate_budget", "department": "health", "amount": 1000}).to_dict()
    bad = apply_action(env.world, "mayor", {"type": "set_tax", "sector": "Q", "rate": 9}).to_dict()
    out = env.apply("mayor", Decision(executed=[res, bad]))
    assert out == [res, bad]
    assert env.invalid_counts["mayor"] == 1
    assert env.turns[-1]["results"] == [res, bad]
    assert env.world.dept_funds["health"] == 1000          # not applied a second time


def test_agents_get_bound_to_the_env():
    from civitas.agents.base import Agent
    seen = {}

    class Probe(Agent):
        def bind(self, env):
            seen["env"] = env
        def act(self, obs):
            return Decision()
    env = _env()
    Probe("mayor").bind(env)
    assert seen["env"] is env
```

**Step 2:** FAIL (`unexpected keyword argument 'executed'`).

**Step 3: implement.**
- `Decision`: add `executed: list[dict] = field(default_factory=list)  # results of actions an agent already ran (tool agents)`.
- `CivitasEnv.apply`: `results = list(decision.executed)` before applying `decision.actions` (the max-actions cap applies only to `decision.actions`).
- `Agent.bind(self, env) -> None: """Called once by the runner before the episode (tool agents act on the env directly)."""` default no-op.
- `run_episode`: after `a.reset(seed)`, call `a.bind(env)`.

**Step 4:** tests + quick suite pass (the scripted benchmark is untouched). **Step 5:** commit.

---

### Task 3: Fleet memory (`civitas/fleet/memory.py`)

**Files:** Create `civitas/fleet/__init__.py` (empty docstring), `civitas/fleet/memory.py`; Test `tests/test_fleet.py`.

**Spec.**

```python
DEPARTMENTS = ("planning", "utilities", "transport", "police", "fire", "health", "education")
LIMITS = {"goals": 5, "item_chars": 200, "notes_chars": 1500, "list_items": 8}

@dataclass
class FleetState:
    plan: dict            # {"goals": [...], "budgets": {dept: $}, "priorities": {dept: str}, "notes": str, "month": int}
    plan_versions: list   # every accepted plan (copies), for reports
    memory: dict          # {dept_or_mayor: {"commitments": [...], "open_tasks": [...], "lessons": [...]}}
    log: list             # history entries {"month", "role", "kind", ...}; kinds: action, event, report, metrics
    reports: dict         # {role: latest end_turn report text}
    last_turn_month: dict # {role: month of that role's previous turn} (for digests)

    @classmethod
    def of(cls, env) -> "FleetState":   # get-or-create env.fleet
    def update_plan(self, month, **fields) -> str          # validates, caps, merges; returns a short confirmation or raises ValueError
    def update_memory(self, role, **fields) -> str          # replaces given lists, capped
    def record(self, month, role, kind, **data) -> None
    def history(self, kind=None, months=3, role=None, now=None) -> list[dict]   # newest last, max 40 entries
    def digest(self, role, now) -> dict                     # since role's last turn: its failed actions, events, messages count
```

Validation rules for `update_plan`: `goals` list of ≤5 strings (each cut to 200 chars); `budgets` dict with keys in
DEPARTMENTS and non-negative numbers (unknown key → `ValueError` naming the valid ones); `priorities` dict with
DEPARTMENTS keys; `notes` ≤1500 chars. Partial updates merge into the current plan. Every accepted update appends
a copy to `plan_versions`. `update_memory` accepts `commitments` / `open_tasks` / `lessons` lists (≤8 items, 200
chars each); anything else → `ValueError`.

`record(..., kind="metrics")` stores a compact snapshot (population, treasury, happiness, city score) — written by
the ToolAgent at the start of each manager turn from `env.world.history[-1]`.

**Tests:** plan validation (unknown department, negative budget, too many goals cut to 5), merge semantics, versions
kept, memory caps, `history(kind="action", months=1)` filters by month, `digest` returns only this role's entries
since its last turn.

Commit.

---

### Task 4: Tools (`civitas/fleet/tools.py`)

**Files:** Create `civitas/fleet/tools.py`; Test `tests/test_fleet.py`.

**Spec.** `ToolBox(env, role)`:
- `schemas() -> list[dict]` — OpenAI `tools` list for this role (look-ups + turn control + the role's actions).
- `call(name: str, arguments: str | dict) -> dict` — never raises: unknown tool, bad JSON or wrong argument types
  return `{"ok": False, "error": "..."}` with a hint the model can act on.
- `ended: bool`, `report: str`, `executed: list[dict]` (action results, for `Decision.executed`),
  `messages: list[dict]` (for `Decision.messages`).

**Look-up tools** (all roles):

| tool | args | returns |
|---|---|---|
| `city_overview` | – | population, happiness, unemployment, treasury, debt, income/expenses last month, department funds, taxes, demand, power/water/sewage supply vs demand, service coverage %, scores, trend vs 3 months ago (population, treasury, happiness) |
| `inspect` | `x, y, radius` (radius ≤ 4, clipped) | list of non-empty tiles: `{x, y, what}` where `what` = zone letter + level, or the facility tier name (`world.facility_spec`), `road`, `river`; plus per tile `powered`, `watered`, `pollution` (2 dp) for zones. Empty land is summarised as a count. |
| `find_sites` | `purpose` (enum per role), `near_x?`, `near_y?` | up to 6 sites from the observation's site hints (see below), sorted by distance to `near` when given |
| `my_facilities` | – | the role's buildings: for utilities the plants per network (kind, tier, output, crowd, quality); for services `world.service_info` entries of its structs; plus `world.upgrade_options(...)` offers as the observation builds them |
| `unlocks` | – | `world.locked_now()` with the unlock population each, and `world.next_milestone()` |
| `get_plan` | – | the city plan (full for the mayor, the role's slice + goals otherwise) |
| `get_messages` | – | council messages visible to the role in the last `council_window` months |
| `history` | `kind?` (`action`/`failure`/`event`/`report`/`metrics`), `months?` (≤12) | `FleetState.history(...)`; `failure` = action entries with `ok: False` |

`find_sites`: build `observation.build_observation(world, role, ..., site_hints=True)` once per call and pick the
relevant list. Define `SITE_PURPOSES = {role: {purpose: path-into-obs}}` by reading `core/observation.py`
(utilities: `coal_or_gas`, `wind`, `pump`, `sewage`, `near_homes`, `riverbank_2x2`; planning: `zone_R`, `zone_C`,
`zone_I`, `zone_O`, `zone_M`, `road_extension`; police: `station`, `jail`; fire: `station`, `shelter`, `levee`;
health: `clinic`, `hospital`, `landfill`; education: `school`, `high_school`, `university`; transport:
`bus_stops`, `train_station`, `harbour`, `metro_station`, `congested_roads`; mayor: none). Verify each key against
the observation code; a purpose whose list is absent returns `[]` with a note ("unlocks at …" if locked).

**Turn-control tools:** `send_message(to, text)` (to = role name or `all`; appended to `self.messages`),
`update_memory(commitments?, open_tasks?, lessons?)`, `update_plan(goals?, budgets?, priorities?, notes?)` (mayor
only), `end_turn(report)` (sets `ended`, stores the report in `FleetState.reports` and the log).

**Action tools:** one per action in `ROLES[role]["actions"]` except `noop`. Schemas are **derived** from
`actions.action_docs()`: parse the example JSON, drop `type`, map each example value to a JSON type (`int`→
`integer`, `float`→`number`, `str`→`string`, `bool`→`boolean`, `list`→`array` with `items: {}`, `dict`→`object`),
no required fields (the handlers already produce precise errors), description = the doc text. Also add the
optional `tier` integer to build actions whose line has tiers (the doc text already explains it). Execution:
`apply_action(world, role, {"type": name, **args})`, append the result to `executed`, `FleetState.record(month,
role, "action", action=..., ok=..., message=..., cost=...)`, return `{ok, message, cost, budget_left}` where
`budget_left` = `world.dept_funds[ROLES[role]["budget"]]` or the treasury for the mayor.

**Tests:**
- schemas are valid (names unique, every action of every role present, `type` absent from properties);
- `call("build_power_plant", {"kind": "wind", "x": …, "y": …})` on a prepared world builds it and returns `ok`;
- a failing action returns `ok: False` with the handler's message and is recorded as a failure;
- bad JSON / unknown tool / wrong role for `update_plan` → `ok: False`, no exception;
- `inspect` clips radius and names a tier-2 building by its tier name;
- `find_sites` returns ≤ 6 dicts with `x`, `y` for every purpose of every role on a fresh and a 24-month scripted
  world (no `KeyError`);
- `end_turn` sets `ended` and the report.

Commit.

---

### Task 5: Briefings and prompts (`civitas/fleet/briefing.py`)

**Files:** Create `civitas/fleet/briefing.py`; Test `tests/test_fleet.py`.

**Spec.**
- `system_prompt(role, cfg: FleetAgentConfig, months) -> str`, sections in order:
  1. "How to play" (fixed): you are the <title> in a fleet of AI agents running a city; the manager (mayor) sets the
     plan, departments execute; each turn you call tools; look things up before acting; action tools run
     immediately and report success or failure — fix failures instead of repeating them; stay within your
     budget; call `end_turn` with a short report when done; tool-call limit N per turn.
  2. The role's mission (`ROLES[role]["mission"]`) and short game rules (reuse `agents/prompts.GAME_RULES`, formatted).
  3. "Coach's playbook" — the fleet playbook, then the role's playbook (omit empty ones).
- `briefing(env, role) -> str` (the first user message), compact (target < 600 words):
  - header: date, months left, scores (city / yours), mode;
  - **mayor:** headline metrics + 3-month trend, finances (treasury, income, expenses, department funds), events
    since last turn, latest department reports, current plan (or "no plan yet — write one with update_plan"),
    next milestone;
  - **departments:** its plan slice (budget target, priority, goals), its budget (dept funds), 5–8 key metrics for
    the role (define `KEY_METRICS = {role: [...]}` from the observation `city`/`services` fields), "since your last
    turn" digest (failed actions, events, message count), its memory, next milestone.
- Records the metrics snapshot in the log at the start of the mayor's briefing.

**Tests:** each role's briefing renders on a fresh env and after 12 scripted months; it mentions the budget,
the plan slice and the playbook; word count < 700; the mayor's briefing lists department reports once present.

Commit.

---

### Task 6: `ToolAgent` and fleet configs

**Files:** Create `civitas/fleet/agent.py`, `civitas/fleet/config.py`; Modify `civitas/config.py`
(`build_agents`), `civitas/bench/runner.py` (`agent_signature`, meta); Test `tests/test_fleet.py`.

**Config format** (new `kind = "fleet"`):

```toml
[fleet]
mode = "autonomous"            # or "supervised" (metadata for now)
playbook = """Grow steadily, never let the treasury go negative, ..."""
provider = "arcee"
model = "trinity-large-thinking"
max_steps = 20                 # tool calls per turn
max_tokens = 16000             # per model call
turn_tokens = 80000            # completion tokens per turn, all calls
temperature = 0.4

[agents.mayor]
kind = "fleet"
playbook = "..."               # optional per-role playbook; any [fleet] key can be overridden per role

[agents.utilities]
kind = "fleet"
model = "trinity-large-thinking"
```

- `fleet/config.py`: `FleetAgentConfig` dataclass (mode, fleet_playbook, playbook, provider, model, max_steps,
  max_tokens, turn_tokens, temperature, extra_body) and `fleet_agent_config(cfg, role) -> FleetAgentConfig` merging
  `[fleet]` and `[agents.<role>]`. `fleet_version(cfg) -> str` = first 12 hex of sha1 over the canonical JSON of
  `[fleet]` + the fleet agents' specs.
- `build_agents`: `kind == "fleet"` → `ToolAgent(base, cfg=fleet_agent_config(cfg, role), provider=provider_config(...), months=game.months)`;
  raise `ValueError("fleet agents support single-city games only")` when the role has a city prefix.
- `agent_signature`: fleet roles render as `role=fleet:<model>`; append ` [autonomous|supervised]` once when any
  fleet agent exists. `run_episode` meta gains `"fleet_version"` when the config has a `[fleet]` section.

**`ToolAgent` loop** (core — write it close to this):

```python
class ToolAgent(Agent):
    kind = "fleet"

    def bind(self, env):
        self.env = env
        FleetState.of(env)

    def describe(self):
        return f"fleet:{self.cfg.model}"

    def act(self, obs: dict) -> Decision:
        env, role = self.env, self.role
        box = ToolBox(env, role)
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": briefing(env, role)}]
        tools, trace, usage, nudged = box.schemas(), [], {"prompt_tokens": 0, "completion_tokens": 0}, False
        error = None
        for step in range(self.cfg.max_steps):
            try:
                out = self.client.chat(self.cfg.model, messages, temperature=self.cfg.temperature,
                                       max_tokens=self.cfg.max_tokens, extra={"tools": tools, **(self.cfg.extra_body or {})})
            except LLMError as e:
                error = f"api_error: {e}"
                break
            for k in usage:
                usage[k] += int((out["usage"] or {}).get(k) or 0)
            calls = out["tool_calls"]
            messages.append({"role": "assistant", "content": out["content"] or "", **({"tool_calls": calls} if calls else {})})
            if not calls:
                if nudged:
                    break
                nudged = True
                messages.append({"role": "user", "content": "Use the tools to act, then call end_turn with a short report."})
                continue
            for c in calls:
                res = box.call(c["function"]["name"], c["function"].get("arguments") or "{}")
                trace.append({"step": step, "tool": c["function"]["name"], "args": c["function"].get("arguments"),
                              "result": res})
                messages.append({"role": "tool", "tool_call_id": c.get("id") or f"call_{step}", "content": json.dumps(res)[:4000]})
            if box.ended or usage["completion_tokens"] >= self.cfg.turn_tokens:
                break
        d = Decision(executed=box.executed, messages=box.messages, thinking=box.report or "(no report)")
        d.parse_error = error
        d.meta = {"model": self.cfg.model, "messages": messages, "tool_trace": trace, "usage": usage,
                  "steps": len(trace), "ended": box.ended, "finish_reason": "end_turn" if box.ended else "limit"}
        return d
```

Notes: the assistant message sent back must not include `reasoning_content`; tool results are truncated to 4000
chars; an exception inside a tool is already caught by `ToolBox.call`. `runner.run_episode` logs `meta["messages"]`
as `prompt` today; also log `tool_trace`, `steps`, `ended` when present (small change in the record building).

**Tests — fake tool-calling server** (`tests/test_fleet.py`): an HTTP handler whose responses come from a
per-role script keyed by the role named in the system prompt (e.g. search the system message for the role title).
Script: mayor → `update_plan(budgets={"utilities": 3000})`, `allocate_budget(department="utilities", amount=3000)`,
`end_turn("plan set")`; utilities → `find_sites(purpose="wind")`, then `build_power_plant(kind="wind", x, y)` using a
site from the previous tool result (the handler can parse the last tool message), `end_turn("built wind")`; other
roles → `end_turn("nothing to do")`. Serve plain JSON (`stream` ignored). Run `run_episode` on a 2-month config with
all roles `kind = "fleet"` pointing at the fake server and assert:
- utilities funds were allocated and a wind turbine exists;
- `trajectories.jsonl` has 16 records with `tool_trace`; the mayor's plan is in `env.fleet.plan`;
- the signature contains `fleet:fake` and `[autonomous]`; `meta.json` has `fleet_version`;
- a role whose model never calls tools ends after one nudge (no infinite loop), and a scripted limit of
  `max_steps = 3` stops a looping model.

Commit.

---

### Task 7: End-of-run fleet report (`civitas/fleet/report.py`)

**Files:** Create `civitas/fleet/report.py`; Modify `civitas/bench/runner.py`; Test `tests/test_fleet.py`.

**Spec.** `fleet_report(env, summary, trajectories: list[dict]) -> (dict, str)` — JSON and Markdown:
- scoreboard: final city + role scores, population, treasury, bankrupt;
- the final plan and number of plan versions; per role: turns, tool calls, actions, failed actions (and top 3
  failure messages), tokens (prompt/completion), turns that hit the step or token limit, lessons in memory;
- coaching hints (rules, each only when triggered): a role with ≥ 3 failures sharing a message prefix; a department
  whose funds stayed above 2× its plan budget for 3+ months (unused money); a department with 0 actions for 6+
  months; treasury negative for any month; turns ending on the limit > 20%.
- `run_episode` writes `fleet_report.json` and `fleet_report.md` into the run directory when any agent is a
  `ToolAgent` (collect the records it already logs).

**Tests:** on the fake-server run from Task 6 the report files exist, the Markdown names every role, and a
constructed failure pattern produces the repeated-failure hint.

Commit.

---

### Task 8: Example config, docs, live smoke run

**Files:** Create `configs/fleet_trinity.toml`, `configs/fleet_smoke.toml`; Modify `README.md`, `CLAUDE.md`,
`docs/MECHANICS.md` (roles/agents section only).

- `fleet_trinity.toml`: all eight roles `kind = "fleet"`, trinity-large-thinking, a sensible default fleet playbook
  (keep a treasury reserve, fund utilities first, then services as the city grows, use tiers when unlocked) and
  short department playbooks; 120 months, `act_every = 3`.
- `fleet_smoke.toml`: same fleet, `months = 12`, `act_every = 3`.
- README: a "Fleets" section — manager + departments, tools list, memory layers, playbooks, modes, config format,
  report files. CLAUDE.md: a short architecture paragraph on `civitas/fleet/` and the `bind`/`executed` hooks.
- Run `civitas run -c configs/fleet_smoke.toml --seed 0` (live, needs `ARCEE_API_KEY`; run in the background and
  watch it). Report: tool calls per turn, failed-action rate vs the old agent's 19%, turns ending on limits,
  tokens per turn, and what the report's hints say. If the key is missing, say so.
- Full `pytest -q` passes.

Commit. Then use superpowers:finishing-a-development-branch.
