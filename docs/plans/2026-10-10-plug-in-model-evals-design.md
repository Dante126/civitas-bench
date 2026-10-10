# Plug-in model evals: design

Goal: add a model, press Evaluate, read a ranked, comparable result, all from the viewer (`civitas serve`). The
commands stay underneath for scripts and tests. Models come from the Arcee API and third-party APIs (OpenRouter,
OpenAI, Anthropic's OpenAI-compatible endpoint); no self-hosted checkpoints for now.

## 1. Model registry: `configs/models.toml`

One entry per model that can be evaluated; machine-written by the viewer (a fixed header comment, entries sorted),
hand-editable too.

```toml
[models.trinity-large-thinking]
provider = "arcee"
temperature = 0.4
max_tokens = 16000
[models.gpt-5]
provider = "openai"
id = "gpt-5"                 # the served id when it differs from the key
temperature = "default"      # send no temperature: reasoning models take only their own
extra_body = {}
notes = "..."
[providers.together]         # a provider not built in
base_url = "https://api.together.xyz/v1"
api_key_env = "TOGETHER_API_KEY"
```

- Model keys: `provider`, `id`, `temperature`, `max_tokens`, `extra_body`, `notes`. Environment keys (the compute
  budget, equal for every model: `max_steps`, `turn_tokens`, `act_every`, `max_actions_per_turn`, playbooks) are
  refused in an entry.
- `temperature` is a number, unset (the suite config's, 0.4), or `"default"`: the request then carries no
  temperature at all (`fleet.temperature=default`; the fleet config reads it as None and the client leaves the key
  out, preflight too), since OpenAI's reasoning models refuse any temperature but their own.
- Built-in providers gain `openai` (`https://api.openai.com/v1`, `OPENAI_API_KEY`, sends `max_completion_tokens`)
  and `anthropic` (`https://api.anthropic.com/v1/`, `ANTHROPIC_API_KEY`). `ProviderConfig.max_tokens_field` names
  the token-limit field.
- The New run dialog's model list comes from the registry.
- API keys live in `./.env`; the viewer shows set / not set and can write a pasted key (localhost and launching
  allowed only; a key is never sent back to the browser).

## 2. Environment versions

A version (`civitas-v1`) is a frozen, named fingerprint stored in `configs/envs/<name>.json` (committed), with a git
tag `env/<name>` at freeze. The standard suite names its version (`env = "civitas-v1"` in the suite file).

The fingerprint is behavioural: a probe replays every suite seed with the scripted fleet (the yardstick's line-up,
no model, ~10 s for 8 seeds in parallel) and hashes, per part:

- `config`: the suite and each entry's effective config with the model keys removed;
- `prompts` / `tools`: each role's system prompt and tool schemas;
- per seed: `briefings` and `results` (the briefing text and tool results the scripted fleet sees after the
  hand-over) and `metrics` (the monthly metrics);
- per seed: `model`, what only a model-backed agent meets (ToolAgent's loop and its ToolBox flags): on a copy of the
  city at the hand-over each fleet role plays one real ToolAgent turn against a canned client (no network) that
  replies without a tool call, then cut at max_tokens, then with the scripted department's calls and end_turn.

Refactors and comments leave it unchanged; a sim, briefing, tool or config change moves the part it touches. A probe
is cached per code state (commit + diff hash).

An eval runs the probe first; a mismatch refuses unless the user allows drift, and a drifted eval is recorded as such
and never ranked. Bumping is deliberate: freeze `civitas-v2`, then re-run the reference models (about 12M tokens
each).

## 3. Evaluating a model

`civitas eval --model <key>` (and the viewer's Evaluate button): the default suite is `configs/eval_singles.toml`
unless `-s`; the registry entry becomes the run overrides; then

1. preflight: one real request with fleet tool schemas, which must come back as a tool call (catches a wrong key, a
   wrong id, no tool support) before any tokens are spent on runs;
2. the environment probe against the suite's version;
3. the runs, the scorecard, and a paired comparison with the suite's `baseline_model` eval on the same version.

`eval.json` records `env` (name, fingerprint, match), the model entry, the preflight.

## 4. Leaderboard

Per environment version, the latest finished, non-drifted eval of each model: score against the yardstick with its
95% interval, departments, tokens, cost, and the paired verdict against the baseline model. Evals of different
versions are never ranked or compared together. Written to `runs/leaderboard_<env>.md` as well.

## 5. The Evals page (viewer)

A full-screen page from a top-bar button, three tabs:

- **Models**: the registry as a table (provider, id, key set, latest eval on the current version and its score,
  Evaluate button). Add / edit / delete a model with a form; paste an API key per provider. Test (preflight) shows
  the result inline.
- **Evals**: running evals with progress (runs done of total, tokens so far, time left), Stop; finished evals with
  their scorecard and the paired verdict; the existing Compare view for runs.
- **Leaderboard**: the ranked table for the current version (a version picker for older ones).

A strip on top shows the environment: version, "matches" or "changed: briefings, results", and Freeze as a new
version (with the warning that the reference models then need a fresh eval).

## Testing

Probe determinism (twice, and in a fresh process); a balance or briefing change moves the right part; registry
validation; preflight against the test chat server; the leaderboard skips drifted and other-version evals; server
endpoints. No paid runs.
