# RL tasks and training

`civitas/rl/` cuts the game into short, deterministic, verifiable tasks (one department's turn from a saved
city, scored against the scripted turn a few months on) and trains on them with TRL's GRPO on Modal.

Contents:

- [RL tasks](#rl-tasks)
- [RL training](#rl-training)

## RL tasks

A full eval episode (12 months x 8 roles, ~10M tokens) is far too long for RL. `civitas/rl/` cuts the game into
short, verifiable, deterministic tasks: **one department's turn from a saved city**, played through the real fleet
tools and scored against the scripted fleet's own turn a few months on. No model is called by any of it except
`civitas rl eval`, which talks to the endpoint you give it.

```bash
civitas rl bank build --name rl-v1 -c configs/fleet_smoke_late.toml -c configs/fleet_open_late.toml \
    --seeds 100-104,300-301 --months 24,48,72,96 --workers 4       # 448 tasks, ~6 min
civitas rl bank info rl-v1 [--tasks]                               # composition
civitas rl show rl-v1 fleet_smoke_late/s100/m48/planning           # what the model sees (system, briefing, tools)
civitas rl check --bank rl-v1 --split train --n 160                # validate the reward with hand-made policies
civitas rl eval --bank rl-v1 --split test --base-url http://localhost:8000/v1 --model trinity-nano [--n 4]
```

**Tasks and the bank** (`rl/bank.py`, `runs/_tasks/<name>/`). A task is (config, seed, month, role). The build plays
each (config, seed) once with the scripted fleet (the warm-up line-up: deterministic, no model) and at each task
month, before any turn, writes a gzipped pickle of the env and the scripted agents (~110-220 KB); the roles before
the task's role in turn order play that month when the task is loaded. `index.jsonl` has one line per task:
`task_id` (`fleet_smoke_late/s100/m48/planning`), config, seed, month, role, `style` (the config's
`[fleet].briefing`, guided / open), `split`, the hand-over metrics, `actionable` / `why`, what the scripted turn did,
`state_fp` (the city at the turn, checked on every load: a bank built by other sim code is refused) and the cached
scores of the baseline branch (and of the noop branch) for `--horizon-max` (12) months. Seeds: everything below 100
and every seed an eval suite in `configs/` names is refused (the evals play 0, 4, 5, 11, 12, 16, 18, 20 and v1
0-7); training seeds are 100-299, held-out test seeds 300-339 (`split`). A task is *actionable* when its role could
do anything: the mayor always, a department when its funds pay for a ready site or tier upgrade, or when the scripted
turn acted or asked the mayor for money; the rest stay in the index but `TaskBank.tasks()` (and check / eval) leave
them out. Snapshots are not HMAC-signed like session checkpoints (a bank is built once and copied to training
machines, where a per-installation key would refuse it); the index holds each snapshot's SHA-256, checked before
unpickling. A bank is code: load only banks you built or trust.

rl-v1 (above): 448 tasks (train 320, test 128; guided 224, open 224; 112 per month, 56 per role), 412 actionable
(transport 32 of 56: before its first bus line it often has nothing to buy), the scripted turn idle in 200; at 6
months the noop turn is worse than the scripted one in 148, equal in 196 and better in 104 (mean +0.12 city points).

**Episodes** (`rl/episode.py:TaskEpisode`). `reset()` returns the system prompt, the briefing (in the task's style)
and the tool schemas exactly as `fleet/agent.py:ToolAgent` builds them (a ToolAgent is bound for the prompt; the
toolbox is the model's: `branch_dezone`, `max_steps`, `next_line`, so every guard, refusal, hint and end_turn
reminder is there). `step(name, args_json)` returns the tool message content ToolAgent would send (`_budget`'s
`steps_left` / wrap-up notes, compact JSON cut at `TOOL_MESSAGE_CHARS`, "not run" after end_turn or past
`max_steps`); `reply(content, tool_calls)` takes a whole assistant reply as one iteration of ToolAgent's loop (text
call recovery, NUDGE once, LENGTH_NUDGE for cut replies, the turn ending on a further reply without a call);
`replay(messages, check=True)` replays an OpenAI-format conversation and reports tool messages that differ. Done on
a successful end_turn, after `max_steps` (20) calls, or on a reply without a call past the nudge;
`end_at_action_cap=True` also ends it at the action cap (ToolAgent doesn't: a message to the mayor after the cap still
changes what the scripted mayor funds). civitas-v3's token budget: `TaskEpisode(task, turn_budget=...)` (default the
task config's, 0 for v2 banks) with each reply's `usage` passed to `reply()` / `step()`, or a ToolAgent trajectory's
`call_usage` to `replay(messages, usage=...)` (the same cutoff); check `done` before sampling the next reply.
`civitas rl eval` rows keep `turn_budget`, `budget_used`, `budget_cut` and `budget_over` when the task has a budget,
and transcripts each call's `call_usage`. Left to the trainer: `max_tokens`, `turn_tokens` and sampling.
`tests/test_rl.py` checks a ToolAgent turn and the same replies through `reply()` give identical messages, trace and
reward. Prompts are ~20-25k characters (system 7-11k, briefing 2-4k, tool schemas 6-10k).

**Reward** (`rl/reward.py`). From the task's state two branches go on: the scripted fleet agent's turn (the
baseline) and the model's; in both the rest of the month and the next H months are played by the scripted fleet,
every role (the trained role too, after its one turn). The baseline branch is the scripted game itself, cached in the
bank, so a reward is one replay:

    reward = (city_model - city_base) + w_role * (role_model - role_base)
             - 0.05 * failed actions - 0.02 * guard refusals - 0.02 * invalid calls - 0.1 * (no end_turn)

read H = 6 months after the task's month (two turns at act_every 3), components reported apart
(`RewardConfig`: `horizon`, `w_role` = 0.25, `p_failed`, `p_refused`, `p_invalid`, `p_unended`). The defaults come
from the check below: the paired effect size (mean / sd of policy - noop over tasks) of the scripted and the oracle
turn is 0.08 / 0.17 at H = 3, 0.24 / 0.38 at H = 6 and 0.23 / 0.27 at H = 12 (train split), so 6 months carries the
most signal per second (H = 12 costs twice the time and adds drift from the scripted months after the turn); with H
= 6 the city score alone gives 0.20 / 0.38, w_role 0.1-0.25 0.24 / 0.40 and fewer ties (68 -> 60 of 160 tasks where
noop equals the baseline), and larger weights lose signal (role only: 0.21 / 0.23; role scores swing 5x the city
score, correlation 0.4). A penalty unit (0.05) is about a quarter of the mean noop-baseline gap.

**Validation** (`civitas rl check`; rl-v1, guided and open, actionable tasks; 95% t-intervals; the policies in
`rl/policies.py`): *baseline* is the scripted fleet agent's own turn on the prepared state; *scripted* its decision
sent through the episode's model toolbox; *degraded* that with ~half its actions dropped; *random* 1-4 action calls
with random arguments; *oracle* the guided briefing's own calls (a department's Next / This turn / tier-upgrade
calls, the mayor's covered to-fund top-ups) before the scripted ones.

| policy (H = 6) | train, 160 tasks | test, 118 tasks | train: + / 0 / - |
|---|---|---|---|
| oracle | +0.154 [+0.058, +0.249] | +0.161 [+0.026, +0.295] | 29 / 97 / 34 |
| scripted | -0.001 [-0.009, +0.007] | -0.001 [-0.010, +0.007] | 13 / 117 / 30 |
| baseline | -0.004 [-0.006, -0.002] (city and role exactly 0; its own failed calls) | -0.003 | 0 / 148 / 12 |
| degraded | -0.130 [-0.214, -0.045] | -0.120 [-0.289, +0.048] | 29 / 64 / 67 |
| noop | -0.181 [-0.295, -0.068] | -0.207 [-0.402, -0.011] | 38 / 60 / 62 |
| random | -0.299 [-0.416, -0.181] | -0.338 [-0.534, -0.142] | 30 / 0 / 130 |

Ordering oracle > scripted ~ baseline > degraded > noop > random at every horizon on both splits (H = 3: noop's CI
holds 0). Within a task the five episode policies give different rewards in every task (105 of 160 without random;
noop differs from the baseline in 100), mean within-task std 0.27. Determinism: 40 of 40 episodes replayed in fresh
processes gave identical branches. Wall time per reward at H = 6 (prepare + turn + branch, one core): mean 0.5 s,
p90 0.8-1.1 s, a cold task (snapshot loaded, earlier roles played) +0.2 s; H = 3 0.15-0.2 s, H = 12 1.1-1.5 s.
By role at H = 6 the oracle gains most as mayor (+0.98: funding the to-fund line beats the scripted mayor) and noop
loses most as planning (-0.75) and police (-0.30).

**Endpoint runner** (`civitas rl eval`, `rl/evaluate.py`): each task (and `--n` samples) is played by the fleet's own
ToolAgent loop with one ChatClient on `--base-url` / `--model` (the task config's playbooks and limits; `--api-key-env`,
`--temperature`, `--max-tokens`, `--extra-body`, `--no-stream`), then scored. `runs/_rl_evals/<name>_<stamp>/`:
`results.jsonl` (reward and components, counts, tokens, steps, finish reason), `transcripts.jsonl`, `summary.json` /
`summary.md` (mean reward with CI, by role / style / month, the bank's noop reference and model - noop, failed /
refused / invalid per task, tokens, within-task std over samples).

## RL training

GRPO on the RL tasks above with TRL's `GRPOTrainer` (trl 1.15.0) on Modal H100s: Qwen/Qwen3-4B-Instruct-2507 with
LoRA, one department first. Everything is ready to launch, and nothing here launches it: every Modal function costs
money, so the commands below are for when the spend is approved.

```bash
# 1. a bank (locally, free; ~4 workers): training seeds 100-299 (dev seeds 290-299 held out) + test seeds 300-339
civitas rl bank build --name rl-train -c configs/fleet_smoke_late.toml -c configs/fleet_open_late.toml \
    --seeds 100-299,300-339 --workers 4            # 480 games, 15,360 tasks, ~2.5-3.5 h at 4 workers
modal volume create civitas-rl && modal volume put civitas-rl runs/_tasks/rl-train /tasks/rl-train

# 2. the plan, the pins and the cost (no Modal account needed)
python -m civitas.rl.modal_app --print-config [-c configs/rl_qwen3_4b.toml] [--set grpo.max_steps=50]

# 3. on Modal
modal run civitas/rl/modal_app.py::check                                    # image versions + bank (CPU, cents)
modal run --detach civitas/rl/modal_app.py --action baseline                # base model on the test split
modal run --detach civitas/rl/modal_app.py --action sft                     # optional SFT warm start
modal run --detach civitas/rl/modal_app.py --action train                   # GRPO; resumes after preemption
modal run --detach civitas/rl/modal_app.py --action evaluate --run qwen3-4b-planning-v1
modal volume get civitas-rl /evals runs/_rl_evals_modal                      # results back

# local: the env / reward / masking plumbing with a 0.6B model, 2 GRPO steps (CPU or Apple MPS, no vLLM)
uv venv --python 3.12 .venv-train && uv pip install --python .venv-train/bin/python torch trl peft datasets accelerate
civitas rl bank build --name rl-dev -c configs/fleet_smoke_late.toml -c configs/fleet_open_late.toml \
    --seeds 100,300 --months 48,72 --workers 2
PYTHONPATH=$PWD .venv-train/bin/python -m civitas.rl.train -c configs/rl_smoke_local.toml
```

**The pipeline.** bank -> baseline (`civitas rl eval` of the base model on the test split: its failed / invalid
calls per task say whether SFT is needed) -> optional SFT -> GRPO -> evaluate (the same test split, `--n 2`) ->
paired comparison, `civitas rl compare <baseline eval dir> <trained eval dir>`: the per-task differences of the mean
reward on the tasks both scored (mean and 95% t-interval, overall and by role, wins / ties / losses); each
summary.md also gives model - noop per task. With `suite=configs/eval_v2.toml` (`baseline` / `evaluate` functions) the guided eval
suite runs on the served model too (the full 12-month late start, against the scripted yardstick).

**The environment** (`civitas/rl/trl_env.py`, written against trl 1.15.0's source). TRL's `environment_factory` makes
one env instance per rollout, calls `reset(**row)` (its string is appended to the row's user message) and exposes the
public methods of the class as tools, called with the parsed arguments; the reward function gets the instances as
`environments`. `task_set(bank, split, roles=...)` builds the rows (`prompt` = [system prompt, empty user], `task_id`,
`environment`) by resetting each task once; `factories()` generates one class per tool set (role + schema digest,
a dict factory: each row's `environment` picks it) with one method per ToolBox tool, each running the call through
`TaskEpisode.step`, so a tool result is exactly the tool message ToolAgent sends (guards, refusals, hints, refreshed
Next lines, `steps_left`). TRL renders tool schemas from docstrings and type hints, which can't express every
civitas schema, so `CivitasGRPOTrainer` (`trainer_class()`) swaps the exact ToolBox schemas into the trainer after
`__init__` (`install_schemas`: `trainer._env_tools`, read only when rendering a prompt); the generated methods still
carry typed signatures and Google docstrings that `get_json_schema` turns into a close equivalent (tested).
`CivitasReward` scores each rollout with the counterfactual reward of `reward.py` (H 6, w_role 0.25, the penalties)
plus `p_malformed` for a `<tool_call>` TRL couldn't parse, and logs every component and count through `log_metric`
(`civitas/reward`, `city`, `role`, `penalty`, `actions`, `failed`, `refused`, `invalid`, `malformed`, `ended` = the
end_turn rate, `end_turn_first`, `lookups`, `messages`, `money_asks`, `asks_held`, `overrides` (earned confirms),
`loans`, `repaid`, `take_backs`, `top_ups`, `any_failed` ...) and per completion through `log_extra`. With
`[env].reward_workers` the 6-month branches run in a spawned process pool (the env pickled after the turn; identical
rewards, tested). The trainer also drops zero-advantage rollouts (all-equal groups) from the loss
(`drop_zero_std`, logged as `civitas/dropped_zero_std`).

How a TRL rollout differs from ToolAgent's loop, all TRL's doing: no nudges (a reply without a tool call ends the
rollout: unended, `p_unended`), no recovery of calls written as text (`malformed`), and the model sees end_turn's
result and replies once more (calls after end_turn are "not run" and cost nothing). Messages are append-only: TRL
renders the prompt once and grows the sequence by token concatenation (prompt + completion + tool suffix ids), the
env never rewrites anything, and the Qwen3-2507 template renders each message of a turn as a pure token suffix
(`tests/test_rl_trl.py`). Tool tokens are out of the loss (TRL's `tool_mask`); `max_completion_length` counts them.
`CIVITAS_RL_VERBOSE=1` prints each reset, tool call and reward batch.

**Tests.** `tests/test_rl_trl.py` drives the env classes with a stand-in for TRL's loop (no trl needed: tool
discovery, tool messages = TaskEpisode's, append-only conversation, reward and logging, penalties, the reward
workers, SFT export, `civitas rl compare`). With trl installed (the training venv) it also runs a real
`CivitasGRPOTrainer` step with a tiny random Qwen3 and scripted generations (one rollout calls find_sites then
end_turn, the other answers in text): TRL's own loop runs the env's tools, the tool responses are masked out of the
loss, the two rollouts score apart and an optimizer step happens (~90 s on a laptop CPU). On macOS, where TRL's
Triton log-prob kernel doesn't exist, `train.py` swaps in a plain-PyTorch one (`_TorchLogProbs`). The 0.6B smoke
(`configs/rl_smoke_local.toml`, `CIVITAS_TRL_SMOKE=1` for its test) is the same with a real model's generations.
It is too heavy for a busy 16 GB Mac: on 2026-10-10 generation, the env and the reward ran (two rollouts in ~4.5
min, no tool call in 384 tokens, both -2.362: a zero-variance group), but the backward pass over two ~7k-token
sequences hadn't finished after an hour under memory pressure (CPU attention with a padding mask materializes the
7k x 7k matrices; Apple MPS ran out of memory on them). Use a Linux GPU box, or Modal's `check` + a short `train`.

**Config** (`configs/rl_qwen3_4b.toml`, read by `civitas/rl/train.py` and `modal_app.py`; `--set section.key=value`):
[data] bank / split / roles (`["planning"]`) / styles / dev_seeds; [reward] and [env]; [lora] r 32, alpha 64 on
q,k,v,o,gate,up,down; [grpo] passed to `GRPOConfig` as is: lr 2e-5 (constant after 10 warmup steps), beta 0,
group 8, 8 tasks per step (64 completions: batch 1 x 64 accumulation), T 1.0, no top-p / top-k, 20 tool iterations,
`max_completion_length` 20480, epsilon 0.2 / 0.28, `loss_type` dapo, truncated completions masked, vLLM importance
sampling on, 300 steps, a checkpoint every 10 steps, the dev-seed reward every 25 (`eval/civitas/*`); [vllm],
[eval], [modal] (GPUs, volumes, timeout, retries, pins: python 3.12, vllm 0.31.0, trl 1.15.0, peft 0.21.2, torch and
transformers as vLLM resolves them) and [estimate]. `configs/rl_smoke_local.toml` is the laptop smoke.
The launcher leaves files with a [grpo] table out.

**Modal** (`civitas/rl/modal_app.py`). One Volume (`civitas-rl` at /vol: tasks/, train/<run>/, sft/, evals/) and one
for the Hugging Face cache; the repo is mounted at /repo. `train` runs on `H100:2`: `vllm serve` on GPU 0 with NCCL
weight transfer (`--weight-transfer-config '{"backend":"nccl"}' --logprobs-mode processed_logprobs --max-logprobs -1`;
TRL merges the LoRA and pushes the weights after each step, no vLLM adapter path) and the trainer on GPU 1 in TRL's
server mode. GPU functions are preemptible: `retries` (10) restarts the call and `train.py` resumes from the newest
complete checkpoint on the volume (committed after each save). One call is at most 24 h; a longer run is the same
command again. `baseline` / `evaluate` serve the base model / the trained adapter (`--enable-lora`) with the hermes
tool parser (the chat template the trainer used) and run `civitas rl eval` on the test split. `sft` exports
scripted-policy transcripts (`civitas/rl/sft.py`: the better of oracle / scripted per task, kept when it made no
failed or invalid call and scored >= 0; OpenAI chat JSONL with tool calls and the exact schemas) and trains a LoRA
with TRL's SFTTrainer (assistant-only loss). `python -m civitas.rl.sft export --bank rl-train --out x.jsonl` does
the export locally.

**Budget** (estimates; measure step 1 and re-plan). H100 $3.95/GPU-h. GRPO ~3 min a step (generation ~1.5 min: 64
rollouts x ~20 calls; training ~1.5 min: 64 sequences x ~15-25k tokens), 300 steps + 12 dev evals ~16 h on 2 GPUs
~$125; baseline and final evaluate ~$4-12 each; optional SFT $8-24; smoke / debug runs $50-150: ~$200-330 in all.
`--print-config` prints it for the config at hand.

**Metrics to watch** (TRL logs plus `civitas/*`): `reward` mean and `reward_std`; `frac_reward_zero_std` and
`civitas/dropped_zero_std` (no signal when most groups tie: harder tasks, more roles, higher temperature);
`civitas/city` vs `civitas/role` (the role term shouldn't carry the reward alone); `civitas/ended` (end_turn rate)
and `civitas/end_turn_first` (a noop collapse: ending at once is safe while the noop gap is small);
`civitas/failed` / `refused` / `invalid` / `malformed`; `completions/mean_length` and `clipped_ratio` (truncation);
`tools/call_frequency`; `entropy`, `clip_ratio/*`, `sampling/sampling_logp_difference` and
`sampling/importance_sampling_ratio` (vLLM vs trainer mismatch); `grad_norm`; and the dev reward `eval/civitas/reward`
every 25 steps against the baseline's.

**Known risks and reward-hacking signals.** The reward reads the city 6 months on: a turn that borrows (`loans`), takes
money back (`take_backs`), forces refusals through (`overrides`) or spams the mayor (`money_asks`, `asks_held`) can
look good at H = 6 and cost later; inspect those counts against the baseline's and re-score suspicious checkpoints
at H = 12 (`civitas rl eval --horizon 12`). `end_turn_first` rising with flat reward means the policy found the noop
(the scripted turn's edge over noop is ~0.1-0.2 points). A role-score gain without a city gain is the role term
being gamed (`w_role`). Zero-variance groups are frequent (the sim is deterministic: only sampling varies a group).
Truncation at 20k tokens on long turns (20 tool results of up to 4,000 characters) masks those rollouts. The tool
schemas reach the model only through `install_schemas`, which depends on TRL's private `_env_tools` (refused with a
clear error if it's gone). Qwen3.5 is the stretch model (Gated-DeltaNet layers, the `qwen3_coder` parser, reported
silent LoRA no-ops in vLLM: test that generation changes after a weight sync).
