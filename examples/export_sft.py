"""Turn logged trajectories into a chat-format dataset for fine-tuning.

Every run writes runs/<id>/trajectories.jsonl with one record per agent turn:
prompt (LLM agents) or observation_text (scripted agents), the decision, the
action results and the per-turn reward. This script keeps the best turns and
writes {"messages": [system, user, assistant]} lines, ready for SFT or as the
starting point for preference / GRPO-style training.

Fleet turns (agent "fleet:...") are skipped: they are multi-step tool-calling
conversations whose actions ran inside the loop, so they don't fit the
one-shot {system, user, assistant-JSON} format. Their full traces stay in
trajectories.jsonl (prompt + tool_trace).

    python examples/export_sft.py runs/ --role planning --min-reward 0 --out planning_sft.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from civitas.agents.prompts import system_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("--role", default=None)
    ap.add_argument("--min-reward", type=float, default=0.0)
    ap.add_argument("--skip-failed", action="store_true", help="drop turns with any failed action")
    ap.add_argument("--out", default="sft.jsonl")
    args = ap.parse_args()

    n_in = n_out = n_fleet = 0
    with open(args.out, "w") as out:
        for f in Path(args.runs_dir).glob("*/trajectories.jsonl"):
            meta = json.loads((f.parent / "meta.json").read_text())
            for line in f.open():
                t = json.loads(line)
                n_in += 1
                if str(t.get("agent", "")).startswith("fleet:"):
                    n_fleet += 1
                    continue
                if args.role and t["role"] != args.role:
                    continue
                if t["reward"] < args.min_reward or (args.skip_failed and any(not r["ok"] for r in t["results"])):
                    continue
                if "prompt" in t:
                    msgs = [m for m in t["prompt"] if m["role"] in ("system", "user")][-2:]
                else:
                    msgs = [{"role": "system", "content": system_prompt(
                                t["role"], months=meta["months"], target_pop=20000, max_actions=8)},
                            {"role": "user", "content": t.get("observation_text", "")}]
                answer = {"thinking": t["thinking"], "actions": t["actions"], "messages": t["messages"]}
                if t.get("notes"):
                    answer["notes"] = t["notes"]
                msgs.append({"role": "assistant", "content": json.dumps(answer)})
                out.write(json.dumps({"messages": msgs, "reward": t["reward"], "role": t["role"],
                                      "run_id": t["run_id"], "month": t["month"]}) + "\n")
                n_out += 1
    print(f"kept {n_out}/{n_in} turns -> {args.out}"
          + (f" ({n_fleet} fleet turns skipped: tool-calling traces don't fit this format)" if n_fleet else ""))


if __name__ == "__main__":
    main()
