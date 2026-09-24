"""Play a role from a separate process over HTTP.

1) start a game where a role is remote:   civitas run -c configs/remote.toml --serve
2) run this script:                         python examples/remote_agent.py --role planning

The loop is language-agnostic: GET /api/live/turn?role=... returns the full
observation (JSON, plus a ready-made text prompt in observation.text), and
POST /api/live/act submits {"role", "turn_id", "decision": {"actions": [...],
"messages": [...], "notes": "...", "thinking": "..."}}.

By default this script delegates to the built-in scripted agent; replace
`decide()` with your own policy, LLM framework, or human input.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

from civitas.agents.scripted import SCRIPTED


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="planning")
    ap.add_argument("--server", default="http://127.0.0.1:8765")
    args = ap.parse_args()
    policy = SCRIPTED[args.role](args.role)
    policy.reset()

    def decide(obs: dict) -> dict:
        d = policy.act(obs)
        return {"actions": d.actions, "messages": d.messages, "notes": d.notes,
                "thinking": "[remote] " + d.thinking}

    print(f"playing {args.role} against {args.server} ... (Ctrl+C to stop)")
    while True:
        try:
            turn = get(f"{args.server}/api/live/turn?role={args.role}")
        except OSError:
            time.sleep(1.0)
            continue
        if not turn.get("waiting"):
            time.sleep(0.2)
            continue
        obs = turn["observation"]
        res = post(f"{args.server}/api/live/act",
                   {"role": args.role, "turn_id": turn["turn_id"], "decision": decide(obs)})
        print(f"month {obs['month']:>3}: {res['message']}")


if __name__ == "__main__":
    main()
