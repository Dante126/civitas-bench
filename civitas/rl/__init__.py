"""RL tasks: one department's turn from a saved city, scored against the scripted fleet's own turn.

- `bank.py`: the task bank (`civitas rl bank build`): scripted warm-ups on training / test seeds, snapshots of the city
  at chosen months, a JSONL index with the scripted baseline's scores cached per task;
- `episode.py`: `TaskEpisode`, one turn through the real fleet tools (what fleet/agent.py:ToolAgent sends and gets);
- `reward.py`: the counterfactual reward (the city H months on against the scripted baseline's branch);
- `policies.py` / `check.py`: hand-made policies and the validation that the reward means something (`civitas rl check`);
- `evaluate.py`: a model on an OpenAI-compatible endpoint over a bank split (`civitas rl eval`, `civitas rl compare`);
- `trl_env.py` / `train.py`: GRPO with TRL (environment classes, dataset rows, the reward function, the trainer);
- `sft.py`: scripted-policy turns as chat transcripts for an SFT warm start; `modal_app.py`: the Modal app.
"""
