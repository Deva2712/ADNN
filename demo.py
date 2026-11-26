"""Replay a saved agent run on both versions of one task (for the presentation demo).

    python demo.py                                   # first delete task, Qwen, neutral prompt
    python demo.py --pair task-305 --model Mistral-7B-Instruct-v0.3 --cond careful
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen2.5-7B-Instruct")
ap.add_argument("--cond", default="neutral")
ap.add_argument("--pair", default="task-000")
a = ap.parse_args()
for run in map(json.loads, open(f"results/{a.model}/agent_{a.cond}.jsonl")):
    if run["pair"] != a.pair:
        continue
    print("=" * 100)
    print(f"{run['id']}  ({'reversible' if run['version'] == 'rev' else 'IRREVERSIBLE'} version, {a.model}, {a.cond} prompt)\n")
    print(run["messages"][1]["content"], "\n")
    for i, s in enumerate(run["steps"], 1):
        print(f"  step {i}: {s['raw']}\n          -> {s['obs']}")
