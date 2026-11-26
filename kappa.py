"""Agreement on the hand-labelled sample (Section 4.2 quality control).

Two team members fill the labeller_1 / labeller_2 columns of data/human_label_sample.csv with
reversible, partial or irreversible, without looking at data/pairs.jsonl. Then:

    python kappa.py
"""
import csv
import json

from sklearn.metrics import cohen_kappa_score

auto = {r["id"]: r["label"] for r in map(json.loads, open("data/pairs.jsonl"))}
rows = [r for r in csv.DictReader(open("data/human_label_sample.csv")) if r["labeller_1"] and r["labeller_2"]]
assert rows, "fill in the labeller_1 and labeller_2 columns first"
l1, l2, a = [r["labeller_1"].strip().lower() for r in rows], [r["labeller_2"].strip().lower() for r in rows], [auto[r["id"]] for r in rows]
print(f"n = {len(rows)}")
print(f"kappa labeller 1 vs labeller 2: {cohen_kappa_score(l1, l2):.3f}")
print(f"kappa labeller 1 vs automatic:  {cohen_kappa_score(l1, a):.3f}")
print(f"kappa labeller 2 vs automatic:  {cohen_kappa_score(l2, a):.3f}")
