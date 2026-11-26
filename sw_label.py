"""Sample (state, action) pairs from ScienceWorld and label reversibility by search.

    python sw_label.py --procs 20        # writes data/scienceworld_all.jsonl

ScienceWorld has no save/restore, so a state is rebuilt by reloading the task variation and
replaying the action history. For each pair: record the state signature (room description plus
inventory), apply the action, then search for a sequence of at most 3 actions that brings the
signature back. `horizon` is the shortest such sequence (0 = the action changed nothing), or null.

The search is exhaustive at depth 1 over candidate recovery actions (valid actions that involve
an object of the original action, plus walking to another room). At depths
2 and 3 it keeps the BEAM states closest to the original and expands each fully. This is an
approximation of full breadth-first search, chosen for cost; it can miss a recovery sequence.
"""
import argparse
import json
import multiprocessing as mp
import random

from scienceworld import ScienceWorldEnv

BEAM, HORIZON, STATES_PER_VARIATION, ACTIONS_PER_STATE, GOLD_PREFIX = 6, 3, 3, 5, 15
OBSERVE = ("look", "examine", "read", "inventory")          # change nothing; at most one per state
SKIP = ("focus", "wait", "wait1", "task", "reset", "teleport")  # task bookkeeping, time, shortcuts
RISKY = ("eat", "pour", "mix", "use", "dunk", "flush", "activate", "deactivate")  # sampled first, for class balance


def verb(action):
    w = action.split()
    return " ".join(w[:2]) if w[0] in ("pick", "put", "look") else w[0]


class World:
    def __init__(self):
        self.env = ScienceWorldEnv("", envStepLimit=1000)
        self.s = self.env.server

    def goto(self, task, variation, actions):
        self.env.load(task, variation, "easy")
        for a in actions:
            self.s.step(a)
        return self.s.freeActionLook().strip() + "\n" + self.s.freeActionInventory().strip()

    def valid(self):
        return self.env.get_valid_action_object_combinations_with_templates()


def distance(a, b):
    a, b = set(a.split("\n")), set(b.split("\n"))
    return len(a ^ b)


def search(w, task, variation, history, action, objs, target):
    """Shortest number of recovery actions (<= HORIZON) that restores `target`, else None."""
    def candidates():
        out = []
        for v in w.valid():
            vb = verb(v["action"])
            if vb.startswith(OBSERVE) or vb in SKIP:
                continue
            if set(v["obj_ids"]) & objs or (vb == "go" and "door" not in v["action"]):
                out.append(v["action"])
        return out

    sig = w.goto(task, variation, history + [action])
    if sig == target:
        return 0
    frontier = [[]]
    for depth in range(1, HORIZON + 1):
        reached = []
        for seq in frontier:
            w.goto(task, variation, history + [action] + seq)
            for c in candidates():
                sig = w.goto(task, variation, history + [action] + seq + [c])
                if sig == target:
                    return depth
                reached.append((distance(sig, target), seq + [c]))
        reached.sort(key=lambda x: x[0])
        frontier = [seq for _, seq in reached[:BEAM]]
    return None


def work(job):
    task, variation, seed = job
    rng, w, rows = random.Random(seed), World(), []
    w.env.load(task, variation, "easy", generateGoldPath=True)
    gold = w.env.get_gold_action_sequence()[:GOLD_PREFIX]
    desc = w.env.get_task_description()
    for k in sorted(rng.sample(range(len(gold) + 1), min(STATES_PER_VARIATION, len(gold) + 1))):
        history = gold[:k]
        target = w.goto(task, variation, history)
        if w.goto(task, variation, history) != target:  # replay must be deterministic
            continue
        by_verb = {}
        for v in w.valid():
            if verb(v["action"]) not in SKIP:
                by_verb.setdefault("observe" if verb(v["action"]).startswith(OBSERVE) else verb(v["action"]), []).append(v)
        verbs = sorted(by_verb)
        rng.shuffle(verbs)
        verbs = [v for v in verbs if v in RISKY][:3] + [v for v in verbs if v not in RISKY]
        for vb in verbs[:ACTIONS_PER_STATE]:
            v = rng.choice(by_verb[vb])
            h = search(w, task, variation, history, v["action"], set(v["obj_ids"]), target)
            rows.append({"environment": "scienceworld", "state_text": target, "action_text": v["action"],
                         "label": "irreversible" if h is None else "reversible", "horizon": h, "source": "scienceworld_search",
                         "scenario": task, "tool": verb(v["action"]), "group": None, "task": desc,
                         "variation": variation, "history": history})
    w.env.close()
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--tasks", type=int, default=0, help="smoke test: only the first N tasks")
    ap.add_argument("--variations", type=int, default=3)
    a = ap.parse_args()
    env = ScienceWorldEnv("")
    jobs = []
    for task in env.get_task_names()[: a.tasks or None]:
        env.load(task, 0, "easy")
        jobs += [(task, v, len(jobs) + i) for i, v in enumerate(env.get_variations_train()[: a.variations])]
    env.close()
    with mp.Pool(a.procs) as pool, open("data/scienceworld_all.jsonl", "w") as f:
        n = 0
        for rows in pool.imap_unordered(work, jobs):
            for r in rows:
                f.write(json.dumps({"id": f"sw-{n:05d}", **r}) + "\n")
                n += 1
            f.flush()
            print(n, flush=True)
