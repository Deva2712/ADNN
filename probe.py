"""Linear probes on saved hidden states (RQ1), with every control and ablation.

    python probe.py --model Qwen2.5-7B-Instruct --jobs 24

Writes results/<model>/probe.json (one record per experiment, layer, seed), lexical.json,
steer.npz (best layer and the irreversible-minus-reversible direction) and probe_oof.json
(out-of-fold probe predictions for every pair, used for RQ2).
"""
import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CS = [1e-3, 1e-2, 1e-1]
FAMILIES = ["delete_file", "edit_file", "send_email", "transfer_money"]


def read_jsonl(path):
    return [json.loads(l) for l in open(path)]


def bal_acc(y, pred):
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def score(y, proba, classes, rng):
    """Balanced accuracy with a 1,000-resample bootstrap CI, macro-F1 and AUROC."""
    pred = classes[proba.argmax(1)]
    boot = [bal_acc(y[i], pred[i]) for i in rng.integers(0, len(y), (1000, len(y)))]
    try:
        auc = roc_auc_score(y == classes[1], proba[:, 1]) if len(classes) == 2 else \
            roc_auc_score(y, proba, multi_class="ovr", labels=classes)
    except ValueError:  # a class is missing from this test subset
        auc = float("nan")
    return {"n": int(len(y)), "bal_acc": bal_acc(y, pred), "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "acc": float(np.mean(pred == y)), "f1": float(f1_score(y, pred, average="macro")), "auroc": float(auc)}


def new_probe(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def pick_C(X, y, groups):
    """Grouped 3-fold CV on the training split only."""
    cv = np.zeros(len(CS))
    for tr, va in GroupKFold(3).split(X, y, groups):
        for k, C in enumerate(CS):
            cv[k] += bal_acc(y[va], new_probe(C).fit(X[tr], y[tr]).predict(X[va])) / 3
    return CS[int(cv.argmax())], float(cv.max())


class Data:
    def __init__(self, rows):
        col = lambda k: np.array([r[k] for r in rows])
        self.rows, self.label, self.env, self.group = rows, col("label"), col("environment"), col("scenario")
        self.tool, self.minimal = col("tool"), col("source") == "minimal_triple"
        self.horizon = np.array([np.inf if r["horizon"] is None else r["horizon"] for r in rows])
        self.binary = self.label != "partial"
        # identity of an action: tool plus its first argument (sandbox) or the full action (ScienceWorld)
        self.action_id = np.array([r["tool"] + "|" + str(next(iter(json.loads(r["action_text"])["args"].values()), ""))
                                   if r["environment"] == "sandbox" else r["action_text"] for r in rows])

    def split(self, seed):
        """80/20 by scenario (sandbox) and by task (ScienceWorld). Returns a boolean test mask."""
        rng, test = np.random.default_rng(seed), np.zeros(len(self.rows), bool)
        for env in np.unique(self.env):
            g = rng.permutation(np.unique(self.group[self.env == env]))
            test |= np.isin(self.group, g[: max(1, round(0.2 * len(g)))]) & (self.env == env)
        return test

    def control(self, keys, seed, y):
        """Hewitt & Liang control task: one random label per type, drawn from the label distribution."""
        rng = np.random.default_rng(100 + seed)
        table = {k: rng.choice(y) for k in np.unique(keys)}
        return np.array([table[k] for k in keys])

    def experiments(self, pos):
        """Each: name, seed, train mask, test mask, labels, whether to tune C, evaluation subsets."""
        exps, sb, sw = [], self.env == "sandbox", self.env == "scienceworld"
        subsets = {"all": np.ones(len(self.rows), bool), "minimal": self.minimal, "sandbox": sb, "scienceworld": sw}
        for seed in range(3):
            te = self.split(seed)
            exps.append(("binary", seed, ~te & self.binary, te & self.binary, self.label, True, subsets))
            exps.append(("three_class", seed, ~te, te, self.label, True, subsets))
            if pos != "last":
                continue
            b = self.binary
            shuffled = self.label.copy()
            shuffled[~te & b] = np.random.default_rng(seed).permutation(shuffled[~te & b])
            exps.append(("shuffled", seed, ~te & b, te & b, shuffled, False, {}))
            exps.append(("control_verb", seed, ~te & b, te & b, self.control(self.tool, seed, self.label[b]), False, {}))
            seen = np.isin(self.action_id, self.action_id[~te & b])
            exps.append(("control_action", seed, ~te & b, te & b & seen, self.control(self.action_id, seed, self.label[b]), False, {}))
            exps.append(("minimal_only", seed, ~te & b & self.minimal, te & b & self.minimal, self.label, False, {}))
            if sw.any():
                for h in (1, 2, 3):  # ScienceWorld only: reversible iff restored within h steps
                    yh = np.where(self.horizon <= h, "reversible", "irreversible")
                    exps.append((f"horizon_{h}", seed, ~te & sw, te & sw, yh, False, {}))
        if pos == "last":
            b = self.binary
            if sw.any():
                exps.append(("sandbox_to_scienceworld", 0, sb & b, sw & b, self.label, False, {}))
                exps.append(("scienceworld_to_sandbox", 0, sw & b, sb & b, self.label, False, {}))
            for fam in FAMILIES:  # train on three tool families, test on the fourth
                m = self.minimal & b
                exps.append((f"heldout_{fam}", 0, m & (self.tool != fam), m & (self.tool == fam), self.label, False, {}))
        return exps


def run_layer(X, data, pos, layer):
    X, out, C_main = X.astype(np.float32), [], {}
    for name, seed, tr, te, y, tune, subsets in data.experiments(pos):
        rng = np.random.default_rng(seed)
        if tune:
            C, cv = pick_C(X[tr], y[tr], data.group[tr])
            if name == "binary":
                C_main[seed] = C
        else:
            C, cv = C_main[seed], None
        clf = new_probe(C).fit(X[tr], y[tr])
        classes = clf.classes_
        rec = {"exp": name, "pos": pos, "layer": layer, "seed": seed, "C": C, "cv_bal_acc": cv, "n_train": int(tr.sum()),
               **score(y[te], clf.predict_proba(X[te]), classes, rng)}
        for sname, mask in subsets.items():
            if (te & mask).sum() and len(np.unique(y[te & mask])) > 1:
                rec[sname] = score(y[te & mask], clf.predict_proba(X[te & mask]), classes, rng)
        out.append(rec)
    return out


def lexical(data):
    """Bag-of-words baselines (action text only, and state plus action) and the majority class."""
    out = []
    texts = {"bow_action": [r["action_text"] for r in data.rows],
             "bow_state_action": [r["state_text"] + "\n" + r["action_text"] for r in data.rows]}
    for seed in range(3):
        te = data.split(seed)
        for task, keep in (("binary", data.binary), ("three_class", np.ones(len(data.rows), bool))):
            tr_m, y = ~te & keep, data.label
            for name, text in texts.items():
                vec = CountVectorizer() if name == "bow_action" else TfidfVectorizer(ngram_range=(1, 2), min_df=2)
                text = np.array(text)
                clf = make_pipeline(vec, LogisticRegression(C=1.0, max_iter=2000)).fit(text[tr_m], y[tr_m])
                rec = {"exp": name, "task": task, "seed": seed}
                for sname, mask in (("all", keep), ("minimal", keep & data.minimal)):
                    m = te & mask
                    rec[sname] = score(y[m], clf.predict_proba(text[m]), clf.classes_, np.random.default_rng(seed))
                out.append(rec)
            vals, counts = np.unique(y[tr_m], return_counts=True)
            m = te & keep
            out.append({"exp": "majority", "task": task, "seed": seed,
                        "all": {"n": int(m.sum()), "acc": float(np.mean(y[m] == vals[counts.argmax()])), "bal_acc": 1 / len(vals)}})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--envs", default="sandbox,scienceworld")
    a = ap.parse_args()
    out = f"results/{a.model}"
    envs = [e for e in a.envs.split(",") if os.path.exists(f"{out}/hs_{e}_last.npy")]
    rows = [r for e in envs for r in read_jsonl(f"data/{e}.jsonl")]
    assert [r["id"] for r in rows] == [i for e in envs for i in json.load(open(f"{out}/hs_{e}_ids.json"))]
    data = Data(rows)
    print(a.model, envs, len(rows), flush=True)
    json.dump(lexical(data), open(f"{out}/lexical.json", "w"), indent=1)

    records = []
    for pos in ("last", "mean"):
        X = np.concatenate([np.load(f"{out}/hs_{e}_{pos}.npy") for e in envs])  # N, L+1, d
        res = Parallel(n_jobs=a.jobs)(delayed(run_layer)(X[:, l], data, pos, l) for l in range(X.shape[1]))
        records += [r for layer in res for r in layer]
        json.dump(records, open(f"{out}/probe.json", "w"))
        print("done", pos, flush=True)
        if pos == "last":  # best layer by training CV (seed 0), steering direction, out-of-fold predictions
            cv = {r["layer"]: r["cv_bal_acc"] for r in records if r["exp"] == "binary" and r["seed"] == 0 and r["layer"] > 0}
            best = max(cv, key=cv.get)
            tr, sb, Xb = ~data.split(0), data.env == "sandbox", X[:, best].astype(np.float32)
            v = Xb[tr & sb & (data.label == "irreversible")].mean(0) - Xb[tr & sb & (data.label == "reversible")].mean(0)
            m = tr & data.minimal  # same direction from minimal pairs only: tool identity cancels out
            v_min = Xb[m & (data.label == "irreversible")].mean(0) - Xb[m & (data.label == "reversible")].mean(0)
            np.savez(f"{out}/steer.npz", layer=best, direction=v, direction_minimal=v_min, mean_norm=np.linalg.norm(Xb, axis=1).mean())
            print("best layer", best, "cv", cv[best], "|v|", np.linalg.norm(v), "mean |h|", np.linalg.norm(Xb, axis=1).mean(), flush=True)
            C3 = next(r["C"] for r in records if r["exp"] == "three_class" and r["seed"] == 0 and r["layer"] == best)
            oof = np.empty(len(rows), dtype=object)
            for trn, tst in GroupKFold(5).split(Xb, data.label, data.group):
                oof[tst] = new_probe(C3).fit(Xb[trn], data.label[trn]).predict(Xb[tst])
            json.dump({"layer": best, "pred": dict(zip([r["id"] for r in rows], oof.tolist()))}, open(f"{out}/probe_oof.json", "w"))
