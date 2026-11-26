# one-off: add the minimal-pair direction to an existing steer.npz (probe.py now writes it directly)
import json, sys, numpy as np
from probe import Data, read_jsonl
out = f"results/{sys.argv[1]}"
rows = [r for e in ("sandbox", "scienceworld") for r in read_jsonl(f"data/{e}.jsonl")]
data, z = Data(rows), dict(np.load(f"{out}/steer.npz"))
Xb = np.concatenate([np.load(f"{out}/hs_{e}_last.npy", mmap_mode="r")[:, int(z["layer"])] for e in ("sandbox", "scienceworld")]).astype(np.float32)
tr, sb = ~data.split(0), data.env == "sandbox"
v = Xb[tr & sb & (data.label == "irreversible")].mean(0) - Xb[tr & sb & (data.label == "reversible")].mean(0)
assert np.allclose(v, z["direction"], atol=1e-4), "recomputed main direction differs"
m = tr & data.minimal
z["direction_minimal"] = Xb[m & (data.label == "irreversible")].mean(0) - Xb[m & (data.label == "reversible")].mean(0)
np.savez(f"{out}/steer.npz", **z)
cos = float(v @ z["direction_minimal"] / np.linalg.norm(v) / np.linalg.norm(z["direction_minimal"]))
print("layer", int(z["layer"]), "|v|", np.linalg.norm(v), "|v_min|", np.linalg.norm(z["direction_minimal"]), "cosine", cos)
