# Do AI agents know which actions cannot be undone?

Measurement study for ADNN Project Option 1. Two open-weight models are tested for whether they
(1) encode action reversibility in their hidden states, (2) state it correctly when asked, and
(3) act on it as agents. `PHASES.md` lists the phases and the decisions fixed before any test
data was seen. `results/tables.md` holds every number; `report/report.tex` is the write-up (ACL LaTeX, not yet compiled).

All irreversible actions happen inside an in-memory simulation (`sandbox.py`). Nothing touches
real files, mail or accounts. No paid API is used.

## Files

| file | what it does |
|---|---|
| `sandbox.py` | file system, outbox and wallet with save/restore, 33 tools, exact reversibility check |
| `make_data.py` | sandbox pairs (minimal triples), 60 paired behaviour tasks, merge with ScienceWorld |
| `sw_label.py` | ScienceWorld pairs labelled by replay-based search at horizons 1 to 3 |
| `gpu.py` | hidden states, stated answers, agent runs, steering (one model per call) |
| `probe.py` | linear probes with all controls and ablations |
| `analyze.py` | tables and figures from saved results |
| `kappa.py` | agreement on the hand-labelled sample |
| `demo.py` | replays a saved agent run on both versions of a task (for the presentation) |
| `slurm/` | the Slurm job scripts used on the DGX, and `minimal_dir.py` (one-off, see below) |
| `logs/` | job logs from the DGX runs |
| `DATA_CARD.md` | dataset description |

## Reproduce

Python 3.12, one GPU with 20 GB or more, Java 8 or newer for ScienceWorld.

```bash
pip install -r requirements.txt   # torch 2.6.0 was installed from the cu124 wheel index
python make_data.py                                   # data/sandbox.jsonl, data/tasks.jsonl
python sw_label.py --procs 20                         # data/scienceworld_all.jsonl (about 20 min on 20 cores)
python make_data.py --merge                           # data/scienceworld.jsonl, data/pairs.jsonl, blind label sheet
for M in Qwen/Qwen2.5-7B-Instruct mistralai/Mistral-7B-Instruct-v0.3; do
  python gpu.py --model $M --stages extract,stated,agent     # use --bs 8 for Mistral on a 20 GB GPU
  python probe.py --model ${M##*/} --jobs 24                  # also writes steer.npz
  python gpu.py --model $M --stages steer            # run for Qwen only so far
done
python gpu.py --model Qwen/Qwen2.5-7B-Instruct --stages steer_minimal   # post-hoc control, +-2 done, +-4 pending
python analyze.py                                     # results/tables.md, results/summary.json, figs/*.png
```

`probe.py` now writes the minimal-pair direction into `steer.npz` itself. For the runs reported
here it was added to the existing `steer.npz` by `slurm/minimal_dir.py`, which uses the same formula.

Seeds are fixed (0 for data, splits 0 to 2, 0 for the random steering direction). Decoding is
greedy. Agent runs are batched, so bit-exact repetition needs the same batch size.

Every prompt and model output is saved: `results/<model>/stated_*.jsonl` (prompt, raw answer,
parsed label) and `results/<model>/agent_*.jsonl` (full message list, every tool call and
result, final sandbox state). Hidden-state arrays (`hs_*.npy`, about 2 GB per model) are not
kept in Git; `gpu.py --stages extract` regenerates them.

## What each figure and table comes from

| output | produced by |
|---|---|
| `figs/probe_layers.png` (probe accuracy by layer with baselines) | `analyze.py` from `probe.json`, `lexical.json` |
| `figs/stated_probe_behaviour.png` (stated vs probed vs behaviour) | `analyze.py` from `stated_*.jsonl`, `probe_oof.json`, `agent_neutral.jsonl` |
| `figs/blindness_gap.png` (K, A and the gap with intervals) | `analyze.py` from `stated_tasks.jsonl`, `agent_*.jsonl` |
| `figs/steering_curve.png` | `analyze.py` from `agent_steer_*.jsonl` |
| ablation table and all other tables | `results/tables.md` |

## Models

Main model: `Qwen/Qwen2.5-7B-Instruct`. Second family: `mistralai/Mistral-7B-Instruct-v0.3`.
The brief names `meta-llama/Llama-3.1-8B-Instruct` as the second family; it is gated and no
approved token was available. Once access is granted, rerun the three per-model lines above
with that model name.

## Still to do by the team

See the `[!]` and `[~]` items in `PHASES.md`: the individual contribution log, telling the
instructor that Mistral-7B replaced the gated Llama-3.1-8B, and the two optional GPU runs.
The slide deck is `slides/Irreversibility_Deck.pptx`.
