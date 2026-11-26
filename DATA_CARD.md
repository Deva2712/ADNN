# Data card: reversibility-labelled (state, action) pairs

File: `data/pairs.jsonl` (2,710 pairs). One JSON object per line.

| field | meaning |
|---|---|
| `id` | `sb-…` (sandbox) or `sw-…` (ScienceWorld) |
| `environment` | `sandbox` or `scienceworld` |
| `state_text` | the state shown to the model |
| `action_text` | the candidate action (a JSON tool call, or a ScienceWorld command) |
| `label` | `reversible`, `partial` or `irreversible` |
| `horizon` | fewest further actions that restore the exact previous state (0 = the action changed nothing); `null` if none within 3 |
| `source` | `minimal_triple`, `fixed_tool` or `scienceworld_search` |
| `scenario` | split unit: sandbox scenario id, or ScienceWorld task name |
| `tool` | tool name (sandbox) or action verb (ScienceWorld) |
| `group` | id shared by the three members of a minimal triple |
| `task` | task description shown with the state |

ScienceWorld rows also carry `variation` and `history` (the actions replayed to reach the state).

## Class counts

| environment | reversible | partial | irreversible | total |
|---|---|---|---|---|
| sandbox | 720 | 720 | 720 | 2,160 |
| ScienceWorld | 330 | 0 | 220 | 550 |
| total | 1,050 | 720 | 940 | 2,710 |

1,440 pairs (53% of the dataset, 67% of the sandbox part) are minimal-triple items.

## How the sandbox part was built (`sandbox.py`, `make_data.py`)

120 scenarios. Each has a themed workspace (six files, a wallet) and four policy lines
(recycle bin, version history, outgoing mail, transfers). Every state shows all four lines.

- **Minimal triples (1,440 pairs).** For each scenario and each of `delete_file`, `edit_file`,
  `send_email`, `transfer_money`, the same tool call appears three times. Only the one policy line
  that governs that tool changes, which sets the label to reversible, partial or irreversible.
  The action text is identical across the triple.
- **Fixed tools (720 pairs).** Tools whose reversibility does not depend on the policies
  (for example `read_file`, `rename_file`, `convert_currency`, `shred_file`, `send_sms`),
  two per class per scenario, with the four policies drawn at random as distractors.
- Each policy level has three phrasings (explicit, "policy" style, terse); one style per scenario.
- **Check.** Every design label was compared with an exact search over recovery tools:
  a pair is labelled reversible if and only if the exact previous state is restored within 3
  further actions. All 2,160 pairs pass (the generator asserts this).
- "Partial" is set by design: restoration is approximate or has a cost (last night's snapshot,
  a change-log excerpt, an email recall that only removes unread copies, a refund minus a 5% fee).

## How the ScienceWorld part was built (`sw_label.py`)

29 tasks, 3 training variations each, states taken at 3 points along the first 15 steps of the
gold path, up to 5 candidate actions per state with distinct verbs. ScienceWorld has no
save/restore, so each state is rebuilt by reloading the variation and replaying the history.
The state signature is the room description plus the inventory. After applying the action, a
search looks for at most 3 actions that bring the signature back.

1,112 pairs were labelled. The released subset keeps every irreversible pair and subsamples the
reversible ones to 1.5 times that number, with no-op actions (horizon 0) capped at 20% of them.
Six numbered answer-choice actions were removed.

## Agreement scores

Not yet available. `data/human_label_sample.csv` holds 100 randomly drawn pairs without labels.
Two team members must fill it in independently; `python kappa.py` then prints Cohen's kappa
between the two labellers and between each labeller and the automatic label.

## Known limitations

- The sandbox is small: four policy-governed tool families and ten fixed tools. A probe could
  in principle learn twelve (tool, policy level) conjunctions; the held-out-family experiment
  tests for this.
- ScienceWorld has no "partial" class, because the automatic procedure only decides whether the
  exact state comes back.
- The ScienceWorld search is exhaustive only at depth 1. At depths 2 and 3 it keeps the six
  states closest to the original, so it can miss a recovery sequence and label a recoverable
  action irreversible. Candidate recovery actions are limited to those involving an object of
  the original action, plus walking to another room.
- The ScienceWorld signature covers the current room and inventory only. Changes that are not
  visible there (temperature, time) are ignored.
- 124 of the 220 irreversible ScienceWorld pairs use the verb `eat`, so the verb alone predicts
  much of that class. Compare against the action-only bag-of-words baseline.
- A few automatic ScienceWorld labels look odd on inspection (for example `open drawer` labelled
  irreversible). The hand-labelled sample is the place to quantify this.
- Labels use a 3-step horizon. `horizon` allows relabelling at 1 or 2 steps.

## Licences

Sandbox data: generated by this project's code. ScienceWorld text comes from the ScienceWorld
simulator (check the LICENSE file in github.com/allenai/ScienceWorld before redistributing).
