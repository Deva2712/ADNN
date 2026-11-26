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

`data/human_label_sample.csv` holds 100 randomly drawn pairs (77 sandbox, 23 ScienceWorld).
Two team members (Sarthak A and Yashwant Alli) labelled them independently without seeing the
automatic label. Cohen's kappa from `python kappa.py`:

| comparison | all (n = 100) | sandbox (n = 77) | ScienceWorld (n = 23) |
|---|---|---|---|
| labeller 1 vs labeller 2 | 0.320 | 0.432 | 0.044 |
| labeller 1 vs automatic | 0.388 | 0.329 | 0.401 |
| labeller 2 vs automatic | 0.272 | 0.357 | 0.073 |

Raw agreement: 56% between the labellers, 59% and 51% with the automatic label.

Agreement is only fair. Two patterns account for much of it:

- On ScienceWorld rows, labeller 2 used `partial` 11 times out of 23, although the automatic
  procedure has no partial class there. The two labellers read "approximately restored" differently
  for actions such as walking to another room or opening a freezer.
- On sandbox rows, agreement with the automatic label is higher on minimal-triple items (60% and
  65%), where the governing policy line decides the label, than on fixed-tool items (45% and 36%),
  where the label depends on what the tool does and the four policy lines are distractors.

No disagreements were resolved afterwards. The automatic labels in `pairs.jsonl` are unchanged.

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
  irreversible). Human agreement is low on ScienceWorld (kappa 0.044 between labellers), so the
  three-way definition is hard to apply there without the sandbox's policy lines.
- Labels use a 3-step horizon. `horizon` allows relabelling at 1 or 2 steps.

## Licences

Sandbox data: generated by this project's code. ScienceWorld text comes from the ScienceWorld
simulator (check the LICENSE file in github.com/allenai/ScienceWorld before redistributing).
