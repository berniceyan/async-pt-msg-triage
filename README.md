# Async Message Triage

An asynchronous patient-messaging triage system. Each message goes to one of three places: an
urgent escalation for anything that cannot safely wait three days, a physician queue where a doctor
replies asynchronously three or more days later, or automated self-care guidance.

**Project page:** <https://berniceyan.github.io/async-pt-msg-triage/>

It decides in two steps:

1. **LLM urgency screen.** One model call asks one question: *can this safely wait three days
   for a physician?* It answers `safe_to_wait`, `unsure` or `not_safe`. Anything other than
   `safe_to_wait` escalates, and nothing downstream can lower it. A screen that fails escalates.
2. **Judgment.** Only for a message the screen lets wait: a second call decides whether a
   clinician has to act (physician queue) or automated guidance is enough (self-care). It is the
   only step that can clear a message.

A policy layer with no clinical knowledge reads the two calls and routes (`src/policy.py`).

## Results

Built and evaluated on PMR-Bench Reddit messages against one physician's labels. The test set was
scored once, with the prompt frozen.

| test set, 100 messages | urgent missed | non-urgent escalated | agreement |
|---|---|---|---|
| **LLM urgency screen, unsure → urgent** | **2/70 (3%, 1–10)** | **5/30 (17%, 7–34)** | **87/100** |
| LLM urgency screen, unsure → queue | 14/70 (20%) | 3/30 (10%) | 76/100 |
| Judgment only (baseline), no screen | 34/70 (49%) | 1/30 (3%) | 56/100 |

Intervals are Wilson 95%.

- **The screen fixed the miss rate.** The baseline escalated on can't-miss probabilities and
  missed half the urgent messages; asking the question directly misses 2 of 70.
- **The price is escalation.** At a realistic 18% urgent mix, about one escalation in two would
  be truly urgent.
- **One urgent message reached self-care**: a dog scratch with recent bat contact, read as a
  vaccinated pet and unbroken skin.

## Failure modes

All 37 routing errors from dev, val and test, sorted by what went wrong and reviewed by the
physician who labelled them. Dev and val count alongside the test set's 13 so that each mode has
enough cases behind it.

| group | mode | errors | on test |
|---|---|---|---|
| **Under-triaged** (10) | Exposure risk discounted (rabies, HIV, body fluid) | 4 | 1 |
| | Missing warning signs read as safety | 4 | 1 |
| | A benign reading accepted | 2 | 1 |
| **Over-escalated** (14) | Time-sensitive, but the automated reply could answer | 4 | 2 |
| | A recent in-person assessment discounted | 3 | 3 |
| | Needs a workup this week, not within days | 7 | 0 |
| **Over-referred** (13) | A test or prescription question taken as needing a clinician | 10 | 4 |
| | Distress treated as clinical need | 3 | 1 |

- **Under-triage comes from exposures and absences.** A policy rule that an exposure question
  never goes to self-care would have blocked the worst test error without changing the prompt.
  The second mode is the screen's own rule, that silence is not reassurance, failing to hold.
- **Over-escalation is the price of the safety margin.** A prompt revision aimed at the first two
  over-escalation modes raised urgent misses from 3 to 11 on dev and from 2 to 6 on val, and was
  rejected.
- **Over-referrals are the cheapest errors.** They cost physician time, not safety.

## How it is built

- **Evaluation data:** PMR-Bench (Gatto et al., *Medical Triage as Pairwise Ranking*, arXiv
  2601.13178; HF `PortalPal-AI`, **CC BY-NC 4.0**). `make data` fetches the pinned revision
  (`data/pmr_manifest.json`) and builds three sets: dev is PMR's dev split (`pmr_dev`)
  plus 40 urgent messages from PMR's training split, which nothing here trains on; val and test
  are carved from the same split by `tools/split_extended.py` (ids in `data/ext_split.json`).
- **Labels:** `data/clinician_labels.csv` (ids and labels only, no message text), one
  physician's three-way call on every dev, val and test message, is the primary reference.
  PMR's six levels, folded to three in `src/pmr.py`, are reported alongside.
- **Model:** `anthropic/claude-sonnet-5.5` via OpenRouter, reasoning effort `low`. Prompts:
  `prompts/screen.txt` (frozen under `prompts/screen_freeze_rule.txt`) and `prompts/judgment.txt`.
  Every call is cached in `results/`, keyed on model, prompt, message, budget and effort, so a
  rerun reproduces offline.
- **Cost:** $0.0095 per message on the shipped route; an escalation is decided in about 2 s.

## Quickstart

```bash
python3 -m venv .venv                # 3.11+
source .venv/bin/activate
make install                         # requirements.txt
cp .env.example .env                 # add OPENROUTER_API_KEY
python check_setup.py                # packages, key, one live screen call
make data                            # PMR-Bench at the pinned revision, then the split
make test
```

`src/env.py` reads the key when the model layer is imported; a value exported in the shell wins
over the file. Without a key, a live call raises rather than landing as `MODEL_FAILURE`, so a
keyless run cannot file every cache miss as a failure and write a plausible set of results.

## Run order

```bash
make eval-dev                        # tuning: the dev set
make eval-val                        # once per tuning round
make eval                            # test, once, at the frozen prompt
```

Every run writes `results/<dataset>.<config>.json` for three configurations — `screen` (shipped),
`screen_uq` (unsure → queue) and `model` (baseline, no screen) — scored from the same calls. The
evaluation makes both calls on every message; the shipped route skips the judgment when the
screen escalates, and `tests/test_pipeline.py` checks the two decide identically.

## Design history

- **Keyword and semantic-retrieval screen:** a deterministic screen ahead of the model, with no
  LLM call. One list of guideline-cited red-flag criteria and patient phrasings fed two
  recognizers: a keyword matcher (literal phrase match) and semantic retrieval, which embedded
  each sentence with a local encoder (`all-mpnet-base-v2`) and escalated when it was close enough
  to a criterion. Both sat behind negation and experiencer guards. On PMR's Reddit test split (362
  messages, 51 urgent, scored on PMR's levels), the keyword matcher alone escalated 4 messages:
  1 urgent caught and 3 of 311 non-urgent over-escalated. Adding semantic retrieval caught 25 of
  51 urgent but over-escalated 87 of 311 non-urgent. It was removed.
- **Judgment only (the baseline):** escalating when a can't-miss probability crossed a threshold
  tuned on `pmr_dev`. The thresholds did not survive PMR's test split.
- **LLM urgency screen:** replaces the thresholds for escalation; the judgment keeps the
  queue-versus-self-care decision. Tuned and tested on fresh sets from PMR's training split.

| design | urgent missed | non-urgent escalated | scored on |
|---|---|---|---|
| Judgment only (baseline) | 34 of 70 | 1 of 30 | the test set, the physician's labels |
| Keyword screen alone | 50 of 51 | 3 of 311 | PMR's test split, PMR's levels |
| Keyword and semantic retrieval | 26 of 51 | 87 of 311 | PMR's test split, PMR's levels |
| **LLM urgency screen** | **2 of 70** | **5 of 30** | the test set, the physician's labels |

The two halves of the table are not directly comparable, but the gap is not subtle.

## Data and attribution

Messages are from PMR-Bench: Gatto, Seegmiller, Burdick, Resnik, Rahat, DeLozier and Preum,
*Medical Triage as Pairwise Ranking: A Benchmark for Urgency in Patient Portal Messages*,
arXiv 2601.13178 (2026), distributed under
[**CC BY-NC 4.0**](https://creativecommons.org/licenses/by-nc/4.0/). The Reddit-derived portion is
also subject to Reddit's data policy. This project uses it non-commercially and does not
redistribute the text: `make data` rebuilds it from the pinned revision.

**Changes from the original.** The message text is used as published, from the Reddit portion
only. The labels are not PMR-Bench's:

- PMR's six urgency levels are folded to this project's three destinations (`src/pmr.py`).
- The author relabelled every dev, val and test message against this project's own definition of
  urgent: cannot safely wait three days for an asynchronous reply. Those labels
  (`data/clinician_labels.csv`) are the primary reference for every result here; PMR's folded
  levels are reported alongside.
- Val and test are subsets drawn from PMR's training split (`tools/split_extended.py`), sampled
  to include more urgent messages than the benchmark's natural mix.

## License

The code and prompts are MIT-licensed (`LICENSE`). PMR-Bench messages are not covered by it: they
remain under CC BY-NC 4.0 and Reddit's data policy, as above.
