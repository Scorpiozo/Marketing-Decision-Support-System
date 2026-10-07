# Faculty / Viva Guide

How to explain this project simply and accurately. Every number below comes from an
experiment in this repository (`reports/experiments/*.json`, `artifacts/model_card.json`).
If an examiner asks "where does that number come from?", the answer is always: *a script
in `src/leadintel/experiments/` produced it, and you can re-run it with one command.*

---

## 30-second explanation

> A bank's sales team calls the same leads again and again. My system looks at each lead's
> **interaction history** (whether they said yes to a previous campaign, how recently they
> were contacted, how many times we've called them this campaign) plus their profile. It
> estimates how likely they are to subscribe, and then a separate, transparent **decision
> policy** turns that estimate into a recommended action: prioritise, follow up, change
> approach, human review, deprioritise, or stop following up. Every recommendation comes
> with the evidence behind it. A person makes the final decision.

## 1-minute explanation

**Problem →** Sales teams waste calls on unpromising leads and over-contact people.
**Data →** UCI Bank Marketing: 45,211 phone-campaign records from a Portuguese bank, 2008–2010. 11.7% subscribed.
**ML →** A gradient-boosting model predicts the probability of subscribing (ROC-AUC 0.81 on unseen leads, with well-calibrated probabilities).
**Decision →** An explicit cost rule (call if probability × value > cost) plus contact-fatigue and uncertainty rules turn the probability into an action.
**Explanation →** SHAP shows exactly which signals pushed each lead's estimate up or down.
**Application →** A web app where you can see the portfolio, open a lead, read why, and run a what-if.

## 3-minute explanation (end-to-end)

1. **Audit.** The original project reported 0.92 AUC, but it used `duration` (call length), which is only known *after* the call that decides the outcome. That's data leakage. Its probabilities were also 2–3× too high because of class weighting, and it used a 1D CNN on tabular data.
2. **Fix the ML.** I removed `duration` (AUC drops to an honest 0.81), trained without class weights so probabilities are calibrated, and compared five model families plus the original CNN under cross-validation.
3. **Test the research question.** In an ablation, adding interaction-history features to profile + campaign context raised PR-AUC from 0.395 to 0.466. When testing on a later time period, interaction history was the most robust feature group.
4. **Separate prediction from decision.** The model only estimates a probability. A policy with editable assumptions decides the action. If re-trained versions of the model disagree about which side of the line a lead falls on, it says **human review**.
5. **Explain.** Exact TreeSHAP per lead, global importance, interaction analysis, and a what-if simulator.
6. **Build and verify.** FastAPI web app, 56 automated tests, accessibility-checked, ~27 ms to re-score a what-if.

## The dataset (UCI Bank Marketing)

* Direct-marketing **phone campaigns** of a Portuguese bank, May 2008 – Nov 2010; `bank-full` variant, 45,211 rows.
* Target `y`: did the client subscribe to a **term deposit**? Yes for 11.7% of rows.
* Citation: Moro, Cortez & Rita (2014), *Decision Support Systems* 62:22–31. Licence: CC BY 4.0.
* What it does **not** contain: call transcripts, messages, sentiment, objections, CRM notes, salesperson identity, or a full per-lead event history.

## Features (15 used, 1 excluded)

| Group | Fields | Role |
|---|---|---|
| Customer profile | age, job, marital, education, default, balance, housing, loan | who the customer is |
| Campaign context | contact channel, month, day | how/when they were reached; partly reflects the collection period |
| **Interaction history** | `campaign` (contacts this campaign), `previous` (contacts before), `pdays` (days since previous contact), `poutcome` (previous outcome) | **the "interaction pattern"** |
| Excluded | `duration` | leakage: known only after the deciding call |

## Target: what are we predicting?

The probability that a lead subscribes to a term deposit. We do **not** predict "yes/no"
directly. We predict a probability, then a policy decides what to do with it.

## Models: which, and why

| Model | Why it was tested | Test ROC-AUC | Verdict |
|---|---|---|---|
| Logistic Regression | Transparent baseline; coefficients are directly readable | 0.772 | Good baseline, ~0.04 behind |
| Decision Tree (depth 5) | A model a human can read as rules | 0.728 | Too simple |
| Random Forest | Strong non-linear ensemble; kept from the original | 0.802 | Statistically tied with boosting, but 14 MB and slower |
| **Gradient Boosting (HistGB)** | Standard best-in-class for tabular data | **0.808** | **Chosen**: best ranking + best probability quality, small, fast, exact SHAP |
| Neural network (MLP) | The *appropriate* neural baseline for tables | 0.794 | Close, but less accurate and harder to explain |
| 1D CNN (original architecture) | Kept to evaluate the original choice fairly | 0.792 (3 seeds) | Not justified (see below) |

**Why Random Forest?** It was in the original project and it is strong: 0.802 AUC,
essentially tied with gradient boosting. We kept it in the comparison. We chose boosting
because it was marginally better on every quality metric, about 34× smaller (417 KB vs 14 MB),
and its explanations are faster to compute.

**Why not the CNN?** A CNN slides filters over *neighbouring* inputs, which makes sense
for pixels or time steps. Here the "neighbours" are just the order in which table columns
were written. Test: I randomly shuffled the column order and retrained. Performance didn't
change (0.792 vs 0.789), which shows the convolution isn't finding any real structure. It
behaves like a generic neural network, scores below boosting, and takes ~12 s to train
instead of ~4 s. This matches the literature (Grinsztajn et al., 2022: tree ensembles
generally beat deep nets on medium-sized tabular data). The CNN stays in the project as
a documented experiment.

**Why logistic regression / tree / boosting?** Together they span the interpretability ↔
accuracy spectrum: LR (most transparent) → tree (readable rules) → boosting (most accurate,
explained with SHAP).

## Interaction patterns: what exactly are they here?

Four observable facts about the bank's contact with a lead:
previous-campaign outcome, contacts before this campaign, days since that contact,
and contacts already made in this campaign. The app labels each lead with a readable
pattern, e.g. *"Previously converted · 2–3 contacts"*, and shows how often similar leads
subscribed:

* Previously converted: **68%** subscribe after 1 contact, 49% after 4+.
* New to campaigns: **11%** after 1 contact, 7% after 4+.
* Every history type converts less as contacts pile up. That's the evidence for the "contact fatigue" proxy.

**Did interaction features improve performance?** Yes, measured:

| Features | PR-AUC (same period) | ROC-AUC (later period) |
|---|---|---|
| Profile only | 0.265 | 0.639 |
| Profile + campaign context | 0.395 | 0.596 |
| Profile + interaction history | 0.395 | 0.686 |
| All three (deployed) | **0.466** | 0.665 |

Interaction history adds on top of context, and on a later time period it is the most robust
group. Context (month/channel) *stops* helping there, because it partly encodes when the data was collected.

**Most important interaction effect:** recency × previous outcome. For leads contacted within
~200 days, recency lifts the estimate 2–3× more after a past success than after a past failure.
**Not supported:** "contact frequency × previous outcome". Repeated contact lowers everyone's
chances by a similar amount. I report this rather than force a narrative.

## Decision engine: how a prediction becomes a recommendation

```
probability ──► is the model unsure which side of the line? ──yes──► HUMAN REVIEW
                     │no
                     ├─ ≥ priority threshold (35%) ───────────────► PRIORITISE
                     ├─ ≥ break-even (cost/value = 10%) + 4+ calls ► FOLLOW UP, CHANGE APPROACH
                     ├─ ≥ break-even ─────────────────────────────► FOLLOW UP
                     ├─ < break-even + 4+ calls ──────────────────► STOP REPEATED FOLLOW-UP
                     └─ otherwise ────────────────────────────────► DEPRIORITISE
```

* **Break-even:** call when probability × value of a subscription > cost of a call. With the illustrative €5 / €50, the line is 10%. That only makes sense because the probabilities are calibrated.
* **Uncertainty:** 15 versions of the model, each trained on a resample of the data. If they disagree about which side of the line a lead is on → human review (~11% of leads).
* **Back-test on 9,043 unseen leads:** contacting the 33% the policy recommends captures **74%** of all subscriptions (random choice would capture 33%). "Prioritise" leads subscribed 55% of the time; "Stop" leads 4.5%.
* All assumptions are editable on the Decision-policy page, and the back-test updates live.

## Explainability: how do we know *why*?

* **SHAP** splits a prediction into contributions from each field. They add up *exactly*
  to the prediction (checked automatically: max error 1.6e-14).
* Shown as **odds multipliers**: "previous outcome = success → odds ×2.3".
* **Global:** which fields matter most overall (contact channel, days since previous contact, month, previous outcome).
* **What-if:** change a field and see the new estimate, the new action, and which effects moved, including "interaction effects" where an unchanged field now matters differently.
* A pitfall I caught: the SHAP library silently gives *wrong* explanations for scikit-learn boosting when categories are handled natively (error 2.7 log-odds). I changed the encoding and added a test.

## Evaluation: what do the metrics mean?

| Metric | Plain meaning | Our model |
|---|---|---|
| Precision | Of the leads we said to call, what share subscribed? | 30% at the 10% line |
| Recall | Of all subscribers, what share did we catch? | 70% at the 10% line |
| F1 | Balance of precision and recall | 0.50 (at its best threshold) |
| ROC-AUC | Chance a random subscriber is ranked above a random non-subscriber | 0.81 |
| PR-AUC | Ranking quality focused on the rare "yes" class (random = 0.117) | 0.47 |
| Calibration | Does "30%" really mean 30%? | Mean estimate 11.5% vs actual 11.7%; error 0.9% |

Accuracy alone is misleading: predicting "no" for everyone is 88% accurate and useless.

## Leakage: what is it, and did we check?

Leakage = the model sees information it wouldn't have at decision time, so test scores look
better than reality. We checked every feature for when it becomes available
(`docs/DATA_LEAKAGE_AND_FEATURE_AVAILABILITY.md`). `duration` was the big one: 0.94 AUC with it, 0.81 without.
Also: preprocessing fitted only on training folds, thresholds chosen on cross-validation and
never on the test set, and the test set used once.

## Technology

* **Frontend:** server-rendered HTML (Jinja2) + HTMX for live updates + a hand-written stylesheet; charts are SVG drawn on the server. No JavaScript build step.
* **Backend:** FastAPI (Python). Also a JSON API (`POST /api/v1/assess`) showing how a CRM could call the model.
* **Model serving:** the model is trained offline (`uv run leadintel-train`, ~30 s) and saved. The web app loads it **once** at start-up (~0.3 s) and never retrains.
* **Why not Streamlit?** Streamlit reruns the whole script on every click. That's fine for analysis notebooks, but awkward for a multi-page decision workflow with a JSON API. The original also required TensorFlow just to score one row.

## Performance: how did we make it fast?

| | Original | New |
|---|---|---|
| Score a what-if lead | ~1,050 ms first time, ~100 ms after | **27 ms** (p50) |
| Page response (server) | full script rerun per click | 0.5–10 ms |
| Install | failed on Apple Silicon (tensorflow-cpu) | `uv sync` with a lockfile |

Main tricks: pre-compute scores + explanations for the portfolio; load the model once;
no TensorFlow in serving; single-thread scoring for one row (161 ms → 24 ms).

## Limitations (say these before the examiner does)

1. No conversation data: "interaction pattern" means contact history, not what was said.
2. 2008–2010, one Portuguese bank, financial-crisis period. Strong drift: conversion rises from ~3% to ~32% across the file, and on a later time period ROC-AUC drops to ~0.67.
3. Channel and month partly encode *when* data was collected. They're flagged in the app and never offered as levers.
4. SHAP explains the model, not cause and effect.
5. Costs are illustrative; "fatigue" is a proxy (contact count).
6. No regulatory claim (e.g. RBI). This is a research prototype.

## Future work (with real CRM data)

Full event histories per lead, call notes and transcripts (audited for leakage like `duration`),
time-based validation with drift monitoring and recalibration, real cost inputs, fairness
review, consent/do-not-call handling, and capturing what relationship managers actually did,
to measure whether the recommendations help.

---

## 10 difficult examiner questions, with short accurate answers

1. **"Your AUC dropped from 0.92 to 0.81. Isn't your model worse?"**
   No. The 0.92 used `duration`, which is only known after the call that decides the outcome, so it can't be used for a follow-up decision. 0.81 is the honest number. UCI's own documentation and the original authors (Moro et al., 2014) both exclude it.

2. **"Why gradient boosting and not deep learning?"**
   Measured: boosting 0.808 AUC vs MLP 0.794 vs CNN 0.792, with better calibration, 4 s to train and exact explanations. On medium tabular data this matches published benchmarks.

3. **"Prove the CNN isn't learning spatial patterns."**
   I retrained it with randomly shuffled column orders. If convolution found local structure, shuffling would hurt. It didn't (0.792 → 0.789).

4. **"How is this different from ordinary classification?"**
   Three layers on top of the classifier: an explicit cost-based decision policy, an uncertainty-based human-review state, and per-lead explanations. Plus a back-test showing the decisions work: 33% of calls capture 74% of subscriptions.

5. **"Why should a banker trust 23%?"**
   Because it's calibrated: across every segment, average estimates match actual rates (e.g. "Prioritise": 54.5% estimated vs 55.3% actual). And the app shows the range and flags uncertain cases for review.

6. **"Did interaction features actually help, or is that your story?"**
   Measured ablation: PR-AUC 0.395 → 0.466 when added to profile + context; most robust group out-of-time. But I also report what was *not* supported (contact frequency × outcome interaction).

7. **"Your top feature is contact channel. Isn't that suspicious?"**
   Yes, and the app says so. `contact = unknown` only occurs in the earliest records, so it partly encodes the collection period. It's flagged in every explanation, and the out-of-time test shows context features don't generalise.

8. **"How did you choose the thresholds?"**
   Break-even = cost ÷ value (decision theory), with illustrative costs the user can change. The review band (middle 50% of 15 bootstrap models, the standard interquartile range) was checked on a validation split carved from the training data, never the test set. A wider band flagged 23% of leads, which isn't workable; the chosen band flags ~13% there (~11% on the portfolio), and those leads are somewhat harder to predict. Honestly, the review flag mainly marks leads sitting right on the line.

9. **"Would this work in an Indian bank today?"**
   Not as-is. The data is from Portugal, 2008–2010, with strong drift. The *architecture* (model → explicit policy → explanation → human) transfers. It would need retraining on the bank's own CRM data, recalibration, fairness review, and regulatory approval. I make no compliance claim.

10. **"Is SHAP telling us what causes subscriptions?"**
    No. SHAP explains what the *model* relies on, which reflects associations in historical data. "Fewer contacts → higher estimate" doesn't prove that calling less causes subscriptions. Proving causation would need an experiment such as a randomised contact policy.

## The 10 brief questions, answered in one line each

1. *Models used?* LR, decision tree, random forest, gradient boosting, MLP, plus the original 1D CNN.
2. *Why?* To span transparent → accurate, and to evaluate the original choices fairly.
3. *Why Random Forest?* Strong and tied with boosting (0.802); not chosen because it's 34× larger and slower to explain.
4. *Why CNN / why rejected?* Kept as an experiment; column-shuffle test shows no spatial learning, and it scores below boosting.
5. *What is XAI?* Methods that show *why* a model made a prediction.
6. *Why SHAP?* Exact, additive, fast for tree models; contributions sum to the prediction.
7. *How does XAI explain one lead?* A waterfall of per-field contributions shown as odds multipliers, plus a plain-English summary built from those numbers.
8. *Most important interaction features?* Days since previous contact and previous outcome (and their interaction).
9. *Did interaction features improve performance?* Yes: PR-AUC 0.395 → 0.466; most robust out-of-time.
10. *How does prediction become a recommendation?* Explicit policy: uncertainty check → priority threshold → break-even (cost ÷ value) → contact-count fatigue rule.

## 5-minute live demo script

1. **Overview** (30 s): read the headline, the 5-step pipeline, "33% of calls → 74% of subscriptions".
2. **Leads** → filter "Stop repeated follow-up", then open a top "Prioritise" lead, e.g. `L43013` (1 min).
3. On the lead page: action, 88% with range, "stable", the SHAP waterfall, the context-flag caveat, the interaction pattern (1 min).
4. **What-if**: set previous outcome to *failure* and contacts to 8 → recommendation changes to "Follow up — change approach" at 12%; point out the "interaction effects" table (1 min).
5. **Decision policy**: raise call cost to €15 → contacts drop from 33% to 12%; explain break-even (45 s).
6. **Model & XAI**: comparison table, calibration chart, CNN shuffle test, ablation (45 s).
7. **Method & limits**: name the three biggest limitations (15 s).
