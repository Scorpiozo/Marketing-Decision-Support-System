# Limitations

This is a research prototype for decision support. The limitations below are part of the result, not footnotes to it.

## Data

| Limitation | Consequence |
|---|---|
| One Portuguese bank, phone campaigns, May 2008 – Nov 2010 (financial-crisis period) | Not representative of other banks, countries (including India), products, channels or today's customers |
| No conversation data: no transcripts, messages, sentiment, objections, CRM notes or per-agent actions | "Interaction pattern" means contact **history** only (`campaign`, `previous`, `pdays`, `poutcome`). The system cannot judge a customer's stated intent; every lead card says so |
| Only three summary counts, not a touch-by-touch timeline | Recency, cadence and channel sequences can't be modelled |
| Missingness encoded as "unknown" (poutcome 81.7 %, contact 28.8 %) | Much of the history signal is absent for most leads |
| No customer IDs | Lead IDs are derived from row position |

## Time drift (the most important limitation)

* Subscription rate rises from 3.4 % in the earliest fifth of the file to 31.6 % in the latest.
* Out-of-time evaluation (train earliest 80 %, test latest 20 %): deployed GB ROC-AUC falls from **0.808 → 0.665**, PR-AUC from 0.466 → 0.482 against a base rate that rose to 31.6 %, and calibration error rises to ~20 %. The most robust out-of-time model was a simpler logistic regression on profile + interaction history (0.723).
* Implication: a real deployment would need time-based validation, drift monitoring, regular retraining and recalibration. The headline in-period metrics overstate how well this model would do on next quarter's leads.

## Era-proxy features

`contact`, `month` and `day` are available at decision time, but in this file they partly encode the collection period (`contact = unknown` is almost exclusively early; day × month reconstructs the date). They are the model's strongest global drivers. They are kept for in-period accuracy, flagged ⚑ in every explanation, and never offered as levers. Removing them trades 0.01–0.06 in-period AUC for better out-of-time behaviour, mainly for logistic regression ([EXPERIMENTS.md](EXPERIMENTS.md#3b-dropping-calendar-proxies-exploratory-probe)).

## Modelling and explanation

* **Associations, not causes.** Neither the model nor SHAP shows that changing a field would change the outcome. The what-if simulator shows how the *model* responds, not how a customer would.
* **Correlated features share credit** in SHAP (e.g. `pdays` / `previous` / `poutcome`).
* **Uncertainty is epistemic only.** The bootstrap band measures sensitivity to the training sample, not data shift. It will not widen for a lead from a new era or a different bank.
* **Leakage judgement.** `campaign` counts the last contact too; we treat it as "contacts so far", which is available at decision time. A stricter design would use contacts *before* the current call.
* **Modest tuning.** A small grid only; gains within families were ~0.01 AUC.
* **Policy numbers are a back-test, not a field trial.** The policy's segment results are measured on held-out historical leads. The review band was re-selected on dev-only data after an initial look at the test set (disclosed in EXPERIMENTS.md §7). The review flag carries only modest information (Brier 0.086 vs 0.081 on validation).

## Decision policy

* **Economics are assumed.** €5 per contact and €50 per subscription are illustrative; results are sensitive to them ([DECISION_POLICY.md](DECISION_POLICY.md#sensitivity)).
* **"Contact fatigue" is a proxy** (contact count ≥ 4), justified by falling conversion rates, not by any fatigue measurement.
* **"Change approach" can't say which approach.** The data has no record of what channel, offer or script would work better.
* **Single outcome.** The label is term-deposit subscription. Real CRM value includes other products, lifetime value, complaints, consent and customer experience.
* **The portfolio is the held-out test set**, not live leads. "Back-test" outcomes are historical.

## Fairness, governance and compliance

* Age, job, marital status and education are model inputs. Only a basic subgroup check was done (calibration holds; ROC-AUC 0.75–0.85 across groups; [MODEL_EVALUATION.md](MODEL_EVALUATION.md#subgroup-check)). There has been **no fairness audit**, no assessment of disparate impact, and no review of whether these attributes may lawfully be used for contact prioritisation.
* **No regulatory claim.** The prototype is not reviewed, approved or endorsed by any bank or regulator, including the RBI, and makes no claim of compliance with any banking, privacy or model-risk framework. It is an academic demonstration of how such a system *could* be structured.
* The deployed demo uses a shared password gate only. There is no user management or audit log.

## What would change the conclusions

| If… | Then… |
|---|---|
| Real CRM event histories and conversation signals were available | The "interaction pattern" could become genuinely behavioural, and the missed-subscriber pattern (mostly leads with no history) would likely shrink; each new signal would need the same leakage audit as `duration` |
| Data came from a stable period or another bank | Context features might stop acting as era proxies; the out-of-time gap should be re-measured |
| Real costs and capacity were supplied | Break-even and priority thresholds, and therefore the action mix, would change ([sensitivity](DECISION_POLICY.md#sensitivity)) |
| A time-based training/validation scheme were adopted for the deployed model | The model choice might shift towards the simpler, more robust profile + interaction logistic regression |
| Outcomes of actual relationship-manager decisions were logged | The policy could be evaluated as an intervention rather than only back-tested |
