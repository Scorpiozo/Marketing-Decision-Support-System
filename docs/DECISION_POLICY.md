# Decision policy

## Prediction is not a decision

| Layer | Question | Lives in | Changes when… |
|---|---|---|---|
| Model | How likely is this lead to subscribe? | `models.py`, `artifacts/model.joblib` | data or features change (retrain) |
| Policy | Given our costs and contact history, what should the team do next? | `policy.py` (`PolicyConfig`, `decide`, `decide_many`) | business assumptions change (no retraining) |
| Human | Do I agree, given what I know about this customer? | the relationship manager | always |

A probability of 0.73 does not mean "continue" by itself. It becomes an action only through explicit, editable rules. In the app these are set on the **Decision policy** page and stored per browser; every page updates.

## Rules (first match wins)

| # | Condition | Action | Next step shown to the user |
|---|---|---|---|
| 1 | Uncertainty band straddles the break-even line | **Human review** | Estimate too uncertain; a relationship manager decides using context the model cannot see |
| 2 | p ≥ priority threshold | **Prioritise follow-up** | Contact in the next calling block, ahead of standard leads |
| 3 | p ≥ break-even **and** contacts this campaign ≥ heavy-contact limit | **Follow up — change approach** | Still worth pursuing, but repeated contact has not converted; review channel, timing or offer first |
| 4 | p ≥ break-even | **Follow up** | Keep in the standard queue |
| 5 | p < break-even **and** heavily contacted | **Stop repeated follow-up** | Another call is below cost and the lead has been contacted repeatedly |
| 6 | otherwise | **Deprioritise** | No active follow-up this campaign; keep for future campaigns |

Every decision carries reasons generated from its numbers, e.g. "Estimate 12 % is above the 10 % break-even line", "Already contacted 8 times this campaign (≥ 4)…", and "Expected value of one more contact ≈ €x (assumes …)". The vectorised `decide_many` is tested to match `decide` exactly on 3,000 random cases.

## Break-even (why "cost ÷ value")

Following cost-sensitive decision theory (Elkan, 2001), one more contact is worth making when its expected value is positive:

    p × V − C > 0   ⇔   p > C / V

where p is the calibrated probability, V the net value of a subscription and C the cost of a contact. This only works because the model's probabilities are **calibrated** (held-out ECE 0.86 %). The legacy class-balanced models, with probabilities inflated 2–3×, would have made every lead look worth calling.

## Defaults are illustrative assumptions

| Parameter | Default | Basis |
|---|---|---|
| Contact cost C | €5 | Assumption (no cost data in the dataset) |
| Subscription value V | €50 | Assumption |
| Break-even C/V | 10 % | Derived |
| Priority threshold | 35 % | ≈ 3× base rate; flags ~8 % of leads, which convert at 55 % in the back-test |
| Heavy contact | 4 contacts this campaign | Observed conversion by contacts: 14.6 % (1) → 11.2 % (2–3) → 9.0 % (4) → 7.9 % (5) → … → 4.2 % (10+) |
| Review band | 25–75 % bootstrap interval | Flags ~11 % of leads; see [EXPERIMENTS.md](EXPERIMENTS.md#7-human-review-band) |

**Contact fatigue is a proxy.** The data shows conversion falling steadily with contacts per campaign, in every history group ([EXPERIMENTS.md](EXPERIMENTS.md#4-interaction-effects)), but there is no fatigue label. "Heavily contacted" is a heuristic, labelled as such in the UI.

**Human review.** The 15 bootstrap re-fits of the model are asked "which side of the line is this lead on?". If the middle half of them disagree, the policy refuses to decide. Those leads convert at ~10 %, essentially at the break-even rate, and the model is less accurate on them (Brier 0.093 vs 0.079).

## Back-test (defaults)

On the 9,043 held-out leads, against real outcomes:

| Action | Leads | Share | Actually subscribed | Share of subscriptions |
|---|---|---|---|---|
| Prioritise | 747 | 8.3 % | 55.3 % | 39.0 % |
| Follow up | 1,091 | 12.1 % | 22.5 % | 23.2 % |
| Change approach | 162 | 1.8 % | 16.7 % | 2.6 % |
| Human review | 999 | 11.0 % | 9.7 % | 9.2 % |
| Deprioritise | 4,512 | 49.9 % | 4.6 % | 19.6 % |
| Stop repeated follow-up | 1,532 | 16.9 % | 4.5 % | 6.5 % |

The policy contacts 33.2 % of leads (review included) and captures **73.9 %** of subscriptions. Random selection of the same size would capture ~33 %. It avoids 6,044 calls and forgoes 276 of 1,058 subscriptions. Illustrative value under the assumptions: €24,105 vs €7,685 for "contact everyone". This is not a revenue forecast.

## Sensitivity

`ScoringService().policy_backtest(PolicyConfig(contact_cost=C, conversion_value=V, priority_threshold=P))`:

| C (€) | V (€) | Break-even | Contacted | Subscriptions captured | Calls avoided | Subscriptions forgone | Review leads | Value: policy / contact-all |
|---|---|---|---|---|---|---|---|---|
| 3 | 60 | 5 % | 71.4 % | 92.7 % | 2,586 | 77 | 1,487 | €39,489 / €36,351 |
| 5 | 50 | 10 % | 33.2 % | 73.9 % | 6,044 | 276 | 999 | €24,105 / €7,685 |
| 10 | 50 | 20 % | 15.4 % | 57.3 % | 7,650 | 452 | 322 | €16,370 / −€37,530 |
| 15 | 50 | 30 % | 11.5 % | 48.7 % | 8,001 | 543 | 287 | €10,120 / −€82,745 |

(The priority threshold here is 35 %.) The priority threshold only splits the contacted group, so it leaves the contacted share unchanged:

| Priority threshold (C=5, V=50) | Prioritised leads | Their observed rate |
|---|---|---|
| 25 % | 1,051 | 49.3 % |
| 35 % | 747 | 55.3 % |
| 50 % | 403 | 62.8 % |

**Reading it.** When calls are cheap relative to a subscription, the policy approaches "call almost everyone". When they are expensive, it concentrates effort and the gap over "contact everyone" widens. The ranking stays useful across the range, but the right setting depends on economics the dataset does not contain. If the priority threshold is set *below* break-even (e.g. C=15, V=50, priority 25 %), rule 2 fires first, so a few leads below break-even are prioritised (contacted share 11.9 % vs 11.5 %). The UI shows both lines so a user can see this.

## What the policy must never do

* Present itself as the decision-maker. All copy says "recommended action · decision support".
* Act on a lead whose history is internally impossible. Invalid inputs are rejected with an explanation (schema validation).
* Hide the cost assumptions. They appear in every decision's reasons and on the policy page.
* Treat context fields (channel, month, day) as levers in the what-if simulator.
* Expose held-out outcomes through the API.

## How a bank would set it

1. Replace C and V with real per-contact cost (agent time, channel cost) and margin or lifetime value per product, ideally per segment.
2. Add capacity: pick the priority threshold so the "Prioritise" volume matches calling capacity.
3. Replace the contact-count fatigue proxy with contact-frequency rules and consent / do-not-call status from the CRM.
4. Recalibrate the model on recent data before using break-even (out-of-time ECE here is ~19 %).
5. Monitor: log recommended vs actual action and outcome, and re-run this back-test monthly.
