# Product Vision

*Status as of 2026-10-07. ✔ = implemented and tested; ○ = planned / out of scope for the prototype.*

## Problem statement

A bank's sales or relationship team contacts the same leads repeatedly across campaigns.
Without decision support, effort is spread by habit: promising leads wait while
low-probability leads receive their sixth or seventh call. Existing "lead scoring" demos
answer *"will this customer subscribe?"* but not *"what should we do next, and why?"*.

## Research question

> Can customer and campaign interaction patterns be transformed into an interpretable lead
> decision system that helps determine which leads deserve continued engagement, which
> require a different response strategy, and which should be de-prioritised or have
> follow-up terminated?

The system is **decision support**: it recommends, explains and flags uncertainty; a person decides.

## Personas

| Persona | Goal | Needs from the system |
|---|---|---|
| Relationship manager | Spend calling time where it matters | A clear next action per lead, the reasons, and when to stop |
| Campaign / analytics lead | Plan capacity and judge the approach | Portfolio segmentation, back-tested results, editable cost assumptions |
| Faculty / research reviewer | Judge validity | Leakage audit, honest metrics, model comparison, ablation, limitations |

## User stories

| # | Story | Status |
|---|---|---|
| U1 | As a relationship manager, I want each lead to show a recommended action so that I know what to do next. | ✔ |
| U2 | As a relationship manager, I want to see which signals drove the estimate so that I can judge whether to trust it. | ✔ |
| U3 | As a relationship manager, I want borderline leads flagged for my judgement so that the system does not pretend to certainty. | ✔ |
| U4 | As a relationship manager, I want to be told when repeated follow-up should stop so that customers are not over-contacted. | ✔ |
| U5 | As an analytics lead, I want a ranked, filterable lead list so that I can plan the next calling block. | ✔ |
| U6 | As an analytics lead, I want to change cost assumptions and see the resulting segmentation so that thresholds match our economics. | ✔ |
| U7 | As an analytics lead, I want to test "what if" a lead's history were different so that I understand the model's behaviour. | ✔ |
| U8 | As a reviewer, I want to see model comparison, calibration and ablation so that I can judge the evidence. | ✔ |
| U9 | As a reviewer, I want leakage and limitations stated plainly so that claims are not overstated. | ✔ |
| U10 | As a CRM integrator, I want a JSON scoring endpoint so that the logic can be called from another system. | ✔ (`POST /api/v1/assess`) |
| U11 | As a manager, I want to record what action was actually taken so that recommendations can be evaluated over time. | ○ |

## Decision workflow

```
Lead record ─▶ validate ─▶ model estimate P(subscribe) ─▶ bootstrap uncertainty
           ─▶ TreeSHAP explanation ─▶ interaction-pattern label
           ─▶ policy (costs, priority line, contact intensity) ─▶ recommended action + reasons
           ─▶ relationship manager reviews and decides
```

## Scope

**In scope (done):** leak-free calibrated model; model comparison incl. the legacy CNN;
feature ablation in-period and out-of-time; interaction-effect analysis; explicit policy
with uncertainty-based human review; per-lead and global explanations; what-if simulator;
policy back-test; password-gated web app; JSON API.

**Non-goals:** autonomous contact decisions; real-time CRM integration; conversation or
sentiment analysis (no such data); LLM-generated explanations; real ROI forecasts;
regulatory compliance claims (RBI or otherwise); any claim of use by a real bank.

## Dataset limitations (summary)

One Portuguese bank, 2008–2010; 16 summary fields; no transcripts, notes, agent actions or
per-touch timestamps; strong time drift; channel/month/day partly encode the collection
period. Full list: [LIMITATIONS](LIMITATIONS.md).

## Failure modes and how the system fails gracefully

| Failure mode | Behaviour | Status |
|---|---|---|
| Hallucinated explanation | Not possible by design: no LLM; every sentence is a template filled from model output, SHAP values, observed rates or policy numbers | ✔ |
| Impossible input (e.g. never contacted but previous outcome "failure") | Refused with a plain-language message; nothing is scored | ✔ |
| Out-of-range / unknown category | Rejected by schema validation (422 on API, message in UI) | ✔ |
| Model uncertain near the decision line | Action becomes **Human review recommended** with the range shown | ✔ |
| Prediction driven by era-proxy fields (channel, month, day) | Flagged ⚑ in charts and a caveat on the lead card; never offered as a lever | ✔ |
| Sparse pattern (few similar training leads) | Caveat stating the sample size | ✔ |
| Time drift / new period | Stated as a headline limitation; out-of-time results published | ✔ (monitoring ○) |
| Missing model artifacts | Start-up fails with "Run `uv run leadintel-train` first" | ✔ |
| Invalid policy assumptions | Rejected with an error; previous policy kept | ✔ |
| Silent explanation error (SHAP library) | Additivity checked on every training run and in tests | ✔ |
| Over-contacting | "Stop repeated follow-up" for low-value leads with ≥ 4 contacts (proxy rule) | ✔ |

**Hard rules:** never use `duration`; never expose held-out outcomes through the API; never
present costs as real economics; never present the system as the decision-maker.

## Scenarios (real leads from the held-out portfolio, default policy)

| # | Input | Expected result | Verified |
|---|---|---|---|
| S1 | **L43013** — previously converted, 2 contacts this campaign, last contact 183 days ago | **Prioritise follow-up**, estimate 88 % (range 87–90 %), stable; top drivers recency and previous success; caveat that day/channel/month are context | ✔ (browser) |
| S2 | **L41241** — previously converted, 1 contact | Prioritise, estimate 90 %; listed first in "top priority leads" | ✔ |
| S3 | **L30983** — new to campaigns, 5 contacts this campaign, telephone | **Stop repeated follow-up**, estimate 4.5 % (below 10 % line, heavily contacted) | ✔ |
| S4 | **L32248** — new to campaigns, 1 contact | **Human review recommended**: estimate 9.8 %, range 9.3–11.2 % straddles the 10 % line | ✔ |
| S5 | **L37712** — previously declined, 4 contacts | **Follow up — change approach**: estimate 16.8 %, above the line but heavily contacted | ✔ |
| S6 | What-if on L43013: previous outcome → failure, contacts → 8 | Estimate 88 % → 12 %; action Prioritise → Change approach; unchanged "days since previous contact" shown as an interaction effect | ✔ (browser) |
| S7 | What-if with previous = 0 and outcome = failure | "This combination can't be scored" with the reason | ✔ (test) |
| S8 | Policy: contact cost €5 → €15 | Follow-up line 10 % → 30 %; contacted share 33 % → 12 % | ✔ (browser) |
