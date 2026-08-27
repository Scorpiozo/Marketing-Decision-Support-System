# Project Overview (plain-language summary)

## What we built

A **prototype banking marketing analytics dashboard** for the UCI Bank
Marketing dataset. It helps someone unfamiliar with the project answer,
in order: *How did the campaign perform? Who converts? What factors
matter? Who should the bank target next, and what should they do?* —
built for a college project, not connected to a real bank's live systems.

## The dashboard's four core pages

1. **Executive Overview** — top-line campaign numbers: how many customers
   were contacted, how many subscribed, the overall conversion rate, and
   average contacts per customer — all computed directly from the real,
   historical dataset (no model involved).
2. **Campaign Drivers** — which campaign factors go with higher
   conversion: the month contacted, how many times a customer was
   contacted, the previous campaign's outcome, and the last call's
   duration. Shown as conversion-rate charts, not raw counts.
3. **Customer Segments** — which customer characteristics (age group, job,
   education, marital status, housing loan, personal loan, credit
   default) go with higher conversion. You pick one characteristic at a
   time so the page isn't a wall of charts.
4. **Actionable Targeting** — turns the analysis into action: which
   customers the model thinks are most likely to convert, ranked and
   ready to prioritize, with a suggested next step for each.

A sidebar lets you filter all four pages at once (by age group, job,
education, marital status, housing/personal loan, campaign month, and
previous outcome) so every number on screen reflects the same slice of
customers.

Three supporting pages round things out: **Customer Lookup** (look up one
customer's full profile), **New Customer Analysis** (score a customer you
type in yourself, live), and **Model Performance** (the technical
comparison between models, for anyone who wants to check the numbers
behind the scenes).

## Two different kinds of number — and why the distinction matters

This is the most important thing to understand about the dashboard:

- **Historical / observed** (Executive Overview, Campaign Drivers,
  Customer Segments): these numbers come straight from what actually
  happened to all 45,211 real customers in the dataset. "43% of customers
  contacted in March subscribed" is a fact about the past — not a
  prediction, and not proof that calling in March *causes* people to
  subscribe. Other things (like who tends to get called in March) could
  explain part of that pattern too.
- **Model-predicted** (Actionable Targeting, Customer Lookup, New Customer
  Analysis): these numbers are the trained CNN's *estimate* for a specific
  customer — "78% conversion probability" is a guess, not a guarantee, and
  it's only shown for customers the model was tested on (or one you type
  in yourself), not the customers it trained on.

The dashboard labels every page with which of these two it's showing, so
the two are never presented as the same thing.

## What comes out, per customer (Actionable Targeting / Customer Lookup / New Customer Analysis)

1. **Conversion probability** — a percentage chance they'll subscribe to a
   term deposit (the only number that actually comes from the trained
   model — everything below is a rule applied on top of it)
2. **Lead temperature** — HOT, WARM, or COLD, so the team can prioritize at
   a glance
3. **Interaction pattern** — a short label describing their history
   (Interested / Repeatedly Contacted / No Previous Interaction /
   Previously Unsuccessful / Unclear)
4. **Contact fatigue** — LOW, MEDIUM, or HIGH, flagging customers who have
   already been contacted a lot this campaign
5. **Recommendation** — a plain-English suggestion of what to do next,
   combining all three decision-layer outputs above

Plus a ranked list of scored customers (using a project-generated Customer
Record ID, since the UCI dataset has no real customer ID), and a Top
Priority list of the ones most worth calling first.

## What the CNN does

The probability comes from a **1D Convolutional Neural Network**. Normally
CNNs look at images — but here, each customer's information is first
converted into a list of numbers (an "engineered feature representation"),
in a **fixed, documented order** (numerical customer features, then
customer categorical features, then campaign/contact features — saved to
`models/feature_order.json` so it can always be checked), and the CNN
slides small filters across that list the same way it would slide filters
across a row of pixels. This lets it pick up on useful combinations of
nearby features instead of looking at each one in isolation.

**This is a tabular-data adaptation of CNNs, not image recognition** — the
dataset is never treated as an image, and the CNN has not discovered any
natural "spatial" relationship the way it would with actual pixels. It is
included as an experimental/academic deep-learning approach — required by
the project brief, and useful for comparing deep learning against
tree-based models — not because tabular features are assumed to have
image-like structure.

**Is the CNN the "best" model?** Not automatically, and we don't claim
that. In our testing, Random Forest, the CNN, and even a simple balanced
Decision Tree each come out ahead on a different metric (F1-score, ROC-AUC,
and Recall respectively) — they're all genuinely close. The CNN is used to
produce the final probabilities mainly because the project brief requires
an advanced deep-learning model, and because it stays competitive with the
best classical model. The honest, side-by-side numbers are in the README
and in the dashboard's "Model Performance" page.

## What the interaction-pattern and contact-fatigue features do

Beyond just predicting a probability, the system looks at each customer's
**campaign history** — how many times they've been called this campaign,
how many times before, and how the last campaign ended for them — and
sorts them into easy-to-understand behavioural categories using simple,
transparent rules (not a black-box model):

- **Interaction pattern** looks at their longer-term history (previous
  campaign outcome + how many past contacts).
- **Contact fatigue** looks specifically at how many times they've been
  contacted in the *current* campaign, to flag over-contacting risk.

This gives the marketing team context, not just a number: it's the
difference between "70% likely to subscribe" and "70% likely to
subscribe, they've already said yes to us once before, and we haven't
over-contacted them — lead with that."

## Small samples are flagged, not hidden

Some segments (a particular job in a particular month, say) only have a
handful of customers once you filter down far enough. Rather than either
hiding those bars or showing them exactly like a 10,000-customer group,
the dashboard shows every group's sample size in its tooltip and dims any
group with fewer than 30 customers, so a small, noisy result doesn't get
mistaken for a reliable pattern.

## Both remain academic prototypes

Neither the historical analysis nor the model-predicted scoring is
connected to a live bank CRM or a real, ongoing customer database. The
historical pages work off a static, offline dataset; New Customer Analysis
only proves the scoring path can run on any single row shaped like a UCI
record — it doesn't read from or write to any external system.

## How it all fits together

```
Historical, all customers ──► Executive Overview / Campaign Drivers /
(actual outcomes)              Customer Segments  (observed conversion rates)

Customer data (test split) ──► Preprocessing ──► 1D CNN ──► Conversion Probability
                                                                    │
                                                                    ▼
                                                    Lead Temperature (HOT/WARM/COLD)
                                                                    │
Customer history ──► Interaction Pattern rules                      │
Contact count     ──► Contact Fatigue rules                          │
                                                                    ▼
                                                    Recommendation (combines all three)
                                                                    │
                                                                    ▼
                                        Actionable Targeting page / Customer Lookup
```

See `README.md` for the full technical write-up, the honest model
comparison numbers, and known limitations.
