# Project Overview (plain-language summary)

## What we built

A **prototype banking marketing analytics dashboard** for the UCI Bank
Marketing dataset. It helps someone unfamiliar with the project answer,
in order: *How did the campaign perform? Who converts? What factors
matter? Who should the bank target next, and what should they do?* —
built for a college project, not connected to a real bank's live systems.

## The dashboard's four sections

It runs in a web browser (`python run.py`, then open http://localhost:5000).

1. **Executive overview** — one sentence with the headline result (how
   many contacted customers subscribed, and the conversion rate), plus a
   few supporting numbers, monthly charts and a snapshot of how many
   customers the model rates hot, warm or cold.
2. **Campaign insights** — which campaign factors (month, number of
   contacts, previous outcome, call duration) and which customer
   characteristics (age group, job, education, marital status, loans,
   credit default) go with higher conversion. You pick one customer
   characteristic at a time, so the page isn't a wall of charts.
3. **Actionable targeting** — turns the analysis into action: customers
   ranked by how likely the model thinks they are to convert, each with a
   suggested next step. You can click any customer for their full
   profile, switch between the two models, download the list, or score a
   customer you type in yourself.
4. **Model performance** — the technical comparison of the two models,
   for anyone who wants to check the numbers behind the scenes.

One row of filters (age group, job, education, marital status,
housing/personal loan, campaign month, previous outcome) applies to the
first three sections at once, so every number on screen reflects the same
slice of customers.

## Two different kinds of number — and why the distinction matters

This is the most important thing to understand about the dashboard:

- **Historical / observed** (Executive overview, Campaign insights): these numbers come straight from what actually
  happened to all 45,211 real customers in the dataset. "43% of customers
  contacted in March subscribed" is a fact about the past — not a
  prediction, and not proof that calling in March *causes* people to
  subscribe. Other things (like who tends to get called in March) could
  explain part of that pattern too.
- **Model-predicted** (Actionable targeting, and the lead snapshot on the
  overview): these numbers are the trained model's *estimate* for a specific
  customer — "78% conversion probability" is a guess, not a guarantee, and
  it's only shown for customers the model was tested on (or one you type
  in yourself), not the customers it trained on.

The dashboard labels every section with which of these two it's showing, so
the two are never presented as the same thing.

## What comes out, per customer (Actionable targeting)

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

## Where the probability comes from

The probability comes from one of two tree-based machine-learning models,
both trained on the same data: **Gradient Boosting** (many small decision
trees built one after another, each fixing the previous ones' mistakes)
and **Random Forest** (many independent decision trees that vote). Each
customer's information is first converted into a fixed list of 51 numbers
in a documented order (saved to `models/feature_order.json`).

**Which one is used?** Whichever scores higher on ROC-AUC on the held-out
test customers becomes the "production model" for the dashboard. In the
shipped run that is Gradient Boosting (ROC-AUC 0.930 vs 0.924) — the two
are close, and the side-by-side numbers are in the README and on the
dashboard's "Model Performance" page.

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
historical sections work off a static, offline dataset; the "score a new
customer" form only proves the scoring path can run on any single row shaped like a UCI
record — it doesn't read from or write to any external system.

## How it all fits together

```
Historical, all customers ──► Executive overview / Campaign insights
(actual outcomes)              (observed conversion rates)

Customer data (test split) ──► Preprocessing ──► GB / RF ──► Conversion Probability
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
                                        Actionable targeting section
```

See `README.md` for the full technical write-up, the honest model
comparison numbers, and known limitations.
