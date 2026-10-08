# Lead Decision Intelligence — Final Handoff Guide

*User guide · how it works · ML and explainable AI · architecture · deployment · viva preparation*

| | |
|---|---|
| **Live application** | https://web-production-4cf5b6.up.railway.app |
| **Code** | branch `redesign` (final). The original student version is preserved at git tag `legacy-v1`. |
| **Verified against** | final source code, trained artifacts, test suite (59 tests passing) and the live deployment on 2026-10-08 |
| **Who this is for** | anyone who must understand, demonstrate or defend the project, especially a student who is not a programmer |

> **How to read this guide.** Part A (sections 1–7) is enough to understand and demo the
> product. Part B (8–14) explains the machine learning and explainable AI. Part C (15–21)
> covers the technology and deployment. Part D (22–26) covers impact, novelty and limits.
> Part E (27–33) is viva preparation: questions, scripts, a revision sheet and a glossary.
> In a hurry? Read **section 1**, then **section 32 (5-minute revision)**.

---

## Contents

**Part A — The product**
1. Lead Intelligence in one page
2. The problem, the users and a real-world story
3. The six pages (application walkthrough)
4. How to use Lead Intelligence (step by step)
5. What the prediction means, and what it does not mean
6. Prediction vs decision: the decision layer
7. What-if simulation

**Part B — Data, machine learning and explainable AI**
8. The dataset
9. Data preparation (preprocessing)
10. Target leakage: why `duration` was removed
11. Model comparison: six approaches
12. The final model: gradient boosting
13. Explainable AI with SHAP (local and global)
14. How confident is the system? (likely range and human review)

**Part C — Technology and deployment**
15. Architecture
16. Frontend, backend and API
17. Model serving and storage
18. Technology stack
19. Deployment on Railway
20. How to run Lead Intelligence locally
21. Testing, performance and accessibility

**Part D — Value, novelty and honesty**
22. Business impact
23. Technical and practical feasibility
24. What is different: product, research, IP
25. Limitations and what the system does not do
26. How the project evolved, and future scope

**Part E — Viva and presentation kit**
27. Faculty viva: questions and short answers
28. The 30-second explanation
29. The 2-minute technical explanation
30. For a business person / for a technical expert
31. Live demo script
32. 5-minute revision sheet
33. Glossary

---

# PART A — THE PRODUCT

## 1. Lead Intelligence in one page

**What it is.** Lead Decision Intelligence is a web application for sales and relationship
teams. For every lead it **estimates** how likely that person is to say yes, **explains**
which factors drove that estimate, and **recommends a next action** (for example
"Prioritise follow-up" or "Stop repeated follow-up"). A person makes the final call.

**The problem it solves.** Sales teams have more leads than time. A bare prediction
("78 %") does not say whom to call first, why, or what to do next. Lead Intelligence turns
a prediction into an explained, actionable recommendation.

**Who uses it.** Relationship managers and sales agents (which lead next, and why?),
sales or campaign managers (how should we allocate calls?), and analysts (how good is the
model, and what drives it?).

**How it works, in one picture:**

```text
Lead data (profile + interaction history)
        │
        ▼
Validation  ──────────── impossible records are rejected with a clear message
        │
        ▼
ML prediction  ───────── gradient boosting → likelihood to subscribe (e.g. 88 %)
        │
        ▼
Explanation  ─────────── SHAP: which fields pushed the likelihood up or down
        │
        ▼
Decision layer  ──────── cost-aware rules → recommended action + reasons + confidence
        │
        ▼
Prioritised lead list, lead page, what-if simulator
        │
        ▼
Human decision  ──────── the relationship manager decides what to actually do
```

**Headline results (measured on 9,043 leads the model never saw during training):**

| What | Result |
|---|---|
| Ranking accuracy (ROC-AUC) | **0.808** |
| Contacting the 33 % of leads the system recommends… | …reaches **74 %** of everyone who subscribed (random calling would reach 33 %) |
| Leads marked "Prioritise follow-up" | **55 %** of them actually subscribed (average lead: 11.7 %) |
| Probability honesty (calibration error) | **0.9 %**: when it says ~30 %, about 30 % happen |
| Time to re-score a lead in the what-if simulator | ~**27 ms** on the server |

> **One sentence to remember:** *Lead Intelligence turns "how likely is this lead to say
> yes?" into "what should we do next, and why?", and keeps a human in charge.*

---

## 2. The problem, the users and a real-world story

### The problem

A bank (or any sales team) runs phone campaigns. It has thousands of leads and limited
agent time. It calls some people again and again, while others who might have said yes are
never reached. A machine-learning model can estimate likelihood, but **prediction alone is
not enough**. A salesperson still needs to know:

- **whom** to prioritise;
- **why** a lead is promising (or not);
- **what** to do next: call now, try a different approach, or stop;
- **what would change** the picture.

Lead Intelligence is built to bridge exactly that gap.

### Who uses it

| User | Their question | Where they find the answer |
|---|---|---|
| Relationship manager / sales agent | "Which lead should I call next, and why?" | Leads list → lead page |
| Sales or campaign manager | "If a call costs €X, how many leads should we contact?" | Strategy page |
| Analyst / reviewer | "How good is the model, and what drives it?" | Model and Insights pages |
| IT / CRM team | "Can our system ask it for a score?" | JSON API (About page, `/api/docs`) |

### A real-world story (easy to remember)

> Priya leads a team of five relationship managers. This week they have **9,043 leads** and
> time for about **3,000 calls**.
>
> 1. **Data in.** Each lead's profile (age, job, balance, loans) and interaction history
>    (contacted before? what happened last time? how many calls already this campaign?) is in
>    the system.
> 2. **Checked.** The system checks that every record makes sense.
> 3. **Predicted.** The model estimates each lead's likelihood to subscribe.
> 4. **Explained.** For each lead it shows which factors raised or lowered that likelihood.
> 5. **Decided.** A transparent rule turns likelihood into an action, using Priya's own numbers
>    (a call costs €5, a subscription is worth €50).
> 6. **Prioritised.** About 33 % of leads are recommended for contact, and the top 8 % are
>    marked "Prioritise follow-up".
> 7. **Inspected.** An agent opens lead **L43013**: **88 %** likely, "Prioritise follow-up",
>    mainly because this person said yes in a previous campaign.
> 8. **Explored.** The agent asks: *what if this lead had declined last time and we had already
>    called 8 times?* The likelihood drops to **12 %** and the advice changes to
>    "Follow up — change approach".
> 9. **Human decides.** The agent calls L43013 first. The system advised; the person decided.

---

## 3. The six pages (application walkthrough)

The application has **six pages in the top menu**, plus a **lead page** that opens when you
click any lead. Every page is server-rendered (see section 16) and works on desktop and
phone, in light and dark mode (the "Theme" button).

### 3.1 Dashboard (`/`)

| | |
|---|---|
| **What you see** | A headline ("Know which leads to call next, and why."), a 5-step workflow, four key-figure tiles, a ring chart of recommended actions, a **targeting-power (gains) curve**, a bar chart of conversion by interaction history, and the top priority leads. |
| **What you do** | Read the summary; click **Open the top lead** or **Browse all leads**; hover any chart for details. |
| **Purpose** | The 30-second picture of the whole lead portfolio. |
| **Behind the scenes** | The server applies the decision rules to all 9,043 leads and draws the charts as SVG (~10 ms). |
| **Decision it supports** | "How big is the opportunity, and where should I start?" |

Key figures shown: 9,043 leads scored · 33 % recommended to contact · 74 % of subscribers
reached · ROC-AUC 0.81. The gains curve shows that ranking leads by the model reaches
subscribers far faster than calling at random (the dashed diagonal).

### 3.2 Leads (`/leads`)

| | |
|---|---|
| **What you see** | Clickable action chips with counts (e.g. "Prioritise follow-up 747"), filters (action, interaction history, sort order, lead-ID search) and a ranked table: Lead · Likelihood · Likely range · Recommended action · Interaction pattern · Contacts · Profile. 25 leads per page. |
| **What you do** | Filter (e.g. only "Stop repeated follow-up"), search for an ID, page through, click a lead. |
| **Purpose** | The working list: who to call, in what order. |
| **Behind the scenes** | Filters update the table without a full page reload (HTMX, section 16). |
| **Decision it supports** | "Which leads go into today's calling list?" |

### 3.3 Lead page (`/leads/{id}`, e.g. `/leads/L43013`)

The most important screen. Reached by clicking any lead.

| Part | What it shows |
|---|---|
| Header | Lead ID, recommended-action badge, interaction-pattern label (e.g. "Previously converted · 2–3 contacts") and a one-paragraph plain-English summary |
| **Decision card** | Recommended action and next step, a **likelihood gauge** (with tick marks at the follow-up and priority lines), likely range, confidence ("High" or "Needs review"), value of the next contact in €, and the reasons |
| **Why this recommendation** | SHAP chart: each field's push up (blue) or down (red), shown as odds multipliers (×2.33 = doubles the odds), plus "Working for this lead" and "Working against" lists |
| **Interaction profile** | Previous campaigns, this campaign, how often similar leads subscribed, customer profile |
| **What-if simulator** | Editable form with every field; results appear instantly beside it (section 7) |

*Decision it supports:* "Should I call this person, and what should I say or do differently?"

### 3.4 Insights (`/patterns`)

| | |
|---|---|
| **What you see** | A heat map of subscription rate by previous-campaign history × contacts this campaign; a chart showing that interaction signals add predictive power (+18 %); conversion by number of contacts, by history and by age group; per-value SHAP effects for the four interaction-history signals. |
| **Purpose** | Understand the **patterns** behind the predictions. |
| **Decision it supports** | Campaign planning: "Which kinds of leads respond, and when does calling again stop paying off?" |

Example insights shown: previously-converted leads subscribe at 68 % (1 contact) vs 11 % for
leads new to campaigns; conversion falls from 15 % at the 1st contact to 5 % at 8+ contacts.

### 3.5 Model (`/model`)

| | |
|---|---|
| **What you see** | Four tiles (ROC-AUC 0.808 · top-10 % hit rate 53 % · 70 % of subscribers caught at the follow-up line · calibration error 0.9 %); a six-model comparison chart and table; ROC curve; global SHAP importance; predicted-vs-actual chart; confusion matrix; a summary of the explainable-AI features. |
| **Purpose** | Evidence that the model is accurate, honest and explainable. |
| **Decision it supports** | "Can we trust these numbers?" |

### 3.6 Strategy (`/policy`)

| | |
|---|---|
| **What you see** | Four sliders (cost of a contact, value of a subscription, priority threshold, how many contacts count as "heavily contacted"), the rules in plain words, live results (leads to contact, subscribers reached, calls saved), the action-mix ring chart, a predicted-vs-actual table per action, and campaign value under your assumptions. |
| **What you do** | Move a slider; everything recalculates in milliseconds. Your settings are remembered in your browser and apply on every page. |
| **Decision it supports** | "Given our real costs, how aggressive should our follow-up be?" |

Example: raising the cost per call from €5 to €15 cuts recommended contacts from 33 % to 12 %
of leads.

### 3.7 About (`/about`)

The six-step workflow, the data, the model, the technology, and an example of the
**integration API** for CRM systems (with a link to interactive API documentation at
`/api/docs`).

---

## 4. How to use Lead Intelligence (step by step)

```text
1. Open the application  →  Dashboard: see the big picture
2. Go to Leads           →  click an action chip (e.g. "Prioritise follow-up")
3. Open a lead           →  read the recommended action and the likelihood gauge
4. Check confidence      →  "High" or "Needs review" (human judgement needed)
5. Read the reasons      →  "Why this recommendation": what is working for and against
6. Look at the history   →  Interaction profile: past campaigns, calls so far
7. Try what-if           →  change a field and watch the likelihood and action update
8. Decide                →  you choose the next sales action
(Managers) Strategy      →  set real costs and thresholds; all pages follow
```

> **Note.** In this application the leads are **existing records** (the 9,043 held-out leads
> from the dataset). There is no "add new lead" form on the website. New leads are scored through
> the **what-if simulator** (by editing any lead) or through the **API** (`POST /api/v1/assess`),
> which is how a CRM would connect.

---

## 5. What the prediction means, and what it does not mean

**What the model predicts:** the probability that a lead **subscribes to a term deposit**
(the outcome recorded in the dataset), given the lead's profile and interaction history.

| A likelihood of 88 % **means** | It does **not** mean |
|---|---|
| Among leads that look like this one, the model expects roughly 88 in 100 to subscribe. | "This person will definitely buy." |
| The estimate is *calibrated*: across all leads, predicted and actual rates match closely (error 0.9 %). | Anything about why the person will buy. The model finds patterns, not causes. |
| It is learned from historical campaign data. | That it knows today's market or this person's intentions. |

**Likely range** (e.g. 87 %–90 %): the spread of estimates from 15 re-trained versions of the
model (section 14). A narrow range means the estimate is stable.

---

## 6. Prediction vs decision: the decision layer

This separation is the heart of the project.

```text
Machine learning            "How likely is this lead to subscribe?"        → 88 %
        │
        ▼
Decision layer (rules)      "Given our costs and this lead's history,      → Prioritise follow-up
                             what should we consider doing?"
        │
        ▼
Human                       "What will I actually do?"                     → calls the lead
```

**Why separate them?** The same 20 % likelihood can be worth a call when calls are cheap and
not worth it when calls are expensive. Business priorities change faster than models. Keeping
the policy separate means managers can change it (Strategy page) **without retraining the
model**, and anyone can read exactly why an action was chosen.

### The actual rules (first matching rule wins)

Inputs: the likelihood **p**, its likely range, and **contacts this campaign**. Default
settings (editable on the Strategy page): contact cost **€5**, subscription value **€50**,
priority threshold **35 %**, heavily contacted at **4** contacts.

**Follow-up line (break-even)** = cost ÷ value = 5 ÷ 50 = **10 %**. A call is worth it when
p × €50 > €5.

| # | Rule | Action |
|---|---|---|
| 1 | The likely range crosses the follow-up line (the model is unsure which side the lead is on) | **Human review recommended** |
| 2 | p ≥ priority threshold (35 %; never below the follow-up line) | **Prioritise follow-up** |
| 3 | p ≥ follow-up line **and** already contacted ≥ 4 times this campaign | **Follow up — change approach** |
| 4 | p ≥ follow-up line | **Follow up** |
| 5 | p < follow-up line **and** contacted ≥ 4 times | **Stop repeated follow-up** |
| 6 | everything else | **Deprioritise** |

**Value of the next contact** shown on the lead page = p × value − cost (for L43013:
0.877 × €50 − €5 ≈ **€39**).

> **Important wording.** These are **transparent business rules applied to a model's
> estimate**. They are *not* "AI decisions". The rules are deterministic: the same inputs
> always give the same action, and a person is responsible for the final choice.

### Lead prioritisation, with real numbers (default settings, 9,043 leads)

| Action | Leads | Share | Predicted rate | **Actually subscribed** |
|---|---|---|---|---|
| Prioritise follow-up | 747 | 8 % | 54.5 % | **55.3 %** |
| Follow up | 1,091 | 12 % | 19.2 % | 22.5 % |
| Follow up — change approach | 162 | 2 % | 18.9 % | 16.7 % |
| Human review recommended | 999 | 11 % | 9.9 % | 9.7 % |
| Deprioritise | 4,512 | 50 % | 5.0 % | 4.6 % |
| Stop repeated follow-up | 1,532 | 17 % | 4.7 % | 4.5 % |

Two things to notice: (1) the categories genuinely separate good from poor leads (55 % vs
4.5 %); (2) predicted and actual rates match in every row, so the probabilities can be taken at
face value.

### Follow-up recommendations

Each action comes with a plain next step, for example:

- **Prioritise follow-up:** "Contact in the next calling block, ahead of standard leads."
- **Follow up — change approach:** still worth pursuing, but repeated contact has not
  converted. Review channel, timing or offer before calling again.
- **Stop repeated follow-up:** another call is not worth its cost and the lead has already been
  contacted repeatedly.
- **Human review:** the estimate is too uncertain to call either way. A relationship manager
  should decide using context the model cannot see.

Why "4 contacts"? In the data, conversion falls steadily with each extra contact in a campaign
(15 % at the 1st contact, 9 % at the 4th, 5 % at 8+). Contact count is used as a **proxy** for
contact fatigue; the dataset has no direct "fatigue" label.

---

## 7. What-if simulation

**What it is:** change any input on a lead and immediately see the new likelihood, the new
explanation and the new recommended action.

**How it works:**

```text
1. Start from a real lead (e.g. L43013: 88 %, Prioritise follow-up)
2. Change one or more fields (e.g. previous outcome → failure, contacts → 8)
3. The browser sends the edited lead to the server
4. The server validates it, re-runs the same model, SHAP and decision rules
5. The page shows Before → After (88 % → 12 %, Prioritise → Follow up — change approach)
6. Two tables explain the change:
   • "Fields you changed": the effect of each edit on the odds
   • "Linked effects (feature interactions)": fields you did NOT change whose
     effect changed anyway, because the model combines signals
```

**Speed:** about **27 ms** on the server (p50; 33 ms p95) for the full re-score, including
the 15-model likely range and the SHAP explanation. Over the internet the round trip is
~0.5 s, mostly network distance.

**Guard-rail:** impossible combinations are refused with a clear message. Example: "never
contacted before" together with a previous-campaign outcome of "failure" gives *"This
combination can't be scored. Inconsistent history…"*.

> **What-if is a simulation, not a promise.** It shows how the *model* responds to different
> inputs. It does not prove that changing that thing in real life would change the customer's
> decision (see section 25).

---

# PART B — DATA, MACHINE LEARNING AND EXPLAINABLE AI

## 8. The dataset

| | |
|---|---|
| **Name** | UCI Bank Marketing dataset (`bank-full` version), UCI Machine Learning Repository id 222 |
| **Source** | Moro, Cortez & Rita (2014), *Decision Support Systems* 62:22–31. Licence CC BY 4.0. |
| **What it is** | 45,211 records of direct-marketing **phone calls** by a Portuguese bank, May 2008 – Nov 2010 |
| **Target (what we predict)** | `y`: did the client subscribe to a term deposit? Yes for **11.7 %** |
| **Inputs used** | 15 fields in three groups (below) |

| Group | Fields |
|---|---|
| Customer profile | age, job, marital status, education, credit in default, account balance, housing loan, personal loan |
| Campaign context | contact channel, contact month, day of month |
| **Interaction history** | contacts this campaign (`campaign`), contacts before this campaign (`previous`), days since previous contact (`pdays`), previous campaign outcome (`poutcome`) |

**Why this dataset?** It is a well-known public benchmark of real sales-call outcomes, it
contains genuine **interaction-history** fields (the focus of this project), it is free and
licensed for reuse, and results can be compared with published work.

**Honest limits:** it is a research benchmark, not a modern CRM. It covers one bank in one
country over 2008–2010, has no conversation content (no transcripts, notes or sentiment),
and its success rate rises strongly over time (section 25).

---

## 9. Data preparation (preprocessing)

What actually happens before the model sees the data:

| Step | What it is | Why it is needed |
|---|---|---|
| **Validation** | Every record is checked: correct columns, allowed values, and two consistency rules ("never contacted" ⇔ "days since contact = −1"; never contacted ⇒ no previous outcome). | Garbage in, garbage out. Impossible records are refused, never silently scored. |
| **Target encoding** | `y` = yes/no → 1/0 | Models need numbers. |
| **Excluding `duration`** | The call-length field is removed. | Target leakage (section 10). |
| **Sentinel handling** | `pdays = −1` ("never contacted") is converted to "missing" rather than treated as "−1 days". | −1 is a code, not a number; treating it as a number would mislead the model. |
| **Categorical encoding** | Text categories (job, month, outcome…) become fixed numeric codes from a fixed vocabulary; unseen values become "missing" instead of crashing. | Tree models need numbers. This encoding also keeps SHAP explanations exact (section 13). |
| **"unknown" kept** | Values such as `job = unknown` are kept as their own category. | In this dataset "unknown" is information, not an error. |
| **No scaling for the final model** | Tree models don't need it. (The logistic-regression and neural-network comparisons use one-hot encoding, scaling and log transforms inside their own pipelines.) | Each model gets the preparation it needs, packaged with it. |
| **Train/test split** | 80 % for development (36,168 leads), 20 % held out (9,043 leads), stratified so both have 11.7 % subscribers, fixed random seed 42. | The held-out 20 % measures performance honestly; these are the leads the app displays. |
| **Cross-validation** | 5-fold, on the development set only, for model comparison and tuning. | Choosing a model on its average across five different splits is more reliable than one lucky split. |

**Training and live use stay consistent** because preprocessing and model are saved together
as one *pipeline*: the app runs exactly the same steps as training did.

---

## 10. Target leakage: why `duration` was removed

> **Target leakage** = the model is given information that would **not be available at the
> moment you need the prediction**, usually because it is a *consequence* of the outcome.
> The model looks brilliant in testing and is useless in reality.

**The `duration` field** is the length of the last phone call. It is only known **after the
call has ended**, and that last call is the very call in which the customer says yes or no. Long
calls are, largely, the sales themselves. UCI's own documentation says it "should be discarded
if the intention is to have a realistic predictive model"; the original authors (Moro et al.)
excluded it too.

**Measured effect (same gradient-boosting model, same test set):**

| Features | ROC-AUC |
|---|---|
| With `duration` (leaky) | 0.936 |
| **Without `duration` (deployed)** | **0.808** |

The original student version reported 0.92 because it used `duration`. Removing it gives a
lower but **honest** number, one that could hold up when deciding whether to make the next call.
This is an example of responsible ML design.

---

## 11. Model comparison: six approaches

Rather than picking a model blindly, six were trained and compared on exactly the same data
and test set (results from `reports/experiments/comparison.json` and `cnn.json`):

| Model | In plain words | ROC-AUC | PR-AUC | Notes |
|---|---|---|---|---|
| **Gradient Boosting (HistGB)** | Many small decision trees built one after another, each fixing earlier mistakes | **0.808** | **0.466** | **Selected.** Best accuracy and probability quality; 417 KB; exact explanations |
| Random Forest | Many decision trees built independently, then averaged | 0.802 | 0.461 | Nearly as good, but ~34× larger (14 MB) |
| Neural network (MLP) | Layers of connected "neurons" | 0.794 | 0.440 | Good, slightly worse, harder to explain |
| 1D CNN | Neural network with sliding filters (from the original project) | 0.792 | 0.429 | Average of 3 runs; built for images/sequences, not tables |
| Logistic Regression | One weighted formula, the most transparent | 0.772 | 0.411 | Clear baseline, ~0.04 behind |
| Decision Tree (depth 5) | A single readable flowchart of rules | 0.728 | 0.323 | Easy to read, least accurate |

**Why compare several?** To show that the choice is justified by evidence, to span the range
from "most transparent" to "most accurate", and to evaluate the original project's choices
(Random Forest and CNN) fairly.

**Metrics used, and why accuracy is not enough:** only 11.7 % of leads subscribe, so a model
that always says "no" is 88 % "accurate" and useless. The project therefore uses:

- **ROC-AUC:** how well the model ranks a subscriber above a non-subscriber (1.0 = perfect, 0.5 = coin flip);
- **PR-AUC:** ranking quality focused on the rare "yes" class (random = 0.117);
- **Brier score and calibration error:** whether the probabilities are honest;
- **F1, precision, recall:** at a chosen threshold (the comparison table's F1 uses a threshold chosen during cross-validation; the confusion matrix uses the 10 % follow-up line);
- **Gains and lift:** how many subscribers you reach by calling the top x % (the top 10 % of leads subscribe at 53 %, 4.5× the average).

### Why not just use the most complex model (e.g. a neural network)?

More complex is not automatically better. On medium-sized tables of business data, tree
ensembles usually match or beat neural networks, and here they did (0.808 vs 0.794 / 0.792).
Gradient boosting also gave better-calibrated probabilities, trains in ~4 s instead of ~12 s for
the CNN, and can be explained **exactly** with TreeSHAP. Simple, accurate and explainable beats
complicated.

**About the CNN:** a convolution looks for patterns among *neighbouring* inputs, which makes
sense for pixels in an image, where neighbours are related. In a table, the "neighbours" are
just the order in which columns happen to be listed. When the columns were randomly reshuffled,
the CNN's score did not change (0.792 vs 0.789), so it was not finding any real local structure.
It is kept in the comparison as an evaluated experiment, not used in the product.

---

## 12. The final model: gradient boosting

**How it works (simple version):** gradient boosting builds many small decision trees **one
after another**. Each new tree focuses on the mistakes the previous trees made. Together they
form a strong model from many weak ones.

| Detail | Value |
|---|---|
| Implementation | scikit-learn `HistGradientBoostingClassifier` |
| Settings | 300 trees, learning rate 0.03, up to 31 leaves per tree, at least 40 leads per leaf, L2 regularisation 1.0 (chosen by 5-fold cross-validation) |
| Trained on | 36,168 leads; evaluated once on 9,043 unseen leads |
| Class balance | trained **without** re-weighting, so probabilities stay honest (re-weighting gave the same ranking but probabilities ~3× too high: mean 37 % vs actual 11.7 %) |
| Determinism | fixed random seed; retraining gives identical results |

**Why it was selected:** it gave the best ROC-AUC, PR-AUC and Brier score in this evaluation,
well-calibrated probabilities without extra correction, a small and fast model file, and
exact SHAP explanations. It was selected **for this project based on the observed
evaluation and practical requirements**, not because it is "always the best model".

---

## 13. Explainable AI with SHAP (local and global)

### Why explainability matters

A salesperson should not be asked to accept "the model says 78 %". They need to know **why**.
Explanations build trust, help catch model mistakes, connect the numbers to business sense,
and keep humans responsible for decisions.

### What SHAP is (plain language)

> **SHAP** answers: *"Which factors pushed this particular prediction up, and which pushed it
> down, and by how much?"*

It starts from the average lead (8 % on the model's internal scale) and shows how each field
moved this lead away from that average. The contributions **add up exactly** to the
prediction: the training process checks this every time (largest error: 0.000000000000016).

The app shows each contribution as an **odds multiplier**: ×2.33 means "this factor more
than doubled the odds of subscribing"; ×0.53 means "roughly halved them".

**Method used:** *TreeSHAP*, an exact and fast SHAP algorithm for tree models (≈0.14 ms per
lead).

### Global vs local explanation

| Type | Question it answers | Where in the app |
|---|---|---|
| **Local** | Why did the model give **this lead** this likelihood? | Lead page → "Why this recommendation" (bars + "Working for / against") |
| **Global** | What generally influences the model **across all leads**? | Model page → "What drives the predictions"; Insights page → per-value effects |

**Example local explanation, lead L43013 (88 %):**
days since previous contact = 183 days → ×4.39; previous campaign outcome = success → ×2.33;
day of month, contact channel and month → smaller positive pushes; no strong negatives.

**Global top drivers** (average impact): contact channel, days since previous contact,
contact month, previous campaign outcome, housing loan.

**Interaction effects:** the model also combines signals. The strongest genuine combination
is *recency × previous outcome*: a recent previous contact helps much more after a past
success than after a past failure. The what-if simulator makes this visible in its "Linked
effects" table.

> **SHAP explains the model, not the world.** It shows what the model relies on. It does
> not prove that changing a factor would *cause* a customer to subscribe.

---

## 14. How confident is the system? (likely range and human review)

Besides the main model, **15 extra versions** of the model were trained on slightly different
random resamples of the training data. For each lead the app shows the middle range of their
estimates (the 25th–75th percentile, centred on the main estimate) as the **likely range**.

- If all versions agree which side of the follow-up line the lead is on → **Confidence: High**.
- If they disagree → **Human review recommended** (about 11 % of leads).

The review leads really are the borderline ones: they subscribe at ~10 %, right at the
break-even line, where either decision is defensible.

---

# PART C — TECHNOLOGY AND DEPLOYMENT

## 15. Architecture

```text
                    ┌──────────────── OFFLINE (once, ~30 s) ─────────────────┐
                    │ data/bank-full.csv → validate → split 80/20             │
                    │ → train gradient boosting + 15 bootstrap versions       │
                    │ → score + SHAP-explain the 9,043 held-out leads         │
                    │ → save artifacts/ (model, scored leads, model card)     │
                    └────────────────────────────┬───────────────────────────┘
                                                 │ loaded once at start-up
                                                 ▼
 Browser ──HTTPS──▶ Railway ──▶ FastAPI web application (one Python process)
   ▲                              │
   │                              ├─ page routes  → Jinja2 HTML + SVG charts
   │                              ├─ HTMX routes  → HTML fragments (what-if, filters, strategy)
   │                              └─ JSON API     → /api/v1/assess, /api/v1/leads/{id}
   │                              │
   │                              ▼
   │                        ScoringService
   │                          ├─ validate lead (Pydantic schema)
   │                          ├─ model.predict_proba          → likelihood
   │                          ├─ 15 bootstrap models          → likely range
   │                          ├─ TreeSHAP                     → explanation
   │                          ├─ interaction pattern label    → e.g. "Previously converted"
   │                          └─ decision rules (policy)      → action + reasons
   └──────────────────────── HTML / JSON response ◀───────────┘
```

**Two halves:** training happens **offline** with one command and produces saved files. The
web app only **loads** those files and never trains. That keeps the app fast and predictable.

---

## 16. Frontend, backend and API

### Frontend: what the user sees

| Piece | What it is |
|---|---|
| **Jinja2 templates** | HTML pages generated on the server ("server-rendered"); no separate frontend app to build |
| **HTMX** (one small JavaScript library, included in the project) | Lets parts of a page update without reloading: what-if results, lead filters, strategy sliders |
| **Hand-written CSS** | Layout, light/dark themes, responsive design for phones |
| **SVG charts** | Charts drawn on the server as vector graphics: ring chart, gains curve, bars, gauge, ROC |
| **A few lines of plain JavaScript** | Hover tooltips on charts and the theme switch |

### Backend: where the work happens

**FastAPI** (a Python web framework) receives every request, validates inputs with Pydantic,
calls the scoring service, and returns either an HTML page, an HTML fragment or JSON.

### How the frontend talks to the backend (actual flow, what-if example)

```text
User changes "Contacts this campaign" to 8
   ↓ HTMX sends the form:            POST /leads/L43013/whatif
FastAPI receives it
   ↓ Pydantic validates the lead     (invalid → friendly error box)
   ↓ ScoringService: model + 15 versions + SHAP + decision rules   (~27 ms)
   ↓ Jinja2 renders the "What-if result" HTML fragment
HTMX swaps the fragment into the page   → user sees Before → After
```

> The web pages do **not** call the JSON API. They use server-rendered HTML and HTMX. The
> JSON API exists **for other systems** (such as a CRM) to integrate with.

### The actual routes

| Route | Type | Purpose |
|---|---|---|
| `GET /` | page | Dashboard |
| `GET /leads` | page / fragment | Lead list (fragment when filtered via HTMX) |
| `GET /leads/{id}` | page | Lead page |
| `POST /leads/{id}/whatif` | fragment | What-if re-score |
| `GET /patterns` | page | Insights |
| `GET /model` | page | Model |
| `GET /policy`, `POST /policy` | page / fragment | Strategy; saving settings in a browser cookie |
| `GET /about` | page | About (`/method` redirects here) |
| `POST /api/v1/assess` | JSON API | Score any lead sent as JSON |
| `GET /api/v1/leads/{id}` | JSON API | Score and explain a portfolio lead |
| `GET /api/docs` | docs | Interactive API documentation |
| `GET /healthz` | health | `{"status":"ok","model_loaded":true}` for monitoring |

**An API**, in one line: a structured way for one program to ask another for something and
get a predictable answer. Here a CRM can send a lead's fields and receive the likelihood,
action, reasons and explanation as JSON.

**Why FastAPI?** It is a modern, fast Python framework, so the web layer and the ML code share
one language. It validates inputs automatically (Pydantic) and generates API documentation
for free.

---

## 17. Model serving and storage

| Question | Answer |
|---|---|
| Where does the trained model live? | `artifacts/model.joblib`: the full pipeline (preprocessing + gradient boosting) plus the 15 bootstrap models |
| What else is saved? | `artifacts/portfolio.csv` (the 9,043 scored and SHAP-explained leads) and `artifacts/model_card.json` (metrics, curves, global explanations, data fingerprint) |
| When is it loaded? | **Once**, when the server starts (~0.3–0.5 s). It is never reloaded per request. |
| How are portfolio pages so fast? | Their scores and explanations were pre-computed at training time; pages just look them up (~1 ms). |
| How are what-if / API leads scored? | Live, with the same saved pipeline, so preprocessing is identical to training. |
| Is there a database? | **No.** Everything fits in memory (9,043 rows). Strategy settings live in a browser cookie, the theme choice in browser local storage. Nothing a user does is stored on the server. |

---

## 18. Technology stack

| Technology | Simple meaning | Role here | Why chosen |
|---|---|---|---|
| Python 3.12 | Programming language | Everything | The standard language for ML |
| uv | Package manager | Installs exact, locked versions | Reproducible setup with one command |
| pandas 3.0, NumPy 2.5 | Data tables and maths | Data handling | Industry standard |
| scikit-learn 1.9 | ML library | All models, pipelines, metrics | Reliable, well documented, no special hardware |
| SHAP 0.52 | Explainability library | TreeSHAP explanations | The standard method for explaining tree models |
| Pydantic | Data validation | Checks every lead and setting | Clear errors instead of crashes |
| joblib | Saving Python objects | Stores the trained model | Standard for scikit-learn models |
| FastAPI 0.142 + Uvicorn | Web framework + server | Pages, fragments, JSON API | Fast, Python-native, automatic API docs |
| Jinja2 | HTML templates | Server-rendered pages | Simple; no frontend build step |
| HTMX 2.0 | Tiny JS library | Live updates without page reloads | Interactivity without a complex frontend framework |
| CSS + SVG | Styling and vector graphics | Look and feel, all charts | Fast, accessible, works without heavy chart libraries |
| loguru | Logging | Server logs | Readable logs |
| pytest, ruff, pyright | Testing, linting, type checking | Quality checks | Catch errors before users do |
| Docker | Container packaging | Bundles app + model into one image (1.11 GB) | Runs the same everywhere |
| Railway | Cloud hosting platform | Runs the public site | Simple deployment from the command line |
| TensorFlow | Deep-learning library | **Only** for the optional CNN experiment, not in the deployed app | Keeps the product light |

---

## 19. Deployment on Railway

**What Railway is:** a cloud platform that takes the project, builds it into a running server,
and gives it a public web address.

**What is deployed:** one service (`web`) in the Railway project `lead-decision-intelligence`,
built from the project's `Dockerfile`. The image contains the app, the trained model and the
experiment reports. It does not contain the raw dataset, tests or development tools, and it
runs as a non-administrator user.

**How a visitor reaches it:**

```text
Browser ──HTTPS──▶ Railway edge ──▶ container: uvicorn → FastAPI app (port 8000)
```

**Health check:** Railway calls `/healthz` before switching traffic to a new version
(configured in `railway.json`), and restarts the service if it fails.

**Access and safety:** the site is public with no password. To protect the hosting budget,
live scoring (what-if, API, strategy updates) is limited to 120 requests per minute per visitor.
An optional password gate exists in the code (`APP_PASSWORD`) but is switched off.

**Why Railway:** simple command-line deployment, Docker support, a public URL for
demonstrations. Being hosted on Railway does **not** by itself make the system
enterprise-ready (section 23).

**How it was deployed:** `railway init` → `railway add --service web` →
`railway variables --set "PORT=8000"` → `railway up` → `railway domain` (full procedure in
`docs/DEPLOYMENT.md`).

---

## 20. How to run Lead Intelligence locally

**In plain language:** "running locally" means the application runs on your own computer
instead of on Railway. A **developer** sets it up once; after that, anyone opens a browser
address on that computer.

### Option A: Python with uv (verified)

**Requirements:** a Mac, Linux or Windows computer; [uv](https://docs.astral.sh/uv/) installed;
internet for the first install. uv downloads Python 3.12 by itself.

```bash
# 1. Get the code (from your copy of the repository) and enter the folder
cd Marketing-Decision-Support-System
git switch redesign

# 2. Install the exact dependency versions (first time takes a few minutes)
uv sync

# 3. Start the app (the trained model is already included; no training needed)
uv run uvicorn leadintel.web.app:app --reload
```

**Open:** http://127.0.0.1:8000

**You should see:** the Dashboard ("Know which leads to call next, and why.") with the four
key-figure tiles. Health check: http://127.0.0.1:8000/healthz shows
`{"status":"ok","model_loaded":true}`.

**Optional:**

```bash
uv run pytest                 # run the 59 tests
uv run leadintel-train        # retrain the model (~30 s; results are identical)
uv run leadintel-experiments  # re-run all experiments (~4 min)
```

### Option B: Docker

**Requirements:** Docker Desktop.

```bash
docker build -t leadintel .
docker run --rm -p 8000:8000 leadintel
# open http://localhost:8000
```

(Add `-e APP_PASSWORD=... -e SESSION_SECRET=...` to `docker run` if you want a login page.)

### Railway vs local

| | Railway (hosted) | Local |
|---|---|---|
| Purpose | Public demonstration, easy sharing | Private use, development, offline demos |
| Internet | Required | Only for the first install |
| Setup | Already running | One-time installation by a developer |
| Where it runs | Railway's cloud | Your computer |
| Speed | ~0.5 s per page (network distance) | Milliseconds |
| Best for | Showing faculty or clients a link | A viva room with unreliable Wi-Fi; experimentation |

---

## 21. Testing, performance and accessibility

### Testing: 59 automated tests, all passing

| Test file | Tests | What it proves |
|---|---|---|
| `test_schema.py` | 9 | Valid leads are accepted; impossible histories and out-of-range values are rejected; `duration` cannot be used |
| `test_data.py` | 3 | The dataset is the expected one (45,211 rows, 11.7 % yes) and the validation rules work |
| `test_features.py` | 4 | Preprocessing works as designed (`duration` excluded, −1 → missing, unknown values don't crash) |
| `test_policy.py` | 12 | Each decision rule; break-even = cost ÷ value; priority never below break-even; the fast and simple versions of the rules agree on 3,000 random cases |
| `test_uncertainty.py` | 1 | The likely range always contains the estimate |
| `test_model_and_service.py` | 7 | The model loads; probabilities are valid and calibrated; **SHAP adds up exactly**; pre-computed and live scoring agree; end-to-end assessment works |
| `test_web.py` | 23 | Every page renders; filters, what-if, strategy, API, errors, 404s, redirects, password gate, rate limit; no "prototype/academic/faculty" wording on the site |

Several are **regression tests**: they lock in fixes for mistakes found during the project
(leakage, inflated probabilities, the SHAP pitfall, the priority-threshold edge case), so those
mistakes cannot quietly return.

### Performance (measured, with context)

| Measurement | Result | Conditions |
|---|---|---|
| Page response (server time) | 0.6–10.5 ms (p50) | Local benchmark, 30 requests per page, Apple-silicon Mac |
| What-if re-score | 26.6 ms p50, 32.6 ms p95 | Main model + 15 bootstrap models + SHAP, single lead |
| Server start-up | ~2.8 s to healthy | Includes loading the model |
| Memory | ~110 MB | Running server |
| Live site, per page | ~0.45–0.55 s | Measured from India; mostly network distance to the hosting region |

**Why it matters:** the what-if simulator feels instant, so users can explore freely. These
numbers come from `scripts/benchmark.py` (results in `reports/performance.json`).

### Accessibility

**Accessibility** means the application can be used by people with different abilities: for
example colour-blind users, keyboard-only users and screen-reader users.

What was actually done: automated checks with **axe** (an industry-standard accessibility
checker) against the WCAG 2.2 AA rules found **0 violations on all 7 live pages**. Chart
colours were checked for colour-blind safety, every colour has a text label, links are
underlined, text contrast was fixed, motion respects the "reduce motion" setting, and pages
fit phone screens. This is **automated testing, not a formal WCAG certification**.

---

# PART D — VALUE, NOVELTY AND HONESTY

## 22. Business impact

### Demonstrated by the project (on historical, held-out data)

- Contacting the recommended **33 %** of leads would have reached **74 %** of subscribers.
- "Prioritise" leads subscribed at **55 %** vs an 11.7 % average.
- About **6,000 calls** flagged as not worth making, at the cost of 276 of 1,058 subscriptions (default assumptions).
- Probabilities matched reality in every action group, so they can be used for planning.
- Every recommendation is explained and can be explored with what-if.

### Potential real-world impact (not proven by this project)

Less time spent on unlikely leads, less over-contacting of customers, more consistent
prioritisation across agents, and predictive analytics that sales staff can actually understand
and question. **No revenue increase or conversion improvement is claimed**. That would
require a live trial with real CRM data.

---

## 23. Technical and practical feasibility

**Technically feasible:** it uses established, open-source libraries (scikit-learn, SHAP,
FastAPI). It needs no GPU, no database and no special infrastructure. Scoring a lead takes
milliseconds on an ordinary server, it runs in a standard Docker container, and it is already
hosted publicly.

**Practically feasible:**

| Question | Answer |
|---|---|
| Who can use it? | Sales agents, relationship managers, campaign managers, analysts |
| What do they need? | A browser. The lead data must contain profile and contact-history fields. |
| Training needed? | Short: the action, the gauge and the "why" lists are self-explanatory; odds multipliers need one sentence of explanation |
| Integration | The JSON API lets a CRM send a lead and receive the score, action and explanation |

**What would have to change for a real bank deployment:** retraining on that bank's own,
recent CRM data; real cost and value figures; regular retraining and monitoring (the data
drifts over time); user accounts and access control; data-protection, consent and do-not-call
handling; fairness review; and model-risk approval under the bank's regulator. The prototype
makes **no** regulatory or compliance claim.

---

## 24. What is different: product, research, IP

### Product differentiation (all present in the application)

- **Prediction + explanation + action on one screen.** Not just a score.
- An **explicit, adjustable decision layer** (Strategy page) separate from the model.
- **Confidence-aware**: borderline leads are sent for human review instead of being guessed.
- **What-if simulator** with instant re-scoring and visible interaction effects.
- **Honest probabilities** (calibrated), so "55 %" really means about 55 %.
- A fast, accessible, mobile-friendly interface and an **integration API**.

### Research contribution (what the project investigated and showed)

1. **Prediction ≠ decision:** separating the model from a transparent, cost-based decision policy, and back-testing that policy on held-out data.
2. **Interaction-aware decision support:** interaction-history signals measurably improve prediction (PR-AUC 0.395 → 0.466, +18 %), and recency × previous outcome is the strongest interaction.
3. **Leakage-aware design:** identifying and removing `duration`, and quantifying its effect (0.936 → 0.808).
4. **Evidence-based model choice:** six models compared under the same protocol; the CNN tested with a column-shuffle experiment.
5. **Calibration over class weighting:** class re-weighting gave the same ranking but inflated probabilities ~3×, so imbalance is handled in the decision threshold instead.
6. **Uncertainty-triggered human review** using an ensemble of re-trained models.

### Potential IP direction

No patentability claim is made. *Any patentability assessment would require a separate
professional prior-art and claim analysis.*

---

## 25. Limitations and what the system does not do

### Limitations (say these before an examiner does)

| Limitation | Why it matters |
|---|---|
| **Benchmark data** | One Portuguese bank, 2008–2010, during the financial crisis. Not representative of every bank, country or today's market. |
| **Drift over time** | Success rates rise from ~3 % to ~32 % across the file. On the most recent 20 % of records, ROC-AUC falls to ~0.67. Real use would need regular retraining. |
| **Context fields** | Contact channel, month and day partly reflect *when* the data was collected, not customer behaviour, even though "contact channel" is the model's top driver. |
| **No conversation data** | No transcripts, notes or sentiment. "Interaction" means contact history only. |
| **Assumed economics** | €5 per call and €50 per subscription are illustrative defaults. "Heavily contacted" is a proxy for fatigue. |
| **Back-test, not field trial** | Results are measured on historical held-out leads, not on live campaigns. |
| **Explanations** | SHAP shows associations the model learned, not causes. |

### What Lead Intelligence does **not** do

- It does **not** guarantee that a customer will subscribe.
- It does **not** know the future or a customer's intentions.
- It does **not** replace the salesperson. It advises; people decide.
- It does **not** guarantee optimal decisions. The rules are only as good as the cost assumptions.
- Its explanations do **not** prove causation.
- The what-if simulator does **not** predict what would happen if you changed something in real life.
- It does **not** connect to a live CRM yet (only an API is provided), and it does **not** store user data.

---

## 26. How the project evolved, and future scope

### How the project evolved

The project began as a student implementation (preserved at git tag `legacy-v1`): a Streamlit
dashboard where a 1D CNN predicted subscription, with a reported AUC of 0.92. An audit found
that the 0.92 relied on the leaky `duration` field, that displayed probabilities were 2–3× too
high (class re-weighting), that explanations were fixed text, and that the CNN was not
justified. The redesign (`redesign` branch) rebuilt it: leakage removed, six models compared,
calibrated gradient boosting, exact SHAP explanations, a separate decision layer, uncertainty
and human review, a FastAPI application, tests, Docker and Railway deployment.

### Future scope

- Train on a real organisation's **recent CRM history** (every call, channel, timestamp, agent).
- Add conversation signals (call notes, transcripts, sentiment), each checked for leakage the way `duration` was.
- **Live CRM integration** through the existing API, with feedback on what agents actually did.
- Time-based validation, drift monitoring and scheduled recalibration.
- Real cost and value inputs, consent and do-not-call handling, user accounts.
- A fairness review across customer groups.
- A controlled field trial to measure real business impact.

---

# PART E — VIVA AND PRESENTATION KIT

## 27. Faculty viva: questions and short answers

### Basic project questions

**What is Lead Intelligence?**
A web application that estimates how likely each lead is to subscribe, explains why, and
recommends a next action. A human makes the final decision.

**Why did you choose this problem?**
Sales teams have more leads than time, and a bare prediction doesn't tell them whom to call or
why. I wanted to turn prediction into explained, actionable decisions.

**Who is the target user?**
Relationship managers and sales agents, plus campaign managers (Strategy page) and analysts
(Model and Insights pages).

**What is the input?**
A lead's profile (age, job, balance, loans…), campaign context (channel, month, day) and
interaction history (contacts this campaign, contacts before, days since last contact,
previous outcome): 15 fields.

**What is the output?**
A likelihood to subscribe, a likely range, a recommended action with reasons, a confidence
level, and a SHAP explanation.

### Dataset questions

**Which dataset did you use?**
The UCI Bank Marketing dataset (`bank-full`): 45,211 phone-campaign records from a
Portuguese bank, 2008–2010.

**Why UCI Bank Marketing?**
It is a public, well-known benchmark of real sales-call outcomes, and it contains genuine
interaction-history fields, which is what this project is about.

**What is the target variable?**
`y`: whether the client subscribed to a term deposit (yes for 11.7 %).

**What are the dataset's limitations?**
One bank, one country, 2008–2010, no conversation content, and strong drift over time.

**What is target leakage?**
Giving the model information that would not be available when the prediction is actually
needed, usually something caused by the outcome itself.

**Why was `duration` removed?**
It is the length of the last call, known only after the call that decided the outcome. With
it AUC is 0.94; without it 0.81. The honest number is 0.81.

### ML questions

**What models did you compare?**
Logistic regression, decision tree, random forest, gradient boosting, a neural network (MLP)
and a 1D CNN.

**Why compare multiple models?**
To choose on evidence, to span from transparent to accurate, and to evaluate the original
project's choices fairly.

**Which model was selected, and why?**
Gradient boosting: best ROC-AUC (0.808), PR-AUC (0.466) and Brier score; calibrated; small;
and exactly explainable with TreeSHAP.

**What is gradient boosting?**
Many small decision trees built one after another, each correcting the previous ones'
mistakes.

**What is supervised learning?**
Learning from examples where the answer is known (here, whether each past lead subscribed) in
order to predict the answer for new cases.

**What metrics did you use?**
ROC-AUC, PR-AUC, Brier score, calibration error, precision, recall, F1, and gains/lift.

**Why isn't accuracy enough?**
Only 11.7 % say yes, so a model that always says "no" is 88 % accurate and useless.

**Why not a neural network or CNN?**
They scored lower (0.794 / 0.792), are harder to explain and slower to train. The CNN found no
real structure: shuffling the column order didn't change its score.

### Explainability questions

**What is SHAP?**
A method that splits a prediction into each field's contribution: which factors pushed it up
or down, and by how much. The contributions add up exactly to the prediction.

**What is local explainability?**
Why the model gave *this* lead *this* likelihood: the bars on the lead page.

**What is global explainability?**
What influences the model across all leads: the importance chart on the Model page.

**Why is explainability important?**
Trust, catching mistakes, business understanding, and keeping humans responsible.

**Does SHAP prove causation?**
No. It explains what the model relies on, not what causes customers to subscribe.

### Decision-intelligence questions

**What is the difference between prediction and decision?**
Prediction: "how likely is a yes?" Decision: "given that, our costs and this lead's history,
what should we do?"

**Why have a decision layer?**
So business rules are explicit, readable and adjustable without retraining the model.

**How is lead prioritisation performed?**
Likelihood ≥ 35 % → Prioritise; ≥ the break-even line (cost ÷ value = 10 %) → Follow up
(or "change approach" after 4+ contacts); below it → Deprioritise, or Stop after 4+ contacts.
Uncertain leads → Human review.

**How are recommendations generated?**
By deterministic rules applied to the model's likelihood, its likely range and the contact
count. They are rules, not "AI decisions".

**What is what-if analysis?**
Changing a lead's inputs and instantly seeing the new likelihood, explanation and action. It is
a model simulation, not a guarantee.

**How do you handle uncertainty?**
Fifteen re-trained versions of the model give a likely range. If they disagree about which side
of the break-even line a lead falls on, the system recommends human review.

### Technical-architecture questions

**What is the frontend?**
Server-rendered HTML pages (Jinja2) with HTMX for live updates, hand-written CSS, and SVG
charts.

**What is the backend?**
A FastAPI (Python) application that validates inputs, runs the model, SHAP and decision rules,
and returns pages or JSON.

**Why FastAPI?**
It is fast and Python-native, so it lives alongside the ML code. It validates inputs
automatically and documents its API automatically.

**What is an API?**
A structured way for one program to request something from another and get a predictable
answer.

**How does the frontend communicate with the backend?**
Pages are requested normally. For what-if, filters and strategy, HTMX sends the form to the
server and swaps in the returned HTML. The JSON API is for external systems such as a CRM.

**How is the model served?**
It is trained offline, saved with joblib, and loaded once at start-up. Portfolio scores are
pre-computed; what-if and API leads are scored live with the same pipeline (~27 ms).

**Is there a database?**
No. All the data fits in memory, and settings are stored in the user's browser.

### Deployment questions

**How is the application deployed?**
As a Docker container on Railway, with a public HTTPS address and a `/healthz` health check.

**Why Railway?**
Simple command-line deployment with Docker support and a public URL for demonstrations.

**How can it run locally?**
`uv sync`, then `uv run uvicorn leadintel.web.app:app --reload`, then open
http://127.0.0.1:8000. Or use Docker.

**What are the deployment requirements?**
A container host with ~110 MB of RAM for the app and no GPU. The image is 1.11 GB.

### Business questions

**How can this help sales teams?**
By focusing calls where they are most likely to convert, avoiding over-contacting, and
explaining each recommendation.

**What is the practical impact?**
On held-out data, contacting the recommended 33 % of leads reached 74 % of subscribers.
Real-world impact would need a live trial.

**Can it guarantee better conversion?**
No. It improves where effort goes on historical data; real results depend on the organisation
and its data.

**Can a salesperson rely entirely on the system?**
No. It is decision support. Borderline cases are explicitly sent for human review.

### Research questions

**What is novel?**
Combining prediction, exact explanation, an explicit cost-based decision layer, uncertainty-
triggered human review and what-if exploration around interaction-history signals, with
leakage removed and the policy back-tested.

**What is the research contribution?**
Showing that interaction-history signals add measurable value (+18 % PR-AUC), that prediction
should be separated from decision, and that leakage-aware, calibrated, explainable models make
recommendations trustworthy.

**What did the experiments show?**
Gradient boosting was best (0.808). `duration` was leakage. Class re-weighting inflated
probabilities. The CNN found no real structure. Interaction history adds signal.

**What are the limitations?**
Old single-bank data, drift over time, no conversation data, assumed costs, back-test rather
than field trial.

**What would you improve next?**
Real recent CRM data, live integration, drift monitoring, real costs, and a field trial.

---

## 28. The 30-second explanation

> Sales teams have more leads than time, and a prediction alone doesn't tell them whom to call
> or why. **Lead Intelligence** estimates each lead's likelihood to subscribe using gradient
> boosting on the UCI Bank Marketing data, with the leaky call-duration field removed. It
> explains every estimate with SHAP and turns it into a recommended action, such as
> "Prioritise" or "Stop repeated follow-up", through transparent cost-based rules, with
> uncertain cases sent for human review. On unseen data, contacting the recommended third of
> leads reached 74 % of subscribers. It advises; people decide.

---

## 29. The 2-minute technical explanation

> The data is the UCI Bank Marketing dataset: 45,211 phone-campaign records with an 11.7 %
> subscription rate. I validate every record, remove `duration` because it is target leakage
> (known only after the deciding call), encode categoricals with a fixed vocabulary and treat
> `pdays = −1` as missing. The data is split 80/20, stratified, with seed 42.
>
> I compared six models with 5-fold cross-validation: logistic regression, decision tree,
> random forest, gradient boosting, an MLP and a 1D CNN. Scikit-learn's HistGradientBoosting won
> on ROC-AUC (0.808), PR-AUC (0.466) and Brier score. It was trained unweighted, which keeps
> it calibrated (0.9 % error); class re-weighting inflated probabilities about three-fold. Fifteen
> bootstrap re-fits give each lead a likely range.
>
> TreeSHAP gives exact local explanations, shown as odds multipliers that add up to the
> prediction, plus global importance. A separate decision layer applies explicit rules:
> break-even = cost ÷ value, a priority threshold, a contact-fatigue proxy, and human review
> when the bootstrap range crosses the break-even line.
>
> The FastAPI application loads the saved pipeline once at start-up and serves Jinja2 pages
> with HTMX for the what-if simulator and strategy sliders (~27 ms per live re-score), plus a
> JSON API for CRM integration. It is deployed as a Docker container on Railway, with 59
> automated tests and no accessibility violations found by axe.

---

## 30. For a business person / for a technical expert

### Explain it to a business person (no jargon)

> Your team can't call everyone, so who should they call first? This system looks at each
> customer's history with you (did they say yes last time? how many times have we called this
> month?) and tells you how likely they are to say yes. It suggests what to do: call now, try a
> different approach, or stop calling. It also shows the reasons in plain terms. You can change
> your assumptions, such as what a call costs, and see the plan update instantly. Your people
> stay in charge; the system helps them spend their time where it counts.

### Explain it to a technical expert (concise)

> Offline pipeline: validated UCI bank-full data, `duration` excluded as leakage, ordinal-encoded
> categoricals (which keeps TreeSHAP exact; native categorical splits gave wrong SHAP values),
> unweighted HistGradientBoosting selected from six candidates by 5-fold CV (test ROC-AUC 0.808,
> PR-AUC 0.466, ECE 0.009), and 15 bootstrap re-fits for an interquartile likely range. Serving:
> one FastAPI process that loads the joblib pipeline and pre-scored portfolio at start-up.
> Jinja2 + HTMX for the UI, a JSON API for integration, and single-thread scoring for ~27 ms
> live re-scores including SHAP. The decision layer is an explicit, deterministic policy
> (break-even, priority, fatigue proxy, uncertainty-triggered review), back-tested on the
> held-out set. Delivered as a Docker image on Railway, with 59 tests and axe-clean pages.

---

## 31. Live demo script (about 5 minutes)

Open **https://web-production-4cf5b6.up.railway.app** before you start. If the internet is
unreliable, run it locally (section 20) as a backup.

| # | Do | Say |
|---|---|---|
| 1 | — | "Sales teams have more leads than time, and a prediction alone doesn't say whom to call or why." |
| 2 | Show the **Dashboard** headline and workflow | "Lead Intelligence predicts, explains and recommends; a person decides." |
| 3 | Point to the **key-figure tiles** and the **gains curve** | "Contacting the recommended 33 % reaches 74 % of subscribers, against 33 % at random." |
| 4 | **Leads** → click the "Prioritise follow-up" chip | "Every lead is ranked and given an action." |
| 5 | Open **L43013** | "88 % likely. The range is narrow and confidence is high, so: Prioritise follow-up." |
| 6 | Scroll to **Why this recommendation** | "This is SHAP. The previous success and the recent contact push the odds up the most. This is a *local* explanation." |
| 7 | **What-if:** set Previous outcome = failure, Contacts = 8 | "Now 12 %, and the advice becomes 'change approach'. The linked-effects table shows the model combining signals." |
| 8 | **Strategy:** move the call cost to €15 | "If calls cost more, we contact 12 % instead of 33 %. The model didn't change; only the business rule did. That is prediction versus decision." |
| 9 | **Model** page | "Six models compared, gradient boosting best at 0.808. Here is the *global* explanation, and predicted vs actual on the diagonal." |
| 10 | (Optional) **Insights** | "Interaction-history signals add 18 % to model quality, and conversion falls with every extra call." |
| 11 | **About** page | "FastAPI backend, server-rendered pages, an API for CRMs, deployed on Railway with Docker." |
| 12 | — | "We removed call duration because it is only known after the call: target leakage. Honest 0.81 beats a misleading 0.94." |
| 13 | — | "Limitations: 2008–2010 data from one bank, no conversation data, assumed costs. Next step: real CRM data and a field trial." |
| 14 | — | "In short: prediction, explanation, decision, action, with a human in charge." |

---

## 32. 5-minute revision sheet

| Topic | One sentence |
|---|---|
| **Problem** | Sales teams have more leads than time, and a bare prediction doesn't say whom to call, why, or what to do next. |
| **Solution** | Lead Intelligence predicts each lead's likelihood, explains it and recommends a next action, with a human deciding. |
| **Dataset** | UCI Bank Marketing (`bank-full`): 45,211 phone-campaign records, Portuguese bank, 2008–2010. |
| **Target** | Did the client subscribe to a term deposit (yes 11.7 %)? |
| **Preprocessing** | Validate records, remove `duration`, encode categories, treat `pdays = −1` as missing, split 80/20 stratified. |
| **Leakage** | `duration` is known only after the deciding call, so it was removed (AUC 0.94 → honest 0.81). |
| **Models** | Six compared: logistic regression, decision tree, random forest, gradient boosting, MLP, 1D CNN. |
| **Final model** | Gradient boosting (scikit-learn HistGB): ROC-AUC 0.808, calibrated, small, exactly explainable. |
| **SHAP** | Splits each prediction into factors that push it up or down; local per lead, global across leads. |
| **Decision layer** | Transparent rules: break-even = cost ÷ value (10 %), priority ≥ 35 %, 4+ contacts = fatigue, uncertain → human review. |
| **Prioritisation** | Six actions; "Prioritise" leads subscribe at 55 % vs 4.5 % for "Stop". |
| **What-if** | Edit any field and see the new likelihood, explanation and action in ~27 ms; a simulation, not a guarantee. |
| **Tech stack** | Python, scikit-learn, SHAP, FastAPI, Jinja2 + HTMX, SVG charts, Docker, Railway. |
| **Deployment** | Docker container on Railway with health check, public URL and rate limiting; also runs locally with `uv`. |
| **Novelty** | Prediction + exact explanation + explicit decision layer + uncertainty review + what-if, built around interaction history. |
| **Impact** | On held-out data, calling the recommended 33 % of leads reached 74 % of subscribers. |
| **Limitation** | Old single-bank benchmark data with drift and no conversation content; results are a back-test, not a field trial. |

---

## 33. Glossary

| Term | Meaning |
|---|---|
| **Dataset** | The table of past records the model learns from. |
| **Feature** | One input column, e.g. age or previous campaign outcome. |
| **Target** | What the model predicts: here, whether the client subscribed. |
| **Prediction / probability** | The model's estimate of how likely the target is (e.g. 88 %). |
| **Classification** | Predicting which of a set of outcomes (yes/no) applies. |
| **Supervised learning** | Learning from past examples whose answers are known. |
| **Decision tree** | A flowchart of yes/no questions that ends in a prediction. |
| **Random forest** | Many decision trees built independently and averaged. |
| **Gradient boosting** | Many small trees built in sequence, each fixing earlier errors. |
| **Neural network (MLP)** | Layers of connected units that learn patterns; flexible but harder to explain. |
| **CNN** | A neural network with sliding filters, designed for images or sequences. |
| **Feature engineering / preprocessing** | Preparing raw data for the model: cleaning, encoding, handling codes like −1. |
| **Target leakage** | Using information at training time that wouldn't be available when predicting for real. |
| **Cross-validation** | Testing a model on several different splits of the data to get a reliable average score. |
| **ROC-AUC** | Chance the model ranks a random subscriber above a random non-subscriber (1 = perfect, 0.5 = random). |
| **PR-AUC** | Ranking quality focused on the rare positive class. |
| **Calibration** | Whether a "30 %" prediction really comes true about 30 % of the time. |
| **SHAP** | A method that splits a prediction into each feature's push up or down. |
| **Local explainability** | Explaining one specific prediction. |
| **Global explainability** | Explaining what drives the model across all predictions. |
| **Odds multiplier** | How much a factor multiplies the odds of "yes" (×2 = doubles them). |
| **Decision layer** | The business rules that turn a probability into a recommended action. |
| **Break-even line** | The probability at which a call pays for itself: cost ÷ value. |
| **Likely range** | The spread of estimates from 15 re-trained versions of the model. |
| **What-if analysis** | Changing inputs to see how the model's output responds. |
| **Frontend** | The part of the app users see in the browser. |
| **Backend** | The server-side part that does the computation. |
| **API** | A structured way for programs to request and exchange data. |
| **FastAPI** | The Python web framework used for the backend. |
| **HTMX** | A small library that updates parts of a page without reloading it. |
| **Model serving** | Making a trained model available to answer requests. |
| **Docker** | A tool that packages an app and everything it needs into one portable container. |
| **Deployment** | Putting the application on a server so others can use it. |
| **Railway** | The cloud platform hosting the public version of the app. |
| **Accessibility (WCAG)** | Designing so people with different abilities can use the app; WCAG is the standard. |

---

*Sources of truth for every number in this guide: `artifacts/model_card.json`,
`reports/experiments/*.json`, `reports/performance.json`, the test suite, and the live
application, all checked on 2026-10-08. Deeper detail: `docs/EXPERIMENTS.md`,
`docs/DECISION_POLICY.md`, `docs/EXPLAINABILITY.md`, `docs/LIMITATIONS.md`,
`docs/FACULTY_VIVA_GUIDE.md`.*
