"""
interaction_analysis.py
------------------------
Transparent, RULE-BASED "interaction pattern" analysis for the Bank
Marketing dataset.

IMPORTANT: These categories (Interested / Repeatedly Contacted /
No Previous Interaction / Previously Unsuccessful / Unclear) are NOT
official UCI labels. They are behavioural categories DEFINED FOR THIS
PROJECT, built from the following existing UCI fields:

    * campaign  - number of contacts performed during the CURRENT campaign
    * previous  - number of contacts performed BEFORE the current campaign
    * poutcome  - outcome of the PREVIOUS marketing campaign
                  ('success', 'failure', 'other', 'unknown')

The rules are intentionally simple and explainable (no black-box logic)
so that a marketing team - or a college evaluator - can immediately see
WHY a customer was placed into a given category.

Thresholds used (documented here and in the README):
    * "relatively low" current-campaign contact count  -> campaign <= 3
    * "high" current-campaign contact count             -> campaign >= 4
"""

import pandas as pd

# Threshold used to decide whether the current campaign's contact count is
# "low" or "high". Chosen based on the distribution of the `campaign`
# column (median ~2, most customers contacted 1-3 times).
LOW_CAMPAIGN_CONTACTS = 3


CATEGORY_DESCRIPTIONS = {
    "Interested": (
        "The customer responded positively to a PREVIOUS campaign "
        "(poutcome = success) and has not been over-contacted in the "
        "current campaign. Historically receptive customer."
    ),
    "Repeatedly Contacted": (
        "The current campaign has already contacted this customer many "
        "times (campaign >= {}) while the previous campaign outcome was "
        "not a success. Risk of contact fatigue.".format(LOW_CAMPAIGN_CONTACTS + 1)
    ),
    "No Previous Interaction": (
        "The customer was never contacted in any previous campaign "
        "(previous == 0). No historical signal is available."
    ),
    "Previously Unsuccessful": (
        "The previous campaign contact for this customer ended in failure "
        "(poutcome = failure)."
    ),
    "Unclear": (
        "None of the other rules apply clearly to this customer; the "
        "historical signal is mixed or ambiguous."
    ),
}


def classify_interaction_pattern(row):
    """
    Apply the rule-based interaction-pattern logic to a single customer
    record (a pandas Series or dict-like with 'campaign', 'previous',
    and 'poutcome' keys).

    Rule order matters: rules are checked top-to-bottom, and the FIRST
    matching rule wins. This keeps the logic simple and deterministic.
    """
    campaign = row.get("campaign", 0)
    previous = row.get("previous", 0)
    poutcome = str(row.get("poutcome", "unknown")).lower()

    # Rule 1: Interested
    #   Previous campaign was successful AND current contact count is low.
    if poutcome == "success" and campaign <= LOW_CAMPAIGN_CONTACTS:
        return "Interested"

    # Rule 2: Repeatedly Contacted
    #   Many contacts in the CURRENT campaign AND previous outcome was NOT
    #   a success (failure / other / unknown).
    if campaign > LOW_CAMPAIGN_CONTACTS and poutcome != "success":
        return "Repeatedly Contacted"

    # Rule 3: No Previous Interaction
    if previous == 0:
        return "No Previous Interaction"

    # Rule 4: Previously Unsuccessful
    if poutcome == "failure":
        return "Previously Unsuccessful"

    # Rule 5: Fallback
    return "Unclear"


def add_interaction_patterns(df):
    """
    Vectorised convenience wrapper: adds an 'interaction_pattern' column to
    a whole DataFrame by applying classify_interaction_pattern() row-wise.
    """
    df = df.copy()
    df["interaction_pattern"] = df.apply(classify_interaction_pattern, axis=1)
    return df


def interaction_pattern_summary(df):
    """Return counts and percentages for each interaction pattern category."""
    counts = df["interaction_pattern"].value_counts()
    percentages = (counts / len(df) * 100).round(2)
    summary = pd.DataFrame({"count": counts, "percentage": percentages})
    return summary


if __name__ == "__main__":
    # Small self-test with a handful of synthetic records
    sample = pd.DataFrame([
        {"campaign": 2, "previous": 1, "poutcome": "success"},   # Interested
        {"campaign": 5, "previous": 2, "poutcome": "failure"},   # Repeatedly Contacted
        {"campaign": 1, "previous": 0, "poutcome": "unknown"},   # No Previous Interaction
        {"campaign": 2, "previous": 1, "poutcome": "failure"},   # Previously Unsuccessful
        {"campaign": 2, "previous": 1, "poutcome": "other"},     # Unclear
    ])
    result = add_interaction_patterns(sample)
    print(result)
