import pandas as pd
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
    campaign = row.get("campaign", 0)
    previous = row.get("previous", 0)
    poutcome = str(row.get("poutcome", "unknown")).lower()

    if poutcome == "success" and campaign <= LOW_CAMPAIGN_CONTACTS:
        return "Interested"

    if campaign > LOW_CAMPAIGN_CONTACTS and poutcome != "success":
        return "Repeatedly Contacted"
    
    if previous == 0:
        return "No Previous Interaction"

    if poutcome == "failure":
        return "Previously Unsuccessful"

    return "Unclear"


def add_interaction_patterns(df):
    df = df.copy()
    df["interaction_pattern"] = df.apply(classify_interaction_pattern, axis=1)
    return df


def interaction_pattern_summary(df):
    counts = df["interaction_pattern"].value_counts()
    percentages = (counts / len(df) * 100).round(2)
    summary = pd.DataFrame({"count": counts, "percentage": percentages})
    return summary


if __name__ == "__main__":
    sample = pd.DataFrame([
        {"campaign": 2, "previous": 1, "poutcome": "success"},   # Interested
        {"campaign": 5, "previous": 2, "poutcome": "failure"},   # Repeatedly Contacted
        {"campaign": 1, "previous": 0, "poutcome": "unknown"},   # No Previous Interaction
        {"campaign": 2, "previous": 1, "poutcome": "failure"},   # Previously Unsuccessful
        {"campaign": 2, "previous": 1, "poutcome": "other"},     # Unclear
    ])
    result = add_interaction_patterns(sample)
    print(result)
