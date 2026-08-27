"""
prediction.py
-------------
Turns raw model probabilities into the full Marketing Decision Support
System output. This module clearly separates three different kinds of
information, which is worth explaining in a viva:

    1. MODEL OUTPUT (from the ML/CNN model)
       conversion_probability - "probability that the customer subscribes"

    2. DECISION LAYER (simple thresholds/rules applied to the model output
       and to existing UCI fields - all project-created, not UCI labels)
       lead_temperature   -> HOT / WARM / COLD           (from probability)
       interaction_pattern -> behavioural category         (from campaign history,
                               see interaction_analysis.py)
       contact_fatigue     -> LOW / MEDIUM / HIGH          (from `campaign` count)

    3. RECOMMENDATION (project-created, combines all of the above)
       recommendation -> suggested marketing action

Also builds the final ranked prediction dataframe and the
"Top Priority Customers" table used by the notebook and the Streamlit app.
"""

import pandas as pd

from src.interaction_analysis import add_interaction_patterns, classify_interaction_pattern

# ---------------------------------------------------------------------------
# Lead temperature thresholds (documented in the README)
# ---------------------------------------------------------------------------
HOT_THRESHOLD = 0.70
WARM_THRESHOLD = 0.40

# ---------------------------------------------------------------------------
# Contact Fatigue thresholds (documented in the README)
# ---------------------------------------------------------------------------
# Based on the `campaign` field: number of contacts made during the CURRENT
# campaign for this customer. This is a project-created derived indicator,
# NOT an official UCI label.
FATIGUE_LOW_MAX = 2      # campaign <= 2  -> LOW
FATIGUE_MEDIUM_MAX = 4   # 3 <= campaign <= 4 -> MEDIUM
                          # campaign >= 5 -> HIGH


def classify_lead_temperature(probability):
    """
    Map a conversion probability to a HOT / WARM / COLD label using the
    project's default thresholds:
        probability >= 0.70            -> HOT
        0.40 <= probability < 0.70     -> WARM
        probability < 0.40             -> COLD
    """
    if probability >= HOT_THRESHOLD:
        return "HOT"
    elif probability >= WARM_THRESHOLD:
        return "WARM"
    else:
        return "COLD"


def classify_contact_fatigue(campaign_count):
    """
    Map the number of contacts made during the CURRENT campaign
    (`campaign` field) to a simple Contact Fatigue indicator:
        campaign <= 2   -> LOW
        3 <= campaign <= 4 -> MEDIUM
        campaign >= 5   -> HIGH

    Contact Fatigue is a project-created derived feature meant to flag
    customers who are at risk of being over-contacted. It is NOT an
    official UCI label.
    """
    campaign_count = int(campaign_count)
    if campaign_count <= FATIGUE_LOW_MAX:
        return "LOW"
    elif campaign_count <= FATIGUE_MEDIUM_MAX:
        return "MEDIUM"
    else:
        return "HIGH"


# ---------------------------------------------------------------------------
# Marketing recommendation logic
# ---------------------------------------------------------------------------
def recommend_action(lead_temperature, interaction_pattern, contact_fatigue="LOW"):
    """
    Simple, transparent, explainable recommendation lookup based on the
    combination of (lead_temperature, interaction_pattern, contact_fatigue).

    The table intentionally keeps only a handful of rules so a human
    reader can verify the logic at a glance in a viva. Contact Fatigue is
    checked first within each lead-temperature bucket because "avoid
    over-contacting this customer" is usually the most important caveat
    to a recommendation, regardless of interaction pattern.
    """
    # HOT leads
    if lead_temperature == "HOT":
        if contact_fatigue == "HIGH":
            return "High conversion potential, but avoid excessive contact."
        elif interaction_pattern == "Interested":
            return "Prioritize follow-up soon - strong positive signal."
        else:
            return "High conversion probability - prioritize follow-up soon."

    # WARM leads
    if lead_temperature == "WARM":
        if contact_fatigue == "HIGH":
            return "Moderate potential, but slow down - contact frequency is already high."
        elif interaction_pattern == "No Previous Interaction":
            return "Good potential - attempt a first meaningful contact."
        else:
            return "Follow up and monitor response over next campaign."

    # COLD leads
    if lead_temperature == "COLD":
        if contact_fatigue == "HIGH":
            return "Low priority and already over-contacted - stop further outreach for now."
        elif interaction_pattern == "Previously Unsuccessful":
            return "Low priority - previous attempt failed; contact later if at all."
        else:
            return "Low priority - consider contacting later or deprioritizing."

    # Fallback (should not normally trigger)
    return "Review manually."


def build_prediction_dataframe(raw_df, probabilities, id_offset=0):
    """
    Build the final prediction dataframe described in the project spec.

    Parameters
    ----------
    raw_df : pd.DataFrame
        The ORIGINAL (untransformed) customer records - must include the
        columns needed for display and for the interaction-pattern rules
        (age, job, balance, contact, campaign, previous, poutcome).
    probabilities : array-like
        Model-predicted probability of subscription, aligned by row order
        with raw_df.
    id_offset : int
        Optional starting value for the generated customer_record_id column
        (useful when concatenating multiple batches).

    Returns
    -------
    pd.DataFrame sorted by conversion_probability (descending).
    """
    df = raw_df.reset_index(drop=True).copy()
    df["conversion_probability"] = pd.Series(probabilities).round(4).values

    # Decision layer: thresholds/rules applied to the model output and to
    # existing UCI fields. None of these three columns are official UCI
    # labels - see the module docstring above.
    df["lead_temperature"] = df["conversion_probability"].apply(classify_lead_temperature)
    df = add_interaction_patterns(df)
    df["contact_fatigue"] = df["campaign"].apply(classify_contact_fatigue)

    # Recommendation layer: combines all three decision-layer outputs.
    df["recommendation"] = df.apply(
        lambda r: recommend_action(
            r["lead_temperature"], r["interaction_pattern"], r["contact_fatigue"]
        ),
        axis=1,
    )

    # Customer Record ID: generated from the dataframe row position for this
    # project only. The UCI dataset does not provide a real customer ID.
    df.insert(0, "customer_record_id", range(id_offset, id_offset + len(df)))

    required_display_cols = [
        "customer_record_id", "age", "job", "balance", "contact", "campaign",
        "previous", "poutcome", "conversion_probability", "lead_temperature",
        "interaction_pattern", "contact_fatigue", "recommendation",
    ]
    # Keep required columns first, then any extra columns the caller passed in
    other_cols = [c for c in df.columns if c not in required_display_cols]
    df = df[required_display_cols + other_cols]

    df = df.sort_values("conversion_probability", ascending=False).reset_index(drop=True)
    return df


def get_top_priority_customers(prediction_df, top_n=50):
    """Return the top N customers by conversion probability."""
    return prediction_df.head(top_n).reset_index(drop=True)


def predict_single_customer(customer_dict, preprocessor, cnn_model, reshape_fn):
    """
    Run the full pipeline for ONE new customer record supplied as a dict
    (e.g. from the Streamlit dashboard's manual-entry form).

    Returns a dict with probability, lead_temperature, interaction_pattern,
    contact_fatigue, and recommendation.
    """
    single_df = pd.DataFrame([customer_dict])
    X_proc = preprocessor.transform(single_df)
    X_cnn = reshape_fn(X_proc)
    prob = float(cnn_model.predict(X_cnn, verbose=0).ravel()[0])

    lead_temp = classify_lead_temperature(prob)
    pattern = classify_interaction_pattern(customer_dict)
    fatigue = classify_contact_fatigue(customer_dict.get("campaign", 0))
    action = recommend_action(lead_temp, pattern, fatigue)

    return {
        "conversion_probability": round(prob, 4),
        "lead_temperature": lead_temp,
        "interaction_pattern": pattern,
        "contact_fatigue": fatigue,
        "recommendation": action,
    }
