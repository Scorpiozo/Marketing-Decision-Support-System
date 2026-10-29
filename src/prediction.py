import pandas as pd

from src.interaction_analysis import add_interaction_patterns, classify_interaction_pattern
from src.model import predict_proba
from src.preprocessing import transform_new_records


HOT_THRESHOLD = 0.70
WARM_THRESHOLD = 0.40

FATIGUE_LOW_MAX = 2
FATIGUE_MEDIUM_MAX = 4


def classify_lead_temperature(probability):
    if probability >= HOT_THRESHOLD:
        return "HOT"
    elif probability >= WARM_THRESHOLD:
        return "WARM"
    else:
        return "COLD"


def classify_contact_fatigue(campaign_count):
    campaign_count = int(campaign_count)
    if campaign_count <= FATIGUE_LOW_MAX:
        return "LOW"
    elif campaign_count <= FATIGUE_MEDIUM_MAX:
        return "MEDIUM"
    else:
        return "HIGH"



def recommend_action(lead_temperature, interaction_pattern, contact_fatigue="LOW"):
    if lead_temperature == "HOT":
        if contact_fatigue == "HIGH":
            return "High conversion potential, but avoid excessive contact."
        elif interaction_pattern == "Interested":
            return "Prioritize follow-up soon - strong positive signal."
        else:
            return "High conversion probability - prioritize follow-up soon."

    if lead_temperature == "WARM":
        if contact_fatigue == "HIGH":
            return "Moderate potential, but slow down - contact frequency is already high."
        elif interaction_pattern == "No Previous Interaction":
            return "Good potential - attempt a first meaningful contact."
        else:
            return "Follow up and monitor response over next campaign."

    if lead_temperature == "COLD":
        if contact_fatigue == "HIGH":
            return "Low priority and already over-contacted - stop further outreach for now."
        elif interaction_pattern == "Previously Unsuccessful":
            return "Low priority - previous attempt failed; contact later if at all."
        else:
            return "Low priority - consider contacting later or deprioritizing."

    return "Review manually."


def build_prediction_dataframe(raw_df, probabilities, id_offset=0):
    df = raw_df.reset_index(drop=True).copy()
    df["conversion_probability"] = pd.Series(probabilities).round(4).values

    df["lead_temperature"] = df["conversion_probability"].apply(classify_lead_temperature)
    df = add_interaction_patterns(df)
    df["contact_fatigue"] = df["campaign"].apply(classify_contact_fatigue)

    df["recommendation"] = df.apply(
        lambda r: recommend_action(
            r["lead_temperature"], r["interaction_pattern"], r["contact_fatigue"]
        ),
        axis=1,
    )

    df.insert(0, "customer_record_id", range(id_offset, id_offset + len(df)))

    required_display_cols = [
        "customer_record_id", "age", "job", "balance", "contact", "campaign",
        "previous", "poutcome", "conversion_probability", "lead_temperature",
        "interaction_pattern", "contact_fatigue", "recommendation",
    ]

    other_cols = [c for c in df.columns if c not in required_display_cols]
    df = df[required_display_cols + other_cols]

    df = df.sort_values("conversion_probability", ascending=False).reset_index(drop=True)
    return df


def get_top_priority_customers(prediction_df, top_n=50):
    return prediction_df.head(top_n).reset_index(drop=True)


def predict_single_customer(customer_dict, preprocessor, model):
    """Score one raw customer record with the saved preprocessor + model.

    ``model`` is any fitted estimator exposing ``predict_proba`` (the
    production Gradient Boosting or Random Forest model).
    """
    single_df = pd.DataFrame([customer_dict])
    X_proc = transform_new_records(single_df, preprocessor)
    prob = float(predict_proba(model, X_proc)[0])

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
