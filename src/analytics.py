import pandas as pd

from src.config import DATA_PATH
from src.preprocessing import load_raw_data, clean_data

MIN_RELIABLE_SAMPLE_SIZE = 30

AGE_BINS = [17, 25, 35, 45, 55, 65, 120]
AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]

CAMPAIGN_BINS = [0, 1, 2, 3, 4, 1000]
CAMPAIGN_LABELS = ["1", "2", "3", "4", "5+"]

DURATION_BINS = [-1, 100, 300, 600, 100000]
DURATION_LABELS = ["<100s", "100-300s", "300-600s", "600s+"]

MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun",
               "jul", "aug", "sep", "oct", "nov", "dec"]

SEGMENT_COLUMNS = {
    "Age Group": "age_group",
    "Job": "job",
    "Education": "education",
    "Marital Status": "marital",
    "Housing Loan": "housing",
    "Personal Loan": "loan",
    "Credit Default": "default",
}


def add_age_group(age_series):
    """Bin ages into the dashboard's age groups (shared by all pages)."""
    return pd.cut(age_series, bins=AGE_BINS, labels=AGE_LABELS)


def load_campaign_dataset(data_path=DATA_PATH):
    df = clean_data(load_raw_data(data_path))  # clean_data already returns a copy

    df["converted"] = (df["y"].astype(str).str.lower() == "yes").astype(int)
    df["age_group"] = add_age_group(df["age"])
    df["contacts_bucket"] = pd.cut(
        df["campaign"], bins=CAMPAIGN_BINS, labels=CAMPAIGN_LABELS
    )
    df["duration_bucket"] = pd.cut(
        df["duration"], bins=DURATION_BINS, labels=DURATION_LABELS
    )
    df["month"] = pd.Categorical(df["month"], categories=MONTH_ORDER, ordered=True)

    return df


def apply_filters(df, filters):
    out = df
    for col, selected in filters.items():
        if not selected:
            continue
        if col not in out.columns:
            continue
        out = out[out[col].isin(selected)]
    return out


def compute_overview_kpis(df):
    n = len(df)
    if n == 0:
        return {
            "customers_contacted": 0, "subscriptions": 0,
            "conversion_rate": None, "avg_contacts": None,
        }
    subscriptions = int(df["converted"].sum())
    return {
        "customers_contacted": n,
        "subscriptions": subscriptions,
        "conversion_rate": subscriptions / n,
        "avg_contacts": float(df["campaign"].mean()),
    }


def conversion_rate_by(df, column, min_n=MIN_RELIABLE_SAMPLE_SIZE, order=None):
    if df.empty:
        return pd.DataFrame(columns=[column, "count", "conversions", "conversion_rate", "reliable"])

    grouped = df.groupby(column, observed=True)["converted"].agg(["count", "sum"]).reset_index()
    grouped = grouped.rename(columns={"sum": "conversions"})
    grouped["conversion_rate"] = grouped["conversions"] / grouped["count"]
    grouped["reliable"] = grouped["count"] >= min_n

    if order is not None:
        grouped[column] = pd.Categorical(grouped[column], categories=order, ordered=True)
        grouped = grouped.sort_values(column)
    else:
        grouped = grouped.sort_values("conversion_rate", ascending=False)

    return grouped.reset_index(drop=True)


def best_performing_segment(df, column="job", min_n=MIN_RELIABLE_SAMPLE_SIZE):
    rates = conversion_rate_by(df, column, min_n=min_n)
    reliable = rates[rates["reliable"]]
    if reliable.empty:
        return None
    best = reliable.sort_values("conversion_rate", ascending=False).iloc[0]
    return {
        "category": best[column],
        "conversion_rate": float(best["conversion_rate"]),
        "count": int(best["count"]),
    }
