"""
analytics.py
------------
Historical, descriptive campaign analytics on the FULL UCI Bank Marketing
dataset (all 45,211 customers, actual observed `y` outcomes).

This is deliberately separate from `prediction.py`, which works with
MODEL-PREDICTED probabilities on the held-out test split. Keeping the two
apart lets the dashboard clearly distinguish:

    * Historical observed conversion  (this module - real outcomes, all data)
    * Model-predicted probability     (prediction.py - CNN output, test split)

All functions here compute simple counts and rates directly from the data -
no modeling, no inference - so every number shown in the "Executive
Overview", "Campaign Drivers", and "Customer Segments" dashboard pages can
be traced straight back to a groupby on the raw dataset.
"""

import numpy as np
import pandas as pd

from src.preprocessing import load_raw_data, clean_data

# Minimum group size before a conversion rate is treated as a reliable
# finding. Groups smaller than this are still shown (so nothing is hidden)
# but are flagged as low-confidence rather than presented as a reliable
# pattern - this directly avoids implying "the -100% conversion group of 2
# customers is a real trend."
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


def load_campaign_dataset(data_path="data/bank-full.csv"):
    """
    Load and lightly enrich the full historical dataset for descriptive
    analytics: cleans it (reusing src/preprocessing.py::clean_data, so
    there is only one implementation of "unknown"-handling), adds a binary
    `converted` column, and adds convenience bucket columns
    (`age_group`, `contacts_bucket`, `duration_bucket`) used across the
    Overview / Campaign Drivers / Customer Segments pages.
    """
    df = load_raw_data(data_path)
    df = clean_data(df)
    df = df.copy()

    df["converted"] = (df["y"].astype(str).str.lower() == "yes").astype(int)
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS)
    df["contacts_bucket"] = pd.cut(
        df["campaign"], bins=CAMPAIGN_BINS, labels=CAMPAIGN_LABELS
    )
    df["duration_bucket"] = pd.cut(
        df["duration"], bins=DURATION_BINS, labels=DURATION_LABELS
    )
    df["month"] = pd.Categorical(df["month"], categories=MONTH_ORDER, ordered=True)

    return df


def apply_filters(df, filters):
    """
    Apply a dict of {column: selected_values} filters to a dataframe.
    An empty list / None / "All" for a filter means "no restriction" on
    that column. Used to keep every KPI and chart on a page consistent
    with the same sidebar filter selections.

    `filters` keys expected (all optional): age_group, job, education,
    marital, housing, loan, month, poutcome.
    """
    out = df
    for col, selected in filters.items():
        if not selected:
            continue
        if col not in out.columns:
            continue
        out = out[out[col].isin(selected)]
    return out


def compute_overview_kpis(df):
    """
    Core, honestly-denominated executive KPIs for the CURRENT filtered
    dataframe. Every percentage here divides by len(df) (the filtered
    customer count), never by a mismatched denominator.
    """
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
    """
    Group the filtered dataframe by `column` and compute, per category:
    count of customers, count of conversions, conversion rate, and a
    `reliable` flag (True if the group has at least `min_n` customers).

    This is the single function behind every driver/segment chart in the
    dashboard, so every one of those charts is computed the same way.
    """
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
    """
    Return the single best-performing category of `column` by conversion
    rate, but ONLY among categories with at least `min_n` customers.
    Returns None if no category meets the minimum sample size, so the
    dashboard can honestly say "not enough data" instead of highlighting
    an unreliable small group.
    """
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
