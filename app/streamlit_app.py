import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_preprocessor, get_categorical_options
from src.model import load_cnn_model, reshape_for_cnn
from src.prediction import predict_single_customer
from src.analytics import (
    load_campaign_dataset, apply_filters, compute_overview_kpis,
    conversion_rate_by, best_performing_segment,
    AGE_LABELS, CAMPAIGN_LABELS, DURATION_LABELS, MONTH_ORDER, SEGMENT_COLUMNS,
    MIN_RELIABLE_SAMPLE_SIZE,
)


PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, "outputs", "final_predictions.csv")
TOP_PRIORITY_PATH = os.path.join(PROJECT_ROOT, "outputs", "top_priority_customers.csv")
MODEL_COMPARISON_PATH = os.path.join(PROJECT_ROOT, "outputs", "model_comparison.csv")

#
# Colour palette
#
NAVY = "#1B2631"
NAVY_DARK = "#102746"

TEAL = "#087A75"
TEAL_LIGHT = "#EAF5F3"

CREAM = "#D7C9AA"
CREAM_LIGHT = "#F5F6F8"

TEXT_DARK = "#172033"
TEXT_BODY = "#202020"
TEXT_MUTED = "#4A4A4A"

WHITE = "#FFFFFF"
BORDER = "#DDE2E8"

WARM = "#B8934B"
BRICK = "#7B2D26"


#
# Global styling
#
st.markdown(
    f"""
    <style>

    /* GLOBAL */

    .stApp {{
        background: {CREAM_LIGHT};
        color: {TEXT_BODY};
    }}

    .main {{
        background: {CREAM_LIGHT};
    }}

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }}

    /* Remove Streamlit's excessive top spacing */
    header[data-testid="stHeader"] {{
        background: transparent;
    }}

    /* Normal text */
    .stApp p {{
        color: {TEXT_BODY};
    }}

    /* Headings */
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6 {{
        color: {TEXT_DARK};
    }}

    .stApp h1 {{
        font-weight: 750;
        letter-spacing: -0.025em;
    }}

    .stApp h2,
    .stApp h3 {{
        font-weight: 700;
        letter-spacing: -0.015em;
    }}


    /* SIDEBAR */

    section[data-testid="stSidebar"] {{
        background: {NAVY};
        border-right: 1px solid rgba(255,255,255,0.08);
    }}

    section[data-testid="stSidebar"] > div {{
        background: {NAVY};
    }}

    /* Sidebar overall spacing */
    section[data-testid="stSidebar"] .block-container {{
        padding-top: 1.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }}

    /* Sidebar headings */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {{
        color: {WHITE} !important;
    }}

    /* Sidebar text */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {{
        color: #E8EDF4 !important;
    }}

    /* Sidebar captions */
    section[data-testid="stSidebar"]
    [data-testid="stCaptionContainer"] p {{
        color: #AEBBCB !important;
        font-size: 0.76rem !important;
        line-height: 1.45 !important;
    }}


    /* SIDEBAR NAVIGATION */

    section[data-testid="stSidebar"] [role="radiogroup"] {{
        gap: 0.15rem;
    }}

    section[data-testid="stSidebar"] [role="radio"] {{
        min-height: 38px;
        padding: 0.45rem 0.7rem;
        border-radius: 7px;
        transition: background 0.15s ease;
    }}

    section[data-testid="stSidebar"] [role="radio"]:hover {{
        background: rgba(255,255,255,0.07);
    }}

    /* Selected navigation item */
    section[data-testid="stSidebar"]
    [role="radio"][aria-checked="true"] {{
        background: rgba(255,255,255,0.13);
    }}

    section[data-testid="stSidebar"]
    [role="radio"][aria-checked="true"] p {{
        color: {WHITE} !important;
        font-weight: 700 !important;
    }}

    /* Radio text */
    section[data-testid="stSidebar"] [role="radio"] p {{
        color: #DCE4EE !important;
        font-size: 0.86rem !important;
        margin: 0 !important;
    }}

    /* Radio circle */
    section[data-testid="stSidebar"] [role="radio"] div:first-child {{
        border-color: #8FA4BC !important;
    }}

    section[data-testid="stSidebar"]
    [role="radio"][aria-checked="true"] div:first-child {{
        border-color: {TEAL} !important;
        background: {TEAL} !important;
    }}


    /* SIDEBAR DIVIDERS */

    section[data-testid="stSidebar"] hr {{
        border: none;
        border-top: 1px solid rgba(255,255,255,0.12);
        margin: 1rem 0;
    }}


    /*
       SIDEBAR FILTER LABELS
    */

    section[data-testid="stSidebar"]
    [data-testid="stWidgetLabel"] {{
        margin-bottom: 0.2rem !important;
    }}

    section[data-testid="stSidebar"]
    [data-testid="stWidgetLabel"] p {{
        color: #E8EDF4 !important;
        font-size: 0.76rem !important;
        font-weight: 650 !important;
        margin-bottom: 0.15rem !important;
    }}


    /*
       SIDEBAR SELECTS
    */

    section[data-testid="stSidebar"]
    [data-baseweb="select"] > div {{
        background: {WHITE} !important;
        border: 1px solid #D6DDE6 !important;
        border-radius: 6px !important;
        min-height: 38px !important;
        box-shadow: none !important;
    }}

    section[data-testid="stSidebar"]
    [data-baseweb="select"] > div:hover {{
        border-color: #AAB8C8 !important;
    }}

    section[data-testid="stSidebar"]
    [data-baseweb="select"] > div:focus-within {{
        border-color: {TEAL} !important;
        box-shadow: 0 0 0 1px {TEAL} !important;
    }}

    section[data-testid="stSidebar"]
    [data-baseweb="select"] span {{
        color: {TEXT_BODY} !important;
        font-size: 0.81rem !important;
    }}

    section[data-testid="stSidebar"]
    [data-baseweb="select"] input {{
        color: {TEXT_BODY} !important;
    }}

    section[data-testid="stSidebar"]
    [data-baseweb="select"] svg {{
        color: {NAVY} !important;
    }}


    /*
       MULTISELECT TAGS
    */

    section[data-testid="stSidebar"]
    [data-baseweb="tag"] {{
        background: {TEAL_LIGHT} !important;
        border-radius: 4px !important;
    }}

    section[data-testid="stSidebar"]
    [data-baseweb="tag"] span {{
        color: {NAVY} !important;
        font-size: 0.75rem !important;
    }}


    /*
       RESET BUTTON
    */

    section[data-testid="stSidebar"] .stButton > button {{
        width: 100%;
        min-height: 36px;
        margin-top: 0.45rem;

        background: transparent !important;
        color: #DCE5F0 !important;

        border: 1px solid rgba(255,255,255,0.25) !important;
        border-radius: 6px !important;

        font-size: 0.8rem;
        font-weight: 600;
    }}

    section[data-testid="stSidebar"] .stButton > button:hover {{
        background: rgba(255,255,255,0.08) !important;
        border-color: rgba(255,255,255,0.4) !important;
    }}


    /*
       MAIN HEADER
    */

    .dss-header {{
        position: relative;

        background: {WHITE};

        padding: 1.35rem 1.6rem 1.25rem 1.6rem;

        border: 1px solid {BORDER};
        border-radius: 10px;

        margin-bottom: 1.35rem;

        box-shadow: 0 1px 3px rgba(16,39,70,0.035);

        overflow: hidden;
    }}

    /* Small teal accent at top */
    .dss-header::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;

        height: 3px;
        background: {TEAL};
    }}

    .dss-header h1 {{
        color: {TEXT_DARK} !important;

        margin: 0 0 0.3rem 0;

        font-size: 1.72rem;
        line-height: 1.2;

        font-weight: 750;
        letter-spacing: -0.025em;
    }}

    .dss-header p {{
        color: {TEXT_MUTED} !important;

        margin: 0;

        font-size: 0.88rem;
        line-height: 1.45;
    }}


    /*
       PAGE SUBHEADINGS
    */

    .stApp h3 {{
        margin-top: 0.5rem;
    }}

    .stApp [data-testid="stCaptionContainer"] p {{
        color: {TEXT_MUTED};
        font-size: 0.82rem;
        line-height: 1.5;
    }}


    /*
       METRIC CARDS
    */

    .metric-card {{
        background: {WHITE};

        border: 1px solid {BORDER};
        border-left: 4px solid {TEAL};

        border-radius: 8px;

        padding: 0.95rem 1rem;

        min-height: 80px;

        box-shadow: 0 1px 3px rgba(16,39,70,0.035);
    }}

    .metric-card.hot {{
        border-left-color: {BRICK};
    }}

    .metric-card.warm {{
        border-left-color: {WARM};
    }}

    .metric-card.cold {{
        border-left-color: {NAVY};
    }}

    .metric-card.fatigue {{
        border-left-color: {BRICK};
    }}

    .metric-card .metric-value {{
        font-size: 1.65rem;
        line-height: 1.15;

        font-weight: 750;

        color: {NAVY} !important;
    }}

    .metric-card .metric-label {{
        margin-top: 0.3rem;

        font-size: 0.7rem;
        line-height: 1.25;

        color: {TEXT_MUTED} !important;

        text-transform: uppercase;
        letter-spacing: 0.045em;

        font-weight: 650;
    }}


    /*
       INFO CARDS
    */

    .info-card {{
        background: {WHITE};

        border: 1px solid {BORDER};
        border-radius: 9px;

        padding: 1.15rem 1.25rem;

        box-shadow: 0 1px 3px rgba(16,39,70,0.035);
    }}

    .info-card .metric-label {{
        color: {TEXT_MUTED} !important;
    }}

    .info-card .metric-value {{
        color: {NAVY} !important;
    }}
    

    /*
       BADGES
    */

    .badge {{
        display: inline-block;

        padding: 0.25rem 0.7rem;

        border-radius: 5px;

        font-weight: 700;
        font-size: 0.78rem;

        letter-spacing: 0.02em;

        color: {WHITE} !important;
    }}

    .badge-hot {{
        background: {BRICK};
    }}

    .badge-warm {{
        background: {WARM};
    }}

    .badge-cold {{
        background: {NAVY};
    }}

    .badge-fatigue-low {{
        background: {TEAL};
    }}

    .badge-fatigue-medium {{
        background: {WARM};
    }}

    .badge-fatigue-high {{
        background: {BRICK};
    }}


    /*
       RECOMMENDATION BOX
    */

    .recommendation-box {{
        background: #F2EFE7;

        border: 1px solid #E4DDCD;
        border-left: 4px solid {TEAL};

        padding: 0.9rem 1rem;

        border-radius: 7px;

        font-size: 0.92rem;
        line-height: 1.5;

        color: {NAVY} !important;

        font-weight: 600;
    }}

    .recommendation-box * {{
        color: {NAVY} !important;
    }}


    /*
       STREAMLIT ALERTS
    */

    div[data-testid="stAlert"] {{
        border-radius: 7px;
        border-width: 1px;
    }}


    /*
       DATAFRAME
    */

    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 8px;
        overflow: hidden;
    }}


    /*
       STANDARD BUTTONS
    */

    .stButton > button {{
        background: {NAVY};
        color: {WHITE} !important;

        border: 1px solid {NAVY};
        border-radius: 6px;

        font-weight: 600;
    }}

    .stButton > button:hover {{
        background: {TEAL};
        border-color: {TEAL};
        color: {WHITE} !important;
    }}


    /*
       DOWNLOAD BUTTON
    */

    .stDownloadButton > button {{
        background: {TEAL};
        color: {WHITE} !important;

        border: 1px solid {TEAL};
        border-radius: 6px;

        font-weight: 600;
    }}

    .stDownloadButton > button:hover {{
        background: {NAVY};
        border-color: {NAVY};
    }}


    /*
       FORM SUBMIT BUTTON
    */

    button[kind="primary"] {{
        background: {TEAL} !important;
        color: {WHITE} !important;
        border-color: {TEAL} !important;
        border-radius: 6px !important;
        font-weight: 650 !important;
    }}

    button[kind="primary"]:hover {{
        background: {NAVY} !important;
        border-color: {NAVY} !important;
    }}


    /*
       EXPANDERS
    */

    div[data-testid="stExpander"] {{
        background: {WHITE};

        border: 1px solid {BORDER};
        border-radius: 8px;

        box-shadow: none;
    }}

    div[data-testid="stExpander"] summary {{
        font-weight: 650;
        color: {TEXT_DARK} !important;
    }}


    /*
       INPUTS IN MAIN CONTENT
    */

    .stTextInput input,
    .stNumberInput input,
    .stDateInput input {{
        border-radius: 6px !important;
        border: 1px solid {BORDER} !important;
    }}

    .stTextInput input:focus,
    .stNumberInput input:focus,
    .stDateInput input:focus {{
        border-color: {TEAL} !important;
        box-shadow: 0 0 0 1px {TEAL} !important;
    }}


    /*
       MAIN SELECTBOXES
    */

    [data-baseweb="select"] > div {{
        border-radius: 6px !important;
    }}


    /*
       DIVIDERS
    */

    hr {{
        border: none;
        border-top: 1px solid {BORDER};
        margin: 1.25rem 0;
    }}


    /*
       PLOTLY CONTAINERS
    */

    .js-plotly-plot {{
        border-radius: 8px;
    }}


    /*
       MOBILE / SMALL SCREENS
    */

    @media (max-width: 900px) {{
        .block-container {{
            padding-left: 1rem;
            padding-right: 1rem;
        }}

        .dss-header {{
            padding: 1.15rem 1.2rem;
        }}

        .dss-header h1 {{
            font-size: 1.45rem;
        }}
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_predictions():
    if not os.path.exists(PREDICTIONS_PATH):
        return None
    df = pd.read_csv(PREDICTIONS_PATH)
    df["age_group"] = pd.cut(df["age"], bins=[17, 25, 35, 45, 55, 65, 120], labels=AGE_LABELS)
    return df


@st.cache_data
def load_campaign_data():
    """Full historical dataset (all 45,211 customers, actual outcomes)."""
    return load_campaign_dataset(os.path.join(PROJECT_ROOT, "data", "bank-full.csv"))


@st.cache_data
def load_top_priority():
    if os.path.exists(TOP_PRIORITY_PATH):
        return pd.read_csv(TOP_PRIORITY_PATH)
    return None


@st.cache_data
def load_model_comparison():
    if os.path.exists(MODEL_COMPARISON_PATH):
        return pd.read_csv(MODEL_COMPARISON_PATH)
    return None


@st.cache_resource
def load_scoring_artifacts():
    """
    Load the SAME saved preprocessor and SAME saved CNN used during
    training (see models/). Cached with st.cache_resource so these are
    loaded from disk once per app session - the models are never retrained
    here, only loaded and used for inference.
    """
    preprocessor = load_preprocessor(os.path.join(PROJECT_ROOT, "models", "preprocessor.joblib"))
    cnn_model = load_cnn_model(os.path.join(PROJECT_ROOT, "models", "cnn_model.keras"))
    categorical_options = get_categorical_options(preprocessor)
    return preprocessor, cnn_model, categorical_options


def lead_badge_html(temp):
    cls = {"HOT": "badge-hot", "WARM": "badge-warm", "COLD": "badge-cold"}.get(temp, "badge-cold")
    return f'<span class="badge {cls}">{temp}</span>'


def fatigue_badge_html(fatigue):
    cls = {
        "LOW": "badge-fatigue-low",
        "MEDIUM": "badge-fatigue-medium",
        "HIGH": "badge-fatigue-high",
    }.get(fatigue, "badge-fatigue-low")
    return f'<span class="badge {cls}">{fatigue}</span>'


def metric_card(label, value, extra_class=""):
    st.markdown(
        f"""
        <div class="metric-card {extra_class}">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


PLOTLY_FONT = dict(
    family="Segoe UI, Helvetica, Arial, sans-serif",
    color="#172033",
    size=13,
)


def rate_bar_chart(rates_df, category_col, title, x_title=None, min_n=MIN_RELIABLE_SAMPLE_SIZE):
    """
    Build a horizontal-friendly conversion-rate bar chart from a
    conversion_rate_by()-style dataframe (category_col, count, conversions,
    conversion_rate, reliable). Bars for groups below the minimum reliable
    sample size are shown at reduced opacity and flagged in the tooltip,
    so small groups are visible but not visually presented as equally
    trustworthy as large ones.
    """
    df = rates_df.copy()
    if df.empty:
        return None

    df["opacity"] = df["reliable"].map(lambda r: 1.0 if r else 0.35)
    df["reliability_label"] = df["reliable"].map(
        lambda r: "Reliable (n \u2265 {})".format(min_n) if r else "Small sample - interpret with caution"
    )
    df["rate_pct"] = (df["conversion_rate"] * 100).round(1)

    fig = go.Figure()
    fig.add_bar(
        x=df[category_col].astype(str),
        y=df["conversion_rate"],
        marker=dict(color=TEAL, opacity=df["opacity"]),
        customdata=df[["count", "rate_pct", "reliability_label"]],
        hovertemplate=(
            "<b>%{x}</b><br>Conversion rate: %{customdata[1]}%"
            "<br>Customers: %{customdata[0]}<br>%{customdata[2]}<extra></extra>"
        ),
    )
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                family="Segoe UI, Helvetica, Arial, sans-serif",
                size=17,
                color="#172033",
            ),
        ),

        font=PLOTLY_FONT,

        xaxis=dict(
            title=dict(
                text=x_title or category_col,
                font=dict(size=13, color="#202020"),
            ),
            tickfont=dict(
                size=12,
                color="#202020",
            ),
            linecolor="#C9CDD3",
            tickcolor="#C9CDD3",
            gridcolor="rgba(0,0,0,0)",
        ),

        yaxis=dict(
            title=dict(
                text="Conversion rate",
                font=dict(size=13, color="#202020"),
            ),
            tickfont=dict(
                size=12,
                color="#202020",
            ),
            tickformat=".0%",
            linecolor="#C9CDD3",
            tickcolor="#C9CDD3",
            gridcolor="#E2E5E9",
            zerolinecolor="#C9CDD3",
        ),

        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",

        margin=dict(
            t=65,
            l=15,
            r=15,
            b=20,
        ),

        height=380,
    )

    return fig


DISPLAY_COLS = [
    "customer_record_id", "age", "job", "balance", "contact", "campaign",
    "previous", "poutcome", "conversion_probability", "lead_temperature",
    "interaction_pattern", "contact_fatigue", "recommendation",
]


def render_header():
    st.markdown(
        """
        <div class="dss-header">
            <h1>Marketing Decision Support System</h1>
            <p>Bank Campaign Lead and Interaction Analysis &middot; Prototype built on the UCI Bank Marketing dataset</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_filter_sidebar(campaign_df):
    """
    Sidebar filter panel shown only on the four historical/business pages
    (Executive Overview, Campaign Drivers, Customer Segments, Actionable
    Targeting). Returns a filters dict consumed by
    src.analytics.apply_filters(), applied consistently to whichever
    dataframe each page uses.
    """
    st.sidebar.markdown(
    """
    <div style="
        padding: 0.4rem 0 0.8rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.12);
        margin-bottom: 1rem;
    ">
        <div style="
            color: white;
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        ">Filters</div>
        <div style="
            color: #B9C6D8;
            font-size: 0.75rem;
            line-height: 1.4;
        ">
            Applies to Overview, Drivers, Segments & Targeting
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

    filters = {}
    filters["age_group"] = st.sidebar.multiselect("Age group", AGE_LABELS)
    filters["job"] = st.sidebar.multiselect("Job", sorted(campaign_df["job"].unique()))
    filters["education"] = st.sidebar.multiselect("Education", sorted(campaign_df["education"].unique()))
    filters["marital"] = st.sidebar.multiselect("Marital status", sorted(campaign_df["marital"].unique()))
    filters["housing"] = st.sidebar.multiselect("Housing loan", sorted(campaign_df["housing"].unique()))
    filters["loan"] = st.sidebar.multiselect("Personal loan", sorted(campaign_df["loan"].unique()))
    filters["month"] = st.sidebar.multiselect("Campaign month", MONTH_ORDER)
    filters["poutcome"] = st.sidebar.multiselect("Previous campaign outcome", sorted(campaign_df["poutcome"].unique()))

    if st.sidebar.button("Reset filters"):
        st.rerun()

    return filters


def page_executive_overview(campaign_df, filters):
    render_header()
    st.subheader("Executive Overview")
    st.caption(
        "Historical, observed results across the full UCI Bank Marketing "
        "campaign dataset (45,211 customers) - not model predictions."
    )

    filtered = apply_filters(campaign_df, filters)
    kpis = compute_overview_kpis(filtered)

    if kpis["customers_contacted"] == 0:
        st.warning("No customers match the current filters. Adjust filters in the sidebar.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Customers Contacted", f"{kpis['customers_contacted']:,}")
    with c2:
        metric_card("Subscriptions", f"{kpis['subscriptions']:,}", "hot")
    with c3:
        metric_card("Conversion Rate", f"{kpis['conversion_rate']*100:.1f}%", "warm")
    with c4:
        metric_card("Avg. Contacts / Customer", f"{kpis['avg_contacts']:.1f}", "cold")

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.markdown("#### Contacted vs. Subscribed")
        fig = go.Figure()
        fig.add_bar(
            x=["Contacted", "Subscribed"],
            y=[kpis["customers_contacted"], kpis["subscriptions"]],
            marker_color=[NAVY, TEAL],
            text=[f"{kpis['customers_contacted']:,}", f"{kpis['subscriptions']:,}"],
            textposition="outside",
        )
        fig.update_layout(
            font=dict(
                family="Segoe UI, Helvetica, Arial, sans-serif",
                size=13,
                color="#172033",
            ),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            margin=dict(t=30, l=15, r=15, b=25),
            height=320,
            showlegend=False,

            xaxis=dict(
                tickfont=dict(
                    size=13,
                    color="#202020",
                ),
                linecolor="#C9CDD3",
                tickcolor="#C9CDD3",
            ),

            yaxis=dict(
                tickfont=dict(
                    size=12,
                    color="#202020",
                ),
                gridcolor="#E2E5E9",
                linecolor="#C9CDD3",
                tickcolor="#C9CDD3",
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### Best-Performing Segment")
        st.caption(f"Only shown if a segment has at least {MIN_RELIABLE_SAMPLE_SIZE} customers.")
        best = best_performing_segment(filtered, "job", min_n=MIN_RELIABLE_SAMPLE_SIZE)
        if best is None:
            st.info("No job segment in the current filter selection meets the minimum sample size for a reliable comparison.")
        else:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="metric-label">JOB</div>
                    <div class="metric-value" style="font-size:1.4rem;">{best['category']}</div>
                    <div style="margin-top:0.4rem;">
                        Conversion rate: <b>{best['conversion_rate']*100:.1f}%</b>
                        &nbsp;·&nbsp; Sample size: <b>{best['count']:,}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.caption(
            "For the full breakdown across all customer characteristics, "
            "see the Customer Segments page."
        )


def page_campaign_drivers(campaign_df, filters):
    render_header()
    st.subheader("Campaign Drivers")
    st.caption(
        "Which campaign factors are associated with a customer subscribing? "
        "Rates are historical/observed, computed on the full dataset."
    )

    filtered = apply_filters(campaign_df, filters)
    if filtered.empty:
        st.warning("No customers match the current filters. Adjust filters in the sidebar.")
        return

    row1_a, row1_b = st.columns(2)
    with row1_a:
        rates = conversion_rate_by(filtered, "month", order=MONTH_ORDER)
        fig = rate_bar_chart(rates, "month", "Subscription Rate by Month", "Month")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    with row1_b:
        rates = conversion_rate_by(filtered, "contacts_bucket", order=CAMPAIGN_LABELS)
        fig = rate_bar_chart(
            rates, "contacts_bucket",
            "Subscription Rate by Number of Contacts (Current Campaign)",
            "Contacts made",
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    row2_a, row2_b = st.columns(2)
    with row2_a:
        rates = conversion_rate_by(filtered, "poutcome")
        fig = rate_bar_chart(rates, "poutcome", "Subscription Rate by Previous Campaign Outcome", "Previous outcome")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    with row2_b:
        rates = conversion_rate_by(filtered, "duration_bucket", order=DURATION_LABELS)
        fig = rate_bar_chart(rates, "duration_bucket", "Subscription Rate by Last Call Duration", "Last call duration")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Duration is only known after a call ends, so this chart describes what "
            "happened historically - it does not imply a call should be made longer "
            "to force a subscription. See README.md \u00a7 16 (post-interaction assumption)."
        )

    st.caption(
        "Bars shown at reduced opacity have fewer than "
        f"{MIN_RELIABLE_SAMPLE_SIZE} customers in the current filter selection - "
        "treat those rates as low-confidence, not a reliable pattern."
    )


def page_customer_segments(campaign_df, filters):
    render_header()
    st.subheader("Customer Segments")
    st.caption(
        "Which customer characteristics are associated with subscribing? "
        "Pick one characteristic at a time to avoid overwhelming charts."
    )

    filtered = apply_filters(campaign_df, filters)
    if filtered.empty:
        st.warning("No customers match the current filters. Adjust filters in the sidebar.")
        return

    segment_label = st.selectbox("Segment by", list(SEGMENT_COLUMNS.keys()))
    column = SEGMENT_COLUMNS[segment_label]
    order = AGE_LABELS if column == "age_group" else None

    rates = conversion_rate_by(filtered, column, order=order)
    fig = rate_bar_chart(rates, column, f"Subscription Rate by {segment_label}", segment_label)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Bars shown at reduced opacity have fewer than "
        f"{MIN_RELIABLE_SAMPLE_SIZE} customers in the current filter selection - "
        "treat those rates as low-confidence, not a reliable pattern."
    )

    with st.expander("View underlying numbers"):
        display = rates.rename(columns={
            column: segment_label, "count": "Customers",
            "conversions": "Subscriptions", "conversion_rate": "Conversion Rate",
            "reliable": f"Reliable (n \u2265 {MIN_RELIABLE_SAMPLE_SIZE})",
        })
        st.dataframe(
            display.style.format({"Conversion Rate": "{:.1%}"}),
            use_container_width=True,
        )


def page_actionable_targeting(pred_df, top_df, filters):
    render_header()
    st.subheader("Actionable Targeting")
    st.caption(
        "MODEL-PREDICTED probabilities on held-out customers the CNN never trained on "
        "(the test split) - not the historical/observed rates shown on the previous pages. "
        "Use this page to decide who to prioritize next."
    )

    filtered = apply_filters(pred_df, filters)
    if filtered.empty:
        st.warning("No customers match the current filters. Adjust filters in the sidebar.")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Customers Scored", f"{len(filtered):,}")
    with c2:
        metric_card("Hot Leads", f"{(filtered['lead_temperature'] == 'HOT').sum():,}", "hot")
    with c3:
        metric_card("Warm Leads", f"{(filtered['lead_temperature'] == 'WARM').sum():,}", "warm")
    with c4:
        metric_card("Cold Leads", f"{(filtered['lead_temperature'] == 'COLD').sum():,}", "cold")
    with c5:
        metric_card("High Contact Fatigue", f"{(filtered['contact_fatigue'] == 'HIGH').sum():,}", "fatigue")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Recommended Action Mix")
    st.caption("What the recommendation engine suggests across the currently filtered, scored customers.")
    rec_counts = filtered["recommendation"].value_counts().reset_index()
    rec_counts.columns = ["recommendation", "count"]
    fig = go.Figure()
    fig.add_bar(
        y=rec_counts["recommendation"], x=rec_counts["count"], orientation="h",
        marker_color=TEAL,
    )
    fig.update_layout(
        font=PLOTLY_FONT, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=10, l=10, r=10, b=10), height=280,
        yaxis=dict(autorange="reversed", tickfont=dict(size=12, color=TEXT_DARK)),
    )
    fig.update_xaxes(
        gridcolor="#e9e2cf", 
        title=dict(text="Customers", font=dict(size=13, color=TEXT_DARK)),
        tickfont=dict(size=12, color=TEXT_DARK),
        linecolor="#C9CDD3",
        tickcolor="#C9CDD3",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Top Priority Customers")
    st.caption("Ranked by model-predicted conversion probability (descending). Filters above are applied.")

    top_n = st.slider("Number of customers to show", min_value=10, max_value=100, value=50, step=10)
    ranked = filtered.sort_values("conversion_probability", ascending=False).head(top_n)
    cols = [c for c in DISPLAY_COLS if c in ranked.columns]

    st.dataframe(
        ranked[cols].style.format({"conversion_probability": "{:.1%}"}),
        use_container_width=True,
        height=520,
    )

    st.download_button(
        "Download this table as CSV",
        data=ranked[cols].to_csv(index=False).encode("utf-8"),
        file_name="top_priority_customers.csv",
        mime="text/csv",
    )


def page_customer_analysis(df):
    render_header()

    st.subheader("Customer Lookup")
    st.caption(
        "Search and filter test-set customers using their model-predicted "
        "decision profiles. All filters are applied together."
    )

    # =========================================================
    # SEARCH & FILTERS
    # =========================================================
    st.markdown("#### Search & Filters")

    search_col, temp_col, fatigue_col = st.columns([1.6, 1, 1])

    with search_col:
        record_id_search = st.text_input(
            "Customer Record ID",
            placeholder="Search by Customer Record ID...",
        )

    with temp_col:
        temp_filter = st.selectbox(
            "Lead Temperature",
            ["All", "HOT", "WARM", "COLD"],
        )

    with fatigue_col:
        fatigue_filter = st.selectbox(
            "Contact Fatigue",
            ["All", "LOW", "MEDIUM", "HIGH"],
        )

    # =========================================================
    # APPLY FILTERS
    # =========================================================
    filtered = df.copy()

    if temp_filter != "All":
        filtered = filtered[
            filtered["lead_temperature"]
            .astype(str)
            .str.strip()
            .str.upper()
            == temp_filter
        ]

    if fatigue_filter != "All":
        filtered = filtered[
            filtered["contact_fatigue"]
            .astype(str)
            .str.strip()
            .str.upper()
            == fatigue_filter
        ]

    if record_id_search.strip():
        search_value = record_id_search.strip().lower()

        filtered = filtered[
            filtered["customer_record_id"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.contains(search_value, na=False)
        ]

    # =========================================================
    # RESULTS
    # =========================================================
    st.markdown("---")

    if filtered.empty:
        if record_id_search.strip():
            st.warning(
                f"Customer Record ID '{record_id_search.strip()}' "
                "was not found with the selected filters."
            )
        else:
            st.warning(
                "No customers match the selected filters. "
                "Try changing the filters."
            )
        return

    st.markdown(
        f"### Customer Results ({len(filtered):,})"
    )

    active_filters = []

    if temp_filter != "All":
        active_filters.append(f"Temperature: **{temp_filter}**")

    if fatigue_filter != "All":
        active_filters.append(f"Fatigue: **{fatigue_filter}**")

    if record_id_search.strip():
        active_filters.append(
            f"Customer ID: **{record_id_search.strip()}**"
        )

    if active_filters:
        st.caption(" • ".join(active_filters))
    else:
        st.caption("Showing all customers.")

    # =========================================================
    # CUSTOMER RECORDS
    # =========================================================
    for _, row in filtered.iterrows():

        record_id = row["customer_record_id"]
        probability = float(row["conversion_probability"])
        temperature = str(row["lead_temperature"])
        fatigue = str(row["contact_fatigue"])

        # -----------------------------------------------------
        # Compact clickable record
        # -----------------------------------------------------
        with st.expander(
            f"Customer #{record_id}   |   "
            f"{temperature}   |   "
            f"{fatigue} Fatigue   |   "
            f"{probability * 100:.1f}% Conversion",
            expanded=False,
        ):

            st.caption(
                "Customer Record ID is generated by this project for "
                "demonstration purposes - the UCI dataset does not provide "
                "a real customer identifier."
            )

            # =================================================
            # TWO CARDS
            # =================================================
            info_col, decision_col = st.columns(
                [1, 1.3],
                gap="medium",
            )

            # -------------------------------------------------
            # CUSTOMER INFORMATION CARD
            # -------------------------------------------------
            with info_col:
                with st.container(border=True):

                    st.markdown("#### Customer Information")

                    st.write(f"**Age:** {row['age']}")
                    st.write(f"**Job:** {row['job']}")
                    st.write(
                        f"**Marital status:** "
                        f"{row.get('marital', 'n/a')}"
                    )
                    st.write(f"**Balance:** {row['balance']}")
                    st.write(f"**Contact type:** {row['contact']}")
                    st.write(
                        f"**Contacts (current campaign): "
                        f"{row['campaign']}"
                    )
                    st.write(
                        f"**Contacts (previous campaigns): "
                        f"{row['previous']}"
                    )
                    st.write(
                        f"**Previous campaign outcome: "
                        f"{row['poutcome']}"
                    )

            # -------------------------------------------------
            # MODEL PREDICTION CARD
            # -------------------------------------------------
            with decision_col:
                with st.container(border=True):

                    st.markdown("#### Model Prediction")

                    st.metric(
                        "Conversion probability",
                        f"{probability * 100:.1f}%",
                    )

                    st.markdown(
                        f"**Lead Temperature:** "
                        f"{lead_badge_html(temperature)}",
                        unsafe_allow_html=True,
                    )

                    st.markdown("#### Interaction Pattern")

                    st.info(
                        row["interaction_pattern"]
                    )

                    st.markdown("#### Contact Fatigue")

                    st.markdown(
                        fatigue_badge_html(fatigue),
                        unsafe_allow_html=True,
                    )

                    st.markdown("#### Recommendation")

                    st.markdown(
                        f"""
                        <div class="recommendation-box">
                            {row["recommendation"]}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    render_header()

    st.subheader("Customer Lookup")
    st.caption(
        "Search and filter test-set customers using their model-predicted "
        "decision profiles. All filters are applied together."
    )

    # =========================================================
    # SEARCH & FILTERS
    # =========================================================
    st.markdown("#### Search & Filters")

    search_col, temp_col, fatigue_col = st.columns([1.6, 1, 1])

    with search_col:
        record_id_search = st.text_input(
            "Customer Record ID",
            placeholder="Search by Customer Record ID...",
        )

    with temp_col:
        temp_filter = st.selectbox(
            "Lead Temperature",
            ["All", "HOT", "WARM", "COLD"],
        )

    with fatigue_col:
        fatigue_filter = st.selectbox(
            "Contact Fatigue",
            ["All", "LOW", "MEDIUM", "HIGH"],
        )

    # =========================================================
    # APPLY FILTERS
    # =========================================================
    filtered = df.copy()

    if temp_filter != "All":
        filtered = filtered[
            filtered["lead_temperature"]
            .astype(str)
            .str.strip()
            .str.upper()
            == temp_filter.upper()
        ]

    if fatigue_filter != "All":
        filtered = filtered[
            filtered["contact_fatigue"]
            .astype(str)
            .str.strip()
            .str.upper()
            == fatigue_filter.upper()
        ]

    if record_id_search.strip():
        search_value = record_id_search.strip().lower()

        filtered = filtered[
            filtered["customer_record_id"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.contains(search_value, na=False)
        ]

    # =========================================================
    # NO RESULTS
    # =========================================================
    st.markdown("---")

    if filtered.empty:
        if record_id_search.strip():
            st.warning(
                f"Customer Record ID '{record_id_search.strip()}' "
                "was not found with the selected filters."
            )
        else:
            st.warning(
                "No customers match the selected filters. "
                "Try changing the lead temperature or contact fatigue."
            )
        return

    # =========================================================
    # RESULT SUMMARY
    # =========================================================
    active_filters = []

    if temp_filter != "All":
        active_filters.append(f"Temperature: **{temp_filter}**")

    if fatigue_filter != "All":
        active_filters.append(f"Fatigue: **{fatigue_filter}**")

    if record_id_search.strip():
        active_filters.append(
            f"Customer ID: **{record_id_search.strip()}**"
        )

    st.markdown(
        f"### Customer Results "
        f"<span style='font-size: 16px; color: {TEXT_MUTED};'>"
        f"({len(filtered):,})</span>",
        unsafe_allow_html=True,
    )

    if active_filters:
        st.caption(" • ".join(active_filters))
    else:
        st.caption("Showing all customers.")

    # =========================================================
    # COMPACT EXPANDABLE CUSTOMER RECORDS
    # =========================================================
    for _, row in filtered.iterrows():

        record_id = row["customer_record_id"]
        probability = row["conversion_probability"]
        temperature = row["lead_temperature"]
        fatigue = row["contact_fatigue"]

        # Compact clickable customer record
        with st.expander(
            f"Customer #{record_id}  •  "
            f"{temperature}  •  "
            f"{fatigue} Fatigue  •  "
            f"{probability * 100:.1f}% Conversion",
            expanded=False,
        ):

            st.caption(
                "Customer Record ID is generated by this project for "
                "demonstration purposes - the UCI dataset does not provide "
                "a real customer identifier."
            )

            info_col, decision_col = st.columns([1, 1.3])

            # =====================================================
            # CUSTOMER INFORMATION
            # =====================================================
            with info_col:
                with st.container(border=True):
                    st.markdown("#### Customer Information")

                    st.write(f"**Age:** {row['age']}")
                    st.write(f"**Job:** {row['job']}")
                    st.write(
                        f"**Marital status:** "
                        f"{row.get('marital', 'n/a')}"
                    )
                    st.write(f"**Balance:** {row['balance']}")
                    st.write(f"**Contact type:** {row['contact']}")
                    st.write(
                        f"**Contacts (current campaign):** "
                        f"{row['campaign']}"
                    )
                    st.write(
                        f"**Contacts (previous campaigns):** "
                        f"{row['previous']}"
                    )
                    st.write(
                        f"**Previous campaign outcome:** "
                        f"{row['poutcome']}"
                    )

            # =====================================================
            # MODEL PREDICTION
            # =====================================================
            with decision_col:
                with st.container(border=True):
                    st.markdown("#### Model Prediction")

                    st.metric(
                        "Conversion probability",
                        f"{probability * 100:.1f}%",
                    )

                    st.markdown(
                        f"**Lead Temperature:** "
                        f"{lead_badge_html(temperature)}",
                        unsafe_allow_html=True,
                    )

                    st.markdown("#### Interaction Pattern")

                    st.info(row["interaction_pattern"])

                    st.markdown("#### Contact Fatigue")

                    st.markdown(
                        fatigue_badge_html(fatigue),
                        unsafe_allow_html=True,
                    )

                    st.markdown("#### Recommendation")

                    st.markdown(
                        f'<div class="recommendation-box">'
                        f'{row["recommendation"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )



def page_new_customer_analysis():
    render_header()

    st.subheader("New Customer Analysis")
    st.caption(
        "Demonstrates how the trained system could score a customer who is "
        "NOT already present in the pre-computed predictions table. This "
        "uses the SAME saved preprocessor and SAME saved 1D CNN used during "
        "training - no model is retrained here."
    )

    # =========================================================
    # LOAD SCORING ARTIFACTS
    # =========================================================
    try:
        preprocessor, cnn_model, cat_options = load_scoring_artifacts()

    except Exception as e:
        st.error(
            "Could not load the saved preprocessor/model needed for scoring. "
            f"Details: {e}\n\n"
            "Run `python train_and_evaluate.py` from the project root first "
            "so `models/preprocessor.joblib` and "
            "`models/cnn_model.keras` exist."
        )
        return

    # =========================================================
    # NEW CUSTOMER FORM
    # =========================================================
    with st.form("new_customer_form"):

        st.markdown("#### Customer Details")

        c1, c2, c3 = st.columns(3)

        # -----------------------------------------------------
        # Column 1
        # -----------------------------------------------------
        with c1:

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=35,
                step=1,
            )

            job = st.selectbox(
                "Job",
                cat_options["job"],
            )

            marital = st.selectbox(
                "Marital status",
                cat_options["marital"],
            )

            education = st.selectbox(
                "Education",
                cat_options["education"],
            )

            default = st.selectbox(
                "Has credit in default?",
                cat_options["default"],
            )

        # -----------------------------------------------------
        # Column 2
        # -----------------------------------------------------
        with c2:

            balance = st.number_input(
                "Account balance",
                min_value=-10000,
                max_value=150000,
                value=1000,
                step=100,
            )

            housing = st.selectbox(
                "Has housing loan?",
                cat_options["housing"],
            )

            loan = st.selectbox(
                "Has personal loan?",
                cat_options["loan"],
            )

            contact = st.selectbox(
                "Contact type",
                cat_options["contact"],
            )

            day = st.number_input(
                "Last contact day of month",
                min_value=1,
                max_value=31,
                value=15,
                step=1,
            )

        # -----------------------------------------------------
        # Column 3
        # -----------------------------------------------------
        with c3:

            month = st.selectbox(
                "Last contact month",
                cat_options["month"],
            )

            duration = st.number_input(
                "Last call duration (seconds)",
                min_value=0,
                max_value=5000,
                value=180,
                step=10,
            )

            campaign = st.number_input(
                "Contacts in current campaign",
                min_value=1,
                max_value=63,
                value=2,
                step=1,
            )

            pdays = st.number_input(
                "Days since last contact "
                "(previous campaign, -1 = never)",
                min_value=-1,
                max_value=900,
                value=-1,
                step=1,
            )

            previous = st.number_input(
                "Contacts before this campaign",
                min_value=0,
                max_value=275,
                value=0,
                step=1,
            )

            poutcome = st.selectbox(
                "Previous campaign outcome",
                cat_options["poutcome"],
            )

        st.caption(
            "Duration is assumed to be known because this prototype supports "
            "post-interaction follow-up decisions (see README.md, section 16)."
        )

        submitted = st.form_submit_button(
            "Analyze Customer"
        )

    # =========================================================
    # WAIT FOR SUBMISSION
    # =========================================================
    if not submitted:
        return

    # =========================================================
    # BUILD CUSTOMER INPUT
    # =========================================================
    customer_dict = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
    }

    # =========================================================
    # SCORE CUSTOMER
    # =========================================================
    try:
        result = predict_single_customer(
            customer_dict,
            preprocessor,
            cnn_model,
            reshape_for_cnn,
        )

    except Exception as e:
        st.error(
            "Something went wrong while scoring this customer. This can "
            "happen if a required field is missing or has an unexpected "
            f"value. Details: {e}"
        )
        return

    # =========================================================
    # RESULTS
    # =========================================================
    st.markdown("---")
    st.markdown("### Customer Analysis")

    res_col1, res_col2 = st.columns([1, 1.3])

    # ---------------------------------------------------------
    # Left Card
    # ---------------------------------------------------------
    with res_col1:

        st.markdown(
            '<div class="info-card">',
            unsafe_allow_html=True,
        )

        st.markdown("#### Conversion Probability")

        st.metric(
            "",
            f"{result['conversion_probability'] * 100:.1f}%"
        )

        st.markdown("#### Lead Temperature")

        st.markdown(
            lead_badge_html(
                result["lead_temperature"]
            ),
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Right Card
    # ---------------------------------------------------------
    with res_col2:

        st.markdown(
            '<div class="info-card">',
            unsafe_allow_html=True,
        )

        st.markdown("#### Interaction Pattern")

        st.info(
            result["interaction_pattern"]
        )

        st.markdown("#### Contact Fatigue")

        st.markdown(
            fatigue_badge_html(
                result["contact_fatigue"]
            ),
            unsafe_allow_html=True,
        )

        st.markdown("#### Recommendation")

        st.markdown(
            f'<div class="recommendation-box">'
            f'{result["recommendation"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================
    # EXPLANATION
    # =========================================================
    with st.expander("Why this result?"):

        st.markdown(
            f"""
            - **Conversion probability ({result['conversion_probability']*100:.1f}%)**
              comes directly from the trained **1D CNN** model, using the same
              saved preprocessor and the same saved network used for the rest
              of this dashboard.

            - **Lead Temperature ({result['lead_temperature']})** is a
              project-created decision-layer label, derived from that
              probability using fixed thresholds (HOT ≥ 70%, WARM 40-70%,
              COLD < 40%).

            - **Interaction Pattern ({result['interaction_pattern']})** is
              derived from rule-based campaign-history logic
              (`previous`, `poutcome`, `campaign`) - not from the CNN.

            - **Contact Fatigue ({result['contact_fatigue']})** is derived
              from the current-campaign contact count (`campaign`) alone -
              also not from the CNN.

            - **Recommendation** combines the three decision-layer outputs
              above into one suggested marketing action. The CNN does not
              generate the recommendation text itself - it only supplies
              the conversion probability.
            """
        )



def page_model_performance():
    render_header()
    st.subheader("Model Performance")
    st.caption(
        "Technical evaluation of the models trained on the same train/test "
        "split. This is intentionally placed lower in the app - the pages "
        "above are for marketing decisions; this page is for the technical "
        "review."
    )

    comparison_df = load_model_comparison()
    if comparison_df is None:
        st.warning("outputs/model_comparison.csv not found. Run `python train_and_evaluate.py` first.")
        return

    st.markdown("#### Model comparison (same test set, unedited results)")
    st.dataframe(comparison_df, use_container_width=True)

    best_roc = comparison_df.loc[comparison_df["ROC-AUC"].idxmax()]
    best_f1 = comparison_df.loc[comparison_df["F1-score"].idxmax()]
    best_recall = comparison_df.loc[comparison_df["Recall"].idxmax()]

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Best ROC-AUC", best_roc["Model"])
    with c2:
        metric_card("Best F1-score", best_f1["Model"])
    with c3:
        metric_card("Best Recall", best_recall["Model"])

    st.markdown(
        """
        **Reading this table honestly:** whichever model tops each metric is
        reported as-is here - this project does not claim the 1D CNN is the
        best model unless the numbers actually say so. The **1D CNN** is
        used to generate the marketing decision outputs (previous pages)
        because it is the project's required advanced deep-learning model
        and remains competitive with Random Forest, not because it is
        guaranteed to win on every metric. See `README.md` (sections 6 and
        8) for the full discussion of each model's role and this trade-off.

        | Model | Purpose |
        |---|---|
        | Logistic Regression | Simple, fast, highly interpretable baseline |
        | Decision Tree | Interpretable tree-based baseline |
        | Random Forest | Strong classical ML model (ensemble of trees) |
        | 1D CNN | Advanced deep-learning model (academic requirement); learns nonlinear patterns among neighbouring engineered features |
        """
    )

    roc_plot_path = os.path.join(PROJECT_ROOT, "outputs", "roc_curves_all_models.png")
    if os.path.exists(roc_plot_path):
        st.image(roc_plot_path, caption="ROC curves - all models", use_container_width=True)


def page_about():
    render_header()
    st.subheader("About This Project")
    st.markdown(
        """
        <div class="info-card">

        This is a **prototype** Marketing Decision Support System built for a
        college project on the **UCI Bank Marketing dataset**. It is **not**
        connected to a live bank CRM or any real customer database — all
        predictions shown here were generated once (offline) for the dataset's
        test split by `train_and_evaluate.py`, and this dashboard simply displays
        those results.

        **What the system estimates for each customer:**

        1. **Conversion probability** — model output: how likely the customer is
        to subscribe to a term deposit, predicted by a 1D CNN trained on the
        customer's demographic and campaign-history data.
        2. **Lead Temperature** — decision layer: HOT / WARM / COLD, derived from
        the probability using fixed thresholds (HOT ≥ 70%, WARM 40–70%,
        COLD < 40%).
        3. **Interaction Pattern** — decision layer: a transparent, rule-based,
        **project-created** behavioural category (Interested / Repeatedly
        Contacted / No Previous Interaction / Previously Unsuccessful /
        Unclear) — not an official UCI label.
        4. **Contact Fatigue** — decision layer: a **project-created** derived
        indicator (LOW / MEDIUM / HIGH) based on how many times the customer
        has already been contacted in the current campaign.
        5. **Recommendation** — combines all of the above into one plain-English
        suggested next action for the marketing team.

        **Post-interaction assumption:** the dataset's `duration` field (length of
        the last phone call) is only known after a contact attempt has happened.
        This system assumes it is used **after** that first contact, to support
        *follow-up* decisions — not to decide who to call in the first place. See
        `README.md` for the full discussion.

        **Customer Record ID** shown throughout this dashboard is generated from
        the dataset's row position for this project only — the UCI dataset does
        not include a real customer identifier.

        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    df = load_predictions()
    top_df = load_top_priority()
    campaign_df = load_campaign_data()

    if df is None:
        render_header()
        st.error(
            "No predictions found. Please run `python train_and_evaluate.py` "
            "from the project root first — it creates "
            "`outputs/final_predictions.csv`, which this dashboard reads."
        )
        st.stop()

    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Executive Overview",
            "Campaign Drivers",
            "Customer Segments",
            "Actionable Targeting",
            "Customer Lookup",
            "New Customer Analysis",
            "Model Performance",
            "About This Project",
        ],
        label_visibility="collapsed",
    )

    hierarchy_pages = {"Executive Overview", "Campaign Drivers", "Customer Segments", "Actionable Targeting"}
    filters = render_filter_sidebar(campaign_df) if page in hierarchy_pages else {}

    if page == "Executive Overview":
        page_executive_overview(campaign_df, filters)
    elif page == "Campaign Drivers":
        page_campaign_drivers(campaign_df, filters)
    elif page == "Customer Segments":
        page_customer_segments(campaign_df, filters)
    elif page == "Actionable Targeting":
        page_actionable_targeting(df, top_df, filters)
    elif page == "Customer Lookup":
        page_customer_analysis(df)
    elif page == "New Customer Analysis":
        page_new_customer_analysis()
    elif page == "Model Performance":
        page_model_performance()
    else:
        page_about()


if __name__ == "__main__":
    main()
