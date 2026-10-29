"""
Data/model service behind the Flask API.

Everything expensive happens once, in ``DashboardService.__init__``:

* the historical campaign dataset is loaded and cleaned once;
* the saved preprocessor is loaded and checked against feature_order.json;
* both saved models (Gradient Boosting, Random Forest) are loaded once;
* the held-out test split is rebuilt (same deterministic split as training),
  transformed once, and scored once per model;
* each model's decision table (lead temperature, interaction pattern,
  contact fatigue, recommendation) is built once.

Request handlers then only filter/aggregate in-memory DataFrames, and the
aggregates are memoised per (model, filters) key. Nothing is retrained.
"""

import io
import threading
import time
from collections import OrderedDict

import numpy as np

from src import analytics
from src import model as mdl
from src import preprocessing as prep
from src.interaction_analysis import CATEGORY_DESCRIPTIONS
from src.prediction import (
    FATIGUE_LOW_MAX, FATIGUE_MEDIUM_MAX, HOT_THRESHOLD, WARM_THRESHOLD,
    build_prediction_dataframe, predict_single_customer,
)

FILTER_FIELDS = OrderedDict([
    ("age_group", "Age group"),
    ("job", "Job"),
    ("education", "Education"),
    ("marital", "Marital status"),
    ("housing", "Housing loan"),
    ("loan", "Personal loan"),
    ("month", "Campaign month"),
    ("poutcome", "Previous outcome"),
])

DRIVERS = OrderedDict([
    ("month", ("Month of last contact", analytics.MONTH_ORDER)),
    ("contacts_bucket", ("Contacts in current campaign", analytics.CAMPAIGN_LABELS)),
    ("poutcome", ("Previous campaign outcome", None)),
    ("duration_bucket", ("Last call duration", analytics.DURATION_LABELS)),
])

TABLE_COLUMNS = [
    "customer_record_id", "age", "job", "balance", "contact", "campaign",
    "previous", "poutcome", "conversion_probability", "lead_temperature",
    "interaction_pattern", "contact_fatigue", "recommendation",
]
SORTABLE = {"probability": "conversion_probability", "age": "age", "balance": "balance", "campaign": "campaign"}
TEMPERATURES = ["HOT", "WARM", "COLD"]
FATIGUE_LEVELS = ["LOW", "MEDIUM", "HIGH"]

NUMERIC_INPUT_RANGES = OrderedDict([
    ("age", (18, 100)),
    ("balance", (-10000, 150000)),
    ("day", (1, 31)),
    ("duration", (0, 5000)),
    ("campaign", (1, 63)),
    ("pdays", (-1, 900)),
    ("previous", (0, 275)),
])

HISTOGRAM_BIN_WIDTH = 0.05


class ValidationError(ValueError):
    """Bad client input; mapped to HTTP 400 by the API layer."""


class _Memo:
    """Small bounded LRU cache for aggregate responses."""

    def __init__(self, maxsize=512):
        self.maxsize = maxsize
        self._data = OrderedDict()
        self._lock = threading.Lock()

    def get_or_compute(self, key, fn):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
        value = fn()
        with self._lock:
            self._data[key] = value
            if len(self._data) > self.maxsize:
                self._data.popitem(last=False)
        return value


def _rate_rows(rates_df, column):
    return [
        {
            "category": str(r[column]),
            "count": int(r["count"]),
            "conversions": int(r["conversions"]),
            "rate": round(float(r["conversion_rate"]), 4),
            "reliable": bool(r["reliable"]),
        }
        for _, r in rates_df.iterrows()
    ]


def _py(value):
    """Convert numpy scalars to plain Python for JSON."""
    if isinstance(value, np.generic):
        return value.item()
    return value


class DashboardService:
    def __init__(self):
        t0 = time.perf_counter()

        self.campaign_df = analytics.load_campaign_dataset()

        self.preprocessor = prep.load_preprocessor()
        prep.validate_feature_order(self.preprocessor)
        self.categorical_options = prep.get_categorical_options(self.preprocessor)
        self.production_info = mdl.load_production_model_info()
        self.production_key = self.production_info["model_key"]
        self.models = {key: mdl.load_model(key) for key in mdl.MODEL_REGISTRY}

        raw = prep.encode_target(prep.clean_data(prep.load_raw_data()))
        _, X_test, _, y_test = prep.train_test_split_data(raw)
        self.y_test = y_test.values
        X_test = X_test.reset_index(drop=True)
        X_test_proc = self.preprocessor.transform(X_test)

        self.test_probs = {}
        self.predictions = {}
        for key, model in self.models.items():
            probs = mdl.predict_proba(model, X_test_proc)
            self.test_probs[key] = probs
            df = build_prediction_dataframe(X_test, probs)
            df["age_group"] = analytics.add_age_group(df["age"]).astype(str)
            df.index = df["customer_record_id"].to_numpy()
            self.predictions[key] = df

        self.filter_options = OrderedDict()
        for col in FILTER_FIELDS:
            if col == "age_group":
                self.filter_options[col] = list(analytics.AGE_LABELS)
            elif col == "month":
                self.filter_options[col] = list(analytics.MONTH_ORDER)
            else:
                self.filter_options[col] = sorted(self.campaign_df[col].astype(str).unique())

        self._memo = _Memo()
        self._model_report = None
        self.startup_seconds = round(time.perf_counter() - t0, 2)

    def parse_filters(self, getlist):
        """Build a canonical, hashable filters tuple from a ``getlist`` callable."""
        parsed = []
        for col in FILTER_FIELDS:
            values = [v for v in getlist(col) if v != ""]
            if not values:
                continue
            allowed = set(self.filter_options[col])
            bad = [v for v in values if v not in allowed]
            if bad:
                raise ValidationError(f"Unknown value(s) for filter '{col}': {bad}")
            parsed.append((col, tuple(sorted(set(values)))))
        return tuple(parsed)

    def parse_model(self, key):
        if not key:
            return self.production_key
        if key not in mdl.MODEL_REGISTRY:
            raise ValidationError(
                f"Unknown model '{key}'. Use one of: {', '.join(mdl.MODEL_REGISTRY)}"
            )
        return key

    @staticmethod
    def _apply(df, filters):
        return analytics.apply_filters(df, {col: list(vals) for col, vals in filters})

    def meta(self):
        return {
            "filters": [
                {"key": k, "label": label, "options": self.filter_options[k]}
                for k, label in FILTER_FIELDS.items()
            ],
            "models": [
                {"key": k, "name": mdl.display_name(k), "production": k == self.production_key}
                for k in mdl.MODEL_REGISTRY
            ],
            "production_model": {
                "key": self.production_key,
                "name": self.production_info["display_name"],
                "selected_by": self.production_info["selected_by"],
            },
            "segments": [{"key": col, "label": label} for label, col in analytics.SEGMENT_COLUMNS.items()],
            "drivers": [{"key": k, "label": label} for k, (label, _) in DRIVERS.items()],
            "thresholds": {
                "hot": HOT_THRESHOLD, "warm": WARM_THRESHOLD,
                "fatigue_low_max": FATIGUE_LOW_MAX, "fatigue_medium_max": FATIGUE_MEDIUM_MAX,
                "min_reliable_sample": analytics.MIN_RELIABLE_SAMPLE_SIZE,
            },
            "form": {
                "categorical": self.categorical_options,
                "numeric": {k: {"min": lo, "max": hi} for k, (lo, hi) in NUMERIC_INPUT_RANGES.items()},
            },
            "dataset": {"customers": int(len(self.campaign_df)), "test_customers": int(len(self.y_test))},
        }

    def overview(self, filters):
        def compute():
            df = self._apply(self.campaign_df, filters)
            kpis = analytics.compute_overview_kpis(df)
            if kpis["customers_contacted"] == 0:
                return {"kpis": kpis, "by_month": [], "best_segment": None}
            by_month = analytics.conversion_rate_by(df, "month", order=analytics.MONTH_ORDER)
            return {
                "kpis": {k: _py(v) for k, v in kpis.items()},
                "baseline_rate": round(float(self.campaign_df["converted"].mean()), 4),
                "by_month": _rate_rows(by_month, "month"),
                "best_segment": self._best_segment(df),
            }
        return self._memo.get_or_compute(("overview", filters), compute)

    @staticmethod
    def _best_segment(df):
        best = analytics.best_performing_segment(df, "job")
        return None if best is None else {k: _py(v) for k, v in best.items()}

    def drivers(self, filters):
        def compute():
            df = self._apply(self.campaign_df, filters)
            out = {"n": int(len(df)), "drivers": []}
            for col, (label, order) in DRIVERS.items():
                rows = [] if df.empty else _rate_rows(analytics.conversion_rate_by(df, col, order=order), col)
                out["drivers"].append({"key": col, "label": label, "rows": rows})
            return out
        return self._memo.get_or_compute(("drivers", filters), compute)

    def segment(self, filters, column):
        valid = dict((col, label) for label, col in analytics.SEGMENT_COLUMNS.items())
        if column not in valid:
            raise ValidationError(f"Unknown segment '{column}'. Use one of: {', '.join(valid)}")

        def compute():
            df = self._apply(self.campaign_df, filters)
            order = analytics.AGE_LABELS if column == "age_group" else None
            rows = [] if df.empty else _rate_rows(analytics.conversion_rate_by(df, column, order=order), column)
            return {"key": column, "label": valid[column], "n": int(len(df)), "rows": rows}
        return self._memo.get_or_compute(("segment", filters, column), compute)

    def targeting_summary(self, model_key, filters):
        def compute():
            df = self._apply(self.predictions[model_key], filters)
            temp_counts = df["lead_temperature"].value_counts()
            fatigue_counts = df["contact_fatigue"].value_counts()
            rec = df["recommendation"].value_counts()
            n_bins = int(round(1 / HISTOGRAM_BIN_WIDTH))
            bins = np.minimum((df["conversion_probability"].to_numpy() / HISTOGRAM_BIN_WIDTH).astype(int), n_bins - 1)
            hist = np.bincount(bins, minlength=n_bins) if len(df) else np.zeros(n_bins, dtype=int)
            return {
                "model": model_key,
                "kpis": {
                    "scored": int(len(df)),
                    **{t.lower(): int(temp_counts.get(t, 0)) for t in TEMPERATURES},
                    "high_fatigue": int(fatigue_counts.get("HIGH", 0)),
                    "mean_probability": round(float(df["conversion_probability"].mean()), 4) if len(df) else None,
                },
                "recommendations": [{"label": k, "count": int(v)} for k, v in rec.items()],
                "histogram": [
                    {"from": round(i * HISTOGRAM_BIN_WIDTH, 2), "to": round((i + 1) * HISTOGRAM_BIN_WIDTH, 2), "count": int(c)}
                    for i, c in enumerate(hist)
                ],
            }
        return self._memo.get_or_compute(("targeting", model_key, filters), compute)

    def _targeted_frame(self, model_key, filters, temperature, fatigue, query, sort, order):
        df = self._apply(self.predictions[model_key], filters)
        if temperature:
            if temperature not in TEMPERATURES:
                raise ValidationError(f"temperature must be one of {TEMPERATURES}")
            df = df[df["lead_temperature"] == temperature]
        if fatigue:
            if fatigue not in FATIGUE_LEVELS:
                raise ValidationError(f"fatigue must be one of {FATIGUE_LEVELS}")
            df = df[df["contact_fatigue"] == fatigue]
        if query:
            q = query.strip()
            if not q.isdigit():
                raise ValidationError("Search by numeric customer record ID.")
            df = df[df["customer_record_id"].astype(str).str.contains(q, regex=False)]
        if sort not in SORTABLE:
            raise ValidationError(f"sort must be one of {sorted(SORTABLE)}")
        if order not in ("asc", "desc"):
            raise ValidationError("order must be 'asc' or 'desc'")
        return df.sort_values(
            [SORTABLE[sort], "customer_record_id"], ascending=[order == "asc", True], kind="mergesort"
        )

    def customers(self, model_key, filters, temperature=None, fatigue=None, query=None,
                  sort="probability", order="desc", page=1, page_size=25):
        if not (1 <= page_size <= 100):
            raise ValidationError("page_size must be between 1 and 100")
        if page < 1:
            raise ValidationError("page must be >= 1")
        df = self._targeted_frame(model_key, filters, temperature, fatigue, query, sort, order)
        total = int(len(df))
        start = (page - 1) * page_size
        chunk = df.iloc[start:start + page_size][TABLE_COLUMNS]
        return {
            "model": model_key,
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": max(1, -(-total // page_size)),
            "columns": TABLE_COLUMNS,
            "rows": [[_py(v) for v in row] for row in chunk.itertuples(index=False, name=None)],
        }

    def export_csv(self, model_key, filters, temperature=None, fatigue=None, query=None,
                   sort="probability", order="desc"):
        df = self._targeted_frame(model_key, filters, temperature, fatigue, query, sort, order)
        buf = io.StringIO()
        df[TABLE_COLUMNS].to_csv(buf, index=False)
        return buf.getvalue(), int(len(df))

    def customer(self, model_key, record_id):
        df = self.predictions[model_key]
        if record_id not in df.index:
            raise KeyError(record_id)
        row = df.loc[record_id]
        record = {c: _py(row[c]) for c in prep.REQUIRED_INPUT_COLUMNS}
        other = {
            k: round(float(self.predictions[k].loc[record_id, "conversion_probability"]), 4)
            for k in self.predictions
        }
        return {
            "customer_record_id": int(record_id),
            "model": model_key,
            "record": record,
            "conversion_probability": float(row["conversion_probability"]),
            "lead_temperature": row["lead_temperature"],
            "interaction_pattern": row["interaction_pattern"],
            "interaction_description": CATEGORY_DESCRIPTIONS.get(row["interaction_pattern"], ""),
            "contact_fatigue": row["contact_fatigue"],
            "recommendation": row["recommendation"],
            "probability_by_model": other,
        }

    def validate_customer(self, payload):
        if not isinstance(payload, dict):
            raise ValidationError("Request body must be a JSON object with customer fields.")
        missing = [c for c in prep.REQUIRED_INPUT_COLUMNS if c not in payload]
        if missing:
            raise ValidationError(f"Missing fields: {missing}")
        record = {}
        for col, (lo, hi) in NUMERIC_INPUT_RANGES.items():
            value = payload[col]
            if isinstance(value, bool):
                raise ValidationError(f"'{col}' must be a whole number")
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValidationError(f"'{col}' must be a whole number") from None
            if not number.is_integer():
                raise ValidationError(f"'{col}' must be a whole number")
            if not lo <= number <= hi:
                raise ValidationError(f"'{col}' must be between {lo} and {hi}")
            record[col] = int(number)
        for col, options in self.categorical_options.items():
            value = payload[col]
            if value not in options:
                raise ValidationError(f"'{col}' must be one of {options}")
            record[col] = value
        return record

    def predict(self, payload, model_key):
        record = self.validate_customer(payload)
        results = {
            key: predict_single_customer(record, self.preprocessor, model)
            for key, model in self.models.items()
        }
        return {"model": model_key, "result": results[model_key], "probability_by_model": {
            k: r["conversion_probability"] for k, r in results.items()
        }}

    def model_report(self):
        if self._model_report is None:
            rows, roc, confusion, importances = [], {}, {}, {}
            for key, probs in self.test_probs.items():
                pred = (probs >= mdl.DECISION_THRESHOLD).astype(int)
                metrics = mdl.evaluate_model(self.y_test, pred, probs, mdl.display_name(key))
                rows.append({"key": key, **{k: (round(float(v), 4) if k != "Model" else v) for k, v in metrics.items()}})
                roc[key] = mdl.roc_points(self.y_test, probs)
                confusion[key] = mdl.get_confusion_matrix(self.y_test, pred).tolist()
                importances[key] = mdl.grouped_feature_importances(self.models[key], self.preprocessor)
            self._model_report = {
                "production_model": self.production_key,
                "selected_by": self.production_info["selected_by"],
                "threshold": mdl.DECISION_THRESHOLD,
                "test_customers": int(len(self.y_test)),
                "test_positives": int(self.y_test.sum()),
                "metrics": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC"],
                "comparison": rows,
                "roc": roc,
                "confusion": confusion,
                "importances": importances,
            }
        return self._model_report
