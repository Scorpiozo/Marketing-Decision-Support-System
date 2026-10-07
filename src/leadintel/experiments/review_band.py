"""Experiment 5 — choose the human-review band without touching the test set.

The dev split is divided again (75 % fit / 25 % validation). On the validation part we
compare bootstrap-interval widths by (a) share of leads flagged for review and (b) whether
flagged leads really are harder to predict (Brier inside vs outside the flag). Target: a
workable review load (~10 %) where the flag carries real information.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from leadintel.config import SEED
from leadintel.experiments.common import r, random_split, write_report
from leadintel.features import MODEL_FEATURES
from leadintel.models import FINAL_MODEL, build
from leadintel.policy import PolicyConfig
from leadintel.train import bootstrap_interval, fit_bootstrap

BANDS = [(0.10, 0.90), (0.20, 0.80), (0.25, 0.75), (0.30, 0.70)]


def main() -> dict[str, Any]:
    dev = random_split().dev
    fit, val = cast(
        tuple[pd.DataFrame, pd.DataFrame],
        train_test_split(dev, test_size=0.25, stratify=dev["y"], random_state=SEED),
    )
    pipe = build(FINAL_MODEL, MODEL_FEATURES).fit(fit[MODEL_FEATURES], fit["y"])
    members = fit_bootstrap(pipe, fit[MODEL_FEATURES], fit["y"])
    p = pipe.predict_proba(val[MODEL_FEATURES])[:, 1]
    y = val["y"].to_numpy()
    be = PolicyConfig().break_even
    rows = []
    for q in BANDS:
        lo, hi = bootstrap_interval(pipe, members, val[MODEL_FEATURES], p, q)
        flag = (lo < be) & (be <= hi)
        rows.append(
            {
                "band": list(q),
                "review_share": r(flag.mean()),
                "observed_rate_in_review": r(y[flag].mean()),
                "brier_in_review": r(np.mean((y[flag] - p[flag]) ** 2)),
                "brier_outside": r(np.mean((y[~flag] - p[~flag]) ** 2)),
            }
        )
        logger.info(rows[-1])
    payload = {"split": "dev 75/25 (test set untouched)", "break_even": be, "bands": rows}
    write_report("review_band", payload)
    return payload


if __name__ == "__main__":
    main()
