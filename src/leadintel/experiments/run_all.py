"""Reproduce every experiment and the deployed artifacts in one command.

    uv run leadintel-experiments            # ~4 min on a laptop
    uv sync --group dl && uv run leadintel-experiments --cnn   # + legacy CNN (~2 min more)

Outputs: reports/experiments/*.json and artifacts/*. All numbers in the docs and the app
come from these files.
"""

from __future__ import annotations

import argparse
import time

from loguru import logger

from leadintel import train
from leadintel.experiments import ablation, comparison, interactions, review_band, tuning


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn", action="store_true", help="also run the TensorFlow CNN study")
    parser.add_argument("--skip-tuning", action="store_true")
    args = parser.parse_args()

    steps = [("comparison", comparison.main), ("ablation", ablation.main)]
    if not args.skip_tuning:
        steps.insert(0, ("tuning", tuning.main))
    steps += [
        ("interactions", interactions.main),
        ("review_band", review_band.main),
        ("train", train.main),
    ]
    if args.cnn:
        from leadintel.experiments import cnn  # TensorFlow is optional

        steps.append(("cnn", cnn.main))

    for name, fn in steps:
        t0 = time.perf_counter()
        logger.info(f"=== {name} ===")
        fn()
        logger.info(f"=== {name} done in {time.perf_counter() - t0:.0f}s ===")


if __name__ == "__main__":
    main()
