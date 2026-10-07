"""Experiment 3 — is the legacy 1D CNN justified on this tabular data?

A convolution assumes that *neighbouring* inputs form meaningful local patterns (pixels,
time steps). Here the "neighbours" are just the order in which columns were one-hot
encoded. Falsification test: if the CNN is exploiting real local structure, randomly
permuting the column order should hurt it. We also repeat over seeds, because the legacy
CNN was not reproducible run-to-run.

Requires the optional `dl` dependency group:  uv sync --group dl
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from loguru import logger
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from leadintel.config import SEED
from leadintel.experiments.common import r, random_split, write_report
from leadintel.features import MODEL_FEATURES, linear_encoder

SEEDS = (0, 1, 2)
N_PERMUTATIONS = 3


def _build_legacy_cnn(n_features: int, seed: int) -> Any:
    """Architecture copied from the legacy src/model.py (2x Conv1D-BN-Pool, Dense 64/32)."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.keras.utils.set_random_seed(seed)
    inputs = keras.Input(shape=(n_features, 1))
    x = layers.Conv1D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model


def _fit_predict(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, seed: int
) -> tuple[np.ndarray, float]:
    from tensorflow import keras

    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.15, stratify=ytr, random_state=seed)
    model = _build_legacy_cnn(Xtr.shape[1], seed)
    stop = keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
    t0 = time.perf_counter()
    # Unweighted loss (unlike the legacy run) so probabilities are comparable to the others.
    model.fit(
        np.expand_dims(Xa, -1), ya, validation_data=(np.expand_dims(Xv, -1), yv),
        epochs=60, batch_size=256, callbacks=[stop], verbose=0,
    )  # fmt: skip
    secs = time.perf_counter() - t0
    return model.predict(np.expand_dims(Xte, -1), verbose=0).ravel(), secs


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": r(roc_auc_score(y, p)),
        "pr_auc": r(average_precision_score(y, p)),
        "brier": r(brier_score_loss(y, p)),
    }


def main() -> dict[str, Any]:
    split = random_split()
    enc = linear_encoder(MODEL_FEATURES).fit(split.dev[MODEL_FEATURES])
    Xtr = np.asarray(enc.transform(split.dev[MODEL_FEATURES]), dtype="float32")
    Xte = np.asarray(enc.transform(split.test[MODEL_FEATURES]), dtype="float32")
    ytr, yte = split.dev["y"].to_numpy(), split.test["y"].to_numpy()

    original, fit_secs = [], []
    for seed in SEEDS:
        p, secs = _fit_predict(Xtr, ytr, Xte, seed)
        original.append(_metrics(yte, p))
        fit_secs.append(secs)
        logger.info(f"cnn original order seed={seed}: {original[-1]}")

    permuted = []
    rng = np.random.default_rng(SEED)
    for i in range(N_PERMUTATIONS):
        perm = rng.permutation(Xtr.shape[1])
        p, _ = _fit_predict(Xtr[:, perm], ytr, Xte[:, perm], seed=SEEDS[0])
        permuted.append(_metrics(yte, p))
        logger.info(f"cnn permuted columns #{i}: {permuted[-1]}")

    def summary(rows: list[dict[str, float]]) -> dict[str, list[float]]:
        return {
            k: [r(np.mean([x[k] for x in rows])), r(np.std([x[k] for x in rows]))] for k in rows[0]
        }

    payload = {
        "input": f"{Xtr.shape[1]} one-hot/scaled features reshaped to ({Xtr.shape[1]}, 1)",
        "original_order_by_seed": original,
        "original_order_mean_std": summary(original),
        "permuted_columns": permuted,
        "permuted_mean_std": summary(permuted),
        "mean_fit_seconds": r(float(np.mean(fit_secs)), 1),
    }
    write_report("cnn", payload)
    return payload


if __name__ == "__main__":
    main()
