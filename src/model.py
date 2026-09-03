import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

RANDOM_STATE = 42


def set_seeds(seed=RANDOM_STATE):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, balanced=True):
    model = DecisionTreeClassifier(
        max_depth=7,
        class_weight="balanced" if balanced else None,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model



# 1D CNN for tabular data

def build_cnn_model(input_dim, learning_rate=1e-3):
    set_seeds()

    inputs = keras.Input(shape=(input_dim, 1), name="feature_vector")

    x = layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(1, activation="sigmoid", name="subscription_probability")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="bank_marketing_cnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def reshape_for_cnn(X):
    return np.asarray(X).reshape((X.shape[0], X.shape[1], 1))


def train_cnn_model(
    X_train, y_train, X_val=None, y_val=None,
    epochs=40, batch_size=256, learning_rate=1e-3, verbose=1,
):
    set_seeds()

    X_train_cnn = reshape_for_cnn(X_train)
    y_train = np.asarray(y_train)

    model = build_cnn_model(input_dim=X_train.shape[1], learning_rate=learning_rate)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    )

    # Class weights to handle the ~88/12 class imbalance without discarding data
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    class_weight = {0: 1.0, 1: n_neg / max(n_pos, 1)}

    fit_kwargs = dict(
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        class_weight=class_weight,
        verbose=verbose,
    )

    if X_val is not None and y_val is not None:
        X_val_cnn = reshape_for_cnn(X_val)
        history = model.fit(
            X_train_cnn, y_train,
            validation_data=(X_val_cnn, np.asarray(y_val)),
            **fit_kwargs,
        )
    else:
        history = model.fit(
            X_train_cnn, y_train,
            validation_split=0.15,
            **fit_kwargs,
        )

    return model, history


def cnn_predict_proba(model, X):
    """Return P(y=1) for each row in X using the trained CNN."""
    X_cnn = reshape_for_cnn(np.asarray(X))
    probs = model.predict(X_cnn, verbose=0).ravel()
    return probs



# Shared evaluation utilities

def evaluate_model(y_true, y_pred, y_prob=None, model_name="model"):
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)
        metrics["PR-AUC"] = average_precision_score(y_true, y_prob)
    return metrics


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)



# Persistence helpers

def save_sklearn_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_sklearn_model(path):
    return joblib.load(path)


def save_cnn_model(model, path="models/cnn_model.keras"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)


def load_cnn_model(path="models/cnn_model.keras"):
    return keras.models.load_model(path)
