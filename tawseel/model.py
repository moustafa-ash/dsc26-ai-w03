"""Failure-risk baseline for Tawseel COD parcels.

You write this file. Keep every name and signature exactly as it is: the grader imports them.
Allowed dependencies: numpy, pandas, scikit-learn (and the standard library).
"""

from __future__ import annotations

import math
from unittest import result

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "failed"

# The raw columns of make_parcels() that the model may use at decision time (when the parcel
# is created). Nothing known only after the delivery attempt, not the label, not the row id.
# TODO: this is every column of the table. Keep only the ones known at decision time.
FEATURE_COLUMNS: list[str] = [
    "customer_id",
    "merchant_id",
    "area",
    "created_at",
    "promised_slot",
    "cod_amount",
    "n_items",
    "address_text",
]


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df (same index and row order) with two new columns.

    Needs customer_id, created_at and failed.
    - customer_prior_parcels (int): the same customer's parcels with created_at strictly earlier than this row.
    - customer_prior_fail_rate (float): share of those that failed; NaN when there are none.
    Parcels with the same timestamp are not prior to each other. The result must not depend
    on the input row order. Never modify the input.
    """
    result = df.copy()

    groups = (
        result.groupby(["customer_id", "created_at"], as_index=False)
        .agg(parcel_count=("failed", "size"), failure_count=("failed", "sum"))
        .sort_values(["customer_id", "created_at"])
    )
    groups["prior_count"] = (
        groups.groupby("customer_id")["parcel_count"].cumsum() - groups["parcel_count"]
    )

    groups["prior_failures"] = (
        groups.groupby("customer_id")["failure_count"].cumsum()
        - groups["failure_count"]
    )

    groups["prior_fail_rate"] = groups["prior_failures"] / groups["prior_count"]

    history = groups.set_index(["customer_id", "created_at"])[
        ["prior_count", "prior_fail_rate"]
    ].rename(
        columns={
            "prior_count": "customer_prior_parcels",
            "prior_fail_rate": "customer_prior_fail_rate",
        }
    )

    return result.join(history, on=["customer_id", "created_at"])


def time_split(
    df: pd.DataFrame, time_col: str = "created_at", test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test).

    ValueError unless 0 < test_fraction < 1. Sort by time; n_test = ceil(len(df) * test_fraction);
    the boundary is the timestamp of the first of the last n_test rows. Rows at or after the
    boundary are test, rows before it are train.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    sorted_df = df.sort_values(time_col)
    n_test = math.ceil(len(sorted_df) * test_fraction)
    boundary = sorted_df[time_col].iloc[-n_test]

    train = sorted_df[sorted_df[time_col] < boundary]
    test = sorted_df[sorted_df[time_col] >= boundary]

    return train, test


def build_pipeline(seed: int = 0) -> Pipeline:
    """Return an UNFITTED Pipeline.

    Input: a DataFrame with at least FEATURE_COLUMNS plus the two history columns; ignore any
    other column. The final step must support predict_proba. Must not fail on unseen
    categories, missing numbers, missing categories or empty address strings. Same seed and
    same data must give identical predictions.
    """
    numeric_columns = [
        "cod_amount",
        "n_items",
        "customer_prior_parcels",
        "customer_prior_fail_rate",
    ]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_columns = [
        "merchant_id",
        "area",
        "promised_slot",
        "address_text",
    ]

    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )


def evaluate(model, test_df: pd.DataFrame) -> dict:
    """{"roc_auc", "pr_auc", "brier", "base_rate", "n"} from model.predict_proba(test_df)[:, 1]
    against test_df["failed"] (roc_auc_score, average_precision_score, brier_score_loss)."""
    actual = test_df[TARGET]
    probabilities = model.predict_proba(test_df)[:, 1]

    return {
        "roc_auc": roc_auc_score(actual, probabilities),
        "pr_auc": average_precision_score(actual, probabilities),
        "brier": brier_score_loss(actual, probabilities),
        "base_rate": actual.mean(),
        "n": len(test_df),
    }


def compare_to_baseline(
    train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = 0
) -> dict:
    """Fit build_pipeline(seed) and DummyClassifier(strategy="prior") on train_df only, evaluate
    both on test_df, return {"model": {...}, "dummy": {...}, "lift_pr_auc": model - dummy PR-AUC}."""
    model = build_pipeline(seed)
    dummy = DummyClassifier(strategy="prior")

    model.fit(train_df, train_df[TARGET])
    dummy.fit(train_df, train_df[TARGET])

    model_results = evaluate(model, test_df)
    dummy_results = evaluate(dummy, test_df)

    return {
        "model": model_results,
        "dummy": dummy_results,
        "lift_pr_auc": model_results["pr_auc"] - dummy_results["pr_auc"],
    }
