from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)


def build_scenario_features(
    factor_scenarios: pd.DataFrame,
    curve_changes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build machine-learning features from PCA factors
    and simulated yield-curve shocks.

    Features include:

    - PCA factor shocks
    - Parallel yield-curve shift
    - 2Y-10Y slope
    - 5Y curvature
    - Long-end shift
    - Short-end shift
    """

    features = factor_scenarios.copy()

    tenors = np.asarray(
        curve_changes.columns,
        dtype=float,
    )

    curve = curve_changes.copy()

    def nearest_tenor(target: float) -> float:
        return float(
            tenors[
                np.argmin(
                    np.abs(tenors - target)
                )
            ]
        )

    tenor_2y = nearest_tenor(2.0)
    tenor_5y = nearest_tenor(5.0)
    tenor_10y = nearest_tenor(10.0)
    tenor_20y = nearest_tenor(20.0)

    features["ParallelShift"] = (
        curve.mean(axis=1)
    )

    features["ShortEndShift"] = (
        curve[tenor_2y]
    )

    features["LongEndShift"] = (
        curve[tenor_10y]
    )

    features["UltraLongShift"] = (
        curve[tenor_20y]
    )

    features["Slope_2Y_10Y"] = (
        curve[tenor_10y]
        - curve[tenor_2y]
    )

    features["Curvature_2Y_5Y_10Y"] = (
        2.0 * curve[tenor_5y]
        - curve[tenor_2y]
        - curve[tenor_10y]
    )

    features["LongEndSlope_10Y_20Y"] = (
        curve[tenor_20y]
        - curve[tenor_10y]
    )

    return features


def create_loss_labels(
    scenario_pnl: pd.DataFrame,
    loss_quantile: float = 0.95,
) -> pd.Series:
    """
    Create binary loss labels.

    Label = 1 when scenario P&L falls below the
    selected lower-tail percentile.

    Example:

        loss_quantile = 0.95

    means the worst 5% of scenarios are classified
    as high-risk loss scenarios.
    """

    if "PnL_Total_INR" not in scenario_pnl.columns:
        raise ValueError(
            "scenario_pnl must contain "
            "'PnL_Total_INR'."
        )

    if not 0.50 < loss_quantile < 1.0:
        raise ValueError(
            "loss_quantile must be between "
            "0.50 and 1.00."
        )

    pnl = scenario_pnl[
        "PnL_Total_INR"
    ]

    threshold = pnl.quantile(
        1.0 - loss_quantile
    )

    labels = (
        pnl <= threshold
    ).astype(int)

    labels.name = "HighRiskLoss"

    return labels


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
):
    """
    Train a Random Forest classifier.
    """

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X, y)

    return model


def train_neural_network(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
):
    """
    Train a scaled multilayer perceptron classifier.
    """

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(
                        64,
                        32,
                    ),
                    activation="relu",
                    solver="adam",
                    alpha=0.0005,
                    learning_rate_init=0.001,
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.20,
                    random_state=random_state,
                ),
            ),
        ]
    )

    model.fit(X, y)

    return model


def evaluate_classifier(
    model,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """
    Evaluate a binary risk classifier.
    """

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    results = {
        "accuracy": float(
            accuracy_score(
                y,
                predictions,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y,
                probabilities,
            )
        ),
        "classification_report": (
            classification_report(
                y,
                predictions,
                output_dict=True,
                zero_division=0,
            )
        ),
    }

    return results


def random_forest_feature_importance(
    model,
    feature_names,
) -> pd.DataFrame:
    """
    Return Random Forest feature importance.
    """

    importance = pd.DataFrame(
        {
            "Feature": list(
                feature_names
            ),
            "Importance": (
                model.feature_importances_
            ),
        }
    )

    return (
        importance
        .sort_values(
            "Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def classify_risk_probability(
    model,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate ML probability of a high-risk
    loss scenario.
    """

    probability = (
        model.predict_proba(X)[:, 1]
    )

    prediction = (
        probability >= 0.50
    ).astype(int)

    return pd.DataFrame(
        {
            "HighRiskProbability": probability,
            "PredictedHighRisk": prediction,
        },
        index=X.index,
    )