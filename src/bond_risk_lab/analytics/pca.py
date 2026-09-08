import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def prepare_curve_matrix(
    curve_history: pd.DataFrame,
    currency: str = "INR",
) -> pd.DataFrame:
    """
    Convert long-format yield curve data into a date x tenor matrix.
    """

    df = curve_history[
        curve_history["Currency"] == currency
    ].copy()

    matrix = df.pivot_table(
        index="CurveDate",
        columns="Tenor_Years",
        values="Yield",
        aggfunc="mean",
    )

    matrix = matrix.sort_index()

    return matrix.dropna()


def calculate_curve_changes(
    curve_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate daily yield changes in decimal yield units.
    """

    return curve_matrix.diff().dropna()


def run_pca(
    curve_matrix: pd.DataFrame,
    n_components: int = 3,
) -> dict:
    """
    Run PCA on historical yield-curve changes.

    Returns:
        PCA model
        factor scores
        factor loadings
        explained variance
    """

    changes = calculate_curve_changes(
        curve_matrix
    )

    scaler = StandardScaler()

    X = scaler.fit_transform(changes)

    n_components = min(
        n_components,
        X.shape[0],
        X.shape[1],
    )

    model = PCA(
        n_components=n_components
    )

    scores = model.fit_transform(X)

    loadings = pd.DataFrame(
        model.components_.T,
        index=changes.columns,
        columns=[
            f"PC{i + 1}"
            for i in range(n_components)
        ],
    )

    factor_scores = pd.DataFrame(
        scores,
        index=changes.index,
        columns=[
            f"PC{i + 1}"
            for i in range(n_components)
        ],
    )

    explained_variance = pd.DataFrame(
        {
            "Component": [
                f"PC{i + 1}"
                for i in range(n_components)
            ],
            "ExplainedVariance": (
                model.explained_variance_ratio_
            ),
        }
    )

    return {
        "model": model,
        "scaler": scaler,
        "loadings": loadings,
        "factor_scores": factor_scores,
        "explained_variance": explained_variance,
    }