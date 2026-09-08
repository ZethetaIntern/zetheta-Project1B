import pandas as pd

from .pca import (
    prepare_curve_matrix,
    run_pca,
)


def analyze_yield_curve(
    curve_history: pd.DataFrame,
    currency: str = "INR",
) -> dict:
    """
    Complete yield-curve factor analysis.
    """

    matrix = prepare_curve_matrix(
        curve_history,
        currency,
    )

    pca_result = run_pca(
        matrix,
        n_components=3,
    )

    return {
        "curve_matrix": matrix,
        "pca": pca_result,
    }


def latest_curve_factors(
    curve_history: pd.DataFrame,
    currency: str = "INR",
) -> pd.DataFrame:

    result = analyze_yield_curve(
        curve_history,
        currency,
    )

    scores = result["pca"]["factor_scores"]

    return scores.tail(10)