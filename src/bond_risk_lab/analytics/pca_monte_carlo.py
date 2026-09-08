from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_pca_scenarios(
    factor_scores: pd.DataFrame,
    loadings: pd.DataFrame,
    explained_variance: pd.DataFrame,
    n_scenarios: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate yield-curve PCA factor shocks.

    The historical PCA factor scores are used to estimate
    the mean and volatility of each principal component.
    """

    if n_scenarios <= 0:
        raise ValueError(
            "n_scenarios must be greater than zero."
        )

    components = list(loadings.columns)

    missing_components = [
        component
        for component in components
        if component not in factor_scores.columns
    ]

    if missing_components:
        raise ValueError(
            "Missing PCA components in factor_scores: "
            f"{missing_components}"
        )

    rng = np.random.default_rng(seed)

    factor_mean = (
        factor_scores[components]
        .mean()
        .to_numpy(dtype=float)
    )

    factor_std = (
        factor_scores[components]
        .std(ddof=1)
        .to_numpy(dtype=float)
    )

    simulated_factors = rng.normal(
        loc=factor_mean,
        scale=factor_std,
        size=(
            n_scenarios,
            len(components),
        ),
    )

    return pd.DataFrame(
        simulated_factors,
        columns=components,
    )


def reconstruct_curve_changes(
    factor_scenarios: pd.DataFrame,
    loadings: pd.DataFrame,
    scaler=None,
) -> pd.DataFrame:
    """
    Reconstruct yield-curve changes from PCA factors.

    PCA is performed on standardized historical yield
    changes. The PCA factors are therefore first converted
    back into standardized curve shocks and then transformed
    back into the original yield-change units using the
    StandardScaler.
    """

    common_components = [
        component
        for component in factor_scenarios.columns
        if component in loadings.columns
    ]

    if not common_components:
        raise ValueError(
            "No common PCA components found."
        )

    standardized_changes = (
        factor_scenarios[
            common_components
        ]
        .to_numpy()
        @ loadings[
            common_components
        ]
        .to_numpy()
        .T
    )

    if scaler is not None:
        raw_changes = scaler.inverse_transform(
            standardized_changes
        )
    else:
        raw_changes = standardized_changes

    return pd.DataFrame(
        raw_changes,
        columns=loadings.index,
    )


def simulate_curve_scenarios(
    factor_scores: pd.DataFrame,
    loadings: pd.DataFrame,
    explained_variance: pd.DataFrame,
    scaler=None,
    n_scenarios: int = 5000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate PCA factor scenarios and reconstruct
    corresponding yield-curve shocks.
    """

    factor_scenarios = simulate_pca_scenarios(
        factor_scores=factor_scores,
        loadings=loadings,
        explained_variance=explained_variance,
        n_scenarios=n_scenarios,
        seed=seed,
    )

    curve_changes = reconstruct_curve_changes(
        factor_scenarios=factor_scenarios,
        loadings=loadings,
        scaler=scaler,
    )

    return (
        factor_scenarios,
        curve_changes,
    )