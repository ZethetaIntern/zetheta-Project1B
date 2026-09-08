from __future__ import annotations

import numpy as np
import pandas as pd


BUCKET_TENORS = {
    "0-2Y": 1.0,
    "2-5Y": 3.0,
    "5-10Y": 7.0,
    "10-15Y": 12.0,
    "15-25Y": 20.0,
}


def calculate_bond_scenario_pnl(
    market_value,
    duration,
    convexity,
    yield_change,
):
    """
    Calculate bond price P&L using duration and convexity.

    Approximation:

        ΔP ≈ MV × [-D × Δy + 0.5 × C × (Δy)^2]

    Parameters
    ----------
    market_value:
        Bond market value.

    duration:
        Modified duration.

    convexity:
        Bond convexity.

    yield_change:
        Yield change in decimal form.

        Example:
            +100 bps = 0.01
            -50 bps  = -0.005

    Returns
    -------
    numpy.ndarray
        Scenario P&L.
    """

    market_value = np.asarray(
        market_value,
        dtype=float,
    )

    duration = np.asarray(
        duration,
        dtype=float,
    )

    convexity = np.asarray(
        convexity,
        dtype=float,
    )

    yield_change = np.asarray(
        yield_change,
        dtype=float,
    )

    return market_value * (
        -duration * yield_change
        + 0.5 * convexity * yield_change**2
    )


def get_representative_tenor(
    key_rate_bucket: str,
) -> float:
    """
    Return the representative tenor for a key-rate bucket.
    """

    return BUCKET_TENORS.get(
        key_rate_bucket,
        5.0,
    )


def find_nearest_tenor(
    target_tenor: float,
    available_tenors,
) -> float:
    """
    Find the simulated curve tenor closest to target tenor.
    """

    available_tenors = np.asarray(
        available_tenors,
        dtype=float,
    )

    if len(available_tenors) == 0:
        raise ValueError(
            "No curve tenors are available."
        )

    index = np.argmin(
        np.abs(
            available_tenors
            - target_tenor
        )
    )

    return float(
        available_tenors[index]
    )


def price_portfolio_scenarios(
    portfolio: pd.DataFrame,
    curve_changes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reprice the entire bond portfolio under simulated
    yield-curve scenarios.

    Each bond is assigned to a representative tenor based
    on its KeyRateBucket.

    Duration and convexity are then used to estimate the
    bond's scenario P&L.
    """

    required_columns = {
        "MarketValue_INR",
        "ModifiedDuration",
        "Convexity",
        "KeyRateBucket",
    }

    missing_columns = (
        required_columns
        - set(portfolio.columns)
    )

    if missing_columns:
        raise ValueError(
            "Portfolio is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if curve_changes.empty:
        raise ValueError(
            "curve_changes cannot be empty."
        )

    pnl = np.zeros(
        len(curve_changes),
        dtype=float,
    )

    available_tenors = np.asarray(
        curve_changes.columns,
        dtype=float,
    )

    for _, bond in portfolio.iterrows():

        target_tenor = (
            get_representative_tenor(
                bond["KeyRateBucket"]
            )
        )

        nearest_tenor = (
            find_nearest_tenor(
                target_tenor,
                available_tenors,
            )
        )

        yield_change = (
            curve_changes[
                nearest_tenor
            ]
            .to_numpy(
                dtype=float
            )
        )

        bond_pnl = (
            calculate_bond_scenario_pnl(
                market_value=
                    bond["MarketValue_INR"],
                duration=
                    bond["ModifiedDuration"],
                convexity=
                    bond["Convexity"],
                yield_change=
                    yield_change,
            )
        )

        pnl += bond_pnl

    return pd.DataFrame(
        {
            "ScenarioID": np.arange(
                1,
                len(curve_changes) + 1,
            ),
            "PnL_Total_INR": pnl,
        }
    )


def scenario_pnl_summary(
    scenario_pnl: pd.DataFrame,
) -> dict:
    """
    Generate basic descriptive statistics for
    simulated portfolio P&L.
    """

    if "PnL_Total_INR" not in scenario_pnl.columns:
        raise ValueError(
            "scenario_pnl must contain "
            "'PnL_Total_INR'."
        )

    pnl = scenario_pnl[
        "PnL_Total_INR"
    ]

    return {
        "scenario_count": int(len(pnl)),
        "mean_pnl_inr": float(pnl.mean()),
        "std_pnl_inr": float(pnl.std()),
        "minimum_pnl_inr": float(pnl.min()),
        "maximum_pnl_inr": float(pnl.max()),
        "median_pnl_inr": float(pnl.median()),
    }