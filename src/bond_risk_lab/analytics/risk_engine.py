import numpy as np
import pandas as pd


BPS = 10_000


def duration_price_change(
    market_value,
    modified_duration,
    yield_change_bps,
):
    """
    Estimate price/P&L impact using modified duration.

    ΔP/P ≈ -Duration × Δy
    """
    delta_y = np.asarray(yield_change_bps) / BPS

    return -np.asarray(market_value) * np.asarray(modified_duration) * delta_y


def convexity_price_change(
    market_value,
    modified_duration,
    convexity,
    yield_change_bps,
):
    """
    Estimate price/P&L impact using duration + convexity.

    ΔP/P ≈ -D×Δy + 0.5×C×(Δy)^2
    """
    delta_y = np.asarray(yield_change_bps) / BPS

    duration_effect = (
        -np.asarray(modified_duration) * delta_y
    )

    convexity_effect = (
        0.5 * np.asarray(convexity) * delta_y**2
    )

    return np.asarray(market_value) * (
        duration_effect + convexity_effect
    )


def calculate_dv01(
    market_value,
    modified_duration,
):
    """
    Approximate DV01 from market value and modified duration.
    """
    return (
        np.asarray(market_value)
        * np.asarray(modified_duration)
        / BPS
    )


def calculate_portfolio_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Add standard bond-risk metrics to the portfolio."""

    result = df.copy()

    result["Calculated_DV01_INR"] = (
        result["MarketValue_INR"]
        * result["ModifiedDuration"]
        / BPS
    )

    result["DurationRisk_INR"] = (
        result["MarketValue_INR"]
        * result["ModifiedDuration"]
        / BPS
    )

    result["ConvexityRisk_INR"] = (
        result["MarketValue_INR"]
        * result["Convexity"]
        / (2 * BPS**2)
    )

    return result


def portfolio_parallel_shock(
    df: pd.DataFrame,
    shock_bps: float,
) -> dict:
    """
    Estimate portfolio P&L under a parallel yield-curve shock.
    """

    duration_pnl = duration_price_change(
        df["MarketValue_INR"],
        df["ModifiedDuration"],
        shock_bps,
    )

    total_pnl = convexity_price_change(
        df["MarketValue_INR"],
        df["ModifiedDuration"],
        df["Convexity"],
        shock_bps,
    )

    return {
        "shock_bps": shock_bps,
        "duration_pnl_inr": float(np.sum(duration_pnl)),
        "duration_convexity_pnl_inr": float(np.sum(total_pnl)),
        "convexity_adjustment_inr": float(
            np.sum(total_pnl - duration_pnl)
        ),
    }


def risk_summary(df: pd.DataFrame) -> dict:
    """Produce portfolio-level risk statistics."""

    market_value = df["MarketValue_INR"].sum()

    weighted_duration = (
        df["MarketValue_INR"] * df["ModifiedDuration"]
    ).sum() / market_value

    weighted_convexity = (
        df["MarketValue_INR"] * df["Convexity"]
    ).sum() / market_value

    dv01 = (
        df["MarketValue_INR"] * df["ModifiedDuration"] / BPS
    ).sum()

    return {
        "market_value_inr": float(market_value),
        "modified_duration": float(weighted_duration),
        "convexity": float(weighted_convexity),
        "dv01_inr_per_bp": float(dv01),
        "bond_count": int(len(df)),
    }