import numpy as np
import pandas as pd


KEY_RATE_BUCKETS = {
    "0-2Y": (0.0, 2.0),
    "2-5Y": (2.0, 5.0),
    "5-10Y": (5.0, 10.0),
    "10-15Y": (10.0, 15.0),
    "15-25Y": (15.0, 25.0),
}


def calculate_key_rate_dv01(
    portfolio: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate approximate DV01 by key-rate bucket.
    """

    df = portfolio.copy()

    df["DV01_INR"] = (
        df["MarketValue_INR"]
        * df["ModifiedDuration"]
        / 10_000
    )

    result = (
        df.groupby("KeyRateBucket")
        .agg(
            MarketValue_INR=(
                "MarketValue_INR",
                "sum",
            ),
            DV01_INR=(
                "DV01_INR",
                "sum",
            ),
            AverageDuration=(
                "ModifiedDuration",
                "mean",
            ),
            AverageConvexity=(
                "Convexity",
                "mean",
            ),
        )
        .reset_index()
    )

    result["DV01_Percentage"] = (
        result["DV01_INR"]
        / result["DV01_INR"].sum()
    )

    return result.sort_values(
        "DV01_INR",
        ascending=False,
    )


def create_key_rate_shock(
    bucket: str,
    shock_bps: float,
) -> dict:
    """
    Create a localized key-rate shock.
    """

    if bucket not in KEY_RATE_BUCKETS:
        raise ValueError(
            f"Unknown key-rate bucket: {bucket}"
        )

    return {
        "bucket": bucket,
        "shock_bps": shock_bps,
    }


def apply_key_rate_shock(
    portfolio: pd.DataFrame,
    bucket: str,
    shock_bps: float,
) -> pd.DataFrame:
    """
    Estimate bond-level P&L from a localized
    key-rate shock.
    """

    df = portfolio.copy()

    df["Shock_bps"] = np.where(
        df["KeyRateBucket"] == bucket,
        shock_bps,
        0.0,
    )

    delta_y = (
        df["Shock_bps"] / 10_000
    )

    df["Duration_PnL_INR"] = (
        -df["MarketValue_INR"]
        * df["ModifiedDuration"]
        * delta_y
    )

    df["Convexity_PnL_INR"] = (
        0.5
        * df["MarketValue_INR"]
        * df["Convexity"]
        * delta_y**2
    )

    df["Total_PnL_INR"] = (
        df["Duration_PnL_INR"]
        + df["Convexity_PnL_INR"]
    )

    return df