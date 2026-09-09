from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_duration_pnl(
    market_value,
    duration,
    yield_change,
):
    """
    Calculate the first-order duration P&L contribution.

    Formula:

        Duration P&L = MV × (-Duration × Δy)

    Parameters
    ----------
    market_value:
        Bond market value.

    duration:
        Modified duration.

    yield_change:
        Yield change in decimal form.

        +100 bps = 0.01
        -50 bps  = -0.005
    """

    market_value = np.asarray(
        market_value,
        dtype=float,
    )

    duration = np.asarray(
        duration,
        dtype=float,
    )

    yield_change = np.asarray(
        yield_change,
        dtype=float,
    )

    return (
        market_value
        * -duration
        * yield_change
    )


def calculate_convexity_pnl(
    market_value,
    convexity,
    yield_change,
):
    """
    Calculate the second-order convexity P&L contribution.

    Formula:

        Convexity P&L =
            MV × 0.5 × Convexity × (Δy)^2
    """

    market_value = np.asarray(
        market_value,
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

    return (
        market_value
        * 0.5
        * convexity
        * yield_change**2
    )


def calculate_total_duration_convexity_pnl(
    market_value,
    duration,
    convexity,
    yield_change,
):
    """
    Calculate total duration-plus-convexity P&L.
    """

    duration_pnl = calculate_duration_pnl(
        market_value=market_value,
        duration=duration,
        yield_change=yield_change,
    )

    convexity_pnl = calculate_convexity_pnl(
        market_value=market_value,
        convexity=convexity,
        yield_change=yield_change,
    )

    return duration_pnl + convexity_pnl


def attribute_scenario_pnl(
    portfolio: pd.DataFrame,
    curve_changes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attribute each simulated scenario P&L into:

        1. Duration contribution
        2. Convexity contribution
        3. Total P&L

    Bonds are mapped to representative key-rate tenors.
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

    bucket_tenors = {
        "0-2Y": 1.0,
        "2-5Y": 3.0,
        "5-10Y": 7.0,
        "10-15Y": 12.0,
        "15-25Y": 20.0,
    }

    available_tenors = np.asarray(
        curve_changes.columns,
        dtype=float,
    )

    duration_pnl = np.zeros(
        len(curve_changes),
        dtype=float,
    )

    convexity_pnl = np.zeros(
        len(curve_changes),
        dtype=float,
    )

    bucket_contributions = {}

    for _, bond in portfolio.iterrows():

        bucket = bond["KeyRateBucket"]

        target_tenor = bucket_tenors.get(
            bucket,
            5.0,
        )

        nearest_index = np.argmin(
            np.abs(
                available_tenors
                - target_tenor
            )
        )

        nearest_tenor = (
            available_tenors[
                nearest_index
            ]
        )

        yield_change = (
            curve_changes[
                nearest_tenor
            ]
            .to_numpy(
                dtype=float
            )
        )

        bond_duration_pnl = (
            calculate_duration_pnl(
                market_value=
                    bond["MarketValue_INR"],
                duration=
                    bond["ModifiedDuration"],
                yield_change=
                    yield_change,
            )
        )

        bond_convexity_pnl = (
            calculate_convexity_pnl(
                market_value=
                    bond["MarketValue_INR"],
                convexity=
                    bond["Convexity"],
                yield_change=
                    yield_change,
            )
        )

        duration_pnl += (
            bond_duration_pnl
        )

        convexity_pnl += (
            bond_convexity_pnl
        )

        if bucket not in bucket_contributions:
            bucket_contributions[bucket] = {
                "DurationPnL_INR":
                    np.zeros(
                        len(curve_changes),
                        dtype=float,
                    ),
                "ConvexityPnL_INR":
                    np.zeros(
                        len(curve_changes),
                        dtype=float,
                    ),
            }

        bucket_contributions[bucket][
            "DurationPnL_INR"
        ] += bond_duration_pnl

        bucket_contributions[bucket][
            "ConvexityPnL_INR"
        ] += bond_convexity_pnl

    result = pd.DataFrame(
        {
            "ScenarioID": np.arange(
                1,
                len(curve_changes) + 1,
            ),
            "DurationPnL_INR":
                duration_pnl,
            "ConvexityPnL_INR":
                convexity_pnl,
        }
    )

    result["TotalPnL_INR"] = (
        result["DurationPnL_INR"]
        + result["ConvexityPnL_INR"]
    )

    return result


def bucket_risk_attribution(
    portfolio: pd.DataFrame,
    curve_changes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate average absolute risk contribution
    by key-rate bucket.
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

    bucket_tenors = {
        "0-2Y": 1.0,
        "2-5Y": 3.0,
        "5-10Y": 7.0,
        "10-15Y": 12.0,
        "15-25Y": 20.0,
    }

    available_tenors = np.asarray(
        curve_changes.columns,
        dtype=float,
    )

    records = []

    for bucket, group in portfolio.groupby(
        "KeyRateBucket"
    ):
        bucket_label = bucket
        if isinstance(bucket_label, bytes):
            bucket_label = bucket_label.decode(
                "utf-8"
            )
        elif not isinstance(bucket_label, str):
            bucket_label = str(bucket_label)

        target_tenor = bucket_tenors.get(
            bucket_label,
            5.0,
        )
        nearest_index = np.argmin(
            np.abs(
                available_tenors
                - target_tenor
            )
        )

        nearest_tenor = (
            available_tenors[
                nearest_index
            ]
        )

        yield_changes = (
            curve_changes[
                nearest_tenor
            ]
            .to_numpy(
                dtype=float
            )
        )

        bucket_duration = np.zeros(
            len(yield_changes),
            dtype=float,
        )

        bucket_convexity = np.zeros(
            len(yield_changes),
            dtype=float,
        )

        for _, bond in group.iterrows():

            bucket_duration += (
                calculate_duration_pnl(
                    market_value=
                        bond["MarketValue_INR"],
                    duration=
                        bond["ModifiedDuration"],
                    yield_change=
                        yield_changes,
                )
            )

            bucket_convexity += (
                calculate_convexity_pnl(
                    market_value=
                        bond["MarketValue_INR"],
                    convexity=
                        bond["Convexity"],
                    yield_change=
                        yield_changes,
                )
            )

        total = (
            bucket_duration
            + bucket_convexity
        )

        records.append(
            {
                "KeyRateBucket": bucket,
                "MarketValue_INR":
                    float(
                        group[
                            "MarketValue_INR"
                        ].sum()
                    ),
                "AverageDurationPnL_INR":
                    float(
                        bucket_duration.mean()
                    ),
                "AverageConvexityPnL_INR":
                    float(
                        bucket_convexity.mean()
                    ),
                "AverageTotalPnL_INR":
                    float(
                        total.mean()
                    ),
                "AverageAbsolutePnL_INR":
                    float(
                        np.abs(total).mean()
                    ),
                "WorstPnL_INR":
                    float(
                        total.min()
                    ),
                "BestPnL_INR":
                    float(
                        total.max()
                    ),
            }
        )

    result = pd.DataFrame(records)

    return result.sort_values(
        "AverageAbsolutePnL_INR",
        ascending=False,
    ).reset_index(
        drop=True
    )


def calculate_portfolio_risk_contributions(
    portfolio: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate static portfolio contributions from
    market value, duration and convexity.
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

    result = portfolio.copy()

    result["DV01_INR"] = (
        result["MarketValue_INR"]
        * result["ModifiedDuration"]
        * 0.0001
    )

    result["ConvexityExposure"] = (
        result["MarketValue_INR"]
        * result["Convexity"]
    )

    total_market_value = (
        result["MarketValue_INR"].sum()
    )

    total_dv01 = (
        result["DV01_INR"].sum()
    )

    result["MarketValueWeight"] = (
        result["MarketValue_INR"]
        / total_market_value
    )

    result["DV01Weight"] = (
        result["DV01_INR"]
        / total_dv01
    )

    return result
