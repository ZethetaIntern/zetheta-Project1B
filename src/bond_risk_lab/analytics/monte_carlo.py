import numpy as np
import pandas as pd


def calculate_var(
    pnl: pd.Series,
    confidence: float = 0.99,
) -> float:
    """
    Historical/Monte Carlo VaR.

    Returns VaR as a positive loss number.
    """
    percentile = np.percentile(
        pnl,
        (1 - confidence) * 100,
    )

    return max(0.0, -float(percentile))


def calculate_expected_shortfall(
    pnl: pd.Series,
    confidence: float = 0.99,
) -> float:
    """
    Expected Shortfall / Conditional VaR.
    """
    cutoff = np.percentile(
        pnl,
        (1 - confidence) * 100,
    )

    tail = pnl[pnl <= cutoff]

    if len(tail) == 0:
        return 0.0

    return max(0.0, -float(tail.mean()))


def monte_carlo_summary(
    scenarios: pd.DataFrame,
    confidence_levels=(0.95, 0.99),
) -> dict:

    pnl = scenarios["PnL_Total_INR"]

    result = {
        "scenario_count": int(len(scenarios)),
        "mean_pnl_inr": float(pnl.mean()),
        "median_pnl_inr": float(pnl.median()),
        "std_pnl_inr": float(pnl.std()),
        "min_pnl_inr": float(pnl.min()),
        "max_pnl_inr": float(pnl.max()),
    }

    for confidence in confidence_levels:
        key = int(confidence * 100)

        result[f"var_{key}_inr"] = calculate_var(
            pnl,
            confidence,
        )

        result[f"es_{key}_inr"] = calculate_expected_shortfall(
            pnl,
            confidence,
        )

    return result


def scenario_factor_correlation(
    scenarios: pd.DataFrame,
) -> pd.DataFrame:

    columns = [
        "ParallelShift_bps",
        "TwistFactor_bps",
        "ButterflyFactor_bps",
        "PnL_Total_INR",
    ]

    return scenarios[columns].corr()