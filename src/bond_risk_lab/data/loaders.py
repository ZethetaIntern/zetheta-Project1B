from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_bond_portfolio() -> pd.DataFrame:
    """Load and lightly validate the bond portfolio dataset."""
    path = RAW_DATA_DIR / "bond_portfolio_data.csv"

    df = pd.read_csv(path)

    required_columns = {
        "BondID",
        "Issuer",
        "Sector",
        "CreditRating",
        "YieldToMaturity",
        "ModifiedDuration",
        "Convexity",
        "MarketValue_INR",
        "PortfolioWeight",
        "KeyRateBucket",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Bond portfolio is missing required columns: {sorted(missing)}"
        )

    return df


def load_monte_carlo_scenarios() -> pd.DataFrame:
    """Load Monte Carlo scenario data."""
    path = RAW_DATA_DIR / "monte_carlo_scenarios.csv"

    df = pd.read_csv(path)

    required_columns = {
        "ScenarioID",
        "ParallelShift_bps",
        "TwistFactor_bps",
        "ButterflyFactor_bps",
        "PnL_Total_INR",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Monte Carlo dataset is missing required columns: {sorted(missing)}"
        )

    return df


def load_yield_curve_history() -> pd.DataFrame:
    """Load historical yield curve observations."""
    path = RAW_DATA_DIR / "yield_curve_history.csv"

    df = pd.read_csv(path)

    required_columns = {
        "CurveDate",
        "Currency",
        "Tenor_Years",
        "TenorLabel",
        "Yield",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Yield curve dataset is missing required columns: {sorted(missing)}"
        )

    df["CurveDate"] = pd.to_datetime(df["CurveDate"])

    return df