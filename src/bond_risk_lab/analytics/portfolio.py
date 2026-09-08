import pandas as pd

from .risk_engine import (
    calculate_portfolio_risk,
    risk_summary,
)


def analyze_portfolio(df: pd.DataFrame) -> dict:
    """Run the core portfolio risk analysis."""

    enriched = calculate_portfolio_risk(df)

    summary = risk_summary(enriched)

    sector_risk = (
        enriched.groupby("Sector", as_index=False)
        .agg(
            MarketValue_INR=("MarketValue_INR", "sum"),
            DV01_INR=("Calculated_DV01_INR", "sum"),
            Duration=("ModifiedDuration", "mean"),
            Convexity=("Convexity", "mean"),
        )
        .sort_values("MarketValue_INR", ascending=False)
    )

    tenor_risk = (
        enriched.groupby("KeyRateBucket", as_index=False)
        .agg(
            MarketValue_INR=("MarketValue_INR", "sum"),
            DV01_INR=("Calculated_DV01_INR", "sum"),
            AverageDuration=("ModifiedDuration", "mean"),
            AverageConvexity=("Convexity", "mean"),
        )
        .sort_values("MarketValue_INR", ascending=False)
    )

    return {
        "portfolio": enriched,
        "summary": summary,
        "sector_risk": sector_risk,
        "tenor_risk": tenor_risk,
    }