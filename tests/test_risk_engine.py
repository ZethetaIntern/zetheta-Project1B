import numpy as np

from bond_risk_lab.data.loaders import (
    load_bond_portfolio,
    load_yield_curve_history,
)
from bond_risk_lab.analytics.curve_risk import analyze_yield_curve
from bond_risk_lab.analytics.pca_monte_carlo import simulate_curve_scenarios
from bond_risk_lab.analytics.scenario_pricing import price_portfolio_scenarios
from bond_risk_lab.analytics.risk_attribution import (
    attribute_scenario_pnl,
    bucket_risk_attribution,
)


def build_test_scenarios(n_scenarios=100):
    portfolio = load_bond_portfolio()
    curve = load_yield_curve_history()

    analysis = analyze_yield_curve(curve)
    pca = analysis["pca"]

    _, curve_changes = simulate_curve_scenarios(
        pca["factor_scores"],
        pca["loadings"],
        pca["explained_variance"],
        pca["scaler"],
        n_scenarios,
        42,
    )

    return portfolio, curve_changes


def test_portfolio_and_curve_load():
    portfolio = load_bond_portfolio()
    curve = load_yield_curve_history()

    assert len(portfolio) == 300
    assert len(curve) == 264
    assert "MarketValue_INR" in portfolio.columns
    assert "KeyRateBucket" in portfolio.columns
    assert "Yield" in curve.columns


def test_pca_structure():
    curve = load_yield_curve_history()
    analysis = analyze_yield_curve(curve)
    pca = analysis["pca"]

    assert pca["explained_variance"].shape == (3, 2)
    assert pca["loadings"].shape[1] == 3

    # PCA factor scores are based on changes between observation dates,
    # so one fewer observation is expected than the number of curve dates.
    unique_dates = curve["CurveDate"].nunique()
    assert len(pca["factor_scores"]) == unique_dates - 1
    assert len(pca["factor_scores"]) > 0

    # Three principal components should be available.
    assert list(pca["factor_scores"].columns) == ["PC1", "PC2", "PC3"]


def test_scenario_pricing_output():
    portfolio, curve_changes = build_test_scenarios(100)

    scenarios = price_portfolio_scenarios(portfolio, curve_changes)

    assert len(scenarios) == 100
    assert "ScenarioID" in scenarios.columns
    assert "PnL_Total_INR" in scenarios.columns
    assert scenarios["PnL_Total_INR"].notna().all()
    assert np.isfinite(scenarios["PnL_Total_INR"]).all()


def test_risk_attribution_output():
    portfolio, curve_changes = build_test_scenarios(100)

    attribution = attribute_scenario_pnl(portfolio, curve_changes)
    bucket = bucket_risk_attribution(portfolio, curve_changes)

    assert len(attribution) == 100
    assert len(bucket) == 9

    assert "ScenarioID" in attribution.columns
    assert "DurationPnL_INR" in attribution.columns
    assert "ConvexityPnL_INR" in attribution.columns
    assert "TotalPnL_INR" in attribution.columns
    assert "KeyRateBucket" in bucket.columns

    expected_buckets = {
        "0-1Y",
        "1-2Y",
        "2-3Y",
        "3-5Y",
        "5-7Y",
        "7-10Y",
        "10-15Y",
        "15-20Y",
        "20Y+",
    }

    assert set(bucket["KeyRateBucket"]) == expected_buckets


def test_scenario_pnl_has_both_wins_and_losses():
    portfolio, curve_changes = build_test_scenarios(500)

    scenarios = price_portfolio_scenarios(portfolio, curve_changes)
    pnl = scenarios["PnL_Total_INR"]

    assert pnl.min() < 0
    assert pnl.max() > 0
