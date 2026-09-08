from bond_risk_lab.data.loaders import (
    load_bond_portfolio,
    load_yield_curve_history,
)

from bond_risk_lab.analytics.curve_risk import (
    analyze_yield_curve,
)

from bond_risk_lab.analytics.pca_monte_carlo import (
    simulate_curve_scenarios,
)

from bond_risk_lab.analytics.scenario_pricing import (
    price_portfolio_scenarios,
)

from bond_risk_lab.analytics.monte_carlo import (
    monte_carlo_summary,
)


def main():

    print("=" * 70)
    print("ZHETA BOND RISK LAB - PHASE 3")
    print("PCA-BASED YIELD CURVE MONTE CARLO")
    print("=" * 70)

    portfolio = load_bond_portfolio()
    curve_history = load_yield_curve_history()

    print("\nDATA")
    print(f"Portfolio bonds: {len(portfolio):,}")
    print(f"Yield curve observations: {len(curve_history):,}")

    result = analyze_yield_curve(curve_history)

    curve_matrix = result["curve_matrix"]
    pca_results = result["pca"]

    pca_model = pca_results["model"]
    scaler = pca_results["scaler"]
    loadings = pca_results["loadings"]
    factor_scores = pca_results["factor_scores"]
    explained_variance = pca_results["explained_variance"]

    print("\n" + "=" * 70)
    print("PCA EXPLAINED VARIANCE")
    print("=" * 70)

    print(
        explained_variance.to_string(
            index=False,
            formatters={
                "ExplainedVariance": "{:.4f}".format
            },
        )
    )

    cumulative_variance = (
        explained_variance["ExplainedVariance"].cumsum()
    )

    print("\nCumulative variance explained:")

    for component, value in zip(
        explained_variance["Component"],
        cumulative_variance,
    ):
        print(f"{component}: {value:.2%}")

    print("\n" + "=" * 70)
    print("PCA LOADINGS")
    print("=" * 70)

    print(
        loadings.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    factor_scenarios, curve_changes = simulate_curve_scenarios(
        factor_scores=factor_scores,
        loadings=loadings,
        explained_variance=explained_variance,
        scaler=scaler,
        n_scenarios=5000,
        seed=42,
    )

    print("\n" + "=" * 70)
    print("PCA FACTOR SCENARIOS")
    print("=" * 70)

    print(
        factor_scenarios.describe().to_string(
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print("\n" + "=" * 70)
    print("SIMULATED CURVE SHOCKS")
    print("=" * 70)

    shock_summary = curve_changes.describe().T

    shock_summary["Mean_bps"] = (
        shock_summary["mean"] * 10000
    )

    shock_summary["Std_bps"] = (
        shock_summary["std"] * 10000
    )

    shock_summary["Min_bps"] = (
        shock_summary["min"] * 10000
    )

    shock_summary["Max_bps"] = (
        shock_summary["max"] * 10000
    )

    print(
        shock_summary[
            [
                "Mean_bps",
                "Std_bps",
                "Min_bps",
                "Max_bps",
            ]
        ].to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    scenario_pnl = price_portfolio_scenarios(
        portfolio=portfolio,
        curve_changes=curve_changes,
    )

    print("\n" + "=" * 70)
    print("PCA MONTE CARLO P&L")
    print("=" * 70)

    pnl_stats = scenario_pnl["PnL_Total_INR"].describe()

    print(
        pnl_stats.to_string(
            float_format=lambda x: f"{x:,.2f}"
        )
    )

    summary = monte_carlo_summary(
        scenario_pnl,
        confidence_levels=(0.95, 0.99),
    )

    print("\n" + "=" * 70)
    print("PCA MONTE CARLO RISK")
    print("=" * 70)

    if isinstance(summary, dict):
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:,.2f}")
            else:
                print(f"{key}: {value}")

    else:
        print(summary)

    print("\n" + "=" * 70)
    print("WORST 10 SCENARIOS")
    print("=" * 70)

    worst_scenarios = scenario_pnl.nsmallest(
        10,
        "PnL_Total_INR",
    )

    print(
        worst_scenarios.to_string(
            index=False,
            float_format=lambda x: f"{x:,.2f}"
        )
    )

    print("\n" + "=" * 70)
    print("BEST 10 SCENARIOS")
    print("=" * 70)

    best_scenarios = scenario_pnl.nlargest(
        10,
        "PnL_Total_INR",
    )

    print(
        best_scenarios.to_string(
            index=False,
            float_format=lambda x: f"{x:,.2f}"
        )
    )

    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)

    print(
        f"PCA scenarios: {len(scenario_pnl):,}"
    )

    print(
        f"Portfolio market value: "
        f"₹{portfolio['MarketValue_INR'].sum():,.2f}"
    )

    print(
        f"Mean simulated P&L: "
        f"₹{scenario_pnl['PnL_Total_INR'].mean():,.2f}"
    )

    print(
        f"Worst simulated P&L: "
        f"₹{scenario_pnl['PnL_Total_INR'].min():,.2f}"
    )

    print(
        f"Best simulated P&L: "
        f"₹{scenario_pnl['PnL_Total_INR'].max():,.2f}"
    )


if __name__ == "__main__":
    main()