from pprint import pprint

from bond_risk_lab.data.loaders import (
    load_bond_portfolio,
    load_monte_carlo_scenarios,
    load_yield_curve_history,
)

from bond_risk_lab.analytics.portfolio import (
    analyze_portfolio,
)

from bond_risk_lab.analytics.monte_carlo import (
    monte_carlo_summary,
    scenario_factor_correlation,
)

from bond_risk_lab.analytics.risk_engine import (
    portfolio_parallel_shock,
)

from bond_risk_lab.analytics.yield_curve import (
    curve_factor_summary,
)


def main():

    print("=" * 70)
    print("ZETHETA BOND RISK LAB")
    print("Project 1B - Data Analyst Convexity Sensitivity AI Agent")
    print("=" * 70)

    portfolio = load_bond_portfolio()
    scenarios = load_monte_carlo_scenarios()
    curve = load_yield_curve_history()

    print(f"\nPortfolio bonds: {len(portfolio):,}")
    print(f"Monte Carlo scenarios: {len(scenarios):,}")
    print(f"Yield curve observations: {len(curve):,}")

    print("\nPORTFOLIO RISK SUMMARY")
    print("-" * 70)

    analysis = analyze_portfolio(portfolio)

    pprint(analysis["summary"])

    print("\nPARALLEL SHOCK ANALYSIS")
    print("-" * 70)

    for shock in [-200, -100, -50, 50, 100, 200]:

        result = portfolio_parallel_shock(
            portfolio,
            shock,
        )

        print(
            f"{shock:+4d} bps | "
            f"Duration P&L: "
            f"{result['duration_pnl_inr']:,.2f} | "
            f"Duration+Convexity P&L: "
            f"{result['duration_convexity_pnl_inr']:,.2f}"
        )

    print("\nMONTE CARLO SUMMARY")
    print("-" * 70)

    mc_summary = monte_carlo_summary(
        scenarios
    )

    pprint(mc_summary)

    print("\nSCENARIO FACTOR CORRELATIONS")
    print("-" * 70)

    print(
        scenario_factor_correlation(
            scenarios
        ).round(4)
    )

    print("\nLATEST YIELD CURVE")
    print("-" * 70)

    print(
        curve_factor_summary(curve).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
    