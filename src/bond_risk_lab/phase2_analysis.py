from bond_risk_lab.data.loaders import (
    load_bond_portfolio,
    load_yield_curve_history,
)

from bond_risk_lab.analytics.key_rate import (
    calculate_key_rate_dv01,
    apply_key_rate_shock,
)

from bond_risk_lab.analytics.curve_risk import (
    analyze_yield_curve,
)


def main():

    print("=" * 70)
    print("ZETHETA BOND RISK LAB")
    print("PHASE 2 - KEY-RATE & PCA CURVE ANALYTICS")
    print("=" * 70)

    portfolio = load_bond_portfolio()
    curve = load_yield_curve_history()

    # --------------------------------------------------------
    # KEY-RATE RISK
    # --------------------------------------------------------

    print("\nKEY-RATE DV01")
    print("-" * 70)

    key_rate = calculate_key_rate_dv01(
        portfolio
    )

    print(
        key_rate.to_string(
            index=False,
            float_format=lambda x: f"{x:,.4f}",
        )
    )

    # --------------------------------------------------------
    # KEY-RATE STRESS
    # --------------------------------------------------------

    print("\nKEY-RATE STRESS TEST")
    print("-" * 70)

    for bucket in [
        "0-2Y",
        "2-5Y",
        "5-10Y",
        "10-15Y",
        "15-25Y",
    ]:

        stressed = apply_key_rate_shock(
            portfolio,
            bucket,
            100,
        )

        pnl = stressed[
            "Total_PnL_INR"
        ].sum()

        print(
            f"{bucket:>8} +100 bps | "
            f"P&L: ₹{pnl:,.2f}"
        )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    print("\nYIELD CURVE PCA")
    print("-" * 70)

    result = analyze_yield_curve(
        curve,
        currency="INR",
    )

    explained = result[
        "pca"
    ]["explained_variance"]

    print(
        explained.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nPCA LOADINGS")
    print("-" * 70)

    loadings = result[
        "pca"
    ]["loadings"]

    print(
        loadings.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )


if __name__ == "__main__":
    main()