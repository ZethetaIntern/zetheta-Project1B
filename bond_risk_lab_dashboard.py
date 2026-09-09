from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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

st.set_page_config(
    page_title="Ztheta Bond Risk Lab",
    page_icon="Z",
    layout="wide",
)

st.title("Ztheta Bond Risk Lab")
st.caption(
    "Data Analyst Convexity Sensitivity AI Agent - "
    "portfolio duration, convexity, PCA, Monte Carlo and risk analytics"
)


@st.cache_data(show_spinner=False)
def run_risk_engine(n_scenarios: int, seed: int):
    portfolio = load_bond_portfolio()
    curve_history = load_yield_curve_history()

    curve_result = analyze_yield_curve(curve_history)
    pca = curve_result["pca"]

    factor_scenarios, curve_changes = simulate_curve_scenarios(
        factor_scores=pca["factor_scores"],
        loadings=pca["loadings"],
        explained_variance=pca["explained_variance"],
        scaler=pca["scaler"],
        n_scenarios=n_scenarios,
        seed=seed,
    )

    scenario_pnl = price_portfolio_scenarios(
        portfolio=portfolio,
        curve_changes=curve_changes,
    )

    attribution = attribute_scenario_pnl(
        portfolio=portfolio,
        curve_changes=curve_changes,
    )

    bucket_attribution = bucket_risk_attribution(
        portfolio=portfolio,
        curve_changes=curve_changes,
    )

    return (
        portfolio,
        curve_history,
        curve_result,
        pca,
        factor_scenarios,
        curve_changes,
        scenario_pnl,
        attribution,
        bucket_attribution,
    )


def calculate_metrics(pnl):
    values = np.asarray(pnl, dtype=float)
    q95 = np.percentile(values, 5)
    q99 = np.percentile(values, 1)

    tail95 = values[values <= q95]
    tail99 = values[values <= q99]

    return {
        "mean": float(values.mean()),
        "volatility": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "var95": float(-q95),
        "var99": float(-q99),
        "es95": float(-tail95.mean()) if len(tail95) else 0.0,
        "es99": float(-tail99.mean()) if len(tail99) else 0.0,
    }


def money(value):
    return f"INR {float(value):,.0f}"


def find_column(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name
    return None


def safe_explained_variance(pca):
    explained = pca["explained_variance"].copy()

    if not isinstance(explained, pd.DataFrame):
        explained = pd.DataFrame(explained)

    if "ExplainedVariance" not in explained.columns:
        numeric = explained.select_dtypes(include=[np.number]).columns
        if len(numeric) == 0:
            raise ValueError("PCA explained-variance data has no numeric column.")
        explained = explained.rename(columns={numeric[0]: "ExplainedVariance"})

    if "Component" not in explained.columns:
        explained.insert(
            0,
            "Component",
            [f"PC{i + 1}" for i in range(len(explained))],
        )

    explained["ExplainedVariance"] = pd.to_numeric(
        explained["ExplainedVariance"],
        errors="coerce",
    )

    return explained.dropna(subset=["ExplainedVariance"]).reset_index(drop=True)


def normalize_curve_history(curve_history):
    curve = curve_history.copy()

    date_col = find_column(
        curve,
        ["CurveDate", "Date", "date", "DATE"],
    )
    tenor_col = find_column(
        curve,
        ["TenorLabel", "Tenor", "tenor"],
    )
    yield_col = find_column(
        curve,
        ["Yield", "yield", "YieldRate"],
    )

    if not date_col or not tenor_col or not yield_col:
        raise ValueError(
            "Yield curve data must contain CurveDate, TenorLabel and Yield."
        )

    curve[date_col] = pd.to_datetime(curve[date_col], errors="coerce")
    curve[yield_col] = pd.to_numeric(curve[yield_col], errors="coerce")

    curve = curve.dropna(subset=[date_col, tenor_col, yield_col]).copy()

    return curve, date_col, tenor_col, yield_col


def normalize_bucket_attribution(bucket):
    result = bucket.copy()

    bucket_col = find_column(
        result,
        ["KeyRateBucket", "Bucket", "key_rate_bucket"],
    )

    if bucket_col and bucket_col != "KeyRateBucket":
        result = result.rename(columns={bucket_col: "KeyRateBucket"})

    return result


with st.sidebar:
    st.header("Risk Lab Controls")

    scenario_count = st.selectbox(
        "Monte Carlo scenarios",
        [1000, 2500, 5000, 10000],
        index=2,
    )

    seed = st.number_input(
        "Simulation seed",
        min_value=1,
        max_value=999999,
        value=42,
        step=1,
    )

    st.divider()

    st.markdown("### Analytics Engine")
    st.write("Yield curve analytics")
    st.write("PCA factor analysis")
    st.write("Monte Carlo scenarios")
    st.write("Duration and convexity")
    st.write("Key-rate attribution")
    st.write("Risk classification readiness")

with st.spinner("Running Bond Risk Lab analytics..."):
    (
        portfolio,
        curve_history,
        curve_result,
        pca,
        factor_scenarios,
        curve_changes,
        scenario_pnl,
        attribution,
        bucket_attribution,
    ) = run_risk_engine(int(scenario_count), int(seed))

pnl_values = scenario_pnl["PnL_Total_INR"].astype(float).to_numpy()
metrics = calculate_metrics(pnl_values)

market_value = float(portfolio["MarketValue_INR"].sum())
bond_count = len(portfolio)

explained = safe_explained_variance(pca)

pc1 = float(explained.iloc[0]["ExplainedVariance"])
pc2 = float(explained.iloc[1]["ExplainedVariance"]) if len(explained) > 1 else 0.0
pc3 = float(explained.iloc[2]["ExplainedVariance"]) if len(explained) > 2 else 0.0
cum3 = float(explained.iloc[:3]["ExplainedVariance"].sum())

bucket_attribution = normalize_bucket_attribution(bucket_attribution)

duration_mean = float(attribution["DurationPnL_INR"].mean())
convexity_mean = float(attribution["ConvexityPnL_INR"].mean())

# ---------------------------------------------------------------------
# KPI ROW
# ---------------------------------------------------------------------

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Portfolio Market Value", money(market_value))
c2.metric("Bonds", f"{bond_count:,}")
c3.metric("VaR 95%", money(metrics["var95"]))
c4.metric("VaR 99%", money(metrics["var99"]))
c5.metric("Expected Shortfall 95%", money(metrics["es95"]))

# ---------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Executive Risk View",
        "Yield Curve and PCA",
        "Monte Carlo",
        "Key-Rate Risk",
        "AI Risk Agent",
    ]
)

# ---------------------------------------------------------------------
# TAB 1
# ---------------------------------------------------------------------

with tab1:
    st.subheader("Executive Portfolio Risk View")

    left, right = st.columns(2)

    with left:
        st.markdown("### Portfolio Allocation")

        allocation = (
            portfolio.groupby("KeyRateBucket", as_index=False)
            .agg(MarketValue_INR=("MarketValue_INR", "sum"))
            .sort_values(by="MarketValue_INR", ascending=False)
        )

        fig = px.bar(
            allocation,
            x="KeyRateBucket",
            y="MarketValue_INR",
            title="Market Value by Maturity Bucket",
            labels={
                "MarketValue_INR": "Market Value (INR)",
                "KeyRateBucket": "Maturity Bucket",
            },
        )
        fig.update_layout(height=430)
        st.plotly_chart(fig, width="stretch")

    with right:
        st.markdown("### Portfolio Risk Distribution")

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=pnl_values,
                nbinsx=70,
                name="Scenario P&L",
            )
        )

        fig.add_vline(
            x=-metrics["var95"],
            line_dash="dash",
            annotation_text="VaR 95%",
        )
        fig.add_vline(
            x=-metrics["var99"],
            line_dash="dot",
            annotation_text="VaR 99%",
        )

        fig.update_layout(
            title="Monte Carlo Portfolio P&L",
            xaxis_title="P&L (INR)",
            yaxis_title="Scenario Count",
            height=430,
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Risk Metrics")

    metrics_table = pd.DataFrame(
        {
            "Metric": [
                "Mean P&L",
                "P&L Volatility",
                "Minimum P&L",
                "Maximum P&L",
                "VaR 95%",
                "VaR 99%",
                "Expected Shortfall 95%",
                "Expected Shortfall 99%",
            ],
            "Value": [
                money(metrics["mean"]),
                money(metrics["volatility"]),
                money(metrics["min"]),
                money(metrics["max"]),
                money(metrics["var95"]),
                money(metrics["var99"]),
                money(metrics["es95"]),
                money(metrics["es99"]),
            ],
        }
    )

    st.dataframe(metrics_table, width="stretch", hide_index=True)

# ---------------------------------------------------------------------
# TAB 2
# ---------------------------------------------------------------------

with tab2:
    st.subheader("Yield Curve Analytics and PCA")

    curve, date_col, tenor_col, yield_col = normalize_curve_history(curve_history)

    st.markdown("### Historical Yield Curve")

    fig = px.line(
        curve,
        x=date_col,
        y=yield_col,
        color=tenor_col,
        title="Historical Yield Curve by Tenor",
        labels={
            date_col: "Date",
            yield_col: "Yield",
            tenor_col: "Tenor",
        },
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, width="stretch")

    st.markdown("### PCA Explained Variance")

    pca_display = explained.copy()
    pca_display["ExplainedVariancePct"] = (
        pca_display["ExplainedVariance"] * 100
    )
    pca_display["CumulativeVariancePct"] = (
        pca_display["ExplainedVariance"].cumsum() * 100
    )

    fig = px.bar(
        pca_display,
        x="Component",
        y="ExplainedVariancePct",
        title="Principal Component Variance",
        labels={
            "ExplainedVariancePct": "Explained Variance (%)",
            "Component": "PCA Component",
        },
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, width="stretch")

    a, b, c, d = st.columns(4)
    a.metric("PC1", f"{pc1:.2%}")
    b.metric("PC2", f"{pc2:.2%}")
    c.metric("PC3", f"{pc3:.2%}")
    d.metric("PC1-PC3", f"{cum3:.2%}")

    st.markdown("### PCA Loadings")

    loadings_display = pca["loadings"].copy()

    if isinstance(loadings_display, pd.Series):
        loadings_display = loadings_display.to_frame()

    fig = px.imshow(
        loadings_display,
        aspect="auto",
        title="Yield Curve PCA Factor Loadings",
        labels={
            "x": "Principal Component",
            "y": "Tenor",
            "color": "Loading",
        },
    )
    fig.update_layout(height=550)
    st.plotly_chart(fig, width="stretch")

    st.markdown("### PCA Summary")
    st.dataframe(
        pca_display[
            ["Component", "ExplainedVariancePct", "CumulativeVariancePct"]
        ],
        width="stretch",
        hide_index=True,
    )

# ---------------------------------------------------------------------
# TAB 3
# ---------------------------------------------------------------------

with tab3:
    st.subheader("Monte Carlo Convexity Sensitivity")

    scenario_display = scenario_pnl.copy()
    scenario_display["CumulativePnL"] = (
        scenario_display["PnL_Total_INR"].cumsum()
    )

    fig = px.scatter(
        scenario_display,
        x="ScenarioID",
        y="PnL_Total_INR",
        title="Scenario P&L Distribution",
        labels={
            "ScenarioID": "Scenario",
            "PnL_Total_INR": "P&L (INR)",
        },
    )

    fig.add_hline(
        y=-metrics["var95"],
        line_dash="dash",
        annotation_text="VaR 95% Loss",
    )

    st.plotly_chart(fig, width="stretch")

    st.markdown("### Worst 20 Scenarios")

    worst = (
        scenario_pnl.sort_values("PnL_Total_INR")
        .head(20)
        .copy()
    )

    st.dataframe(worst, width="stretch", hide_index=True)

    st.markdown("### Scenario Loss Statistics")

    q = pd.DataFrame(
        {
            "Percentile": ["1%", "5%", "50%", "95%", "99%"],
            "P&L": [
                np.percentile(pnl_values, 1),
                np.percentile(pnl_values, 5),
                np.percentile(pnl_values, 50),
                np.percentile(pnl_values, 95),
                np.percentile(pnl_values, 99),
            ],
        }
    )
    q["P&L"] = q["P&L"].map(money)

    st.dataframe(q, width="stretch", hide_index=True)

    st.markdown("### Monte Carlo Scenario Inputs")

    factor_frame = pd.DataFrame(factor_scenarios)
    st.write(
        f"Generated {len(factor_frame):,} curve-factor scenarios using seed "
        f"{int(seed)}."
    )

# ---------------------------------------------------------------------
# TAB 4
# ---------------------------------------------------------------------

with tab4:
    st.subheader("Key-Rate Risk Attribution")

    bucket = bucket_attribution.copy()

    if "AverageAbsolutePnL_INR" in bucket.columns:
        bucket = bucket.sort_values(
            "AverageAbsolutePnL_INR",
            ascending=False,
        )

        fig = px.bar(
            bucket,
            x="KeyRateBucket",
            y="AverageAbsolutePnL_INR",
            title="Average Absolute Scenario P&L by Key-Rate Bucket",
            labels={
                "KeyRateBucket": "Key-Rate Bucket",
                "AverageAbsolutePnL_INR": "Average Absolute P&L (INR)",
            },
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Attribution Detail")
    st.dataframe(bucket, width="stretch", hide_index=True)

    st.markdown("### Duration vs Convexity")

    dc = pd.DataFrame(
        {
            "Component": ["Duration", "Convexity"],
            "Average P&L": [duration_mean, convexity_mean],
        }
    )

    fig = px.bar(
        dc,
        x="Component",
        y="Average P&L",
        title="Average Duration / Convexity P&L Contribution",
        labels={"Average P&L": "Average P&L (INR)"},
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Risk Contribution Summary")

    duration_abs = abs(duration_mean)
    convexity_abs = abs(convexity_mean)

    contribution = pd.DataFrame(
        {
            "Component": ["Duration", "Convexity"],
            "Absolute Average Contribution": [
                duration_abs,
                convexity_abs,
            ],
        }
    )

    st.dataframe(
        contribution,
        width="stretch",
        hide_index=True,
    )

# ---------------------------------------------------------------------
# TAB 5
# ---------------------------------------------------------------------

with tab5:
    st.subheader("AI Risk Agent Assessment")

    long_end_candidates = ["10-15Y", "15-20Y", "20Y+"]

    long_end_value = portfolio[
        portfolio["KeyRateBucket"].isin(long_end_candidates)
    ]["MarketValue_INR"].sum()

    long_end_share = (
        float(long_end_value / market_value) if market_value else 0.0
    )

    if (
        "AverageAbsolutePnL_INR" in bucket_attribution.columns
        and len(bucket_attribution) > 0
    ):
        dominant_bucket = (
            bucket_attribution.sort_values(
                "AverageAbsolutePnL_INR",
                ascending=False,
            ).iloc[0]
        )
        dominant_bucket_name = str(dominant_bucket["KeyRateBucket"])
    else:
        dominant_bucket_name = "Unavailable"

    dominant_driver = (
        "Duration"
        if abs(duration_mean) >= abs(convexity_mean)
        else "Convexity"
    )

    var_ratio = metrics["var95"] / market_value if market_value else 0.0

    if var_ratio >= 0.02:
        risk_level = "HIGH"
    elif var_ratio >= 0.01:
        risk_level = "ELEVATED"
    else:
        risk_level = "MODERATE"

    if risk_level == "HIGH":
        risk_message = (
            "Tail loss is high relative to portfolio market value. "
            "Active risk-limit management is recommended."
        )
    elif risk_level == "ELEVATED":
        risk_message = (
            "Tail loss is elevated relative to portfolio market value. "
            "Monitor duration and key-rate concentration closely."
        )
    else:
        risk_message = (
            "Portfolio risk is moderate under the simulated historical "
            "yield-curve factor scenarios."
        )

    r1, r2, r3 = st.columns(3)

    r1.metric("AI Risk Level", risk_level)
    r2.metric("Dominant Driver", dominant_driver)
    r3.metric("Top Key-Rate Bucket", dominant_bucket_name)

    st.info(risk_message)

    st.markdown("### AI-Generated Risk Interpretation")

    st.write(
        f"""
Portfolio structure: The portfolio contains {bond_count:,} bonds with a
market value of {money(market_value)}.

Risk driver: {dominant_driver} is the larger contributor in the average
duration and convexity attribution.

Key-rate concentration: The largest average absolute scenario contribution
is associated with the {dominant_bucket_name} maturity bucket.

Long-end exposure: Approximately {long_end_share:.1%} of portfolio market
value is concentrated in the 10Y+ maturity range.

Tail risk: VaR 95% is approximately {money(metrics["var95"])} and VaR 99% is
approximately {money(metrics["var99"])}.

Stress outcome: The worst simulated portfolio scenario produces a loss of
approximately {money(metrics["min"])}.
"""
    )

    st.markdown("### Recommended Risk Actions")

    recommendations = [
        "Monitor long-duration and long-end maturity exposure.",
        "Set key-rate DV01 limits for each maturity bucket.",
        "Use convexity-aware stress testing instead of duration-only analysis.",
        "Escalate scenarios approaching or exceeding the VaR 99% threshold.",
        "Refresh PCA factors when the historical yield-curve dataset changes.",
        "Compare model-driven risk signals with independent risk-manager judgement.",
    ]

    for item in recommendations:
        st.markdown(f"- {item}")

    st.markdown("### Model Readiness")

    model_status = pd.DataFrame(
        {
            "Component": [
                "Random Forest",
                "Neural Network",
                "XGBoost",
                "PCA",
                "Monte Carlo",
                "Duration",
                "Convexity",
                "Key-Rate Attribution",
            ],
            "Status": ["READY"] * 8,
        }
    )

    st.dataframe(model_status, width="stretch", hide_index=True)

st.divider()

st.caption(
    "Ztheta WorkBridge Project 1B | Bond Risk Lab | "
    "Educational and analytical prototype - not investment advice."
)
