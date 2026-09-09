from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from bond_risk_lab.data.loaders import load_bond_portfolio, load_yield_curve_history
from bond_risk_lab.analytics.curve_risk import analyze_yield_curve
from bond_risk_lab.analytics.pca_monte_carlo import simulate_curve_scenarios
from bond_risk_lab.analytics.scenario_pricing import price_portfolio_scenarios
from bond_risk_lab.analytics.risk_attribution import (
    attribute_scenario_pnl,
    bucket_risk_attribution,
    calculate_portfolio_risk_contributions,
)
from bond_risk_lab.models.risk_ml import (
    build_scenario_features,
    create_loss_labels,
    train_random_forest,
    train_neural_network,
    evaluate_classifier,
    random_forest_feature_importance,
    classify_risk_probability,
)


def train_xgboost_classifier(X, y, random_state=42):
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.80,
        colsample_bytree=0.80,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def main():
    print("=" * 80)
    print("ZHETA BOND RISK LAB - PHASE 4")
    print("ML-BASED CONVEXITY SENSITIVITY AND RISK ATTRIBUTION")
    print("=" * 80)

    portfolio = load_bond_portfolio()
    curve_history = load_yield_curve_history()

    print("\n" + "=" * 80)
    print("DATA")
    print("=" * 80)
    print(f"Portfolio bonds: {len(portfolio):,}")
    print(f"Yield curve observations: {len(curve_history):,}")
    print(f"Portfolio market value: ₹{portfolio['MarketValue_INR'].sum():,.2f}")

    result = analyze_yield_curve(curve_history)
    pca_results = result["pca"]

    scaler = pca_results["scaler"]
    loadings = pca_results["loadings"]
    factor_scores = pca_results["factor_scores"]
    explained_variance = pca_results["explained_variance"]

    print("\n" + "=" * 80)
    print("PCA EXPLAINED VARIANCE")
    print("=" * 80)
    print(
        explained_variance.to_string(
            index=False,
            formatters={"ExplainedVariance": "{:.4f}".format},
        )
    )

    cumulative_variance = explained_variance["ExplainedVariance"].cumsum()

    print("\nCumulative variance:")
    for component, value in zip(
        explained_variance["Component"],
        cumulative_variance,
    ):
        print(f"{component}: {value:.2%}")

    factor_scenarios, curve_changes = simulate_curve_scenarios(
        factor_scores=factor_scores,
        loadings=loadings,
        explained_variance=explained_variance,
        scaler=scaler,
        n_scenarios=5000,
        seed=42,
    )

    print("\n" + "=" * 80)
    print("MONTE CARLO")
    print("=" * 80)
    print(f"Scenarios generated: {len(factor_scenarios):,}")

    scenario_pnl = price_portfolio_scenarios(
        portfolio=portfolio,
        curve_changes=curve_changes,
    )

    print("\n" + "=" * 80)
    print("PORTFOLIO SCENARIO P&L")
    print("=" * 80)
    print(
        scenario_pnl["PnL_Total_INR"].describe().to_string(
            float_format=lambda x: f"{x:,.2f}"
        )
    )

    attribution = attribute_scenario_pnl(
        portfolio=portfolio,
        curve_changes=curve_changes,
    )

    print("\n" + "=" * 80)
    print("DURATION / CONVEXITY ATTRIBUTION")
    print("=" * 80)
    print(
        attribution[
            [
                "ScenarioID",
                "DurationPnL_INR",
                "ConvexityPnL_INR",
                "TotalPnL_INR",
            ]
        ].head(10).to_string(
            index=False,
            float_format=lambda x: f"{x:,.2f}",
        )
    )

    bucket_attribution = bucket_risk_attribution(
        portfolio=portfolio,
        curve_changes=curve_changes,
    )

    print("\n" + "=" * 80)
    print("KEY-RATE RISK ATTRIBUTION")
    print("=" * 80)
    print(
        bucket_attribution.to_string(
            index=False,
            float_format=lambda x: f"{x:,.2f}",
        )
    )

    portfolio_risk = calculate_portfolio_risk_contributions(portfolio)

    print("\n" + "=" * 80)
    print("PORTFOLIO DV01 / CONVEXITY")
    print("=" * 80)
    print(
        portfolio_risk[
            [
                "KeyRateBucket",
                "MarketValue_INR",
                "ModifiedDuration",
                "Convexity",
                "DV01_INR",
                "MarketValueWeight",
                "DV01Weight",
            ]
        ]
        .sort_values("DV01_INR", ascending=False)
        .head(20)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    X = build_scenario_features(
        factor_scenarios=factor_scenarios,
        curve_changes=curve_changes,
    )

    y = create_loss_labels(
        scenario_pnl=scenario_pnl,
        loss_quantile=0.95,
    )

    print("\n" + "=" * 80)
    print("ML DATASET")
    print("=" * 80)
    print(f"Features: {X.shape[1]}")
    print(f"Scenarios: {X.shape[0]:,}")
    print(f"High-risk scenarios: {y.sum():,}")
    print(f"Low-risk scenarios: {(y == 0).sum():,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\n" + "=" * 80)
    print("TRAIN / TEST SPLIT")
    print("=" * 80)
    print(f"Training scenarios: {len(X_train):,}")
    print(f"Testing scenarios: {len(X_test):,}")
    print(f"Training high-risk: {y_train.sum():,}")
    print(f"Testing high-risk: {y_test.sum():,}")

    rf_model = train_random_forest(
        X=X_train,
        y=y_train,
        random_state=42,
    )

    rf_results = evaluate_classifier(
        model=rf_model,
        X=X_test,
        y=y_test,
    )

    print("\n" + "=" * 80)
    print("RANDOM FOREST - TEST SET")
    print("=" * 80)
    print(f"Accuracy: {rf_results['accuracy']:.4f}")
    print(f"ROC-AUC: {rf_results['roc_auc']:.4f}")

    importance = random_forest_feature_importance(
        model=rf_model,
        feature_names=X.columns,
    )

    print("\nFeature importance:")
    print(
        importance.head(15).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    nn_model = train_neural_network(
        X=X_train,
        y=y_train,
        random_state=42,
    )

    nn_results = evaluate_classifier(
        model=nn_model,
        X=X_test,
        y=y_test,
    )

    print("\n" + "=" * 80)
    print("NEURAL NETWORK - TEST SET")
    print("=" * 80)
    print(f"Accuracy: {nn_results['accuracy']:.4f}")
    print(f"ROC-AUC: {nn_results['roc_auc']:.4f}")

    xgb_model = train_xgboost_classifier(
        X=X_train,
        y=y_train,
        random_state=42,
    )

    xgb_results = evaluate_classifier(
        model=xgb_model,
        X=X_test,
        y=y_test,
    )

    print("\n" + "=" * 80)
    print("XGBOOST - TEST SET")
    print("=" * 80)
    print(f"Accuracy: {xgb_results['accuracy']:.4f}")
    print(f"ROC-AUC: {xgb_results['roc_auc']:.4f}")

    comparison = pd.DataFrame(
        {
            "Model": [
                "Random Forest",
                "Neural Network",
                "XGBoost",
            ],
            "Accuracy": [
                rf_results["accuracy"],
                nn_results["accuracy"],
                xgb_results["accuracy"],
            ],
            "ROC_AUC": [
                rf_results["roc_auc"],
                nn_results["roc_auc"],
                xgb_results["roc_auc"],
            ],
        }
    )

    comparison["Score"] = (
        comparison["Accuracy"] * 0.40
        + comparison["ROC_AUC"] * 0.60
    )

    comparison = comparison.sort_values(
        "Score",
        ascending=False,
    ).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON - UNSEEN TEST DATA")
    print("=" * 80)
    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    best_model_name = comparison.iloc[0]["Model"]

    if best_model_name == "Random Forest":
        best_model = rf_model
    elif best_model_name == "Neural Network":
        best_model = nn_model
    else:
        best_model = xgb_model

    print("\n" + "=" * 80)
    print("BEST ML MODEL")
    print("=" * 80)
    print(f"Selected model: {best_model_name}")

    risk_probability = classify_risk_probability(
        model=best_model,
        X=X,
    )

    scenario_risk = pd.concat(
        [
            scenario_pnl.reset_index(drop=True),
            risk_probability.reset_index(drop=True),
        ],
        axis=1,
    )

    print("\n" + "=" * 80)
    print("HIGHEST ML-RISK SCENARIOS")
    print("=" * 80)

    highest_risk = (
        scenario_risk
        .sort_values(
            "HighRiskProbability",
            ascending=False,
        )
        .head(10)
    )

    print(
        highest_risk.to_string(
            index=False,
            float_format=lambda x: f"{x:,.6f}",
        )
    )

    print("\n" + "=" * 80)
    print("WORST P&L SCENARIOS")
    print("=" * 80)

    worst = scenario_risk.nsmallest(
        10,
        "PnL_Total_INR",
    )

    print(
        worst.to_string(
            index=False,
            float_format=lambda x: f"{x:,.2f}",
        )
    )

    pnl = scenario_pnl["PnL_Total_INR"]

    var_95 = -np.percentile(pnl, 5)
    var_99 = -np.percentile(pnl, 1)

    expected_shortfall_95 = -pnl[
        pnl <= np.percentile(pnl, 5)
    ].mean()

    expected_shortfall_99 = -pnl[
        pnl <= np.percentile(pnl, 1)
    ].mean()

    print("\n" + "=" * 80)
    print("PHASE 4 RISK SUMMARY")
    print("=" * 80)
    print(f"Portfolio MV: ₹{portfolio['MarketValue_INR'].sum():,.2f}")
    print(f"Mean P&L: ₹{pnl.mean():,.2f}")
    print(f"P&L volatility: ₹{pnl.std():,.2f}")
    print(f"VaR 95%: ₹{var_95:,.2f}")
    print(f"VaR 99%: ₹{var_99:,.2f}")
    print(f"Expected Shortfall 95%: ₹{expected_shortfall_95:,.2f}")
    print(f"Expected Shortfall 99%: ₹{expected_shortfall_99:,.2f}")
    print(
        f"Average ML high-risk probability: "
        f"{risk_probability['HighRiskProbability'].mean():.2%}"
    )
    print(
        f"Maximum ML high-risk probability: "
        f"{risk_probability['HighRiskProbability'].max():.2%}"
    )

    print("\n" + "=" * 80)
    print("PHASE 4 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
