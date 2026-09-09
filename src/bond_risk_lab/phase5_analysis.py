from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

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

from bond_risk_lab.analytics.risk_attribution import (
    attribute_scenario_pnl,
    bucket_risk_attribution,
)

from bond_risk_lab.models.risk_ml import (
    build_scenario_features,
    create_loss_labels,
    train_random_forest,
    train_neural_network,
    classify_risk_probability,
)

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


OUTPUT_DIR = "reports/generated"


def train_xgboost_classifier(X, y):
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.80,
        colsample_bytree=0.80,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y)

    return model


def calculate_risk_metrics(pnl):
    pnl = np.asarray(pnl, dtype=float)

    var95_threshold = np.percentile(pnl, 5)
    var99_threshold = np.percentile(pnl, 1)

    var95 = -var95_threshold
    var99 = -var99_threshold

    es95 = -pnl[pnl <= var95_threshold].mean()
    es99 = -pnl[pnl <= var99_threshold].mean()

    return {
        "mean_pnl_inr": float(pnl.mean()),
        "pnl_volatility_inr": float(pnl.std()),
        "minimum_pnl_inr": float(pnl.min()),
        "maximum_pnl_inr": float(pnl.max()),
        "var_95_inr": float(var95),
        "var_99_inr": float(var99),
        "expected_shortfall_95_inr": float(es95),
        "expected_shortfall_99_inr": float(es99),
    }


def generate_agent_explanation(
    portfolio,
    attribution,
    bucket_attribution,
    risk_metrics,
    best_model_name,
    best_accuracy,
    best_auc,
    risk_probabilities,
):
    total_duration = attribution["DurationPnL_INR"].mean()
    total_convexity = attribution["ConvexityPnL_INR"].mean()

    dominant_bucket = bucket_attribution.iloc[0]

    average_risk_probability = risk_probabilities.mean()
    maximum_risk_probability = risk_probabilities.max()

    if abs(total_duration) > abs(total_convexity):
        dominant_driver = "duration"
    else:
        dominant_driver = "convexity"

    if average_risk_probability >= 0.05:
        risk_level = "ELEVATED"
    else:
        risk_level = "MODERATE"

    recommendations = [
        "Monitor long-duration holdings because long-end maturity buckets contribute disproportionately to scenario risk.",
        "Use key-rate DV01 limits to control concentration across maturity buckets.",
        "Use convexity-aware stress testing alongside duration-based sensitivity.",
        "Escalate scenarios where the ML high-risk probability exceeds 90%.",
        "Re-run PCA and Monte Carlo scenarios when the yield-curve history is materially updated.",
    ]

    return {
        "risk_level": risk_level,
        "dominant_risk_driver": dominant_driver,
        "dominant_key_rate_bucket": str(
            dominant_bucket["KeyRateBucket"]
        ),
        "dominant_bucket_average_absolute_pnl_inr": float(
            dominant_bucket["AverageAbsolutePnL_INR"]
        ),
        "average_ml_high_risk_probability": float(
            average_risk_probability
        ),
        "maximum_ml_high_risk_probability": float(
            maximum_risk_probability
        ),
        "selected_ml_model": best_model_name,
        "model_accuracy": float(best_accuracy),
        "model_roc_auc": float(best_auc),
        "recommendations": recommendations,
    }


def main():
    print("=" * 80)
    print("ZHETA BOND RISK LAB - PHASE 5")
    print("AI RISK AGENT")
    print("=" * 80)

    portfolio = load_bond_portfolio()
    curve_history = load_yield_curve_history()

    curve_result = analyze_yield_curve(curve_history)
    pca_results = curve_result["pca"]

    factor_scores = pca_results["factor_scores"]
    loadings = pca_results["loadings"]
    explained_variance = pca_results["explained_variance"]
    scaler = pca_results["scaler"]

    factor_scenarios, curve_changes = simulate_curve_scenarios(
        factor_scores=factor_scores,
        loadings=loadings,
        explained_variance=explained_variance,
        scaler=scaler,
        n_scenarios=5000,
        seed=42,
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

    X = build_scenario_features(
        factor_scenarios=factor_scenarios,
        curve_changes=curve_changes,
    )

    y = create_loss_labels(
        scenario_pnl=scenario_pnl,
        loss_quantile=0.95,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    rf_model = train_random_forest(
        X_train,
        y_train,
        random_state=42,
    )

    nn_model = train_neural_network(
        X_train,
        y_train,
        random_state=42,
    )

    xgb_model = train_xgboost_classifier(
        X_train,
        y_train,
    )

    models = {
        "Random Forest": rf_model,
        "Neural Network": nn_model,
        "XGBoost": xgb_model,
    }

    model_results = []

    for name, model in models.items():
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        model_results.append(
            {
                "model": name,
                "accuracy": float(
                    accuracy_score(
                        y_test,
                        predictions,
                    )
                ),
                "roc_auc": float(
                    roc_auc_score(
                        y_test,
                        probabilities,
                    )
                ),
            }
        )

    model_results_df = pd.DataFrame(model_results)

    model_results_df["score"] = (
        model_results_df["accuracy"] * 0.40
        + model_results_df["roc_auc"] * 0.60
    )

    model_results_df = model_results_df.sort_values(
        "score",
        ascending=False,
    ).reset_index(drop=True)

    best_model_name = model_results_df.iloc[0]["model"]
    best_model = models[best_model_name]

    risk_probability = classify_risk_probability(
        model=best_model,
        X=X,
    )

    risk_metrics = calculate_risk_metrics(
        scenario_pnl["PnL_Total_INR"]
    )

    agent = generate_agent_explanation(
        portfolio=portfolio,
        attribution=attribution,
        bucket_attribution=bucket_attribution,
        risk_metrics=risk_metrics,
        best_model_name=best_model_name,
        best_accuracy=model_results_df.iloc[0]["accuracy"],
        best_auc=model_results_df.iloc[0]["roc_auc"],
        risk_probabilities=(
            risk_probability[
                "HighRiskProbability"
            ]
        ),
    )

    worst_scenarios = (
        scenario_pnl
        .sort_values(
            "PnL_Total_INR"
        )
        .head(10)
    )

    report = {
        "generated_at": datetime.now().isoformat(),
        "project": "Ztheta WorkBridge Project 1B",
        "module": "Data Analyst Convexity Sensitivity AI Agent",
        "phase": "Phase 5",
        "portfolio": {
            "bond_count": int(len(portfolio)),
            "market_value_inr": float(
                portfolio[
                    "MarketValue_INR"
                ].sum()
            ),
        },
        "simulation": {
            "scenario_count": int(
                len(scenario_pnl)
            ),
            "pca_components": int(
                len(explained_variance)
            ),
            "cumulative_variance_explained": float(
                explained_variance[
                    "ExplainedVariance"
                ].sum()
            ),
        },
        "risk_metrics": risk_metrics,
        "model_comparison": model_results,
        "ai_agent": agent,
        "worst_scenarios": [
            {
                "scenario_id": int(row["ScenarioID"]),
                "pnl_inr": float(row["PnL_Total_INR"]),
            }
            for _, row in worst_scenarios.iterrows()
        ],
        "key_rate_attribution": (
            bucket_attribution.to_dict(
                orient="records"
            )
        ),
    }

    import os

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    output_path = (
        f"{OUTPUT_DIR}/phase5_ai_risk_report.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 80)
    print("AI AGENT RISK ASSESSMENT")
    print("=" * 80)

    print(
        f"Risk level: {agent['risk_level']}"
    )

    print(
        f"Dominant risk driver: "
        f"{agent['dominant_risk_driver']}"
    )

    print(
        f"Dominant key-rate bucket: "
        f"{agent['dominant_key_rate_bucket']}"
    )

    print(
        f"Selected ML model: "
        f"{agent['selected_ml_model']}"
    )

    print(
        f"Test accuracy: "
        f"{agent['model_accuracy']:.2%}"
    )

    print(
        f"Test ROC-AUC: "
        f"{agent['model_roc_auc']:.4f}"
    )

    print("\nRisk metrics:")

    for key, value in risk_metrics.items():
        print(
            f"{key}: ₹{value:,.2f}"
        )

    print("\nAgent recommendations:")

    for index, recommendation in enumerate(
        agent["recommendations"],
        start=1,
    ):
        print(
            f"{index}. {recommendation}"
        )

    print("\n" + "=" * 80)
    print("PHASE 5 COMPLETE")
    print("=" * 80)

    print(
        f"Report saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
