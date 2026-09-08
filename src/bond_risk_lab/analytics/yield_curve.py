import numpy as np
import pandas as pd


def curve_snapshot(
    curve_history: pd.DataFrame,
    date=None,
    currency="INR",
) -> pd.DataFrame:

    df = curve_history[
        curve_history["Currency"] == currency
    ].copy()

    if date is None:
        date = df["CurveDate"].max()

    snapshot = df[df["CurveDate"] == date].copy()

    return snapshot.sort_values("Tenor_Years")


def curve_changes(
    curve_history: pd.DataFrame,
    currency="INR",
) -> pd.DataFrame:

    df = curve_history[
        curve_history["Currency"] == currency
    ].copy()

    df = df.sort_values(
        ["Tenor_Years", "CurveDate"]
    )

    df["YieldChange"] = (
        df.groupby("Tenor_Years")["Yield"]
        .diff()
    )

    df["ZeroRateChange"] = (
        df.groupby("Tenor_Years")["ZeroRate"]
        .diff()
    )

    return df


def curve_factor_summary(
    curve_history: pd.DataFrame,
    currency="INR",
) -> pd.DataFrame:

    df = curve_changes(
        curve_history,
        currency,
    )

    latest_date = df["CurveDate"].max()

    latest = df[
        df["CurveDate"] == latest_date
    ].copy()

    if latest.empty:
        return latest

    return latest[
        [
            "TenorLabel",
            "Tenor_Years",
            "Yield",
            "YieldChange",
        ]
    ].sort_values("Tenor_Years")

    