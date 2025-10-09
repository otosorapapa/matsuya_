"""Time-series forecasting helpers for sales and cash-flow analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    from prophet import Prophet
except Exception:  # pragma: no cover - dependency guard
    Prophet = None

try:  # pragma: no cover - optional dependency
    from statsmodels.tsa.arima.model import ARIMA
except Exception:  # pragma: no cover - dependency guard
    ARIMA = None


FREQUENCY_MAP = {
    "daily": "D",
    "weekly": "W-MON",
    "monthly": "MS",
    "yearly": "YS",
}


@dataclass
class ForecastResult:
    """Container describing the outcome of a forecasting run."""

    history: pd.DataFrame
    forecast: pd.DataFrame
    model_name: str
    success: bool
    message: str = ""

    def combined(self) -> pd.DataFrame:
        return pd.concat([self.history, self.forecast], ignore_index=True, sort=False)


def _prepare_series(
    df: pd.DataFrame,
    *,
    value_column: str,
    date_column: str = "date",
    frequency: str = "monthly",
) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    frame = df.copy()
    if date_column not in frame.columns:
        raise KeyError(f"Column '{date_column}' is required for forecasting.")
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame = frame.dropna(subset=[date_column])
    frame = frame.sort_values(date_column)
    freq = FREQUENCY_MAP.get(frequency, "MS")
    series = frame.set_index(date_column)[value_column].resample(freq).sum()
    return series.astype(float)


def _run_prophet(series: pd.Series, periods: int, frequency: str) -> Optional[pd.DataFrame]:
    if Prophet is None or series.empty:
        return None
    frame = series.reset_index()
    frame.columns = ["ds", "y"]
    model = Prophet(interval_width=0.8, daily_seasonality=False, weekly_seasonality=False)
    if frequency in ("daily", "weekly"):
        model.weekly_seasonality = True
    if frequency == "daily":
        model.daily_seasonality = True
    if frequency == "monthly":
        model.yearly_seasonality = True
    model.fit(frame)
    future = model.make_future_dataframe(periods=periods, freq=FREQUENCY_MAP.get(frequency, "MS"))
    forecast = model.predict(future)
    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)
    result = result.rename(columns={"ds": "date", "yhat": "forecast"})
    return result


def _run_arima(series: pd.Series, periods: int) -> Optional[pd.DataFrame]:
    if ARIMA is None or series.empty:
        return None
    model = ARIMA(series, order=(1, 1, 1))
    fitted = model.fit()
    forecast_res = fitted.get_forecast(steps=periods)
    frame = forecast_res.summary_frame(alpha=0.2)
    frame = frame.rename(
        columns={
            "mean": "forecast",
            "mean_ci_lower": "forecast_lower",
            "mean_ci_upper": "forecast_upper",
        }
    )
    frame = frame.reset_index().rename(columns={"index": "date"})
    return frame


def build_forecast(
    df: pd.DataFrame,
    *,
    value_column: str,
    date_column: str = "date",
    periods: int = 6,
    frequency: str = "monthly",
) -> ForecastResult:
    """Create a forecast using Prophet or ARIMA and return a :class:`ForecastResult`."""

    series = _prepare_series(
        df,
        value_column=value_column,
        date_column=date_column,
        frequency=frequency,
    )
    history_df = series.reset_index().rename(columns={"index": "date", value_column: "actual"})
    history_df = history_df.rename(columns={series.name or "actual": "actual"})

    forecast_frame: Optional[pd.DataFrame] = None
    model_name = ""
    message = ""

    try:
        forecast_frame = _run_prophet(series, periods, frequency)
        model_name = "Prophet"
    except Exception as exc:  # pragma: no cover - best effort fallback
        message = f"Prophet forecast failed: {exc}"

    if forecast_frame is None:
        try:
            forecast_frame = _run_arima(series, periods)
            model_name = model_name or "ARIMA"
        except Exception as exc:  # pragma: no cover
            message = message or f"ARIMA forecast failed: {exc}"

    if forecast_frame is None or forecast_frame.empty:
        fallback = pd.DataFrame(
            {
                "date": pd.date_range(
                    start=series.index.max() if not series.empty else pd.Timestamp.today(),
                    periods=periods,
                    freq=FREQUENCY_MAP.get(frequency, "MS"),
                ),
                "forecast": np.nan,
                "forecast_lower": np.nan,
                "forecast_upper": np.nan,
            }
        )
        return ForecastResult(
            history=history_df,
            forecast=fallback,
            model_name=model_name or "moving_average",
            success=False,
            message=message or "利用可能な予測モデルがありませんでした。",
        )

    if "forecast_lower" not in forecast_frame.columns:
        forecast_frame["forecast_lower"] = forecast_frame["forecast"].astype(float)
    if "forecast_upper" not in forecast_frame.columns:
        forecast_frame["forecast_upper"] = forecast_frame["forecast"].astype(float)
    forecast_frame = forecast_frame.rename(
        columns={
            "yhat_lower": "forecast_lower",
            "yhat_upper": "forecast_upper",
        }
    )
    forecast_frame["forecast"] = forecast_frame["forecast"].astype(float)

    return ForecastResult(
        history=history_df,
        forecast=forecast_frame,
        model_name=model_name or "Prophet",
        success=True,
        message=message,
    )


def evaluate_accuracy(
    result: ForecastResult,
    actual_df: pd.DataFrame,
    *,
    value_column: str,
    date_column: str = "date",
    frequency: str = "monthly",
) -> pd.DataFrame:
    """Calculate MAPE and absolute errors between forecast and realised values."""

    if result.forecast.empty or actual_df.empty:
        return pd.DataFrame(columns=["date", "forecast", "actual", "abs_error", "ape"])

    realised = actual_df.copy()
    realised[date_column] = pd.to_datetime(realised[date_column], errors="coerce")
    realised = realised.dropna(subset=[date_column])
    realised = realised.sort_values(date_column)
    aggregated = (
        realised.set_index(date_column)
        .resample(FREQUENCY_MAP.get(frequency, "MS"))
        [value_column]
        .sum()
        .reset_index()
    )
    merged = result.forecast.merge(
        aggregated.rename(columns={value_column: "actual"}),
        on="date",
        how="left",
    )
    merged["abs_error"] = (merged["forecast"] - merged["actual"]).abs()
    merged["ape"] = merged["abs_error"] / merged["actual"].replace(0, np.nan)
    merged["ape"] = merged["ape"].fillna(0.0)
    return merged


def forecast_sales(
    df: pd.DataFrame,
    *,
    periods: int = 6,
    frequency: str = "monthly",
) -> ForecastResult:
    return build_forecast(
        df,
        value_column="sales_amount",
        date_column="date",
        periods=periods,
        frequency=frequency,
    )


__all__ = [
    "ForecastResult",
    "build_forecast",
    "evaluate_accuracy",
    "forecast_sales",
]
