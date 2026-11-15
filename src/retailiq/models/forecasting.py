
import numpy as np
import pandas as pd
import mlflow

from sqlalchemy import create_engine, text
from scipy.stats import t
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ENGINE = create_engine("postgresql+psycopg:///retailiq")

TEST_SIZE = 6
CONFIDENCE_LEVEL = 0.95
RANDOM_STATE = 42
MLFLOW_EXPERIMENT = "RetailIQ - Revenue Forecasting"


def load_monthly_sales():
    query = """
    SELECT month, merchandise_revenue
    FROM retail.monthly_sales
    ORDER BY month
    """

    df = pd.read_sql(text(query), ENGINE)
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month").set_index("month")

    series = df["merchandise_revenue"].astype(float).asfreq("MS")

    if series.isna().any():
        raise ValueError("Monthly revenue contains missing months.")

    if len(series) < 18:
        raise ValueError("At least 18 months are required for forecasting.")

    if len(series) <= TEST_SIZE + 12:
        raise ValueError("Insufficient training history for seasonal baseline.")

    return series


def evaluate_forecasts(actual, predictions):
    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))

    nonzero = actual != 0
    mape = (
        np.mean(
            np.abs(
                (actual[nonzero] - predictions[nonzero])
                / actual[nonzero]
            )
        ) * 100
        if nonzero.any()
        else np.nan
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_percent": mape,
    }


def prediction_intervals(forecast, residuals, alpha=0.05):
    """
    Approximate residual-based prediction intervals.

    Forecast uncertainty widens with the forecast horizon.
    These intervals do not fully account for model parameter
    uncertainty or structural changes in the revenue series.
    """
    residuals = pd.Series(residuals).dropna().to_numpy()

    if len(residuals) < 3:
        raise ValueError("Insufficient residuals to estimate intervals.")

    residual_std = np.std(residuals, ddof=1)
    degrees_of_freedom = len(residuals) - 1

    critical_value = t.ppf(
        1 - alpha / 2,
        df=degrees_of_freedom,
    )

    horizons = np.arange(1, len(forecast) + 1)

    horizon_scale = np.sqrt(
        1 + horizons / len(residuals)
    )

    margin = critical_value * residual_std * horizon_scale

    lower = np.asarray(forecast) - margin
    upper = np.asarray(forecast) + margin

    return lower, upper


def evaluate_intervals(actual, lower, upper):
    actual = np.asarray(actual)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    coverage = np.mean(
        (actual >= lower) & (actual <= upper)
    )

    average_width = np.mean(upper - lower)

    return {
        "coverage_percent": coverage * 100,
        "average_interval_width": average_width,
    }


def log_forecast_run(
    model_name,
    parameters,
    metrics,
    comparison,
    forecast_frame,
):
    with mlflow.start_run(run_name=model_name):
        mlflow.set_tags({
            "project": "RetailIQ",
            "task": "monthly_revenue_forecasting",
            "model": model_name,
            "validation": "chronological_six_month_holdout",
            "interval_method": "approximate_residual_based",
        })

        mlflow.log_params(parameters)

        for metric_name, metric_value in metrics.items():
            if pd.notna(metric_value):
                mlflow.log_metric(
                    metric_name.lower(),
                    float(metric_value),
                )

        mlflow.log_metric("test_months", TEST_SIZE)
        mlflow.log_metric(
            "confidence_level",
            CONFIDENCE_LEVEL,
        )

        mlflow.log_table(
            forecast_frame.reset_index().rename(
                columns={"index": "month"}
            ),
            "forecast_results.json",
        )

        mlflow.log_table(
            comparison.reset_index().rename(
                columns={"index": "month"}
            ),
            "actual_vs_forecast.json",
        )


def main():
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    sales = load_monthly_sales()

    train = sales.iloc[:-TEST_SIZE]
    test = sales.iloc[-TEST_SIZE:]

    results = []
    forecasts = {}

    # --------------------------------------------------------
    # SEASONAL NAIVE
    # --------------------------------------------------------

    seasonal_predictions = train.iloc[-12:].iloc[
        np.arange(TEST_SIZE) % 12
    ].to_numpy()

    seasonal_residuals = (
        train.iloc[12:].to_numpy()
        - train.iloc[:-12].to_numpy()
    )

    seasonal_lower, seasonal_upper = prediction_intervals(
        seasonal_predictions,
        seasonal_residuals,
        alpha=1 - CONFIDENCE_LEVEL,
    )

    seasonal_metrics = evaluate_forecasts(
        test.to_numpy(),
        seasonal_predictions,
    )

    seasonal_interval_metrics = evaluate_intervals(
        test.to_numpy(),
        seasonal_lower,
        seasonal_upper,
    )

    seasonal_all_metrics = {
        **seasonal_metrics,
        **seasonal_interval_metrics,
    }

    results.append({
        "model": "Seasonal Naive",
        **seasonal_all_metrics,
    })

    forecasts["Seasonal Naive"] = pd.DataFrame({
        "actual_revenue": test.to_numpy(),
        "forecast": seasonal_predictions,
        "lower_95": seasonal_lower,
        "upper_95": seasonal_upper,
    }, index=test.index)

    log_forecast_run(
        model_name="Seasonal Naive",
        parameters={
            "test_size": TEST_SIZE,
            "seasonal_period": 12,
            "confidence_level": CONFIDENCE_LEVEL,
        },
        metrics=seasonal_all_metrics,
        comparison=forecasts["Seasonal Naive"],
        forecast_frame=forecasts["Seasonal Naive"],
    )

    # --------------------------------------------------------
    # HOLT-WINTERS
    # --------------------------------------------------------

    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal=None,
        initialization_method="estimated",
    )

    fitted = model.fit(optimized=True)
    hw_forecast = fitted.forecast(TEST_SIZE)

    hw_residuals = train - fitted.fittedvalues

    hw_lower, hw_upper = prediction_intervals(
        hw_forecast.to_numpy(),
        hw_residuals,
        alpha=1 - CONFIDENCE_LEVEL,
    )

    hw_metrics = evaluate_forecasts(
        test.to_numpy(),
        hw_forecast.to_numpy(),
    )

    hw_interval_metrics = evaluate_intervals(
        test.to_numpy(),
        hw_lower,
        hw_upper,
    )

    hw_all_metrics = {
        **hw_metrics,
        **hw_interval_metrics,
    }

    results.append({
        "model": "Holt-Winters",
        **hw_all_metrics,
    })

    forecasts["Holt-Winters"] = pd.DataFrame({
        "actual_revenue": test.to_numpy(),
        "forecast": hw_forecast.to_numpy(),
        "lower_95": hw_lower,
        "upper_95": hw_upper,
    }, index=test.index)

    log_forecast_run(
        model_name="Holt-Winters",
        parameters={
            "trend": "add",
            "seasonal": "none",
            "initialization_method": "estimated",
            "optimized": True,
            "test_size": TEST_SIZE,
            "confidence_level": CONFIDENCE_LEVEL,
        },
        metrics=hw_all_metrics,
        comparison=forecasts["Holt-Winters"],
        forecast_frame=forecasts["Holt-Winters"],
    )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    print("\nMONTHLY REVENUE FORECAST EVALUATION")
    print(results_df.to_string(index=False))

    comparison = pd.DataFrame({
        "actual_revenue": test,
        "seasonal_naive_forecast": forecasts["Seasonal Naive"]["forecast"],
        "seasonal_naive_lower_95": forecasts["Seasonal Naive"]["lower_95"],
        "seasonal_naive_upper_95": forecasts["Seasonal Naive"]["upper_95"],
        "holt_winters_forecast": forecasts["Holt-Winters"]["forecast"],
        "holt_winters_lower_95": forecasts["Holt-Winters"]["lower_95"],
        "holt_winters_upper_95": forecasts["Holt-Winters"]["upper_95"],
    })

    print("\nACTUAL VS FORECAST WITH 95% APPROXIMATE INTERVALS")
    print(comparison.round(2).to_string())

    print(
        "\nNOTE: Prediction intervals are residual-based approximations. "
        "Coverage on this six-month holdout is descriptive and does not "
        "establish calibrated 95% coverage."
    )


if __name__ == "__main__":
    main()
