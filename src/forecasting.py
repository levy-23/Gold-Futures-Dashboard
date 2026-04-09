import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings

from .data_loader import get_feature_columns

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class ModelMetrics:
    rmse: float
    mae: float
    directional_accuracy: float


@dataclass
class ForecastResult:
    predicted_return_5d: float
    predicted_return_30d: float
    predicted_volatility: float
    confidence_lower_5d: float
    confidence_upper_5d: float
    confidence_lower_30d: float
    confidence_upper_30d: float
    xgb_metrics_5d: ModelMetrics
    xgb_metrics_30d: ModelMetrics
    rf_metrics_5d: ModelMetrics
    rf_metrics_30d: ModelMetrics
    shap_values_5d: np.ndarray = field(default=None, repr=False)
    shap_values_30d: np.ndarray = field(default=None, repr=False)
    feature_names: list = field(default_factory=list)
    test_predictions_5d: pd.DataFrame = field(default=None, repr=False)
    test_predictions_30d: pd.DataFrame = field(default=None, repr=False)
    vol_forecast_series: pd.Series = field(default=None, repr=False)


def _prepare_xy(df: pd.DataFrame, target_col: str, feature_cols: list[str]):
    valid = df.dropna(subset=feature_cols + [target_col])
    X = valid[feature_cols].values
    y = valid[target_col].values
    dates = valid["date"].values
    return X, y, dates


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    true_dir = np.sign(y_true)
    pred_dir = np.sign(y_pred)
    return np.mean(true_dir == pred_dir)


def _walk_forward_predict(
    model_class,
    model_params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    min_train_size: int = 200,
    retrain_every: int = 50,
) -> tuple[np.ndarray, ModelMetrics]:
    predictions = []
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    train_end = len(X_train)

    model = model_class(**model_params)
    model.fit(X_train, y_train)

    for i in range(len(X_test)):
        idx = train_end + i
        pred = model.predict(X_test[i : i + 1])[0]
        predictions.append(pred)

        if (i + 1) % retrain_every == 0 and idx >= min_train_size:
            model = model_class(**model_params)
            model.fit(X_all[:idx], y_all[:idx])

    predictions = np.array(predictions)
    metrics = ModelMetrics(
        rmse=float(np.sqrt(mean_squared_error(y_test, predictions))),
        mae=float(mean_absolute_error(y_test, predictions)),
        directional_accuracy=float(_directional_accuracy(y_test, predictions)),
    )
    return predictions, metrics


def _fit_garch_volatility(returns: pd.Series, horizon: int = 5) -> tuple[float, pd.Series]:
    returns_pct = returns.dropna() * 100
    if len(returns_pct) < 100:
        return returns.std(), pd.Series(dtype=float)

    try:
        model = arch_model(returns_pct, vol="Garch", p=1, q=1, mean="Zero", rescale=False)
        result = model.fit(disp="off", show_warning=False)
        forecast = result.forecast(horizon=horizon)
        # Annualized volatility from the last forecast variance
        avg_var = forecast.variance.iloc[-1].mean()
        vol_daily = np.sqrt(avg_var) / 100
        conditional_vol = pd.Series(
            np.sqrt(result.conditional_volatility) / 100,
            index=returns_pct.index,
        )
        return float(vol_daily), conditional_vol
    except Exception:
        return float(returns.std()), pd.Series(dtype=float)


def run_forecasting(train_df: pd.DataFrame, test_df: pd.DataFrame) -> ForecastResult:
    feature_cols = get_feature_columns()

    # --- 5-day return prediction ---
    X_train_5, y_train_5, _ = _prepare_xy(train_df, "fwd_return_5d", feature_cols)
    X_test_5, y_test_5, dates_test_5 = _prepare_xy(test_df, "fwd_return_5d", feature_cols)

    xgb_params = {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05,
                   "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42,
                   "verbosity": 0}
    rf_params = {"n_estimators": 200, "max_depth": 8, "random_state": 42, "n_jobs": -1}

    xgb_pred_5, xgb_met_5 = _walk_forward_predict(XGBRegressor, xgb_params, X_train_5, y_train_5, X_test_5, y_test_5)
    rf_pred_5, rf_met_5 = _walk_forward_predict(RandomForestRegressor, rf_params, X_train_5, y_train_5, X_test_5, y_test_5)

    # --- 30-day return prediction ---
    X_train_30, y_train_30, _ = _prepare_xy(train_df, "fwd_return_30d", feature_cols)
    X_test_30, y_test_30, dates_test_30 = _prepare_xy(test_df, "fwd_return_30d", feature_cols)

    xgb_pred_30, xgb_met_30 = _walk_forward_predict(XGBRegressor, xgb_params, X_train_30, y_train_30, X_test_30, y_test_30)
    rf_pred_30, rf_met_30 = _walk_forward_predict(RandomForestRegressor, rf_params, X_train_30, y_train_30, X_test_30, y_test_30)

    # --- SHAP values (train a final model on all training data) ---
    xgb_final_5 = XGBRegressor(**xgb_params)
    xgb_final_5.fit(X_train_5, y_train_5)
    xgb_final_30 = XGBRegressor(**xgb_params)
    xgb_final_30.fit(X_train_30, y_train_30)

    try:
        import shap
        explainer_5 = shap.TreeExplainer(xgb_final_5)
        shap_vals_5 = explainer_5.shap_values(X_test_5)
        explainer_30 = shap.TreeExplainer(xgb_final_30)
        shap_vals_30 = explainer_30.shap_values(X_test_30)
    except Exception:
        shap_vals_5 = None
        shap_vals_30 = None

    # --- Volatility forecast via GARCH ---
    all_returns = pd.concat([train_df["daily_return"], test_df["daily_return"]]).dropna()
    predicted_vol, vol_series = _fit_garch_volatility(all_returns)

    # --- Latest predictions (from last available data point) ---
    latest_pred_5d = float(xgb_pred_5[-1]) if len(xgb_pred_5) > 0 else 0.0
    latest_pred_30d = float(xgb_pred_30[-1]) if len(xgb_pred_30) > 0 else 0.0
    residual_std_5 = float(np.std(y_test_5 - xgb_pred_5)) if len(xgb_pred_5) > 0 else 0.01
    residual_std_30 = float(np.std(y_test_30 - xgb_pred_30)) if len(xgb_pred_30) > 0 else 0.01

    # Test prediction DataFrames for visualization
    test_preds_5d = pd.DataFrame({
        "date": dates_test_5,
        "actual": y_test_5,
        "xgb_pred": xgb_pred_5,
        "rf_pred": rf_pred_5,
    })
    test_preds_30d = pd.DataFrame({
        "date": dates_test_30,
        "actual": y_test_30,
        "xgb_pred": xgb_pred_30,
        "rf_pred": rf_pred_30,
    })

    return ForecastResult(
        predicted_return_5d=latest_pred_5d,
        predicted_return_30d=latest_pred_30d,
        predicted_volatility=predicted_vol,
        confidence_lower_5d=latest_pred_5d - 1.96 * residual_std_5,
        confidence_upper_5d=latest_pred_5d + 1.96 * residual_std_5,
        confidence_lower_30d=latest_pred_30d - 1.96 * residual_std_30,
        confidence_upper_30d=latest_pred_30d + 1.96 * residual_std_30,
        xgb_metrics_5d=xgb_met_5,
        xgb_metrics_30d=xgb_met_30,
        rf_metrics_5d=rf_met_5,
        rf_metrics_30d=rf_met_30,
        shap_values_5d=shap_vals_5,
        shap_values_30d=shap_vals_30,
        feature_names=feature_cols,
        test_predictions_5d=test_preds_5d,
        test_predictions_30d=test_preds_30d,
        vol_forecast_series=vol_series,
    )
