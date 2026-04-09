import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    full: pd.DataFrame


def load_and_clean(csv_path: str = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = Path(__file__).parent.parent / "data" / "gold_price_forecasting_dataset.csv"

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    price_cols = ["open", "high", "low", "close", "adj close"]
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].round(2)

    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Log returns
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Rolling Sharpe ratio (annualized, using risk-free rate ~ 0)
    df["sharpe_21"] = (
        df["daily_return"].rolling(21).mean()
        / df["daily_return"].rolling(21).std()
    ) * np.sqrt(252)

    # Lagged features
    for lag in [1, 5, 10, 20]:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
        df[f"return_lag_{lag}"] = df["daily_return"].shift(lag)
        df[f"volume_lag_{lag}"] = df["volume"].shift(lag)
        df[f"vol7_lag_{lag}"] = df["volatility_7"].shift(lag)

    # Forward returns (targets for forecasting)
    df["fwd_return_5d"] = df["close"].shift(-5) / df["close"] - 1
    df["fwd_return_30d"] = df["close"].shift(-30) / df["close"] - 1

    # Price relative to Bollinger Bands (normalized position)
    bb_range = df["bb_upper"] - df["bb_lower"]
    df["bb_position"] = np.where(
        bb_range > 0,
        (df["close"] - df["bb_lower"]) / bb_range,
        0.5,
    )

    # MACD histogram
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    return df


def split_data(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15) -> DataSplit:
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    return DataSplit(
        train=df.iloc[:train_end].copy(),
        val=df.iloc[train_end:val_end].copy(),
        test=df.iloc[val_end:].copy(),
        full=df.copy(),
    )


def get_feature_columns() -> list[str]:
    base = [
        "open", "high", "low", "close", "volume",
        "ma_7", "ma_30", "ma_90",
        "daily_return", "volatility_7", "volatility_30",
        "rsi", "macd", "macd_signal", "bb_upper", "bb_lower",
        "log_return", "sharpe_21", "bb_position", "macd_hist",
    ]
    lagged = []
    for lag in [1, 5, 10, 20]:
        lagged += [f"close_lag_{lag}", f"return_lag_{lag}", f"volume_lag_{lag}", f"vol7_lag_{lag}"]
    return base + lagged


def load_pipeline(csv_path: str = None) -> DataSplit:
    df = load_and_clean(csv_path)
    df = engineer_features(df)
    return split_data(df)
