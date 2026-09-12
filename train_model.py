"""
Adani Ports Stock Market Prediction - Machine Learning Pipeline
---------------------------------------------------------------
Developer: Abhiraj Kumar (B.Tech CSE)
Description: Complete ML pipeline covering data loading, cleaning, EDA visualization,
             data-leakage-free feature engineering, chronological train-test split,
             MinMaxScaler feature scaling, XGBoost model training, multi-model evaluation,
             and saving model artifacts.
"""

from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# Define project directories using relative paths for Windows/Linux/Streamlit Cloud compatibility
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ADANIPORTS.csv"
MODEL_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets" / "images"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def load_and_clean_data(path: Path) -> pd.DataFrame:
    """Load the Adani Ports CSV dataset, clean column names, format dates, and sort chronologically."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]

    print("========== DATASET SUMMARY ==========")
    print(f"Total Rows: {df.shape[0]:,}")
    print(f"Total Columns: {df.shape[1]}")
    print("Columns:", list(df.columns))

    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError("Dataset must contain 'Date' and 'Close' columns.")

    # Convert Date to datetime and sort chronologically
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Ensure price/volume fields are numeric
    numeric_cols = ["Prev Close", "Open", "High", "Low", "Last", "Close", "VWAP", "Volume", "Turnover"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def generate_eda_plots(df: pd.DataFrame) -> None:
    """Generate static exploratory visualization charts for portfolio artifacts."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Closing Price Trend Over Time
    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["Close"], color="#0052CC", linewidth=1.5, label="Closing Price")
    plt.title("Adani Ports Historical Closing Price (2007 - 2021)", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Closing Price (₹)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "closing_price_trend.png", dpi=200)
    plt.close()

    # 2. Moving Averages Chart
    df_ma = df.copy().set_index("Date")
    ma_7 = df_ma["Close"].rolling(window=7).mean()
    ma_30 = df_ma["Close"].rolling(window=30).mean()

    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["Close"], label="Actual Close", color="#CBD5E1", alpha=0.7, linewidth=1)
    plt.plot(df["Date"], ma_7, label="7-Day Moving Average", color="#FF9900", linewidth=1.5)
    plt.plot(df["Date"], ma_30, label="30-Day Moving Average", color="#10B981", linewidth=1.5)
    plt.title("Adani Ports Moving Averages (7-Day vs 30-Day)", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Price (₹)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "moving_averages.png", dpi=200)
    plt.close()

    # 3. Correlation Heatmap
    num_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    corr = num_df.corr()

    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5)
    plt.title("Adani Ports Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct features without Data Leakage.
    All inputs use t-1 (previous day or historical rolling) information to predict Target Close_t.
    """
    data = df.copy().sort_values("Date").reset_index(drop=True)

    # Historical Lag Features (t-1)
    data["Prev_Close"] = data["Close"].shift(1)
    data["Prev_Open"] = data["Open"].shift(1)
    data["Prev_High"] = data["High"].shift(1)
    data["Prev_Low"] = data["Low"].shift(1)
    data["Prev_Volume"] = data["Volume"].shift(1)

    # Derived Technical Indicators (strictly from shifted close)
    prev_close = data["Prev_Close"]
    data["Daily_Price_Change"] = data["Prev_Close"] - data["Prev_Open"]
    data["Daily_Pct_Change"] = prev_close.pct_change() * 100
    data["MA_7"] = prev_close.rolling(window=7).mean()
    data["MA_30"] = prev_close.rolling(window=30).mean()
    data["Rolling_Std_7"] = prev_close.rolling(window=7).std()

    # Calendar Features
    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Day"] = data["Date"].dt.day
    data["Day_of_Week"] = data["Date"].dt.dayofweek

    # Drop missing rows created by shifting/rolling operations
    feature_cols_check = [
        "Prev_Close", "Prev_Open", "Prev_High", "Prev_Low", "Prev_Volume",
        "Daily_Price_Change", "Daily_Pct_Change", "MA_7", "MA_30", "Rolling_Std_7",
        "Year", "Month", "Day", "Day_of_Week", "Close"
    ]
    data = data.dropna(subset=feature_cols_check).reset_index(drop=True)
    return data


def prepare_datasets(df_engineered: pd.DataFrame):
    """
    Split data chronologically (80% train, 20% test) without shuffling.
    Fit MinMaxScaler exclusively on X_train.
    """
    feature_columns = [
        "Prev_Close", "Prev_Open", "Prev_High", "Prev_Low", "Prev_Volume",
        "Daily_Price_Change", "Daily_Pct_Change", "MA_7", "MA_30", "Rolling_Std_7",
        "Year", "Month", "Day", "Day_of_Week"
    ]

    X = df_engineered[feature_columns]
    y = df_engineered["Close"]

    # Chronological Split (80% Train, 20% Test)
    split_idx = int(len(df_engineered) * 0.80)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = df_engineered["Date"].iloc[split_idx:]

    # Scale Features (Fit scaler strictly on X_train)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        feature_columns,
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        dates_test,
        scaler,
    )


def train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, dates_test):
    """Train primary XGBoost Regressor and comparison baseline models. Evaluate and save metrics/charts."""
    models = {
        "XGBoost Regressor": XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
    }

    metrics_summary = {}
    predictions_dict = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        predictions_dict[name] = preds

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        metrics_summary[name] = {
            "MAE": round(float(mae), 4),
            "MSE": round(float(mse), 4),
            "RMSE": round(float(rmse), 4),
            "R2": round(float(r2), 4)
        }

    print("\n========== MODEL EVALUATION COMPARISON ==========")
    metrics_df = pd.DataFrame(metrics_summary).T
    print(metrics_df.to_string())

    # Selected primary model = XGBoost
    primary_model = models["XGBoost Regressor"]
    xgb_preds = predictions_dict["XGBoost Regressor"]

    # 1. Actual vs Predicted Plot
    plt.figure(figsize=(14, 6))
    plt.plot(dates_test, y_test.values, label="Actual Closing Price", color="#0052CC", linewidth=1.5)
    plt.plot(dates_test, xgb_preds, label="Predicted Closing Price (XGBoost)", color="#FF9900", linewidth=1.5, linestyle="--")
    plt.title("Adani Ports Actual vs Predicted Closing Price (Test Set)", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Closing Price (₹)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "actual_vs_predicted.png", dpi=200)
    plt.close()

    # 2. Feature Importance Chart
    importance = pd.Series(primary_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    plt.figure(figsize=(10, 6))
    importance.plot(kind="barh", color="#0052CC")
    plt.title("XGBoost Feature Importance", fontsize=14, fontweight="bold")
    plt.xlabel("Importance Score", fontsize=11)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "feature_importance.png", dpi=200)
    plt.close()

    return primary_model, metrics_summary, predictions_dict


def save_artifacts(model, scaler, feature_cols, metrics_summary):
    """Save trained model, scaler, feature list, and metrics dictionary using joblib."""
    joblib.dump(model, MODEL_DIR / "xgboost_model.pkl")
    joblib.dump(model, MODEL_DIR / "model.pkl")  # Duplicate reference for portability
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(feature_cols, MODEL_DIR / "feature_columns.pkl")
    joblib.dump(metrics_summary, MODEL_DIR / "model_metrics.pkl")
    print(f"\nModel artifacts successfully saved to: {MODEL_DIR}")


def main():
    print("=" * 75)
    print("ADANI PORTS STOCK MARKET PREDICTION - ML TRAINING PIPELINE")
    print("=" * 75)

    df = load_and_clean_data(DATA_PATH)
    generate_eda_plots(df)

    df_engineered = engineer_features(df)
    (
        feature_cols,
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        dates_test,
        scaler,
    ) = prepare_datasets(df_engineered)

    print("\n========== DATASET SPLIT INFO ==========")
    print(f"Total Processed Samples: {len(df_engineered):,}")
    print(f"Training Samples (80%): {len(X_train):,} ({df_engineered['Date'].iloc[0].strftime('%Y-%m-%d')} to {df_engineered['Date'].iloc[len(X_train)-1].strftime('%Y-%m-%d')})")
    print(f"Testing Samples (20%):  {len(X_test):,} ({dates_test.iloc[0].strftime('%Y-%m-%d')} to {dates_test.iloc[-1].strftime('%Y-%m-%d')})")
    print(f"Feature Count: {len(feature_cols)}")
    print(f"Features: {feature_cols}")

    primary_model, metrics_summary, _ = train_and_evaluate_models(
        X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, dates_test
    )

    save_artifacts(primary_model, scaler, feature_cols, metrics_summary)
    print("\n========== PIPELINE COMPLETE ==========")


if __name__ == "__main__":
    main()
