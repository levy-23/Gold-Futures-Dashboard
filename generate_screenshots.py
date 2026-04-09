"""
Generate documentation screenshots from the dashboard modules.
Runs all analysis and saves key visualizations as PNG images.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from src.data_loader import load_and_clean, engineer_features, split_data, get_feature_columns
from src.user_profile import UserProfile, RiskTolerance
from src.forecasting import run_forecasting
from src.risk_analysis import run_risk_analysis
from src.portfolio_optimizer import (
    optimize_portfolio, compute_portfolio_stats_for_weights,
    EQUITY_ANNUAL_RETURN, CASH_ANNUAL_RETURN,
)
from src.simulation import run_simulation
from src.economics import compute_rebalancing_economics
from src.decision_engine import run_decision_analysis

IMG_DIR = "docs/images"
WIDTH = 1100
HEIGHT = 550

def save(fig, name):
    path = os.path.join(IMG_DIR, f"{name}.png")
    fig.write_image(path, width=WIDTH, height=HEIGHT, scale=2)
    print(f"  Saved {path}")


# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
df = load_and_clean()
df = engineer_features(df)
data = split_data(df)

# ── Define 3 example user profiles ──────────────────────────────────────────
profiles = {
    "conservative": UserProfile(
        capital=500000, current_gold_pct=5, current_equity_pct=40,
        current_cash_pct=55, risk_tolerance=RiskTolerance.CONSERVATIVE,
        investment_horizon_days=365,
    ),
    "moderate": UserProfile(
        capital=100000, current_gold_pct=10, current_equity_pct=60,
        current_cash_pct=30, risk_tolerance=RiskTolerance.MODERATE,
        investment_horizon_days=180,
    ),
    "aggressive": UserProfile(
        capital=25000, current_gold_pct=20, current_equity_pct=75,
        current_cash_pct=5, risk_tolerance=RiskTolerance.AGGRESSIVE,
        investment_horizon_days=90,
    ),
}

# ── Run shared analysis (not user-dependent) ────────────────────────────────
print("Running risk analysis...")
risk_stats = run_risk_analysis(df)

print("Running forecasting (this takes ~30s)...")
forecast = run_forecasting(data.train, data.test)

gold_ann_return = risk_stats.annualized_return
gold_ann_vol = risk_stats.annualized_volatility
gold_equity_corr = 0.05

# ══════════════════════════════════════════════════════════════════════════════
# SCREENSHOT 1: Market Overview - Candlestick
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating Tab 1: Market Overview...")
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["date"], open=df["open"], high=df["high"],
    low=df["low"], close=df["close"], name="OHLC",
))
for ma, color in [("ma_7", "#2196F3"), ("ma_30", "#FF9800"), ("ma_90", "#4CAF50")]:
    fig.add_trace(go.Scatter(
        x=df["date"], y=df[ma], mode="lines",
        name=ma.upper().replace("_", " "), line=dict(width=1.5, color=color),
    ))
fig.add_trace(go.Scatter(
    x=df["date"], y=df["bb_upper"], mode="lines",
    name="BB Upper", line=dict(width=1, dash="dot", color="gray"),
))
fig.add_trace(go.Scatter(
    x=df["date"], y=df["bb_lower"], mode="lines",
    name="BB Lower", line=dict(width=1, dash="dot", color="gray"),
    fill="tonexty", fillcolor="rgba(200,200,200,0.1)",
))
fig.update_layout(
    title="Gold Futures Price History with Technical Indicators",
    xaxis_title="Date", yaxis_title="Price (USD)",
    xaxis_rangeslider_visible=False,
    template="plotly_white",
)
save(fig, "01_candlestick")

# ── Returns Distribution ─────────────────────────────────────────────────────
returns = risk_stats.returns_series.dropna()
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=returns, nbinsx=80, name="Daily Returns",
    marker_color="#2196F3", opacity=0.7,
))
fig.add_vline(x=risk_stats.historical_var_95, line_dash="dash", line_color="red",
              annotation_text="VaR 95%")
fig.add_vline(x=risk_stats.historical_var_99, line_dash="dash", line_color="darkred",
              annotation_text="VaR 99%")
fig.update_layout(
    title="Distribution of Daily Returns (Best Fit: Student-t)",
    xaxis_title="Daily Return", yaxis_title="Frequency",
    template="plotly_white",
)
save(fig, "02_returns_distribution")

# ══════════════════════════════════════════════════════════════════════════════
# SCREENSHOT 2: Forecast Tab
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Tab 2: Forecast...")

# Predicted vs actual
for horizon, preds, label in [
    ("5d", forecast.test_predictions_5d, "5-Day"),
    ("30d", forecast.test_predictions_30d, "30-Day"),
]:
    if preds is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=preds["date"], y=preds["actual"], name="Actual", line=dict(color="#333")))
        fig.add_trace(go.Scatter(x=preds["date"], y=preds["xgb_pred"], name="XGBoost", line=dict(color="#2196F3")))
        fig.add_trace(go.Scatter(x=preds["date"], y=preds["rf_pred"], name="Random Forest", line=dict(color="#FF9800", dash="dash")))
        fig.update_layout(
            title=f"{label} Forward Return: Predicted vs Actual (Test Set)",
            xaxis_title="Date", yaxis_title="Return",
            template="plotly_white",
        )
        save(fig, f"03_forecast_{horizon}")

# SHAP feature importance
if forecast.shap_values_5d is not None:
    mean_shap = np.abs(forecast.shap_values_5d).mean(axis=0)
    feat_imp = pd.DataFrame({
        "Feature": forecast.feature_names,
        "Mean |SHAP|": mean_shap,
    }).sort_values("Mean |SHAP|", ascending=True).tail(15)

    fig = go.Figure(go.Bar(
        x=feat_imp["Mean |SHAP|"], y=feat_imp["Feature"],
        orientation="h", marker_color="#2196F3",
    ))
    fig.update_layout(
        title="Top 15 Features by SHAP Importance (5-Day XGBoost Model)",
        xaxis_title="Mean |SHAP Value|",
        template="plotly_white",
    )
    save(fig, "04_shap_importance")


# ══════════════════════════════════════════════════════════════════════════════
# SCREENSHOTS 3-6: Per-profile results
# ══════════════════════════════════════════════════════════════════════════════
for profile_name, profile in profiles.items():
    print(f"\nGenerating screenshots for {profile_name} profile...")

    # Optimization
    opt = optimize_portfolio(
        gold_ann_return, gold_ann_vol, gold_equity_corr,
        profile.risk_params, profile.current_weights,
    )
    curr_ret, curr_vol, curr_sharpe = compute_portfolio_stats_for_weights(
        profile.current_weights, gold_ann_return, gold_ann_vol, gold_equity_corr,
    )

    # Efficient Frontier
    if opt.efficient_frontier:
        ef = opt.efficient_frontier
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[p["risk"] for p in ef], y=[p["return"] for p in ef],
            mode="lines", name="Efficient Frontier", line=dict(color="#2196F3", width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=[curr_vol], y=[curr_ret], mode="markers",
            name="Current Portfolio", marker=dict(color="red", size=16, symbol="circle"),
        ))
        fig.add_trace(go.Scatter(
            x=[opt.expected_volatility], y=[opt.expected_return],
            mode="markers", name="Optimal Portfolio",
            marker=dict(color="green", size=16, symbol="star"),
        ))
        fig.update_layout(
            title=f"Efficient Frontier — {profile_name.title()} Profile (${profile.capital:,.0f})",
            xaxis_title="Portfolio Risk (Annualized Std Dev)",
            yaxis_title="Expected Annual Return",
            template="plotly_white",
        )
        save(fig, f"05_frontier_{profile_name}")

    # Allocation comparison
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]],
                        subplot_titles=["Current Allocation", "Optimal Allocation"])
    colors = ["#FFD700", "#2196F3", "#4CAF50"]
    fig.add_trace(go.Pie(
        labels=["Gold", "Equity", "Cash"],
        values=[profile.current_weights["gold"], profile.current_weights["equity"], profile.current_weights["cash"]],
        marker_colors=colors,
    ), row=1, col=1)
    fig.add_trace(go.Pie(
        labels=["Gold", "Equity", "Cash"],
        values=[opt.weights["gold"], opt.weights["equity"], opt.weights["cash"]],
        marker_colors=colors,
    ), row=1, col=2)
    fig.update_layout(
        title=f"Allocation Comparison — {profile_name.title()} Profile",
        template="plotly_white",
    )
    save(fig, f"06_allocation_{profile_name}")

    # Simulation
    sim = run_simulation(
        returns=df["daily_return"],
        portfolio_weights=opt.weights,
        capital=profile.capital,
        horizon_days=profile.investment_horizon_days,
        gold_annual_return=gold_ann_return,
        gold_annual_vol=gold_ann_vol,
    )

    # Fan chart
    paths = sim.paths
    days = np.arange(paths.shape[1])
    p5 = np.percentile(paths, 5, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=p95, mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=days, y=p5, fill="tonexty", fillcolor="rgba(33,150,243,0.15)",
                              line=dict(width=0), name="5th-95th Percentile"))
    fig.add_trace(go.Scatter(x=days, y=p75, mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=days, y=p25, fill="tonexty", fillcolor="rgba(33,150,243,0.3)",
                              line=dict(width=0), name="25th-75th Percentile"))
    fig.add_trace(go.Scatter(x=days, y=p50, mode="lines", name="Median",
                              line=dict(color="#2196F3", width=2.5)))
    fig.add_hline(y=profile.capital, line_dash="dash", line_color="red",
                   annotation_text="Initial Capital")
    fig.update_layout(
        title=f"Monte Carlo Simulation — {profile_name.title()} ({profile.investment_horizon_days} Days, 10,000 Paths)",
        xaxis_title="Day", yaxis_title="Portfolio Value (USD)",
        template="plotly_white",
    )
    save(fig, f"07_simulation_{profile_name}")

    # Terminal value distribution
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=sim.terminal_values, nbinsx=80,
                                marker_color="#2196F3", opacity=0.7))
    fig.add_vline(x=profile.capital, line_dash="dash", line_color="red",
                   annotation_text="Initial Capital")
    var_95_val = profile.capital * (1 + sim.var_95)
    fig.add_vline(x=var_95_val, line_dash="dot", line_color="orange",
                   annotation_text="VaR 95%")
    fig.update_layout(
        title=f"Terminal Portfolio Value Distribution — {profile_name.title()}",
        xaxis_title="Portfolio Value (USD)", yaxis_title="Frequency",
        template="plotly_white",
    )
    save(fig, f"08_terminal_{profile_name}")

    # Decision analysis
    decision = run_decision_analysis(df["daily_return"], profile, sim.regime_params)

    # Decision criteria bar chart
    fig = go.Figure()
    for criterion, values in [
        ("Expected Value", decision.expected_value),
        ("Maximin", decision.maximin),
        ("Hurwicz", decision.hurwicz),
    ]:
        fig.add_trace(go.Bar(
            name=criterion,
            x=list(values.keys()),
            y=[v * 100 for v in values.values()],
        ))
    fig.update_layout(
        title=f"Strategy Comparison by Decision Criterion — {profile_name.title()}",
        barmode="group", yaxis_title="Expected Return (%)",
        template="plotly_white",
    )
    save(fig, f"09_decision_{profile_name}")

print("\nAll screenshots generated!")
