import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.data_loader import load_and_clean, engineer_features, split_data, get_feature_columns
from src.user_profile import UserProfile, RiskTolerance
from src.forecasting import run_forecasting
from src.risk_analysis import run_risk_analysis
from src.portfolio_optimizer import (
    optimize_portfolio,
    compute_portfolio_stats_for_weights,
    EQUITY_ANNUAL_RETURN,
    CASH_ANNUAL_RETURN,
)
from src.simulation import run_simulation
from src.economics import compute_rebalancing_economics
from src.decision_engine import run_decision_analysis

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold Futures Decision Support Tool",
    page_icon="$",
    layout="wide",
)
st.title("Gold Futures Investment Decision Support Tool")
st.caption("A personalized, multi-disciplinary decision support system for retail investors")


# ── Data Loading (cached) ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = load_and_clean()
    df = engineer_features(df)
    return df


@st.cache_data
def get_splits(_df):
    return split_data(_df)


@st.cache_resource
def cached_forecast(_train, _test):
    return run_forecasting(_train, _test)


@st.cache_data
def cached_risk_analysis(_df):
    return run_risk_analysis(_df)


df = load_data()
data = get_splits(df)

# ── Sidebar: User Inputs ────────────────────────────────────────────────────
st.sidebar.header("Your Portfolio")
capital = st.sidebar.number_input(
    "Total Investable Capital (USD)", min_value=1000, value=100000, step=5000
)
st.sidebar.markdown("**Current Allocation (%)**")
gold_pct = st.sidebar.slider("Gold", 0, 100, 10)
equity_pct = st.sidebar.slider("Equities", 0, 100 - gold_pct, 60)
cash_pct = 100 - gold_pct - equity_pct
st.sidebar.metric("Cash / Bonds", f"{cash_pct}%")

risk_tolerance = st.sidebar.radio(
    "Risk Tolerance", ["Conservative", "Moderate", "Aggressive"], index=1
)
horizon = st.sidebar.selectbox(
    "Investment Horizon (days)", [30, 90, 180, 365], index=2
)

run_clicked = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

# Validate and create profile
try:
    profile = UserProfile(
        capital=float(capital),
        current_gold_pct=float(gold_pct),
        current_equity_pct=float(equity_pct),
        current_cash_pct=float(cash_pct),
        risk_tolerance=RiskTolerance(risk_tolerance),
        investment_horizon_days=horizon,
    )
except Exception as e:
    st.sidebar.error(str(e))
    st.stop()


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Market Overview",
    "Forecast",
    "Portfolio Optimization",
    "Risk Simulation",
    "Decision Analysis",
    "Recommendation",
    "Methodology",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Market Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Gold Futures Market Overview")

    col1, col2, col3, col4 = st.columns(4)
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price_change = latest["close"] - prev["close"]
    col1.metric("Latest Close", f"${latest['close']:,.2f}", f"{price_change:+,.2f}")
    col2.metric("RSI", f"{latest['rsi']:.1f}")
    col3.metric("7-Day Volatility", f"{latest['volatility_7']:.4f}")
    col4.metric("MACD", f"{latest['macd']:.2f}")

    # Candlestick chart with MAs
    fig_candle = go.Figure()
    fig_candle.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="OHLC",
    ))
    for ma, color in [("ma_7", "#2196F3"), ("ma_30", "#FF9800"), ("ma_90", "#4CAF50")]:
        fig_candle.add_trace(go.Scatter(
            x=df["date"], y=df[ma], mode="lines",
            name=ma.upper().replace("_", " "), line=dict(width=1, color=color),
        ))
    fig_candle.add_trace(go.Scatter(
        x=df["date"], y=df["bb_upper"], mode="lines",
        name="BB Upper", line=dict(width=1, dash="dot", color="gray"),
    ))
    fig_candle.add_trace(go.Scatter(
        x=df["date"], y=df["bb_lower"], mode="lines",
        name="BB Lower", line=dict(width=1, dash="dot", color="gray"),
        fill="tonexty", fillcolor="rgba(200,200,200,0.1)",
    ))
    fig_candle.update_layout(
        title="Gold Futures Price History",
        xaxis_title="Date", yaxis_title="Price (USD)",
        height=500, xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    # Returns distribution
    risk_stats = cached_risk_analysis(df)
    returns = risk_stats.returns_series.dropna()

    col_a, col_b = st.columns(2)
    with col_a:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=returns, nbinsx=80, name="Daily Returns",
            marker_color="#2196F3", opacity=0.7,
        ))
        fig_hist.update_layout(
            title="Distribution of Daily Returns",
            xaxis_title="Daily Return", yaxis_title="Frequency", height=400,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        stats_data = {
            "Metric": [
                "Mean Daily Return", "Annualized Return", "Annualized Volatility",
                "Skewness", "Excess Kurtosis",
                "Historical VaR (95%)", "Historical VaR (99%)",
                "Best Distribution Fit",
            ],
            "Value": [
                f"{risk_stats.mean_return:.6f}",
                f"{risk_stats.annualized_return:.4f} ({risk_stats.annualized_return*100:.2f}%)",
                f"{risk_stats.annualized_volatility:.4f} ({risk_stats.annualized_volatility*100:.2f}%)",
                f"{risk_stats.skewness:.4f}",
                f"{risk_stats.kurtosis:.4f}",
                f"{risk_stats.historical_var_95:.6f} ({risk_stats.historical_var_95*100:.4f}%)",
                f"{risk_stats.historical_var_99:.6f} ({risk_stats.historical_var_99*100:.4f}%)",
                risk_stats.best_distribution,
            ],
        }
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

        # Hypothesis test result
        if risk_stats.mean_significantly_nonzero:
            st.success(
                f"Mean daily return IS statistically significantly different from zero "
                f"(t={risk_stats.ttest_statistic:.3f}, p={risk_stats.ttest_pvalue:.4f})"
            )
        else:
            st.info(
                f"Mean daily return is NOT statistically significantly different from zero "
                f"(t={risk_stats.ttest_statistic:.3f}, p={risk_stats.ttest_pvalue:.4f})"
            )

        if risk_stats.volatility_clustering:
            st.warning(
                f"Volatility clustering detected (Ljung-Box on squared returns p={risk_stats.squared_returns_ljungbox_pvalue:.4f}). "
                f"This justifies using GARCH for volatility modeling."
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Forecast
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Price Forecast (MSCI 446 - Machine Learning)")

    with st.spinner("Training forecasting models (walk-forward validation)..."):
        forecast = cached_forecast(data.train, data.test)

    # Metrics comparison
    st.subheader("Model Performance on Test Set")
    metrics_df = pd.DataFrame({
        "Model": ["XGBoost", "Random Forest", "XGBoost", "Random Forest"],
        "Horizon": ["5-Day", "5-Day", "30-Day", "30-Day"],
        "RMSE": [
            forecast.xgb_metrics_5d.rmse, forecast.rf_metrics_5d.rmse,
            forecast.xgb_metrics_30d.rmse, forecast.rf_metrics_30d.rmse,
        ],
        "MAE": [
            forecast.xgb_metrics_5d.mae, forecast.rf_metrics_5d.mae,
            forecast.xgb_metrics_30d.mae, forecast.rf_metrics_30d.mae,
        ],
        "Directional Accuracy": [
            f"{forecast.xgb_metrics_5d.directional_accuracy:.1%}",
            f"{forecast.rf_metrics_5d.directional_accuracy:.1%}",
            f"{forecast.xgb_metrics_30d.directional_accuracy:.1%}",
            f"{forecast.rf_metrics_30d.directional_accuracy:.1%}",
        ],
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Predicted vs Actual charts
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        if forecast.test_predictions_5d is not None:
            tp = forecast.test_predictions_5d
            fig_5d = go.Figure()
            fig_5d.add_trace(go.Scatter(x=tp["date"], y=tp["actual"], name="Actual", line=dict(color="#333")))
            fig_5d.add_trace(go.Scatter(x=tp["date"], y=tp["xgb_pred"], name="XGBoost", line=dict(color="#2196F3")))
            fig_5d.add_trace(go.Scatter(x=tp["date"], y=tp["rf_pred"], name="Random Forest", line=dict(color="#FF9800", dash="dash")))
            fig_5d.update_layout(title="5-Day Forward Return: Predicted vs Actual", height=400,
                                 xaxis_title="Date", yaxis_title="Return")
            st.plotly_chart(fig_5d, use_container_width=True)

    with col_f2:
        if forecast.test_predictions_30d is not None:
            tp = forecast.test_predictions_30d
            fig_30d = go.Figure()
            fig_30d.add_trace(go.Scatter(x=tp["date"], y=tp["actual"], name="Actual", line=dict(color="#333")))
            fig_30d.add_trace(go.Scatter(x=tp["date"], y=tp["xgb_pred"], name="XGBoost", line=dict(color="#2196F3")))
            fig_30d.add_trace(go.Scatter(x=tp["date"], y=tp["rf_pred"], name="Random Forest", line=dict(color="#FF9800", dash="dash")))
            fig_30d.update_layout(title="30-Day Forward Return: Predicted vs Actual", height=400,
                                  xaxis_title="Date", yaxis_title="Return")
            st.plotly_chart(fig_30d, use_container_width=True)

    # Latest forecast
    st.subheader("Latest Forecast")
    col_l1, col_l2, col_l3 = st.columns(3)
    col_l1.metric("Predicted 5-Day Return", f"{forecast.predicted_return_5d:.4f} ({forecast.predicted_return_5d*100:.2f}%)")
    col_l2.metric("Predicted 30-Day Return", f"{forecast.predicted_return_30d:.4f} ({forecast.predicted_return_30d*100:.2f}%)")
    col_l3.metric("Predicted Daily Volatility (GARCH)", f"{forecast.predicted_volatility:.6f}")

    st.caption(f"95% CI for 5-day: [{forecast.confidence_lower_5d*100:.2f}%, {forecast.confidence_upper_5d*100:.2f}%]")
    st.caption(f"95% CI for 30-day: [{forecast.confidence_lower_30d*100:.2f}%, {forecast.confidence_upper_30d*100:.2f}%]")

    # SHAP feature importance
    if forecast.shap_values_5d is not None:
        st.subheader("Feature Importance (SHAP)")
        mean_shap = np.abs(forecast.shap_values_5d).mean(axis=0)
        feat_imp = pd.DataFrame({
            "Feature": forecast.feature_names,
            "Mean |SHAP|": mean_shap,
        }).sort_values("Mean |SHAP|", ascending=True).tail(15)

        fig_shap = go.Figure(go.Bar(
            x=feat_imp["Mean |SHAP|"], y=feat_imp["Feature"],
            orientation="h", marker_color="#2196F3",
        ))
        fig_shap.update_layout(title="Top 15 Features by SHAP Importance (5-Day Model)",
                                height=450, xaxis_title="Mean |SHAP Value|")
        st.plotly_chart(fig_shap, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Portfolio Optimization
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Portfolio Optimization (MSCI 331/332 - Optimization)")

    # Compute gold parameters from data
    gold_ann_return = risk_stats.annualized_return
    gold_ann_vol = risk_stats.annualized_volatility

    # Estimate gold-equity correlation (use daily return correlation with a proxy)
    # Since we only have gold data, use a typical value
    gold_equity_corr = 0.05  # Gold has near-zero correlation with equities historically

    opt_result = optimize_portfolio(
        gold_annual_return=gold_ann_return,
        gold_annual_vol=gold_ann_vol,
        gold_equity_corr=gold_equity_corr,
        risk_params=profile.risk_params,
        current_weights=profile.current_weights,
    )

    # Current portfolio stats
    curr_ret, curr_vol, curr_sharpe = compute_portfolio_stats_for_weights(
        profile.current_weights, gold_ann_return, gold_ann_vol, gold_equity_corr
    )

    # Metrics comparison
    col_o1, col_o2 = st.columns(2)
    with col_o1:
        st.subheader("Current Portfolio")
        st.metric("Expected Return", f"{curr_ret*100:.2f}%")
        st.metric("Volatility", f"{curr_vol*100:.2f}%")
        st.metric("Sharpe Ratio", f"{curr_sharpe:.3f}")
    with col_o2:
        st.subheader("Optimal Portfolio")
        st.metric("Expected Return", f"{opt_result.expected_return*100:.2f}%",
                   f"{(opt_result.expected_return - curr_ret)*100:+.2f}%")
        st.metric("Volatility", f"{opt_result.expected_volatility*100:.2f}%",
                   f"{(opt_result.expected_volatility - curr_vol)*100:+.2f}%", delta_color="inverse")
        st.metric("Sharpe Ratio", f"{opt_result.sharpe_ratio:.3f}",
                   f"{opt_result.sharpe_ratio - curr_sharpe:+.3f}")

    # Allocation pie charts
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        fig_pie_curr = go.Figure(go.Pie(
            labels=["Gold", "Equity", "Cash"],
            values=[profile.current_weights["gold"], profile.current_weights["equity"], profile.current_weights["cash"]],
            marker_colors=["#FFD700", "#2196F3", "#4CAF50"],
        ))
        fig_pie_curr.update_layout(title="Current Allocation", height=350)
        st.plotly_chart(fig_pie_curr, use_container_width=True)

    with col_p2:
        fig_pie_opt = go.Figure(go.Pie(
            labels=["Gold", "Equity", "Cash"],
            values=[opt_result.weights["gold"], opt_result.weights["equity"], opt_result.weights["cash"]],
            marker_colors=["#FFD700", "#2196F3", "#4CAF50"],
        ))
        fig_pie_opt.update_layout(title="Optimal Allocation", height=350)
        st.plotly_chart(fig_pie_opt, use_container_width=True)

    # Efficient Frontier
    if opt_result.efficient_frontier:
        st.subheader("Efficient Frontier")
        ef = opt_result.efficient_frontier
        ef_risks = [p["risk"] for p in ef]
        ef_returns = [p["return"] for p in ef]

        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(
            x=ef_risks, y=ef_returns, mode="lines",
            name="Efficient Frontier", line=dict(color="#2196F3", width=2),
        ))
        fig_ef.add_trace(go.Scatter(
            x=[curr_vol], y=[curr_ret], mode="markers",
            name="Your Portfolio", marker=dict(color="red", size=14, symbol="circle"),
        ))
        fig_ef.add_trace(go.Scatter(
            x=[opt_result.expected_volatility], y=[opt_result.expected_return],
            mode="markers", name="Optimal Portfolio",
            marker=dict(color="green", size=14, symbol="star"),
        ))
        fig_ef.update_layout(
            title="Efficient Frontier",
            xaxis_title="Portfolio Risk (Std Dev)",
            yaxis_title="Expected Annual Return",
            height=500,
        )
        st.plotly_chart(fig_ef, use_container_width=True)

        st.caption(
            f"Constraints applied: Max gold = {profile.risk_params['max_gold_weight']*100:.0f}%, "
            f"Min cash = {profile.risk_params['min_cash_weight']*100:.0f}% "
            f"(based on {risk_tolerance} risk profile)"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Risk Simulation
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Monte Carlo Risk Simulation (MSCI 333 - Simulation)")

    sim_result = run_simulation(
        returns=df["daily_return"],
        portfolio_weights=opt_result.weights,
        capital=profile.capital,
        horizon_days=profile.investment_horizon_days,
        gold_annual_return=gold_ann_return,
        gold_annual_vol=gold_ann_vol,
    )

    # Key metrics
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Expected Value", f"${sim_result.expected_terminal_value:,.0f}")
    col_s2.metric("Probability of Loss", f"{sim_result.probability_of_loss:.1%}")
    col_s3.metric("VaR 95%", f"{sim_result.var_95:.2%}")
    col_s4.metric("CVaR 95%", f"{sim_result.cvar_95:.2%}")

    # Fan chart (percentile bands)
    st.subheader("Simulated Portfolio Paths")
    paths = sim_result.paths
    n_days = paths.shape[1]
    days = np.arange(n_days)

    p5 = np.percentile(paths, 5, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    fig_fan = go.Figure()
    fig_fan.add_trace(go.Scatter(x=days, y=p95, mode="lines", line=dict(width=0), showlegend=False))
    fig_fan.add_trace(go.Scatter(x=days, y=p5, mode="lines", fill="tonexty",
                                  fillcolor="rgba(33,150,243,0.15)", line=dict(width=0), name="5th-95th"))
    fig_fan.add_trace(go.Scatter(x=days, y=p75, mode="lines", line=dict(width=0), showlegend=False))
    fig_fan.add_trace(go.Scatter(x=days, y=p25, mode="lines", fill="tonexty",
                                  fillcolor="rgba(33,150,243,0.3)", line=dict(width=0), name="25th-75th"))
    fig_fan.add_trace(go.Scatter(x=days, y=p50, mode="lines", name="Median",
                                  line=dict(color="#2196F3", width=2)))
    fig_fan.add_hline(y=profile.capital, line_dash="dash", line_color="red",
                       annotation_text="Initial Capital")
    fig_fan.update_layout(
        title=f"10,000 Simulated Paths ({profile.investment_horizon_days} Days)",
        xaxis_title="Day", yaxis_title="Portfolio Value (USD)", height=500,
    )
    st.plotly_chart(fig_fan, use_container_width=True)

    # Terminal value distribution
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        fig_term = go.Figure()
        fig_term.add_trace(go.Histogram(x=sim_result.terminal_values, nbinsx=80,
                                         marker_color="#2196F3", opacity=0.7, name="Terminal Value"))
        fig_term.add_vline(x=profile.capital, line_dash="dash", line_color="red",
                           annotation_text="Initial Capital")
        var_95_val = profile.capital * (1 + sim_result.var_95)
        fig_term.add_vline(x=var_95_val, line_dash="dot", line_color="orange",
                           annotation_text=f"VaR 95%")
        fig_term.update_layout(title="Distribution of Terminal Portfolio Values",
                                xaxis_title="Portfolio Value (USD)", yaxis_title="Frequency", height=400)
        st.plotly_chart(fig_term, use_container_width=True)

    with col_h2:
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Histogram(x=sim_result.max_drawdowns, nbinsx=60,
                                       marker_color="#F44336", opacity=0.7, name="Max Drawdown"))
        fig_dd.update_layout(title="Distribution of Maximum Drawdowns",
                              xaxis_title="Max Drawdown", yaxis_title="Frequency", height=400)
        st.plotly_chart(fig_dd, use_container_width=True)

    # Regime parameters
    st.subheader("Regime Parameters (Estimated from Data)")
    rp = sim_result.regime_params
    regime_df = pd.DataFrame({
        "Regime": ["Bull", "Bear"],
        "Daily Mean Return": [f"{rp['bull']['mu']:.6f}", f"{rp['bear']['mu']:.6f}"],
        "Daily Volatility": [f"{rp['bull']['sigma']:.6f}", f"{rp['bear']['sigma']:.6f}"],
        "Probability": [f"{rp['bull']['probability']:.1%}", f"{rp['bear']['probability']:.1%}"],
    })
    st.dataframe(regime_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: Decision Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Decision Analysis (MSCI 452 - Decision Making Under Uncertainty)")

    decision = run_decision_analysis(
        returns_data=df["daily_return"],
        user_profile=profile,
        regime_params=sim_result.regime_params,
    )

    # Decision matrix
    st.subheader("Decision Matrix (Expected Portfolio Return by Strategy x Scenario)")
    matrix_df = pd.DataFrame(
        decision.decision_matrix,
        index=decision.strategy_names,
        columns=decision.scenario_names,
    )
    matrix_df = matrix_df.map(lambda x: f"{x:.4f} ({x*100:.2f}%)")
    st.dataframe(matrix_df, use_container_width=True)

    st.caption(
        f"Scenario probabilities — Bull: {decision.scenario_probabilities[0]:.1%}, "
        f"Bear: {decision.scenario_probabilities[1]:.1%}, "
        f"Sideways: {decision.scenario_probabilities[2]:.1%}"
    )

    # Decision criteria results
    st.subheader("Decision Criteria Comparison")
    criteria_data = []
    for strategy in decision.strategy_names:
        criteria_data.append({
            "Strategy": strategy,
            "Expected Value": f"{decision.expected_value[strategy]*100:.2f}%",
            "Maximin": f"{decision.maximin[strategy]*100:.2f}%",
            "Max Regret": f"{decision.minimax_regret[strategy]*100:.2f}%",
            "Hurwicz": f"{decision.hurwicz[strategy]*100:.2f}%",
        })
    st.dataframe(pd.DataFrame(criteria_data), use_container_width=True, hide_index=True)

    # Bar chart comparing strategies
    fig_criteria = go.Figure()
    for criterion, values in [
        ("Expected Value", decision.expected_value),
        ("Maximin", decision.maximin),
        ("Hurwicz", decision.hurwicz),
    ]:
        fig_criteria.add_trace(go.Bar(
            name=criterion,
            x=list(values.keys()),
            y=[v * 100 for v in values.values()],
        ))
    fig_criteria.update_layout(
        title="Strategy Comparison by Decision Criterion",
        barmode="group", yaxis_title="Expected Return (%)", height=450,
    )
    st.plotly_chart(fig_criteria, use_container_width=True)

    # Sensitivity analysis
    st.subheader("Sensitivity Analysis")
    st.markdown("**Which strategy wins under each criterion?**")
    for criterion, winner in decision.sensitivity.items():
        marker = " (PRIMARY)" if criterion == "Expected Value" else ""
        st.markdown(f"- **{criterion}**: {winner}{marker}")

    # Economics
    st.subheader("Rebalancing Economics (MSCI 261/263)")
    expected_returns = {
        "gold": gold_ann_return,
        "equity": EQUITY_ANNUAL_RETURN,
        "cash": CASH_ANNUAL_RETURN,
    }
    econ = compute_rebalancing_economics(
        capital=profile.capital,
        current_weights=profile.current_weights,
        optimal_weights=opt_result.weights,
        expected_returns=expected_returns,
        investment_horizon_days=profile.investment_horizon_days,
    )

    col_e1, col_e2, col_e3 = st.columns(3)
    col_e1.metric("Transaction Cost", f"${econ.transaction_cost_usd:,.2f}")
    col_e2.metric("NPV of Rebalancing", f"${econ.npv_of_rebalancing:,.2f}")
    col_e3.metric("Break-Even", f"{econ.break_even_days} days")

    if econ.worth_rebalancing:
        st.success(econ.explanation)
    else:
        st.warning(econ.explanation)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: Recommendation
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("Personalized Recommendation Summary")

    st.subheader(f"Recommended Strategy: {decision.recommended_strategy}")

    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Gold Allocation", f"{decision.recommended_weights['gold']*100:.0f}%",
                   f"{(decision.recommended_weights['gold'] - profile.current_weights['gold'])*100:+.0f}%")
    col_r2.metric("Equity Allocation", f"{decision.recommended_weights['equity']*100:.0f}%",
                   f"{(decision.recommended_weights['equity'] - profile.current_weights['equity'])*100:+.0f}%")
    col_r3.metric("Cash Allocation", f"{decision.recommended_weights['cash']*100:.0f}%",
                   f"{(decision.recommended_weights['cash'] - profile.current_weights['cash'])*100:+.0f}%")

    st.divider()

    # Summary table
    summary_data = {
        "Aspect": [
            "Investment Horizon",
            "Risk Profile",
            "ML Forecast (5-day return)",
            "ML Forecast (30-day return)",
            "Portfolio Expected Return",
            "Portfolio Volatility",
            "Portfolio Sharpe Ratio",
            "Probability of Loss",
            "Value at Risk (95%)",
            "Rebalancing NPV",
            "Decision Criterion Used",
        ],
        "Value": [
            f"{profile.investment_horizon_days} days",
            risk_tolerance,
            f"{forecast.predicted_return_5d*100:.2f}%",
            f"{forecast.predicted_return_30d*100:.2f}%",
            f"{opt_result.expected_return*100:.2f}%",
            f"{opt_result.expected_volatility*100:.2f}%",
            f"{opt_result.sharpe_ratio:.3f}",
            f"{sim_result.probability_of_loss:.1%}",
            f"{sim_result.var_95:.2%}",
            f"${econ.npv_of_rebalancing:,.2f}",
            "Expected Value (probability-weighted)",
        ],
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    st.divider()
    st.warning(
        "**Disclaimer:** This tool is for educational and informational purposes only. "
        "It does not constitute financial advice. Past performance does not guarantee future results. "
        "Always consult a qualified financial advisor before making investment decisions."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7: Methodology
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.header("Methodology & Course Mapping")
    st.markdown("""
    This decision support tool integrates multiple analytical techniques from the
    University of Waterloo's Management Engineering (MGTE) curriculum.

    | Course | Module | Technique Applied |
    |--------|--------|-------------------|
    | **MSCI 251/253** - Probability & Statistics | Risk Analysis | Distribution fitting (Normal, Student-t, Skew-Normal), hypothesis testing, VaR/CVaR, autocorrelation analysis |
    | **MSCI 261/263** - Engineering Economics | Rebalancing Economics | NPV of rebalancing decision, break-even analysis, transaction cost modeling, opportunity cost |
    | **MSCI 331/332** - Optimization | Portfolio Optimizer | Markowitz mean-variance optimization via constrained SLSQP, efficient frontier computation |
    | **MSCI 333** - Simulation | Monte Carlo Simulation | Regime-switching Geometric Brownian Motion, 10,000 paths, VaR/CVaR from simulation |
    | **MSCI 436** - Decision Support Systems | Dashboard | Interactive Streamlit DSS with personalized user inputs and real-time outputs |
    | **MSCI 446** - Machine Learning | Forecasting | XGBoost and Random Forest with walk-forward validation, GARCH volatility, SHAP explanations |
    | **MSCI 452** - Decision Under Uncertainty | Decision Engine | Decision matrix with Expected Value, Maximin, Minimax Regret, and Hurwicz criteria |

    ### Design Choices
    - **XGBoost over LSTM**: With only ~1,167 data points, tree-based models outperform deep learning.
      This is an intentional, data-informed choice — not a limitation.
    - **Regime-switching simulation**: Gold markets exhibit distinct bull/bear regimes. A single-distribution
      GBM would underestimate tail risk. The regime-switching model captures this behavior.
    - **Multiple decision criteria**: Different investors may prefer different decision frameworks.
      Showing all four criteria allows the user to understand how their recommendation might change
      under different philosophical approaches to uncertainty.
    - **Personalization**: Every output depends on the user's specific portfolio, risk tolerance, and
      time horizon — making this a true decision support tool, not a generic predictor.

    ### Data Source
    - **Dataset**: Gold Futures (GC=F) from Yahoo Finance via Kaggle
    - **Period**: June 2021 - January 2026 (1,167 trading days)
    - **Features**: 17 columns including OHLCV, moving averages, RSI, MACD, Bollinger Bands, volatility measures
    """)
