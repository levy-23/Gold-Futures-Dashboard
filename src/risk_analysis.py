import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox


@dataclass
class DistributionFit:
    name: str
    params: tuple
    ks_statistic: float
    ks_pvalue: float
    ad_statistic: float
    is_best: bool = False


@dataclass
class RiskStats:
    mean_return: float
    std_return: float
    annualized_return: float
    annualized_volatility: float
    skewness: float
    kurtosis: float

    # VaR and CVaR
    historical_var_95: float
    historical_var_99: float
    parametric_var_95: float
    parametric_var_99: float
    historical_cvar_95: float
    historical_cvar_99: float

    # Hypothesis test: is mean return != 0?
    ttest_statistic: float
    ttest_pvalue: float
    mean_significantly_nonzero: bool

    # Autocorrelation
    returns_ljungbox_pvalue: float
    squared_returns_ljungbox_pvalue: float
    volatility_clustering: bool

    # Distribution fits
    distribution_fits: list[DistributionFit]
    best_distribution: str

    # Raw data for plotting
    returns_series: pd.Series = None


def _fit_distributions(returns: np.ndarray) -> list[DistributionFit]:
    fits = []

    # Normal
    mu, sigma = stats.norm.fit(returns)
    ks_stat, ks_p = stats.kstest(returns, "norm", args=(mu, sigma))
    ad_result = stats.anderson(returns, dist="norm")
    fits.append(DistributionFit("Normal", (mu, sigma), ks_stat, ks_p, ad_result.statistic))

    # Student-t
    df_t, loc_t, scale_t = stats.t.fit(returns)
    ks_stat, ks_p = stats.kstest(returns, "t", args=(df_t, loc_t, scale_t))
    # Anderson-Darling not directly available for t, use KS as proxy
    fits.append(DistributionFit("Student-t", (df_t, loc_t, scale_t), ks_stat, ks_p, 0.0))

    # Skewed-t (skew normal as approximation)
    try:
        a, loc_sn, scale_sn = stats.skewnorm.fit(returns)
        ks_stat, ks_p = stats.kstest(returns, "skewnorm", args=(a, loc_sn, scale_sn))
        fits.append(DistributionFit("Skew-Normal", (a, loc_sn, scale_sn), ks_stat, ks_p, 0.0))
    except Exception:
        pass

    # Mark the best fit (highest KS p-value = best fit)
    best = max(fits, key=lambda f: f.ks_pvalue)
    best.is_best = True

    return fits


def run_risk_analysis(df: pd.DataFrame) -> RiskStats:
    returns = df["daily_return"].dropna()
    r = returns.values

    # Basic statistics
    mean_r = float(np.mean(r))
    std_r = float(np.std(r, ddof=1))
    ann_return = float(mean_r * 252)
    ann_vol = float(std_r * np.sqrt(252))
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r))  # excess kurtosis

    # Historical VaR and CVaR
    hist_var_95 = float(np.percentile(r, 5))
    hist_var_99 = float(np.percentile(r, 1))
    hist_cvar_95 = float(np.mean(r[r <= hist_var_95]))
    hist_cvar_99 = float(np.mean(r[r <= hist_var_99]))

    # Parametric VaR (assuming normal — will compare to historical)
    param_var_95 = float(mean_r + stats.norm.ppf(0.05) * std_r)
    param_var_99 = float(mean_r + stats.norm.ppf(0.01) * std_r)

    # T-test: is mean daily return significantly different from zero?
    t_stat, t_pval = stats.ttest_1samp(r, 0)

    # Autocorrelation tests (Ljung-Box, lag=10)
    try:
        lb_returns = acorr_ljungbox(r, lags=10, return_df=True)
        lb_returns_p = float(lb_returns["lb_pvalue"].iloc[-1])
    except Exception:
        lb_returns_p = 1.0

    try:
        lb_squared = acorr_ljungbox(r**2, lags=10, return_df=True)
        lb_squared_p = float(lb_squared["lb_pvalue"].iloc[-1])
    except Exception:
        lb_squared_p = 1.0

    # Distribution fitting
    dist_fits = _fit_distributions(r)
    best_dist = next(f for f in dist_fits if f.is_best)

    return RiskStats(
        mean_return=mean_r,
        std_return=std_r,
        annualized_return=ann_return,
        annualized_volatility=ann_vol,
        skewness=skew,
        kurtosis=kurt,
        historical_var_95=hist_var_95,
        historical_var_99=hist_var_99,
        parametric_var_95=param_var_95,
        parametric_var_99=param_var_99,
        historical_cvar_95=hist_cvar_95,
        historical_cvar_99=hist_cvar_99,
        ttest_statistic=float(t_stat),
        ttest_pvalue=float(t_pval),
        mean_significantly_nonzero=(t_pval < 0.05),
        returns_ljungbox_pvalue=lb_returns_p,
        squared_returns_ljungbox_pvalue=lb_squared_p,
        volatility_clustering=(lb_squared_p < 0.05),
        distribution_fits=dist_fits,
        best_distribution=best_dist.name,
        returns_series=returns,
    )
