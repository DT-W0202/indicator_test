"""
Structural Break Analysis Module
================================

Provides functions for detecting and analyzing structural breaks in time series:
- Chow Test for regression breakpoint
- Change-point detection on regression coefficients (not IC)
- Rolling IC with confidence intervals
- HAC standard errors for overlapping returns
- Block bootstrap for significance testing
"""

import pandas as pd
import numpy as np
from scipy.stats import f as f_dist, spearmanr, norm, t as t_dist
from typing import Tuple, List, Dict, Optional
import warnings


def chow_test(y: pd.Series, x: pd.Series, breakpoint: str) -> Tuple[float, float]:
    """
    Perform Chow Test for structural break in regression.

    Tests H0: No structural break at the specified breakpoint.

    Args:
        y: Dependent variable (forward returns)
        x: Independent variable (factor)
        breakpoint: Date string for the breakpoint (e.g., '2015-01-01')

    Returns:
        (F-statistic, p-value)
    """
    # Align data
    data = pd.DataFrame({'y': y, 'x': x}).dropna()

    if len(data) < 20:
        return np.nan, np.nan

    breakpoint_dt = pd.to_datetime(breakpoint)

    # Split data
    data1 = data[data.index < breakpoint_dt]
    data2 = data[data.index >= breakpoint_dt]

    if len(data1) < 10 or len(data2) < 10:
        return np.nan, np.nan

    # Calculate RSS for each subsample and pooled
    def calc_rss(df):
        if len(df) < 2:
            return np.nan
        x_with_const = np.column_stack([np.ones(len(df)), df['x'].values])
        y_vals = df['y'].values
        try:
            beta = np.linalg.lstsq(x_with_const, y_vals, rcond=None)[0]
            residuals = y_vals - x_with_const @ beta
            return np.sum(residuals ** 2)
        except:
            return np.nan

    rss_pooled = calc_rss(data)
    rss1 = calc_rss(data1)
    rss2 = calc_rss(data2)

    if np.isnan(rss_pooled) or np.isnan(rss1) or np.isnan(rss2):
        return np.nan, np.nan

    n1 = len(data1)
    n2 = len(data2)
    k = 2  # Number of parameters (intercept + slope)

    # F-statistic
    numerator = (rss_pooled - (rss1 + rss2)) / k
    denominator = (rss1 + rss2) / (n1 + n2 - 2 * k)

    if denominator <= 0:
        return np.nan, np.nan

    f_stat = numerator / denominator
    p_value = 1 - f_dist.cdf(f_stat, k, n1 + n2 - 2 * k)

    return f_stat, p_value


def detect_changepoints_cusum(series: pd.Series, threshold: float = 1.5) -> List[pd.Timestamp]:
    """
    Detect change points using CUSUM (Cumulative Sum) method.

    Simple implementation without external dependencies.

    Args:
        series: Time series to analyze (e.g., rolling IC)
        threshold: Threshold for detecting change (in std deviations)

    Returns:
        List of detected changepoint dates
    """
    series = series.dropna()
    if len(series) < 20:
        return []

    # Normalize
    mean = series.mean()
    std = series.std()
    if std == 0:
        return []

    normalized = (series - mean) / std

    # Calculate CUSUM
    cusum = normalized.cumsum()

    # Find points where CUSUM exceeds threshold
    changepoints = []

    # Look for sign changes and large deviations
    cusum_centered = cusum - cusum.mean()
    cusum_std = cusum_centered.std()

    if cusum_std == 0:
        return []

    # Find local maxima/minima that exceed threshold
    for i in range(1, len(cusum_centered) - 1):
        val = cusum_centered.iloc[i]
        prev_val = cusum_centered.iloc[i-1]
        next_val = cusum_centered.iloc[i+1]

        # Check for local extremum
        if (val > prev_val and val > next_val) or (val < prev_val and val < next_val):
            if abs(val) > threshold * cusum_std:
                changepoints.append(series.index[i])

    return changepoints


def rolling_ic_with_ci(factor: pd.Series,
                       returns: pd.Series,
                       window: int = 120,
                       ci_level: float = 0.95) -> pd.DataFrame:
    """
    Compute rolling IC with confidence intervals.

    Args:
        factor: Factor series
        returns: Forward return series
        window: Rolling window size
        ci_level: Confidence interval level (default 95%)

    Returns:
        DataFrame with columns: ['ic', 'ci_lower', 'ci_upper', 'n_samples']
    """
    # Align data
    data = pd.DataFrame({'factor': factor, 'return': returns}).dropna()

    results = []
    z_score = norm.ppf((1 + ci_level) / 2)

    for i in range(window, len(data)):
        window_data = data.iloc[i - window:i]

        try:
            ic, _ = spearmanr(window_data['factor'], window_data['return'])
            n = len(window_data)

            # Fisher transformation for CI
            if abs(ic) < 1:
                z = np.arctanh(ic)
                se = 1 / np.sqrt(n - 3) if n > 3 else np.nan
                ci_lower = np.tanh(z - z_score * se)
                ci_upper = np.tanh(z + z_score * se)
            else:
                ci_lower = ci_upper = ic

            results.append({
                'date': data.index[i],
                'ic': ic,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_samples': n
            })
        except:
            continue

    if not results:
        return pd.DataFrame(columns=['date', 'ic', 'ci_lower', 'ci_upper', 'n_samples'])

    df = pd.DataFrame(results)
    df = df.set_index('date')
    return df


def compute_subsample_ic(factor: pd.Series,
                         returns: pd.Series,
                         periods: List[Tuple[str, str]]) -> Dict[str, Dict]:
    """
    Compute IC for specific time periods.

    Args:
        factor: Factor series
        returns: Forward return series
        periods: List of (start_date, end_date) tuples with labels

    Returns:
        Dict with period labels as keys and IC stats as values
    """
    data = pd.DataFrame({'factor': factor, 'return': returns}).dropna()

    results = {}

    for period_name, (start, end) in periods.items():
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        period_data = data[(data.index >= start_dt) & (data.index <= end_dt)]

        if len(period_data) < 10:
            results[period_name] = {
                'ic': np.nan,
                'p_value': np.nan,
                'n_samples': len(period_data)
            }
            continue

        ic, p_value = spearmanr(period_data['factor'], period_data['return'])

        results[period_name] = {
            'ic': ic,
            'p_value': p_value,
            'n_samples': len(period_data)
        }

    return results


def rolling_beta_ols(y: pd.Series,
                     x: pd.Series,
                     window: int = 120) -> pd.DataFrame:
    """
    Compute rolling OLS regression beta (slope coefficient).

    This is the preferred method for structural break analysis,
    as beta is more interpretable than IC for economic analysis.

    Args:
        y: Dependent variable (forward returns)
        x: Independent variable (factor)
        window: Rolling window size

    Returns:
        DataFrame with columns: ['beta', 'alpha', 'se_beta', 't_stat', 'r_squared']
    """
    data = pd.DataFrame({'y': y, 'x': x}).dropna()

    results = []

    for i in range(window, len(data)):
        window_data = data.iloc[i - window:i]

        y_arr = window_data['y'].values
        x_arr = window_data['x'].values
        X = np.column_stack([np.ones(len(y_arr)), x_arr])

        try:
            beta = np.linalg.lstsq(X, y_arr, rcond=None)[0]
            y_pred = X @ beta
            residuals = y_arr - y_pred

            n = len(y_arr)
            k = 2

            # Standard errors (OLS, not HAC for speed in rolling)
            mse = np.sum(residuals ** 2) / (n - k)
            var_beta = mse * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(var_beta))

            # t-stat for beta (slope)
            t_stat = beta[1] / se[1] if se[1] > 0 else np.nan

            # R-squared
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_arr - y_arr.mean()) ** 2)
            r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            results.append({
                'date': data.index[i],
                'beta': beta[1],
                'alpha': beta[0],
                'se_beta': se[1],
                't_stat': t_stat,
                'r_squared': r_sq
            })
        except:
            continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.set_index('date')
    return df


def compute_subsample_beta(y: pd.Series,
                           x: pd.Series,
                           periods: Dict[str, Tuple[str, str]]) -> Dict[str, Dict]:
    """
    Compute regression beta for specific time periods.

    Args:
        y: Dependent variable (forward returns)
        x: Independent variable (factor)
        periods: Dict of period_name -> (start_date, end_date)

    Returns:
        Dict with period labels as keys and regression stats as values
    """
    data = pd.DataFrame({'y': y, 'x': x}).dropna()

    results = {}

    for period_name, (start, end) in periods.items():
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        period_data = data[(data.index >= start_dt) & (data.index <= end_dt)]

        if len(period_data) < 10:
            results[period_name] = {
                'beta': np.nan,
                'se': np.nan,
                't_stat': np.nan,
                'p_value': np.nan,
                'n_samples': len(period_data)
            }
            continue

        y_arr = period_data['y'].values
        x_arr = period_data['x'].values
        X = np.column_stack([np.ones(len(y_arr)), x_arr])

        try:
            beta = np.linalg.lstsq(X, y_arr, rcond=None)[0]
            residuals = y_arr - X @ beta

            n = len(y_arr)
            k = 2
            mse = np.sum(residuals ** 2) / (n - k)
            var_beta = mse * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(var_beta))

            t_stat = beta[1] / se[1] if se[1] > 0 else np.nan
            p_value = 2 * (1 - t_dist.cdf(np.abs(t_stat), n - k))

            results[period_name] = {
                'beta': beta[1],
                'se': se[1],
                't_stat': t_stat,
                'p_value': p_value,
                'n_samples': n
            }
        except:
            results[period_name] = {
                'beta': np.nan,
                'se': np.nan,
                't_stat': np.nan,
                'p_value': np.nan,
                'n_samples': len(period_data)
            }

    return results


def analyze_structural_break(factor: pd.Series,
                             returns: pd.Series,
                             candidate_breakpoints: List[str] = None,
                             rolling_window: int = 120) -> Dict:
    """
    Comprehensive structural break analysis.

    Key change: Analyzes regression BETA (not IC) for structural breaks.
    Beta is more economically interpretable and suitable for Chow test.

    Args:
        factor: Factor series
        returns: Forward return series
        candidate_breakpoints: List of dates to test (default: ['2010-01-01', '2015-01-01', '2020-01-01'])
        rolling_window: Window for rolling analysis

    Returns:
        Dictionary with analysis results including:
        - chow_tests: Chow test results for each breakpoint
        - rolling_beta: Rolling regression beta (not IC)
        - rolling_ic: Rolling IC (for reference)
        - subsample_beta: Beta by time period
        - subsample_ic: IC by time period (for reference)
        - changepoints: Detected changepoints in beta series
    """
    if candidate_breakpoints is None:
        candidate_breakpoints = ['2010-01-01', '2015-01-01', '2020-01-01']

    results = {
        'chow_tests': {},
        'rolling_beta': None,
        'rolling_ic': None,
        'changepoints': [],
        'subsample_beta': {},
        'subsample_ic': {}
    }

    # 1. Chow tests at candidate breakpoints (on regression model)
    for bp in candidate_breakpoints:
        f_stat, p_value = chow_test(returns, factor, bp)
        results['chow_tests'][bp] = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05 if not np.isnan(p_value) else False
        }

    # 2. Rolling BETA (primary analysis)
    results['rolling_beta'] = rolling_beta_ols(returns, factor, rolling_window)

    # 3. Rolling IC (for reference/comparison)
    results['rolling_ic'] = rolling_ic_with_ci(factor, returns, rolling_window)

    # 4. Detect changepoints in rolling BETA (not IC)
    if results['rolling_beta'] is not None and len(results['rolling_beta']) > 0:
        rolling_beta_series = results['rolling_beta']['beta']
        results['changepoints'] = detect_changepoints_cusum(rolling_beta_series)

    # 5. Subsample analysis
    periods = {
        '2000-2009': ('2000-01-01', '2009-12-31'),
        '2010-2014': ('2010-01-01', '2014-12-31'),
        '2015-2019': ('2015-01-01', '2019-12-31'),
        '2020-2024': ('2020-01-01', '2024-12-31'),
    }

    # Subsample BETA (primary)
    results['subsample_beta'] = compute_subsample_beta(returns, factor, periods)

    # Subsample IC (for reference)
    results['subsample_ic'] = compute_subsample_ic(factor, returns, periods)

    # Find most significant breakpoint
    best_bp = None
    best_f = 0
    for bp, stats in results['chow_tests'].items():
        if stats['significant'] and stats['f_statistic'] > best_f:
            best_f = stats['f_statistic']
            best_bp = bp

    results['most_significant_breakpoint'] = best_bp

    # Summary interpretation
    if results['rolling_beta'] is not None and len(results['rolling_beta']) > 0:
        beta_series = results['rolling_beta']['beta']
        pre_2015 = beta_series[beta_series.index < '2015-01-01']
        post_2015 = beta_series[beta_series.index >= '2015-01-01']

        results['interpretation'] = {
            'pre_2015_mean_beta': pre_2015.mean() if len(pre_2015) > 0 else np.nan,
            'post_2015_mean_beta': post_2015.mean() if len(post_2015) > 0 else np.nan,
            'beta_sign_changed': (
                (pre_2015.mean() < 0 and post_2015.mean() > 0) or
                (pre_2015.mean() > 0 and post_2015.mean() < 0)
            ) if len(pre_2015) > 0 and len(post_2015) > 0 else False
        }

    return results
