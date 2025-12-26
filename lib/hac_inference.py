"""
HAC Inference Module
====================

Provides robust inference for overlapping return regressions:
- Newey-West HAC standard errors
- Block bootstrap for significance testing
- Quantile regression for tail analysis
"""

import pandas as pd
import numpy as np
from scipy.stats import t as t_dist, norm
from typing import Dict, Tuple, Optional, List
import warnings


def newey_west_se(X: np.ndarray, residuals: np.ndarray, lag: int = None) -> np.ndarray:
    """
    Compute Newey-West HAC standard errors.

    Args:
        X: Design matrix (n x k)
        residuals: Regression residuals (n,)
        lag: Number of lags for HAC (default: floor(4*(n/100)^(2/9)))

    Returns:
        HAC standard errors for each coefficient
    """
    n, k = X.shape

    if lag is None:
        lag = int(np.floor(4 * (n / 100) ** (2 / 9)))

    # Compute (X'X)^-1
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return np.full(k, np.nan)

    # Compute S (HAC covariance of X'e)
    # S = sum_{j=-lag}^{lag} w_j * Gamma_j
    # where Gamma_j = (1/n) * sum_t (X_t * e_t) * (X_{t-j} * e_{t-j})'

    Xe = X * residuals.reshape(-1, 1)  # n x k

    # Start with Gamma_0
    S = (Xe.T @ Xe) / n

    # Add lagged terms with Bartlett kernel
    for j in range(1, lag + 1):
        weight = 1 - j / (lag + 1)  # Bartlett kernel
        Gamma_j = (Xe[j:].T @ Xe[:-j]) / n
        S = S + weight * (Gamma_j + Gamma_j.T)

    # HAC variance-covariance matrix
    V = n * XtX_inv @ S @ XtX_inv

    # Standard errors
    se = np.sqrt(np.diag(V))
    return se


def ols_with_hac(y: pd.Series, X: pd.DataFrame, lag: int = None) -> Dict:
    """
    OLS regression with Newey-West HAC standard errors.

    Args:
        y: Dependent variable
        X: Independent variables (DataFrame with column names)
        lag: HAC lag (default: auto based on sample size)

    Returns:
        Dictionary with coefficients, HAC se, t-stats, p-values
    """
    # Align and drop NaN
    data = pd.concat([y.rename('y'), X], axis=1).dropna()

    if len(data) < X.shape[1] + 10:
        return {'error': 'Insufficient data'}

    y_arr = data['y'].values
    X_arr = data.drop('y', axis=1).values

    # Add constant if not present
    col_names = list(X.columns)
    if 'const' not in col_names:
        X_arr = np.column_stack([np.ones(len(y_arr)), X_arr])
        col_names = ['const'] + col_names

    n, k = X_arr.shape

    # OLS coefficients
    try:
        beta = np.linalg.lstsq(X_arr, y_arr, rcond=None)[0]
    except:
        return {'error': 'OLS failed'}

    # Residuals
    y_pred = X_arr @ beta
    residuals = y_arr - y_pred

    # HAC standard errors
    if lag is None:
        # For 12M overlapping returns, use lag = 11
        lag = max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))

    hac_se = newey_west_se(X_arr, residuals, lag)

    # t-stats and p-values
    t_stats = beta / hac_se
    p_values = 2 * (1 - t_dist.cdf(np.abs(t_stats), n - k))

    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_arr - y_arr.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        'coefficients': dict(zip(col_names, beta)),
        'hac_se': dict(zip(col_names, hac_se)),
        't_stats': dict(zip(col_names, t_stats)),
        'p_values': dict(zip(col_names, p_values)),
        'r_squared': r_squared,
        'n_samples': n,
        'hac_lag': lag,
        'residuals': residuals
    }


def block_bootstrap_regression(y: pd.Series,
                                X: pd.DataFrame,
                                n_bootstrap: int = 1000,
                                block_size: int = 12,
                                seed: int = 42) -> Dict:
    """
    Block bootstrap for regression coefficient inference.

    Args:
        y: Dependent variable
        X: Independent variables
        n_bootstrap: Number of bootstrap iterations
        block_size: Size of each block (default 12 for monthly data with 12M horizon)
        seed: Random seed

    Returns:
        Dictionary with bootstrap confidence intervals and p-values
    """
    np.random.seed(seed)

    # Align and drop NaN
    data = pd.concat([y.rename('y'), X], axis=1).dropna()

    if len(data) < block_size * 3:
        return {'error': 'Insufficient data for block bootstrap'}

    y_arr = data['y'].values
    X_arr = data.drop('y', axis=1).values

    # Add constant
    col_names = list(X.columns)
    if 'const' not in col_names:
        X_arr = np.column_stack([np.ones(len(y_arr)), X_arr])
        col_names = ['const'] + col_names

    n, k = X_arr.shape

    # Original OLS estimate
    try:
        beta_orig = np.linalg.lstsq(X_arr, y_arr, rcond=None)[0]
    except:
        return {'error': 'OLS failed'}

    # Number of blocks
    n_blocks = int(np.ceil(n / block_size))

    # Bootstrap
    beta_bootstrap = np.zeros((n_bootstrap, k))

    for b in range(n_bootstrap):
        # Sample blocks with replacement
        block_starts = np.random.randint(0, n - block_size + 1, n_blocks)
        indices = []
        for start in block_starts:
            indices.extend(range(start, min(start + block_size, n)))
        indices = np.array(indices[:n])  # Trim to original size

        # Bootstrap sample
        y_boot = y_arr[indices]
        X_boot = X_arr[indices]

        # OLS on bootstrap sample
        try:
            beta_bootstrap[b] = np.linalg.lstsq(X_boot, y_boot, rcond=None)[0]
        except:
            beta_bootstrap[b] = np.nan

    # Remove failed iterations
    valid_mask = ~np.any(np.isnan(beta_bootstrap), axis=1)
    beta_bootstrap = beta_bootstrap[valid_mask]

    if len(beta_bootstrap) < n_bootstrap * 0.5:
        return {'error': 'Too many bootstrap failures'}

    # Bootstrap statistics
    results = {
        'original_coef': dict(zip(col_names, beta_orig)),
        'bootstrap_mean': dict(zip(col_names, np.mean(beta_bootstrap, axis=0))),
        'bootstrap_se': dict(zip(col_names, np.std(beta_bootstrap, axis=0))),
        'ci_lower': {},
        'ci_upper': {},
        'bootstrap_p_value': {},
        'n_bootstrap_valid': len(beta_bootstrap)
    }

    for i, name in enumerate(col_names):
        # 95% CI (percentile method)
        results['ci_lower'][name] = np.percentile(beta_bootstrap[:, i], 2.5)
        results['ci_upper'][name] = np.percentile(beta_bootstrap[:, i], 97.5)

        # Two-sided p-value (proportion of bootstrap estimates on opposite side of 0)
        if beta_orig[i] > 0:
            p_val = 2 * np.mean(beta_bootstrap[:, i] < 0)
        else:
            p_val = 2 * np.mean(beta_bootstrap[:, i] > 0)
        results['bootstrap_p_value'][name] = min(p_val, 1.0)

    return results


def rolling_beta_with_hac(y: pd.Series,
                          x: pd.Series,
                          window: int = 120,
                          hac_lag: int = 11) -> pd.DataFrame:
    """
    Compute rolling regression beta with HAC standard errors.

    Args:
        y: Dependent variable (forward returns)
        x: Independent variable (factor)
        window: Rolling window size
        hac_lag: HAC lag parameter

    Returns:
        DataFrame with beta, hac_se, t_stat, p_value for each date
    """
    data = pd.DataFrame({'y': y, 'x': x}).dropna()

    results = []

    for i in range(window, len(data)):
        window_data = data.iloc[i - window:i]

        y_arr = window_data['y'].values
        X_arr = np.column_stack([np.ones(len(y_arr)), window_data['x'].values])

        try:
            beta = np.linalg.lstsq(X_arr, y_arr, rcond=None)[0]
            residuals = y_arr - X_arr @ beta
            hac_se = newey_west_se(X_arr, residuals, hac_lag)

            t_stat = beta[1] / hac_se[1] if hac_se[1] > 0 else np.nan
            p_value = 2 * (1 - t_dist.cdf(np.abs(t_stat), window - 2))

            results.append({
                'date': data.index[i],
                'beta': beta[1],
                'hac_se': hac_se[1],
                't_stat': t_stat,
                'p_value': p_value
            })
        except:
            continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.set_index('date')
    return df


def quantile_regression(y: pd.Series,
                        X: pd.DataFrame,
                        quantile: float = 0.05,
                        max_iter: int = 1000) -> Dict:
    """
    Simple quantile regression using iteratively reweighted least squares.

    Args:
        y: Dependent variable
        X: Independent variables
        quantile: Target quantile (default 0.05 for left tail)
        max_iter: Maximum iterations

    Returns:
        Dictionary with quantile regression coefficients
    """
    # Align and drop NaN
    data = pd.concat([y.rename('y'), X], axis=1).dropna()

    if len(data) < X.shape[1] + 10:
        return {'error': 'Insufficient data'}

    y_arr = data['y'].values
    X_arr = data.drop('y', axis=1).values

    # Add constant
    col_names = list(X.columns)
    if 'const' not in col_names:
        X_arr = np.column_stack([np.ones(len(y_arr)), X_arr])
        col_names = ['const'] + col_names

    n, k = X_arr.shape

    # Initialize with OLS
    try:
        beta = np.linalg.lstsq(X_arr, y_arr, rcond=None)[0]
    except:
        return {'error': 'OLS initialization failed'}

    # IRLS for quantile regression
    eps = 1e-6
    for iteration in range(max_iter):
        residuals = y_arr - X_arr @ beta

        # Weights based on check function derivative
        weights = np.where(residuals >= 0, quantile, 1 - quantile)
        weights = weights / (np.abs(residuals) + eps)

        # Weighted least squares
        W = np.diag(weights)
        try:
            beta_new = np.linalg.lstsq(X_arr.T @ W @ X_arr,
                                        X_arr.T @ W @ y_arr,
                                        rcond=None)[0]
        except:
            break

        # Check convergence
        if np.max(np.abs(beta_new - beta)) < 1e-6:
            beta = beta_new
            break
        beta = beta_new

    # Final residuals
    residuals = y_arr - X_arr @ beta

    # Bootstrap for standard errors (simplified)
    n_boot = 200
    beta_boot = np.zeros((n_boot, k))
    np.random.seed(42)

    for b in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        y_b = y_arr[idx]
        X_b = X_arr[idx]

        try:
            # Quick OLS approximation for bootstrap
            beta_boot[b] = np.linalg.lstsq(X_b, y_b, rcond=None)[0]
        except:
            beta_boot[b] = np.nan

    valid = ~np.any(np.isnan(beta_boot), axis=1)
    se = np.std(beta_boot[valid], axis=0) if valid.sum() > 10 else np.full(k, np.nan)

    return {
        'quantile': quantile,
        'coefficients': dict(zip(col_names, beta)),
        'std_errors': dict(zip(col_names, se)),
        'n_samples': n
    }


def compute_tail_quantile_ic(factor: pd.Series,
                              price: pd.Series,
                              horizon: int = 12,
                              quantile: float = 0.05) -> Dict:
    """
    Compute factor's predictive power for tail outcomes.

    Args:
        factor: Factor series (monthly)
        price: Price series (monthly)
        horizon: Forward horizon in months
        quantile: Target quantile (default 5%)

    Returns:
        Dictionary with quantile regression results and interpretation
    """
    # Forward returns
    fwd_return = np.log(price.shift(-horizon) / price)

    # Align
    common_idx = factor.index.intersection(fwd_return.index)
    factor_aligned = factor.loc[common_idx]
    return_aligned = fwd_return.loc[common_idx]

    # Standard regression (mean)
    X = pd.DataFrame({'factor': factor_aligned})
    mean_reg = ols_with_hac(return_aligned, X, lag=horizon - 1)

    # Quantile regression (left tail)
    quant_reg = quantile_regression(return_aligned, X, quantile=quantile)

    results = {
        'mean_regression': mean_reg,
        'quantile_regression': quant_reg,
        'interpretation': {}
    }

    if 'error' not in mean_reg and 'error' not in quant_reg:
        mean_beta = mean_reg['coefficients'].get('factor', np.nan)
        quant_beta = quant_reg['coefficients'].get('factor', np.nan)

        results['interpretation'] = {
            'mean_beta': mean_beta,
            'quantile_beta': quant_beta,
            'tail_effect_stronger': abs(quant_beta) > abs(mean_beta) if not np.isnan(quant_beta) else False,
            'description': (
                f"Factor effect on mean: {mean_beta:.4f}, "
                f"on {quantile*100:.0f}% quantile: {quant_beta:.4f}"
            )
        }

    return results
