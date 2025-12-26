"""
Regime Analysis Module
======================

Provides functions for regime-dependent factor analysis:
- Drawdown computation (current and forward)
- Realized volatility computation
- Regime-conditional IC analysis
- Conditional regression models
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Dict, Tuple, Optional, List
import warnings


# ============== Drawdown Functions ==============

def compute_current_drawdown(price: pd.Series, window: int = 252) -> pd.Series:
    """
    Compute current drawdown from rolling maximum.

    Args:
        price: Price series
        window: Lookback window for rolling max (default 252 = 1 year)

    Returns:
        Series of drawdown values (negative percentages)
    """
    rolling_max = price.rolling(window=window, min_periods=1).max()
    drawdown = (price - rolling_max) / rolling_max
    return drawdown


def compute_forward_max_drawdown(price: pd.Series, horizon: int = 252) -> pd.Series:
    """
    Compute forward maximum drawdown within horizon.

    For each date t, compute the max drawdown that occurs in [t, t+horizon].

    Args:
        price: Price series
        horizon: Forward horizon in trading days (default 252 = 1 year)

    Returns:
        Series of forward max drawdown values (negative percentages)
    """
    result = pd.Series(index=price.index, dtype=float)

    for i in range(len(price) - horizon):
        future_prices = price.iloc[i:i + horizon + 1]
        running_max = future_prices.cummax()
        drawdowns = (future_prices - running_max) / running_max
        result.iloc[i] = drawdowns.min()

    return result


def compute_forward_realized_vol(price: pd.Series,
                                  horizon: int = 126,
                                  annualize: bool = True) -> pd.Series:
    """
    Compute forward realized volatility.

    Args:
        price: Price series
        horizon: Forward horizon in trading days (default 126 = 6 months)
        annualize: Whether to annualize the volatility

    Returns:
        Series of forward realized volatility values
    """
    # Daily returns
    returns = price.pct_change()

    result = pd.Series(index=price.index, dtype=float)

    for i in range(len(returns) - horizon):
        future_returns = returns.iloc[i + 1:i + 1 + horizon]
        vol = future_returns.std()
        if annualize:
            vol = vol * np.sqrt(252)
        result.iloc[i] = vol

    return result


def compute_forward_return(price: pd.Series, horizon: int = 252) -> pd.Series:
    """
    Compute forward log return.

    Args:
        price: Price series
        horizon: Forward horizon in trading days

    Returns:
        Series of forward log returns
    """
    return np.log(price.shift(-horizon) / price)


# ============== Regime Classification ==============

def classify_drawdown_regime(price: pd.Series,
                              threshold: float = -0.05,
                              window: int = 252) -> pd.Series:
    """
    Classify market regime based on current drawdown.

    Args:
        price: Price series
        threshold: Drawdown threshold for "drawdown regime" (default -5%)
        window: Lookback window for rolling max

    Returns:
        Series with 1 for drawdown regime, 0 for normal
    """
    dd = compute_current_drawdown(price, window)
    regime = (dd < threshold).astype(int)
    return regime


def classify_volatility_regime(price: pd.Series,
                                lookback: int = 21,
                                percentile_threshold: float = 75) -> pd.Series:
    """
    Classify market regime based on realized volatility.

    Args:
        price: Price series
        lookback: Lookback window for vol calculation
        percentile_threshold: Percentile above which is "high vol"

    Returns:
        Series with 1 for high vol regime, 0 for normal
    """
    returns = price.pct_change()
    rolling_vol = returns.rolling(window=lookback).std() * np.sqrt(252)

    threshold = rolling_vol.quantile(percentile_threshold / 100)
    regime = (rolling_vol > threshold).astype(int)
    return regime


# ============== Regime-Conditional IC ==============

def compute_regime_ic(factor: pd.Series,
                      returns: pd.Series,
                      regime: pd.Series,
                      method: str = 'spearman') -> Dict:
    """
    Compute IC separately for each regime.

    Args:
        factor: Factor series
        returns: Forward return series
        regime: Regime indicator (0 or 1)
        method: 'spearman' or 'pearson'

    Returns:
        Dictionary with IC stats for each regime
    """
    data = pd.DataFrame({
        'factor': factor,
        'return': returns,
        'regime': regime
    }).dropna()

    results = {}

    for regime_val, regime_name in [(0, 'normal'), (1, 'stress')]:
        regime_data = data[data['regime'] == regime_val]

        if len(regime_data) < 10:
            results[regime_name] = {
                'ic': np.nan,
                'p_value': np.nan,
                'n_samples': len(regime_data)
            }
            continue

        if method == 'spearman':
            ic, p_value = spearmanr(regime_data['factor'], regime_data['return'])
        else:
            ic, p_value = pearsonr(regime_data['factor'], regime_data['return'])

        results[regime_name] = {
            'ic': ic,
            'p_value': p_value,
            'n_samples': len(regime_data)
        }

    # Also compute full sample IC
    if len(data) >= 10:
        if method == 'spearman':
            ic, p_value = spearmanr(data['factor'], data['return'])
        else:
            ic, p_value = pearsonr(data['factor'], data['return'])
        results['full'] = {
            'ic': ic,
            'p_value': p_value,
            'n_samples': len(data)
        }

    return results


# ============== Risk Target IC ==============

def compute_risk_target_ic(factor: pd.Series,
                           price: pd.Series,
                           horizons: List[int] = None) -> Dict:
    """
    Compute IC against various risk targets.

    Args:
        factor: Factor series (monthly)
        price: Price series (daily)
        horizons: Forward horizons in trading days

    Returns:
        Dictionary with IC for each risk target
    """
    if horizons is None:
        horizons = [126, 252]  # 6M, 12M

    # Resample price to monthly for alignment
    price_monthly = price.resample('ME').last()

    results = {}

    for horizon in horizons:
        horizon_name = f'{horizon // 21}M'

        # Forward return
        fwd_return = compute_forward_return(price_monthly, horizon // 21)

        # Forward max drawdown
        fwd_mdd = compute_forward_max_drawdown(price, horizon)
        fwd_mdd_monthly = fwd_mdd.resample('ME').last()

        # Forward realized vol
        fwd_vol = compute_forward_realized_vol(price, horizon)
        fwd_vol_monthly = fwd_vol.resample('ME').last()

        # Align with factor
        common_idx = factor.index.intersection(fwd_return.index)
        common_idx = common_idx.intersection(fwd_mdd_monthly.index)
        common_idx = common_idx.intersection(fwd_vol_monthly.index)

        if len(common_idx) < 20:
            continue

        factor_aligned = factor.loc[common_idx]
        return_aligned = fwd_return.loc[common_idx]
        mdd_aligned = fwd_mdd_monthly.loc[common_idx]
        vol_aligned = fwd_vol_monthly.loc[common_idx]

        # IC vs Return
        valid = pd.DataFrame({'f': factor_aligned, 'r': return_aligned}).dropna()
        if len(valid) >= 10:
            ic_return, p_return = spearmanr(valid['f'], valid['r'])
        else:
            ic_return, p_return = np.nan, np.nan

        # IC vs Max Drawdown (higher factor -> more negative MDD = positive IC)
        valid = pd.DataFrame({'f': factor_aligned, 'mdd': mdd_aligned}).dropna()
        if len(valid) >= 10:
            ic_mdd, p_mdd = spearmanr(valid['f'], valid['mdd'])
        else:
            ic_mdd, p_mdd = np.nan, np.nan

        # IC vs Volatility
        valid = pd.DataFrame({'f': factor_aligned, 'vol': vol_aligned}).dropna()
        if len(valid) >= 10:
            ic_vol, p_vol = spearmanr(valid['f'], valid['vol'])
        else:
            ic_vol, p_vol = np.nan, np.nan

        results[horizon_name] = {
            'ic_return': {'ic': ic_return, 'p_value': p_return},
            'ic_max_drawdown': {'ic': ic_mdd, 'p_value': p_mdd},
            'ic_volatility': {'ic': ic_vol, 'p_value': p_vol}
        }

    return results


def compute_drawdown_event_auc(factor: pd.Series,
                                price: pd.Series,
                                threshold: float = -0.10,
                                horizon: int = 252) -> Dict:
    """
    Compute AUC for predicting drawdown events.

    Args:
        factor: Factor series (monthly)
        price: Price series (daily)
        threshold: Drawdown threshold for event definition
        horizon: Forward horizon

    Returns:
        Dictionary with AUC and related stats
    """
    # Compute forward max drawdown
    fwd_mdd = compute_forward_max_drawdown(price, horizon)
    fwd_mdd_monthly = fwd_mdd.resample('ME').last()

    # Binary event: MDD < threshold
    event = (fwd_mdd_monthly < threshold).astype(int)

    # Align with factor
    common_idx = factor.index.intersection(event.index)
    factor_aligned = factor.loc[common_idx].dropna()
    event_aligned = event.loc[common_idx]

    valid = pd.DataFrame({
        'factor': factor_aligned,
        'event': event_aligned
    }).dropna()

    if len(valid) < 20 or valid['event'].nunique() < 2:
        return {
            'auc': np.nan,
            'n_events': int(valid['event'].sum()) if len(valid) > 0 else 0,
            'n_samples': len(valid),
            'event_rate': np.nan
        }

    # AUC (higher factor should predict event, so no sign flip needed)
    try:
        auc = roc_auc_score(valid['event'], valid['factor'])
        fpr, tpr, thresholds = roc_curve(valid['event'], valid['factor'])
    except Exception as e:
        return {
            'auc': np.nan,
            'n_events': int(valid['event'].sum()),
            'n_samples': len(valid),
            'event_rate': valid['event'].mean(),
            'error': str(e)
        }

    return {
        'auc': auc,
        'n_events': int(valid['event'].sum()),
        'n_samples': len(valid),
        'event_rate': valid['event'].mean(),
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    }


# ============== HAC Helper ==============

def _newey_west_se(X: np.ndarray, residuals: np.ndarray, lag: int = 11) -> np.ndarray:
    """
    Compute Newey-West HAC standard errors.

    Args:
        X: Design matrix (n x k)
        residuals: Regression residuals (n,)
        lag: Number of lags for HAC (default 11 for 12M overlapping returns)

    Returns:
        HAC standard errors for each coefficient
    """
    n, k = X.shape

    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return np.full(k, np.nan)

    Xe = X * residuals.reshape(-1, 1)
    S = (Xe.T @ Xe) / n

    for j in range(1, lag + 1):
        weight = 1 - j / (lag + 1)
        Gamma_j = (Xe[j:].T @ Xe[:-j]) / n
        S = S + weight * (Gamma_j + Gamma_j.T)

    V = n * XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    return se


# ============== Conditional Regression ==============

def run_conditional_regression(factor: pd.Series,
                                returns: pd.Series,
                                regime: pd.Series,
                                use_hac: bool = True,
                                hac_lag: int = 11) -> Dict:
    """
    Run regression with regime interaction term and HAC standard errors.

    Model: R = α + β·F + γ·I(regime) + δ·F·I(regime) + ε

    Key improvement: Uses Newey-West HAC standard errors for overlapping returns.

    Args:
        factor: Factor series
        returns: Forward return series
        regime: Regime indicator (0 or 1)
        use_hac: Whether to use HAC standard errors (default True)
        hac_lag: Lag for HAC (default 11 for 12M returns)

    Returns:
        Dictionary with regression results including HAC-adjusted inference
    """
    from scipy.stats import t as t_dist

    data = pd.DataFrame({
        'F': factor,
        'R': returns,
        'regime': regime
    }).dropna()

    if len(data) < 30:
        return {'error': 'Insufficient data'}

    # Create interaction term
    data['F_regime'] = data['F'] * data['regime']

    # Design matrix
    X = np.column_stack([
        np.ones(len(data)),  # intercept
        data['F'].values,     # factor
        data['regime'].values,  # regime dummy
        data['F_regime'].values  # interaction
    ])
    y = data['R'].values
    col_names = ['alpha', 'beta_factor', 'gamma_regime', 'delta_interaction']

    n = len(y)
    k = X.shape[1]

    # OLS
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_pred = X @ beta
        residuals = y - y_pred

        # Standard errors (HAC or OLS)
        if use_hac:
            se = _newey_west_se(X, residuals, hac_lag)
        else:
            mse = np.sum(residuals ** 2) / (n - k)
            var_beta = mse * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(var_beta))

        # t-stats and p-values
        t_stats = beta / se
        p_values = 2 * (1 - t_dist.cdf(np.abs(t_stats), n - k))

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        return {
            'coefficients': dict(zip(col_names, beta)),
            'hac_se' if use_hac else 'std_errors': dict(zip(col_names, se)),
            't_stats': dict(zip(col_names, t_stats)),
            'p_values': dict(zip(col_names, p_values)),
            'r_squared': r_squared,
            'n_samples': n,
            'hac_lag': hac_lag if use_hac else None,
            'interpretation': {
                'normal_regime_beta': beta[1],
                'stress_regime_beta': beta[1] + beta[3],
                'interaction_significant': p_values[3] < 0.05,
                'factor_effective_in_stress': (beta[1] + beta[3]) < 0
            }
        }
    except Exception as e:
        return {'error': str(e)}


def run_interaction_regression(y: pd.Series,
                                factor: pd.Series,
                                condition: pd.Series,
                                use_hac: bool = True,
                                hac_lag: int = 11) -> Dict:
    """
    General interaction regression with HAC standard errors.

    Model: Y = α + β·F + γ·C + δ·(F×C) + ε

    This is the general form for testing hypotheses like:
    - Factor × Interest Rate interaction
    - Factor × Duration interaction
    - Factor × Drawdown Regime interaction

    Args:
        y: Dependent variable (forward returns)
        factor: Factor series
        condition: Conditioning variable (continuous or binary)
        use_hac: Whether to use HAC standard errors
        hac_lag: Lag for HAC

    Returns:
        Dictionary with regression results
    """
    from scipy.stats import t as t_dist

    data = pd.DataFrame({
        'y': y,
        'F': factor,
        'C': condition
    }).dropna()

    if len(data) < 30:
        return {'error': 'Insufficient data'}

    # Create interaction term
    data['F_C'] = data['F'] * data['C']

    # Design matrix
    X = np.column_stack([
        np.ones(len(data)),
        data['F'].values,
        data['C'].values,
        data['F_C'].values
    ])
    y_arr = data['y'].values
    col_names = ['const', 'factor', 'condition', 'interaction']

    n = len(y_arr)
    k = X.shape[1]

    try:
        beta = np.linalg.lstsq(X, y_arr, rcond=None)[0]
        residuals = y_arr - X @ beta

        if use_hac:
            se = _newey_west_se(X, residuals, hac_lag)
        else:
            mse = np.sum(residuals ** 2) / (n - k)
            var_beta = mse * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(var_beta))

        t_stats = beta / se
        p_values = 2 * (1 - t_dist.cdf(np.abs(t_stats), n - k))

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_arr - y_arr.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        return {
            'coefficients': dict(zip(col_names, beta)),
            'se': dict(zip(col_names, se)),
            't_stats': dict(zip(col_names, t_stats)),
            'p_values': dict(zip(col_names, p_values)),
            'r_squared': r_squared,
            'n_samples': n,
            'use_hac': use_hac,
            'hac_lag': hac_lag if use_hac else None,
            'interpretation': {
                'interaction_coef': beta[3],
                'interaction_significant': p_values[3] < 0.05,
                'interaction_direction': 'positive' if beta[3] > 0 else 'negative'
            }
        }
    except Exception as e:
        return {'error': str(e)}


# ============== Quintile Analysis ==============

def run_quintile_analysis(factor: pd.Series,
                          returns: pd.Series,
                          drawdown_events: pd.Series = None,
                          n_quantiles: int = 5) -> Dict:
    """
    Run quintile (bucket) analysis.

    Args:
        factor: Factor series
        returns: Forward return series
        drawdown_events: Binary series of drawdown events (optional)
        n_quantiles: Number of quantiles (default 5)

    Returns:
        Dictionary with quintile statistics
    """
    data = pd.DataFrame({
        'factor': factor,
        'return': returns
    }).dropna()

    if drawdown_events is not None:
        data['event'] = drawdown_events
        data = data.dropna()

    if len(data) < n_quantiles * 5:
        return {'error': 'Insufficient data'}

    # Create quantile labels
    try:
        data['quintile'] = pd.qcut(data['factor'], q=n_quantiles, labels=False) + 1
    except ValueError:
        # Handle non-unique bin edges
        data['quintile'] = pd.qcut(data['factor'].rank(method='first'),
                                    q=n_quantiles, labels=False) + 1

    results = {
        'quintile_stats': {},
        'monotonicity': {}
    }

    for q in range(1, n_quantiles + 1):
        q_data = data[data['quintile'] == q]

        stats = {
            'n_samples': len(q_data),
            'mean_factor': q_data['factor'].mean(),
            'mean_return': q_data['return'].mean(),
            'median_return': q_data['return'].median(),
            'std_return': q_data['return'].std()
        }

        if drawdown_events is not None and 'event' in q_data.columns:
            stats['event_rate'] = q_data['event'].mean()
            stats['n_events'] = int(q_data['event'].sum())

        results['quintile_stats'][f'Q{q}'] = stats

    # Check monotonicity
    mean_returns = [results['quintile_stats'][f'Q{q}']['mean_return']
                    for q in range(1, n_quantiles + 1)]

    # Spearman correlation between quintile rank and mean return
    from scipy.stats import spearmanr
    quintile_ranks = list(range(1, n_quantiles + 1))
    mono_corr, mono_p = spearmanr(quintile_ranks, mean_returns)

    results['monotonicity'] = {
        'spearman_corr': mono_corr,
        'p_value': mono_p,
        'is_monotonic_increasing': all(mean_returns[i] <= mean_returns[i+1]
                                        for i in range(len(mean_returns)-1)),
        'is_monotonic_decreasing': all(mean_returns[i] >= mean_returns[i+1]
                                        for i in range(len(mean_returns)-1)),
        'q5_minus_q1': mean_returns[-1] - mean_returns[0]
    }

    return results
