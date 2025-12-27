"""
LQI Trend Factors - Full Validation Suite

Same methodology as V1-V8 structure factors:
1. IC Analysis (full sample, high/low rate regimes)
2. Structural Break Detection
3. Forward MDD AUC
4. Quintile Analysis
5. Sub-sample Stability

Factors:
- T1: VIX Term Structure (VIX/VIX3M - 1)
- T2: VVIX
- T3: SKEW
- T4: Credit Spread Proxy (HYG/LQD)
- T5: Flight to Safety (TLT/SPX change)
- T6: Funding Spread (EFFR - SOFR)
- T7: VIX Level
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from trend.V9_LQI_FACTORS.lqi_factors import LQIFactors
from trend.V9_LQI_FACTORS.lqi_data_loader import LQIDataLoader
from lib.transform_layers import TransformPipeline


# ============== Helper Functions ==============

def compute_forward_return(spx: pd.Series, horizon: int = 12) -> pd.Series:
    """Compute forward log return over horizon months"""
    spx_monthly = spx.resample('ME').last()
    spx_monthly.index = spx_monthly.index.to_period('M').to_timestamp('M')

    fwd_return = np.log(spx_monthly.shift(-horizon) / spx_monthly) * 100
    return fwd_return


def compute_forward_mdd(spx: pd.Series, horizon: int = 12) -> pd.Series:
    """Compute forward maximum drawdown over horizon months"""
    spx_monthly = spx.resample('ME').last()
    spx_monthly.index = spx_monthly.index.to_period('M').to_timestamp('M')
    spx_daily = spx.resample('D').ffill()

    mdd_series = {}
    for date in spx_monthly.index:
        end_date = date + pd.DateOffset(months=horizon)
        future_prices = spx_daily[(spx_daily.index > date) & (spx_daily.index <= end_date)]

        if len(future_prices) < 20:
            mdd_series[date] = np.nan
            continue

        running_max = future_prices.expanding().max()
        drawdowns = (future_prices - running_max) / running_max * 100
        mdd_series[date] = drawdowns.min()

    return pd.Series(mdd_series)


def load_ffr_for_regime() -> pd.Series:
    """Load Fed Funds Rate for regime classification"""
    import fredapi
    try:
        fred = fredapi.Fred(api_key=os.environ.get('FRED_API_KEY', ''))
        ffr = fred.get_series('FEDFUNDS')
        ffr.index = pd.to_datetime(ffr.index)
        return ffr
    except:
        # Fallback to EFFR from cache
        loader = LQIDataLoader()
        effr = loader.load_effr()
        return effr.resample('ME').last()


def compute_regime_mask(ffr: pd.Series) -> pd.Series:
    """High rate = FFR > 10Y rolling median"""
    ffr_monthly = ffr.resample('ME').last()
    ffr_monthly.index = ffr_monthly.index.to_period('M').to_timestamp('M')
    rolling_median = ffr_monthly.rolling(120, min_periods=60).median()
    return ffr_monthly > rolling_median


# ============== IC Analysis ==============

def compute_ic_analysis(factor: pd.Series, forward_return: pd.Series,
                        regime_mask: pd.Series = None) -> dict:
    """
    Compute IC analysis with regime breakdown
    """
    # Align data
    common_idx = factor.index.intersection(forward_return.index)
    if regime_mask is not None:
        common_idx = common_idx.intersection(regime_mask.index)

    factor_aligned = factor.reindex(common_idx).dropna()
    return_aligned = forward_return.reindex(factor_aligned.index).dropna()
    common_idx = factor_aligned.index.intersection(return_aligned.index)

    factor_valid = factor_aligned.reindex(common_idx)
    return_valid = return_aligned.reindex(common_idx)

    n_samples = len(factor_valid)
    if n_samples < 30:
        return {'error': f'Insufficient samples: {n_samples}'}

    # Full sample IC
    full_ic, full_p = stats.spearmanr(factor_valid, return_valid)

    result = {
        'n_samples': n_samples,
        'full_ic': full_ic,
        'full_p': full_p,
        'date_start': common_idx.min(),
        'date_end': common_idx.max(),
    }

    # Regime analysis
    if regime_mask is not None:
        regime_aligned = regime_mask.reindex(common_idx)
        high_rate_mask = regime_aligned == True
        low_rate_mask = ~high_rate_mask

        n_high = high_rate_mask.sum()
        n_low = low_rate_mask.sum()

        if n_high >= 20:
            high_ic, high_p = stats.spearmanr(factor_valid[high_rate_mask],
                                               return_valid[high_rate_mask])
            result['high_rate_ic'] = high_ic
            result['high_rate_p'] = high_p
            result['n_high'] = n_high

        if n_low >= 20:
            low_ic, low_p = stats.spearmanr(factor_valid[low_rate_mask],
                                             return_valid[low_rate_mask])
            result['low_rate_ic'] = low_ic
            result['low_rate_p'] = low_p
            result['n_low'] = n_low

    return result


# ============== AUC Analysis ==============

def compute_auc_analysis(factor: pd.Series, forward_mdd: pd.Series,
                         thresholds: list = [-10, -15, -20]) -> dict:
    """
    Compute AUC for crash prediction
    """
    common_idx = factor.index.intersection(forward_mdd.index)
    factor_aligned = factor.reindex(common_idx).dropna()
    mdd_aligned = forward_mdd.reindex(factor_aligned.index).dropna()
    common_idx = factor_aligned.index.intersection(mdd_aligned.index)

    factor_valid = factor_aligned.reindex(common_idx)
    mdd_valid = mdd_aligned.reindex(common_idx)

    result = {'n_samples': len(factor_valid)}

    for thresh in thresholds:
        crash_event = (mdd_valid < thresh).astype(int)
        n_crash = crash_event.sum()
        n_no_crash = len(crash_event) - n_crash

        if n_crash >= 10 and n_no_crash >= 10:
            try:
                auc = roc_auc_score(crash_event, factor_valid)
                result[f'auc_{abs(thresh)}'] = auc
                result[f'n_crash_{abs(thresh)}'] = n_crash
            except Exception as e:
                result[f'auc_{abs(thresh)}'] = np.nan

    return result


# ============== Quintile Analysis ==============

def compute_quintile_analysis(factor: pd.Series, forward_return: pd.Series,
                               forward_mdd: pd.Series) -> pd.DataFrame:
    """
    Compute quintile analysis
    """
    common_idx = factor.index.intersection(forward_return.index).intersection(forward_mdd.index)
    factor_valid = factor.reindex(common_idx).dropna()
    return_valid = forward_return.reindex(factor_valid.index)
    mdd_valid = forward_mdd.reindex(factor_valid.index)

    # Remove any remaining NaN
    valid_mask = ~factor_valid.isna() & ~return_valid.isna() & ~mdd_valid.isna()
    factor_valid = factor_valid[valid_mask]
    return_valid = return_valid[valid_mask]
    mdd_valid = mdd_valid[valid_mask]

    if len(factor_valid) < 50:
        return pd.DataFrame()

    # Create quintiles
    quintiles = pd.qcut(factor_valid, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')

    results = []
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        mask = quintiles == q
        if mask.sum() >= 5:
            q_return = return_valid[mask]
            q_mdd = mdd_valid[mask]
            results.append({
                'Quintile': q,
                'N': mask.sum(),
                'Mean_Return': q_return.mean(),
                'Std_Return': q_return.std(),
                'Mean_MDD': q_mdd.mean(),
                'Crash_Rate_15': (q_mdd < -15).mean() * 100,
                'Crash_Rate_20': (q_mdd < -20).mean() * 100,
            })

    return pd.DataFrame(results)


# ============== Sub-sample Stability ==============

def compute_subsample_ic(factor: pd.Series, forward_return: pd.Series,
                          periods: list = None) -> list:
    """
    Compute IC for sub-sample periods
    """
    if periods is None:
        periods = [
            ('2000-2007', '2000-01-01', '2007-12-31'),
            ('2008-2014', '2008-01-01', '2014-12-31'),
            ('2015-2019', '2015-01-01', '2019-12-31'),
            ('2020-2024', '2020-01-01', '2024-12-31'),
        ]

    results = []
    for name, start, end in periods:
        mask = (factor.index >= start) & (factor.index <= end)
        factor_sub = factor[mask]
        return_sub = forward_return.reindex(factor_sub.index).dropna()

        common_idx = factor_sub.index.intersection(return_sub.index)
        if len(common_idx) >= 20:
            ic, p = stats.spearmanr(factor_sub.reindex(common_idx),
                                    return_sub.reindex(common_idx))
            results.append({
                'Period': name,
                'N': len(common_idx),
                'IC': ic,
                'p_value': p,
                'Significant': p < 0.05
            })

    return results


# ============== Main Validation ==============

def validate_factor(factor_name: str, factor_raw: pd.Series, spx: pd.Series,
                    ffr: pd.Series, percentile_window: int = 252) -> dict:
    """
    Full validation for a single factor
    """
    print(f"\n{'='*70}")
    print(f"Validating: {factor_name}")
    print(f"{'='*70}")

    # Resample factor to monthly
    factor_monthly = factor_raw.resample('ME').last()
    factor_monthly.index = factor_monthly.index.to_period('M').to_timestamp('M')

    # Apply percentile transform
    pipeline = TransformPipeline({'winsorize_limits': (0.01, 0.99)})
    factor_winsorized = pipeline.winsorize(factor_monthly)
    factor_pctl = pipeline.rolling_percentile(factor_winsorized, window=percentile_window)

    print(f"\nFactor stats:")
    print(f"  Raw range: {factor_raw.index.min().date()} to {factor_raw.index.max().date()}")
    print(f"  Monthly samples: {len(factor_monthly.dropna())}")
    print(f"  Percentile window: {percentile_window} months ({percentile_window//12}Y)")
    print(f"  Current percentile: {factor_pctl.iloc[-1]:.1f}%" if not pd.isna(factor_pctl.iloc[-1]) else "  Current: N/A")

    # Compute forward metrics
    print("\nComputing forward metrics...")
    spx_monthly = spx.resample('ME').last()
    spx_monthly.index = spx_monthly.index.to_period('M').to_timestamp('M')

    fwd_return_1m = compute_forward_return(spx, horizon=1)
    fwd_return_3m = compute_forward_return(spx, horizon=3)
    fwd_return_12m = compute_forward_return(spx, horizon=12)
    fwd_mdd = compute_forward_mdd(spx, horizon=12)

    # Regime mask
    regime_mask = compute_regime_mask(ffr)

    # IC Analysis
    print("\n--- IC Analysis ---")
    ic_1m = compute_ic_analysis(factor_pctl, fwd_return_1m, regime_mask)
    ic_3m = compute_ic_analysis(factor_pctl, fwd_return_3m, regime_mask)
    ic_12m = compute_ic_analysis(factor_pctl, fwd_return_12m, regime_mask)

    print(f"\n  1M Forward Return:")
    print(f"    Full IC: {ic_1m.get('full_ic', np.nan):.4f} (p={ic_1m.get('full_p', np.nan):.4f})")
    if 'high_rate_ic' in ic_1m:
        print(f"    High Rate IC: {ic_1m['high_rate_ic']:.4f} (p={ic_1m['high_rate_p']:.4f})")
    if 'low_rate_ic' in ic_1m:
        print(f"    Low Rate IC: {ic_1m['low_rate_ic']:.4f} (p={ic_1m['low_rate_p']:.4f})")

    print(f"\n  3M Forward Return:")
    print(f"    Full IC: {ic_3m.get('full_ic', np.nan):.4f} (p={ic_3m.get('full_p', np.nan):.4f})")

    print(f"\n  12M Forward Return:")
    print(f"    Full IC: {ic_12m.get('full_ic', np.nan):.4f} (p={ic_12m.get('full_p', np.nan):.4f})")
    if 'high_rate_ic' in ic_12m:
        print(f"    High Rate IC: {ic_12m['high_rate_ic']:.4f} (p={ic_12m['high_rate_p']:.4f})")
    if 'low_rate_ic' in ic_12m:
        print(f"    Low Rate IC: {ic_12m['low_rate_ic']:.4f} (p={ic_12m['low_rate_p']:.4f})")

    # AUC Analysis
    print("\n--- AUC Analysis (Crash Prediction) ---")
    auc_result = compute_auc_analysis(factor_pctl, fwd_mdd)
    for thresh in [10, 15, 20]:
        auc = auc_result.get(f'auc_{thresh}', np.nan)
        n_crash = auc_result.get(f'n_crash_{thresh}', 0)
        if not np.isnan(auc):
            print(f"  MDD < -{thresh}%: AUC = {auc:.4f} (n_crash = {n_crash})")

    # Quintile Analysis
    print("\n--- Quintile Analysis ---")
    quintile_df = compute_quintile_analysis(factor_pctl, fwd_return_12m, fwd_mdd)
    if not quintile_df.empty:
        print(quintile_df.to_string(index=False))

        # Monotonicity
        if len(quintile_df) == 5:
            returns = quintile_df['Mean_Return'].values
            monotonicity = stats.spearmanr(range(5), returns)[0]
            print(f"\n  Monotonicity (Spearman): {monotonicity:.4f}")

    # Sub-sample Stability
    print("\n--- Sub-sample Stability ---")
    subsample_results = compute_subsample_ic(factor_pctl, fwd_return_12m)
    for r in subsample_results:
        sig = '*' if r['Significant'] else ''
        print(f"  {r['Period']}: IC = {r['IC']:.4f} (p={r['p_value']:.4f}){sig}, N = {r['N']}")

    return {
        'factor_name': factor_name,
        'ic_1m': ic_1m,
        'ic_3m': ic_3m,
        'ic_12m': ic_12m,
        'auc': auc_result,
        'quintile': quintile_df,
        'subsample': subsample_results,
    }


def main():
    print("=" * 70)
    print(" LQI Trend Factors - Full Validation Suite")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    lqi = LQIFactors()
    loader = LQIDataLoader()
    spx = loader.load_spx()

    # Load FFR for regime
    ffr = load_ffr_for_regime()

    # Define factors to validate
    factors = [
        ('T1: VIX Term Structure', lqi.compute_vix_term_structure(), 60),
        ('T2: VVIX', lqi.compute_vvix(), 60),
        ('T3: SKEW', lqi.compute_skew(), 120),
        ('T4: Credit Spread Proxy', lqi.compute_credit_spread_proxy(), 60),
        ('T5: Flight to Safety', lqi.compute_flight_to_safety(), 60),
        ('T6: Funding Spread', lqi.compute_funding_spread(), 60),
        ('T7: VIX Level', lqi.compute_vix_level(), 60),
    ]

    all_results = {}

    for name, factor, pctl_window in factors:
        try:
            result = validate_factor(name, factor, spx, ffr, percentile_window=pctl_window)
            all_results[name] = result
        except Exception as e:
            print(f"\n{name}: Error - {e}")
            import traceback
            traceback.print_exc()

    # Summary Table
    print("\n" + "=" * 70)
    print(" SUMMARY TABLE")
    print("=" * 70)

    print(f"\n{'Factor':<25} {'IC(1M)':>10} {'IC(12M)':>10} {'HighR IC':>10} {'LowR IC':>10} {'AUC(20)':>10}")
    print("-" * 85)

    for name, result in all_results.items():
        ic_1m = result['ic_1m'].get('full_ic', np.nan)
        ic_12m = result['ic_12m'].get('full_ic', np.nan)
        high_ic = result['ic_12m'].get('high_rate_ic', np.nan)
        low_ic = result['ic_12m'].get('low_rate_ic', np.nan)
        auc_20 = result['auc'].get('auc_20', np.nan)

        print(f"{name:<25} {ic_1m:>10.4f} {ic_12m:>10.4f} {high_ic:>10.4f} {low_ic:>10.4f} {auc_20:>10.4f}")

    return all_results


if __name__ == '__main__':
    results = main()
