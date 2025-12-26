#!/usr/bin/env python3
"""
V2 Factor Validation: Unstable Deposits Ratio - Transform Comparison
=====================================================================

对于结构性漂移变量，水平值几乎必死。尝试以下变换：

1. Δ(12m) - 12个月变化率
2. Z-score (rolling 10y) - 标准化偏离
3. Rolling Percentile (already tested, for comparison)

Hypothesis: 变化率/冲击比水平值更有领先性
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from fredapi import Fred
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from lib import (
    validate_factor,
    STANDARD_CRISIS_PERIODS,
    STANDARD_WALKFORWARD_WINDOWS,
    compute_forward_max_drawdown,
    compute_forward_return,
)

# ============== Configuration ==============
FRED_API_KEY = 'b37a95dcefcfcc0f98ddfb87daca2e34'
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CRASH_THRESHOLD = -0.20
RELEASE_LAG_MONTHS = 1


def load_raw_factor():
    """Load raw Unstable Deposits Ratio"""
    print("\n[Loading Raw Factor Data]")
    fred = Fred(api_key=FRED_API_KEY)

    demand_deposits = fred.get_series('WDDNS')
    total_deposits = fred.get_series('DPSACBW027SBOG')

    # Resample to monthly
    demand_monthly = demand_deposits.resample('ME').last()
    total_monthly = total_deposits.resample('ME').last()

    # Align and compute ratio
    common_idx = demand_monthly.index.intersection(total_monthly.index)
    factor_raw = (demand_monthly.loc[common_idx] / total_monthly.loc[common_idx] * 100).dropna()

    print(f"  Raw ratio range: {factor_raw.index.min()} to {factor_raw.index.max()}")
    print(f"  Points: {len(factor_raw)}")
    print(f"  Current: {factor_raw.iloc[-1]:.2f}%")

    return factor_raw


def compute_transforms(factor_raw: pd.Series) -> pd.DataFrame:
    """
    Compute all transformations:
    1. Δ(12m) - 12-month change
    2. Z-score (rolling 10y)
    3. Rolling Percentile (10y)
    """
    print("\n[Computing Transformations]")

    # 1. Δ(12m) - 12-month change
    delta_12m = factor_raw.diff(12)
    print(f"  Δ(12m): {delta_12m.dropna().shape[0]} valid points")

    # 2. Z-score (rolling 10y = 120 months)
    window = 120
    rolling_mean = factor_raw.rolling(window=window, min_periods=window).mean()
    rolling_std = factor_raw.rolling(window=window, min_periods=window).std()
    zscore_10y = (factor_raw - rolling_mean) / rolling_std
    print(f"  Z-score (10Y): {zscore_10y.dropna().shape[0]} valid points")

    # 3. Rolling Percentile (10y)
    def rolling_percentile(series, window):
        result = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            window_data = series.iloc[i-window:i]
            current = series.iloc[i]
            pct = (window_data < current).sum() / len(window_data) * 100
            result.iloc[i] = pct
        return result

    pctl_10y = rolling_percentile(factor_raw, window)
    print(f"  Percentile (10Y): {pctl_10y.dropna().shape[0]} valid points")

    # Create DataFrame
    df = pd.DataFrame({
        'raw': factor_raw,
        'delta_12m': delta_12m,
        'zscore_10y': zscore_10y,
        'pctl_10y': pctl_10y,
    })

    # Apply publication lag
    df.index = df.index + pd.DateOffset(months=RELEASE_LAG_MONTHS)
    df.index.name = 'as_of_date'

    return df


def run_quick_insample(factor: pd.Series, spx: pd.Series, name: str) -> dict:
    """Quick in-sample analysis"""
    # Forward returns and MDD
    spx_monthly = spx.resample('ME').last()
    fwd_return_12m = compute_forward_return(spx_monthly, horizon=12)
    fwd_mdd_12m = compute_forward_max_drawdown(spx, horizon=252)

    # Align
    common_idx = factor.dropna().index.intersection(fwd_return_12m.dropna().index)
    if len(common_idx) < 50:
        return {'error': f'Insufficient data: {len(common_idx)}'}

    factor_aligned = factor.loc[common_idx]
    return_aligned = fwd_return_12m.loc[common_idx]

    # IC
    ic, p_val = stats.spearmanr(factor_aligned, return_aligned)

    # AUC
    mdd_common = fwd_mdd_12m.reindex(common_idx).dropna()
    factor_mdd = factor_aligned.loc[mdd_common.index]
    is_crash = (mdd_common < CRASH_THRESHOLD).astype(int)

    if is_crash.mean() > 0 and is_crash.mean() < 1:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(is_crash, factor_mdd)
        auc_effective = max(auc, 1 - auc)
        direction = "high→crash" if auc >= 0.5 else "low→crash"
    else:
        auc_effective = np.nan
        direction = "N/A"

    return {
        'name': name,
        'n_samples': len(common_idx),
        'ic': ic,
        'ic_pval': p_val,
        'auc': auc_effective,
        'direction': direction,
        'current': factor.iloc[-1] if not pd.isna(factor.iloc[-1]) else np.nan,
    }


def run_oos_validation(factor: pd.Series, spx: pd.Series, name: str) -> dict:
    """Full OOS 5-Gate validation"""
    print(f"\n[OOS Validation: {name}]")

    fwd_mdd_12m = compute_forward_max_drawdown(spx, horizon=252)
    common_idx = factor.dropna().index.intersection(fwd_mdd_12m.dropna().index)

    if len(common_idx) < 100:
        print(f"  Insufficient data: {len(common_idx)}")
        return {'error': 'Insufficient data', 'n_pass': 0}

    # For percentile-based validation, need to convert to 0-100 scale
    # For delta and zscore, convert to percentile for zone detection
    if 'delta' in name or 'zscore' in name:
        # Convert to percentile within sample
        factor_pctl = factor.rank(pct=True) * 100
    else:
        factor_pctl = factor

    df = pd.DataFrame({
        'percentile': factor_pctl.loc[common_idx],
        'is_crash': (fwd_mdd_12m.loc[common_idx] < CRASH_THRESHOLD).astype(int)
    }).dropna()

    print(f"  Samples: {len(df)}, Crash rate: {df['is_crash'].mean()*100:.1f}%")

    results = validate_factor(
        df,
        factor_col='percentile',
        crash_col='is_crash',
        release_lag_months=RELEASE_LAG_MONTHS,
        crisis_periods=STANDARD_CRISIS_PERIODS,
        walkforward_windows=STANDARD_WALKFORWARD_WINDOWS,
        horizon_months=12
    )

    return results


def load_spx():
    """Load SPX daily data"""
    filepath = os.path.join(PROJECT_ROOT, 'SPX 1D Data (1).csv')
    df = pd.read_csv(filepath, parse_dates=['time'])
    df = df.set_index('time')[['close']].rename(columns={'close': 'SPX'})
    return df['SPX']


def main():
    print("=" * 70)
    print("V2 Transform Comparison: Unstable Deposits Ratio")
    print("=" * 70)
    print("\nTesting: Δ(12m), Z-score(10Y), Percentile(10Y)")

    # Load data
    factor_raw = load_raw_factor()
    factor_df = compute_transforms(factor_raw)
    spx = load_spx()

    # Save transformed data
    factor_df.to_csv(os.path.join(OUTPUT_DIR, 'factor_transforms.csv'))

    # ============================================================
    # Stage 1: In-Sample Comparison
    # ============================================================
    print("\n" + "=" * 70)
    print("STAGE 1: IN-SAMPLE COMPARISON")
    print("=" * 70)

    transforms = {
        'Δ(12m)': 'delta_12m',
        'Z-score(10Y)': 'zscore_10y',
        'Percentile(10Y)': 'pctl_10y',
    }

    insample_results = {}
    for name, col in transforms.items():
        factor = factor_df[col].dropna()
        result = run_quick_insample(factor, spx, name)
        insample_results[name] = result

    # Print comparison table
    print("\n[In-Sample Results]")
    print(f"{'Transform':<18} {'N':<6} {'IC':<10} {'p-val':<10} {'AUC':<10} {'Direction':<12} {'Current':<10}")
    print("-" * 86)

    for name, res in insample_results.items():
        if 'error' in res:
            print(f"{name:<18} ERROR: {res['error']}")
        else:
            current_str = f"{res['current']:.2f}" if not np.isnan(res['current']) else "N/A"
            print(f"{name:<18} {res['n_samples']:<6} {res['ic']:<10.4f} {res['ic_pval']:<10.4f} {res['auc']:<10.3f} {res['direction']:<12} {current_str:<10}")

    # ============================================================
    # Stage 2: OOS Validation for Best Candidates
    # ============================================================
    print("\n" + "=" * 70)
    print("STAGE 2: OUT-OF-SAMPLE 5-GATE VALIDATION")
    print("=" * 70)

    # Test all transforms
    oos_results = {}
    for name, col in transforms.items():
        factor = factor_df[col].dropna()
        result = run_oos_validation(factor, spx, name)
        oos_results[name] = result

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY: TRANSFORM COMPARISON")
    print("=" * 70)

    print("\n[Results]")
    print(f"{'Transform':<18} {'IC':<10} {'AUC':<10} {'Gates':<8} {'Status':<12}")
    print("-" * 58)

    best_transform = None
    best_gates = 0

    for name in transforms.keys():
        ins = insample_results.get(name, {})
        oos = oos_results.get(name, {})

        ic = ins.get('ic', np.nan)
        auc = ins.get('auc', np.nan)
        gates = oos.get('n_pass', 0)

        if oos.get('all_pass', False):
            status = "APPROVED"
        elif gates >= 3:
            status = "CONDITIONAL"
        else:
            status = "REJECTED"

        print(f"{name:<18} {ic:<10.4f} {auc:<10.3f} {gates}/5{'':>4} {status:<12}")

        if gates > best_gates:
            best_gates = gates
            best_transform = name

    print(f"\n[Best Transform: {best_transform} ({best_gates}/5 gates)]")

    # Generate detailed report for best
    if best_transform:
        col = transforms[best_transform]
        factor = factor_df[col].dropna()
        oos = oos_results[best_transform]

        report = f"""# V2 Unstable Deposits Ratio - Transform Comparison Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Transform Comparison

| Transform | IC | AUC | Direction | Gates Passed |
|-----------|-----|-----|-----------|--------------|
"""
        for name in transforms.keys():
            ins = insample_results.get(name, {})
            oos_r = oos_results.get(name, {})
            ic = ins.get('ic', np.nan)
            auc = ins.get('auc', np.nan)
            direction = ins.get('direction', 'N/A')
            gates = oos_r.get('n_pass', 0)
            report += f"| {name} | {ic:.4f} | {auc:.3f} | {direction} | {gates}/5 |\n"

        report += f"""
## Best Transform: {best_transform}

### Gate Details
"""
        if 'gates' in oos:
            for gate_name, gate_result in oos['gates'].items():
                status = 'PASS' if gate_result['pass'] else 'FAIL'
                reason = gate_result.get('reason', 'N/A')
                report += f"- **{gate_name}**: {status} - {reason}\n"

        report += f"""
## Conclusion

**Best Zone**: [{oos.get('best_zone', (0,0))[0]}%, {oos.get('best_zone', (0,0))[1]}%]
**Final Status**: {'APPROVED' if oos.get('all_pass', False) else 'CONDITIONAL' if best_gates >= 3 else 'REJECTED'}

### Key Insights

1. **Δ(12m)** (12-month change): Captures momentum/trend
2. **Z-score(10Y)**: Measures deviation from long-term mean
3. **Percentile(10Y)**: Relative position in historical distribution

For structural drift variables, change-based transforms often outperform level-based measures
because they capture the "shock" rather than the absolute position.

---

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        filepath = os.path.join(OUTPUT_DIR, 'V2_TRANSFORM_COMPARISON.md')
        with open(filepath, 'w') as f:
            f.write(report)
        print(f"\n  Report saved: {filepath}")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
