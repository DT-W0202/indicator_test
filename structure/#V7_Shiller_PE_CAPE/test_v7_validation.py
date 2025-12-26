#!/usr/bin/env python3
"""
V7 Shiller PE (CAPE) - Complete Validation
===========================================

使用 5-Gate OOS Validation Framework 重新验证 CAPE:
1. Gate 0: Real-time (Release Lag < 6 months)
2. Gate 1: Walk-Forward OOS
3. Gate 2: Leave-Crisis-Out (LOCO)
4. Gate 3: Lead Time (Crisis Pre-Signal)
5. Gate 4: Zone Stability

同时测试多种 transforms:
- Raw Percentile (10Y)
- Δ(12M) - 12个月变化
- Z-score (10Y)
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from lib import compute_forward_max_drawdown
from lib.factor_validation_gates import (
    find_best_zone,
    evaluate_zone,
    check_gate0_realtime,
    check_gate1_walkforward,
    check_gate2_leave_crisis_out,
    check_gate3_lead_time,
    check_gate4_zone_stability,
    validate_factor,
)

# ============== Configuration ==============
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CRASH_THRESHOLD = -0.20
RELEASE_LAG_MONTHS = 0  # CAPE is real-time available

CRISIS_PERIODS = {
    'Dot-com': ('2000-03-01', '2002-10-01'),
    'GFC': ('2007-10-01', '2009-03-01'),
    'COVID': ('2020-02-01', '2020-03-31'),
    '2022': ('2022-01-01', '2022-10-01'),
}

PRE_CRISIS_PERIODS = {
    'Dot-com': ('1999-06-01', '2000-02-01'),
    'GFC': ('2007-01-01', '2007-09-01'),
    'COVID': ('2019-06-01', '2020-01-01'),
    '2022': ('2021-06-01', '2021-12-01'),
}


def load_cape_data():
    """Load CAPE data from existing CSV"""
    filepath = os.path.join(OUTPUT_DIR, 'all_methods_data.csv')
    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')

    # 'cape_raw' column contains raw CAPE
    cape = df['cape_raw'].dropna()

    print(f"[CAPE Data]")
    print(f"  Range: {cape.index.min().strftime('%Y-%m')} to {cape.index.max().strftime('%Y-%m')}")
    print(f"  Current CAPE: {cape.iloc[-1]:.1f}")

    return cape


def load_spx():
    """Load SPX data"""
    filepath = os.path.join(PROJECT_ROOT, 'SPX 1D Data (1).csv')
    df = pd.read_csv(filepath, parse_dates=['time'])
    return df.set_index('time')['close']


def compute_transforms(cape: pd.Series) -> pd.DataFrame:
    """Compute all transforms"""

    # 1. Raw Percentile (10Y = 120 months)
    pctl_10y = cape.rolling(120, min_periods=60).apply(
        lambda x: (x.iloc[:-1] < x.iloc[-1]).sum() / len(x.iloc[:-1]) * 100 if len(x) > 1 else 50
    )

    # 2. Flipped Percentile (high CAPE = danger)
    flipped_pctl = 100 - pctl_10y

    # 3. Δ(12M) - 12-month change
    delta_12m = cape.diff(12)

    # 4. Δ(12M) Percentile
    delta_pctl = delta_12m.rolling(120, min_periods=60).apply(
        lambda x: (x.iloc[:-1] < x.iloc[-1]).sum() / len(x.iloc[:-1]) * 100 if len(x) > 1 else 50
    )

    # 5. Z-score (10Y)
    rolling_mean = cape.rolling(120, min_periods=60).mean()
    rolling_std = cape.rolling(120, min_periods=60).std()
    zscore_10y = (cape - rolling_mean) / rolling_std

    # 6. Δ(12M) Z-score
    delta_mean = delta_12m.rolling(120, min_periods=60).mean()
    delta_std = delta_12m.rolling(120, min_periods=60).std()
    delta_zscore = (delta_12m - delta_mean) / delta_std

    return pd.DataFrame({
        'raw': cape,
        'pctl_10y': pctl_10y,
        'flipped_pctl': flipped_pctl,
        'delta_12m': delta_12m,
        'delta_pctl': delta_pctl,
        'zscore_10y': zscore_10y,
        'delta_zscore': delta_zscore,
    })


# Walk-forward windows
WF_WINDOWS = [
    ('1999-01-01', '2005-12-31', '2006-01-01', '2009-12-31'),
    ('1999-01-01', '2009-12-31', '2010-01-01', '2014-12-31'),
    ('1999-01-01', '2014-12-31', '2015-01-01', '2019-12-31'),
    ('1999-01-01', '2019-12-31', '2020-01-01', '2024-12-31'),
]


def validate_transform(transform_name: str, factor: pd.Series, spx: pd.Series,
                       direction: str = 'high_is_danger'):
    """Validate a single transform using 5-Gate framework"""

    print(f"\n{'='*60}")
    print(f"Validating: {transform_name}")
    print(f"Direction: {direction}")
    print(f"{'='*60}")

    # Apply release lag
    factor_lagged = factor.copy()
    factor_lagged.index = factor_lagged.index + pd.DateOffset(months=RELEASE_LAG_MONTHS)

    # Compute forward MDD
    fwd_mdd = compute_forward_max_drawdown(spx, horizon=252)
    fwd_mdd_monthly = fwd_mdd.resample('ME').last()

    # Align
    common_idx = factor_lagged.dropna().index.intersection(fwd_mdd_monthly.dropna().index)

    factor_aligned = factor_lagged.loc[common_idx]
    mdd_aligned = fwd_mdd_monthly.loc[common_idx]
    crash = (mdd_aligned < CRASH_THRESHOLD).astype(int)

    # Create DataFrame for validation
    df = pd.DataFrame({
        'factor': factor_aligned,
        'fwd_mdd': mdd_aligned,
        'crash': crash,
    })

    # Compute IC
    ic, ic_pval = stats.spearmanr(df['factor'], df['fwd_mdd'])

    print(f"\n[Basic Stats]")
    print(f"  N: {len(df)}")
    print(f"  IC (vs MDD): {ic:.3f} (p={ic_pval:.4f})")
    print(f"  Crash rate: {crash.mean()*100:.1f}%")

    # Find optimal danger zone
    best_zone = find_best_zone(df, 'factor', 'crash')
    zone_eval = evaluate_zone(df, best_zone, 'factor', 'crash')

    print(f"\n[Optimal Danger Zone]")
    print(f"  Best zone: {best_zone}")
    print(f"  Lift: {zone_eval['lift']:.2f}x")

    # Gate 0: Real-time
    gate0 = check_gate0_realtime(RELEASE_LAG_MONTHS)
    print(f"\n[Gate 0: Real-time] {'PASS' if gate0['pass'] else 'FAIL'}")
    print(f"  {gate0['reason']}")

    # Gate 1: Walk-Forward
    gate1 = check_gate1_walkforward(df, 'factor', 'crash', WF_WINDOWS)
    print(f"\n[Gate 1: Walk-Forward] {'PASS' if gate1['pass'] else 'FAIL'}")
    print(f"  Avg Lift: {gate1['avg_lift']:.2f}x, Std: {gate1['std_lift']:.2f}, Min: {gate1['min_lift']:.2f}x")

    # Gate 2: Leave-Crisis-Out
    gate2 = check_gate2_leave_crisis_out(df, 'factor', 'crash', CRISIS_PERIODS)
    print(f"\n[Gate 2: Leave-Crisis-Out] {'PASS' if gate2['pass'] else 'FAIL'}")
    print(f"  Min Lift: {gate2['min_test_lift']:.2f}x, Zone Drift: {gate2['zone_drift']}%")

    # Gate 3: Lead Time
    gate3 = check_gate3_lead_time(df, 'factor', best_zone, PRE_CRISIS_PERIODS)
    print(f"\n[Gate 3: Lead Time] {'PASS' if gate3['pass'] else 'FAIL'}")
    print(f"  Signals: {gate3['n_with_signal']}/{gate3['n_total']} crises")

    # Gate 4: Zone Stability
    gate4 = check_gate4_zone_stability(df, 'factor', 'crash')
    print(f"\n[Gate 4: Zone Stability] {'PASS' if gate4['pass'] else 'FAIL'}")
    print(f"  Lower range: {gate4['lower_range']}%, Upper range: {gate4['upper_range']}%")

    # Summary
    gates_passed = sum([gate0['pass'], gate1['pass'], gate2['pass'], gate3['pass'], gate4['pass']])

    print(f"\n{'='*60}")
    print(f"SUMMARY: {gates_passed}/5 Gates Passed")
    print(f"{'='*60}")

    return {
        'transform': transform_name,
        'direction': direction,
        'n': len(df),
        'ic': ic,
        'ic_pval': ic_pval,
        'best_zone': best_zone,
        'best_lift': zone_eval['lift'],
        'gate0': gate0['pass'],
        'gate1': gate1['pass'],
        'gate2': gate2['pass'],
        'gate3': gate3['pass'],
        'gate4': gate4['pass'],
        'gates_passed': gates_passed,
        'gate0_results': gate0,
        'gate1_results': gate1,
        'gate2_results': gate2,
        'gate3_results': gate3,
        'gate4_results': gate4,
    }


def run_production_validation(transform_name: str, factor: pd.Series, spx: pd.Series,
                               direction: str = 'high_is_danger'):
    """Run production validation tests (same as V4)"""

    print(f"\n{'='*70}")
    print(f"PRODUCTION VALIDATION: {transform_name}")
    print(f"{'='*70}")

    factor_lagged = factor.copy()
    factor_lagged.index = factor_lagged.index + pd.DateOffset(months=RELEASE_LAG_MONTHS)

    results = {}

    # Test 1: Strict Event Definition
    print(f"\n[Test 1: Strict Event Definition]")

    for threshold, name in [(-0.10, 'MDD<-10%'), (-0.20, 'MDD<-20%'), (-0.25, 'MDD<-25%')]:
        fwd_mdd = compute_forward_max_drawdown(spx, horizon=252)
        fwd_mdd_monthly = fwd_mdd.resample('ME').last()

        common_idx = factor_lagged.dropna().index.intersection(fwd_mdd_monthly.dropna().index)

        df = pd.DataFrame({
            'factor': factor_lagged.loc[common_idx],
            'fwd_mdd': fwd_mdd_monthly.loc[common_idx],
            'is_crash': (fwd_mdd_monthly.loc[common_idx] < threshold).astype(int),
        })

        # Define danger zone based on direction
        if direction == 'high_is_danger':
            danger_mask = df['factor'] > df['factor'].quantile(0.8)  # Top 20%
        else:
            danger_mask = df['factor'] < df['factor'].quantile(0.2)  # Bottom 20%

        if danger_mask.sum() > 0:
            cr = df.loc[danger_mask, 'is_crash'].mean() * 100
            avg_mdd = df.loc[danger_mask, 'fwd_mdd'].mean() * 100
            print(f"  {name}: Danger zone crash rate = {cr:.1f}%, Avg MDD = {avg_mdd:.1f}%")
            results[f'test1_{name}'] = {'crash_rate': cr, 'avg_mdd': avg_mdd, 'n': danger_mask.sum()}

    # Test 2: Multi-Horizon
    print(f"\n[Test 2: Multi-Horizon]")

    for horizon_days, horizon_name in [(63, '3m'), (126, '6m'), (252, '12m')]:
        fwd_mdd = compute_forward_max_drawdown(spx, horizon=horizon_days)
        fwd_mdd_monthly = fwd_mdd.resample('ME').last()

        common_idx = factor_lagged.dropna().index.intersection(fwd_mdd_monthly.dropna().index)

        df = pd.DataFrame({
            'factor': factor_lagged.loc[common_idx],
            'fwd_mdd': fwd_mdd_monthly.loc[common_idx],
            'is_crash': (fwd_mdd_monthly.loc[common_idx] < -0.20).astype(int),
        })

        if direction == 'high_is_danger':
            danger_mask = df['factor'] > df['factor'].quantile(0.8)
        else:
            danger_mask = df['factor'] < df['factor'].quantile(0.2)

        if danger_mask.sum() > 0:
            cr = df.loc[danger_mask, 'is_crash'].mean() * 100
            avg_mdd = df.loc[danger_mask, 'fwd_mdd'].mean() * 100
            print(f"  {horizon_name}: Danger zone crash rate = {cr:.1f}%, Avg MDD = {avg_mdd:.1f}%")
            results[f'test2_{horizon_name}'] = {'crash_rate': cr, 'avg_mdd': avg_mdd}

    # Test 3: LOCO
    print(f"\n[Test 3: LOCO Validation]")

    fwd_mdd = compute_forward_max_drawdown(spx, horizon=252)
    fwd_mdd_monthly = fwd_mdd.resample('ME').last()
    common_idx = factor_lagged.dropna().index.intersection(fwd_mdd_monthly.dropna().index)

    full_df = pd.DataFrame({
        'factor': factor_lagged.loc[common_idx],
        'fwd_mdd': fwd_mdd_monthly.loc[common_idx],
        'is_crash': (fwd_mdd_monthly.loc[common_idx] < -0.20).astype(int),
    })

    if direction == 'high_is_danger':
        danger_mask = full_df['factor'] > full_df['factor'].quantile(0.8)
    else:
        danger_mask = full_df['factor'] < full_df['factor'].quantile(0.2)

    full_cr = full_df.loc[danger_mask, 'is_crash'].mean() * 100 if danger_mask.sum() > 0 else 0
    print(f"  Full Sample: {full_cr:.1f}%")

    loco_crs = [full_cr]
    for crisis_name, (start, end) in CRISIS_PERIODS.items():
        exclude_mask = (full_df.index >= start) & (full_df.index <= end)
        loco_df = full_df[~exclude_mask]

        if direction == 'high_is_danger':
            loco_danger = loco_df['factor'] > loco_df['factor'].quantile(0.8)
        else:
            loco_danger = loco_df['factor'] < loco_df['factor'].quantile(0.2)

        if loco_danger.sum() > 0:
            cr = loco_df.loc[loco_danger, 'is_crash'].mean() * 100
            print(f"  w/o {crisis_name}: {cr:.1f}%")
            loco_crs.append(cr)

    stability = max(loco_crs) - min(loco_crs)
    print(f"  LOCO Stability: {stability:.1f}pp range")
    results['test3_loco_stability'] = stability

    # Test 4: Crisis Pre-Signal
    print(f"\n[Test 4: Crisis Pre-Signal]")

    for crisis_name, (start, end) in PRE_CRISIS_PERIODS.items():
        mask = (factor_lagged.index >= start) & (factor_lagged.index <= end)
        if mask.sum() > 0:
            pre_crisis_factor = factor_lagged.loc[mask]

            if direction == 'high_is_danger':
                in_danger = (pre_crisis_factor > factor_lagged.quantile(0.8)).mean() * 100
            else:
                in_danger = (pre_crisis_factor < factor_lagged.quantile(0.2)).mean() * 100

            print(f"  {crisis_name}: {in_danger:.0f}% in danger zone")

    return results


def generate_validation_report(all_results: list, production_results: dict, cape_df: pd.DataFrame):
    """Generate comprehensive validation report"""

    # Find best transform
    best = max(all_results, key=lambda x: x['gates_passed'])

    report = f"""# V7 Shiller PE (CAPE) - Complete Validation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

使用 5-Gate OOS Validation Framework 重新验证 Shiller PE (CAPE)。

---

## Transform Comparison

| Transform | N | IC | p-val | Best Zone | Lift | Gates |
|-----------|---|-----|-------|-----------|------|-------|
"""

    for r in all_results:
        status = "CONDITIONAL" if r['gates_passed'] >= 3 else "REJECTED"
        report += f"| {r['transform']} | {r['n']} | {r['ic']:.3f} | {r['ic_pval']:.4f} | {r['best_zone']} | {r['best_lift']:.2f}x | {r['gates_passed']}/5 ({status}) |\n"

    report += f"""
**Best Transform: {best['transform']} ({best['gates_passed']}/5 gates)**

---

## Best Transform Gate Details: {best['transform']}

| Gate | Description | Result | Details |
|------|-------------|--------|---------|
| Gate 0 | Real-time | {'PASS' if best['gate0'] else 'FAIL'} | Lag = {RELEASE_LAG_MONTHS}m |
| Gate 1 | Walk-Forward | {'PASS' if best['gate1'] else 'FAIL'} | Avg={best['gate1_results']['avg_lift']:.2f}x, Std={best['gate1_results']['std_lift']:.2f} |
| Gate 2 | Leave-Crisis-Out | {'PASS' if best['gate2'] else 'FAIL'} | Min={best['gate2_results']['min_test_lift']:.2f}x, Drift={best['gate2_results']['zone_drift']}% |
| Gate 3 | Lead Time | {'PASS' if best['gate3'] else 'FAIL'} | {best['gate3_results']['n_with_signal']}/{best['gate3_results']['n_total']} crises |
| Gate 4 | Zone Stability | {'PASS' if best['gate4'] else 'FAIL'} | Range={best['gate4_results']['lower_range']}-{best['gate4_results']['upper_range']}% |

---

## Production Validation Results

### Test 1: Strict Event Definition

| Threshold | Danger Zone Crash Rate | Avg MDD |
|-----------|------------------------|---------|
"""

    for key in ['test1_MDD<-10%', 'test1_MDD<-20%', 'test1_MDD<-25%']:
        if key in production_results:
            r = production_results[key]
            report += f"| {key.replace('test1_', '')} | {r['crash_rate']:.1f}% | {r['avg_mdd']:.1f}% |\n"

    report += f"""
### Test 2: Multi-Horizon

| Horizon | Danger Zone Crash Rate |
|---------|------------------------|
"""

    for key in ['test2_3m', 'test2_6m', 'test2_12m']:
        if key in production_results:
            r = production_results[key]
            report += f"| {key.replace('test2_', '')} | {r['crash_rate']:.1f}% |\n"

    report += f"""
### Test 3: LOCO Stability

**LOCO Stability: {production_results.get('test3_loco_stability', 0):.1f}pp range**

---

## Current Status

| Metric | Value |
|--------|-------|
| Current CAPE | {cape_df['raw'].iloc[-1]:.1f} |
| 10Y Percentile | {cape_df['pctl_10y'].iloc[-1]:.1f}% |
| 10Y Z-score | {cape_df['zscore_10y'].iloc[-1]:.2f} |
| Δ(12M) | {cape_df['delta_12m'].iloc[-1]:.1f} |

---

## Conclusion

| Status | Details |
|--------|---------|
| **Validation** | {best['gates_passed']}/5 Gates ({'CONDITIONAL' if best['gates_passed'] >= 3 else 'REJECTED'}) |
| **Best Transform** | {best['transform']} |
| **Direction** | {best['direction']} |
| **Optimal Zone** | {best['best_zone']} |

---

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    filepath = os.path.join(OUTPUT_DIR, 'V7_VALIDATION_REPORT.md')
    with open(filepath, 'w') as f:
        f.write(report)
    print(f"\n  Report saved: {filepath}")

    return report


def main():
    print("=" * 70)
    print("V7 Shiller PE (CAPE) - Complete Validation")
    print("=" * 70)

    # Load data
    cape = load_cape_data()
    spx = load_spx()

    # Compute all transforms
    print("\n[Computing Transforms]")
    cape_df = compute_transforms(cape)
    print(f"  Transforms computed: {list(cape_df.columns)}")

    # Validate each transform
    all_results = []

    # 1. Raw Percentile (high = danger)
    result = validate_transform(
        'Percentile(10Y)',
        cape_df['pctl_10y'],
        spx,
        direction='high_is_danger'
    )
    all_results.append(result)

    # 2. Flipped Percentile (low = danger, for consistency check)
    result = validate_transform(
        'Flipped_Pctl',
        cape_df['flipped_pctl'],
        spx,
        direction='low_is_danger'
    )
    all_results.append(result)

    # 3. Δ(12M) Percentile (high change = danger)
    result = validate_transform(
        'Δ(12M)_Pctl',
        cape_df['delta_pctl'],
        spx,
        direction='high_is_danger'
    )
    all_results.append(result)

    # 4. Z-score (high = danger)
    result = validate_transform(
        'Z-score(10Y)',
        cape_df['zscore_10y'],
        spx,
        direction='high_is_danger'
    )
    all_results.append(result)

    # 5. Δ(12M) Z-score
    result = validate_transform(
        'Δ(12M)_Zscore',
        cape_df['delta_zscore'],
        spx,
        direction='high_is_danger'
    )
    all_results.append(result)

    # Find best transform
    best = max(all_results, key=lambda x: x['gates_passed'])

    print("\n" + "=" * 70)
    print("TRANSFORM COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Transform':<20} {'IC':<8} {'Zone':<15} {'Lift':<8} {'Gates':<10}")
    print("-" * 60)
    for r in all_results:
        status = "COND" if r['gates_passed'] >= 3 else "REJ"
        print(f"{r['transform']:<20} {r['ic']:<8.3f} {str(r['best_zone']):<15} {r['best_lift']:<8.2f} {r['gates_passed']}/5 ({status})")

    print(f"\n**Best: {best['transform']} with {best['gates_passed']}/5 gates**")

    # Run production validation on best transform
    print("\n" + "=" * 70)
    print("PRODUCTION VALIDATION")
    print("=" * 70)

    if best['transform'] == 'Percentile(10Y)':
        best_factor = cape_df['pctl_10y']
    elif best['transform'] == 'Flipped_Pctl':
        best_factor = cape_df['flipped_pctl']
    elif best['transform'] == 'Δ(12M)_Pctl':
        best_factor = cape_df['delta_pctl']
    elif best['transform'] == 'Z-score(10Y)':
        best_factor = cape_df['zscore_10y']
    else:
        best_factor = cape_df['delta_zscore']

    production_results = run_production_validation(
        best['transform'],
        best_factor,
        spx,
        direction=best['direction']
    )

    # Generate report
    generate_validation_report(all_results, production_results, cape_df)

    # Save transform data
    cape_df.to_csv(os.path.join(OUTPUT_DIR, 'all_transforms_data.csv'))

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
