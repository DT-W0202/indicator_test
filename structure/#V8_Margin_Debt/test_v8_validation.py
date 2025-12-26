#!/usr/bin/env python3
"""
V8 Margin Debt - Complete Validation with Wilshire 5000 Normalization
======================================================================

新的标准化方式:
- Margin Debt / Wilshire 5000 Market Cap = 杠杆率
- 比 YoY 增速更直观，更能反映相对杠杆水平

数据来源:
- FINRA: Margin Debt (Monthly)
- Wilshire 5000 Market Cap (Monthly)

使用 5-Gate OOS Validation Framework:
1. Gate 0: Real-time (Release Lag < 6 months)
2. Gate 1: Walk-Forward OOS
3. Gate 2: Leave-Crisis-Out (LOCO)
4. Gate 3: Lead Time (Crisis Pre-Signal)
5. Gate 4: Zone Stability
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
)

# ============== Configuration ==============
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CRASH_THRESHOLD = -0.20
RELEASE_LAG_MONTHS = 1  # FINRA data released ~1 month lag

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

WF_WINDOWS = [
    ('1999-01-01', '2005-12-31', '2006-01-01', '2009-12-31'),
    ('1999-01-01', '2009-12-31', '2010-01-01', '2014-12-31'),
    ('1999-01-01', '2014-12-31', '2015-01-01', '2019-12-31'),
    ('1999-01-01', '2019-12-31', '2020-01-01', '2024-12-31'),
]


def load_margin_debt():
    """Load FINRA Margin Debt data"""
    filepath = os.path.join(OUTPUT_DIR, 'all_methods_data.csv')
    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')

    margin = df['margin_debt'].dropna()

    print(f"[Margin Debt Data]")
    print(f"  Range: {margin.index.min().strftime('%Y-%m')} to {margin.index.max().strftime('%Y-%m')}")
    print(f"  Current: ${margin.iloc[-1]/1e6:.2f} Trillion")

    return margin


def load_wilshire5000():
    """Load Wilshire 5000 Market Cap data"""
    filepath = os.path.join(PROJECT_ROOT, '美国-美股总市值 (Wilshire 5000) (1).csv')

    # Read with specific parsing for this CSV format
    df = pd.read_csv(filepath, header=0, names=['date_str', 'value', 'empty'])

    # Skip the header row if it exists
    df = df[df['date_str'].str.contains('T', na=False)].copy()

    # Parse dates
    df['date'] = pd.to_datetime(df['date_str'].str.split('T').str[0])
    df['market_cap'] = pd.to_numeric(df['value'], errors='coerce')

    # Set index and get series
    df = df.set_index('date')
    market_cap = df['market_cap'].dropna()

    # Resample to month end
    market_cap = market_cap.resample('ME').last()

    print(f"\n[Wilshire 5000 Market Cap]")
    print(f"  Range: {market_cap.index.min().strftime('%Y-%m')} to {market_cap.index.max().strftime('%Y-%m')}")
    print(f"  Current: ${market_cap.iloc[-1]/1000:.2f} Trillion")

    return market_cap


def load_spx():
    """Load SPX data"""
    filepath = os.path.join(PROJECT_ROOT, 'SPX 1D Data (1).csv')
    df = pd.read_csv(filepath, parse_dates=['time'])
    return df.set_index('time')['close']


def compute_margin_ratio(margin: pd.Series, market_cap: pd.Series) -> pd.DataFrame:
    """
    Compute Margin Debt / Market Cap ratio and transforms

    Note: Margin Debt is in millions, Wilshire 5000 is in billions
    We need to convert to same units
    """

    # Align data
    common_idx = margin.dropna().index.intersection(market_cap.dropna().index)

    margin_aligned = margin.loc[common_idx]
    mcap_aligned = market_cap.loc[common_idx]

    # Convert: Margin is in millions, Wilshire is in billions (index value ~50000 = $50T)
    # Wilshire 5000 index: 1 point ≈ $1 billion
    # So current ~55000 ≈ $55 trillion
    mcap_in_millions = mcap_aligned * 1000  # Convert billions to millions

    # Margin Debt / Market Cap ratio (as percentage)
    ratio = (margin_aligned / mcap_in_millions) * 100

    print(f"\n[Margin Debt / Market Cap Ratio]")
    print(f"  Aligned observations: {len(ratio)}")
    print(f"  Date range: {ratio.index.min().strftime('%Y-%m')} to {ratio.index.max().strftime('%Y-%m')}")
    print(f"  Current ratio: {ratio.iloc[-1]:.3f}%")
    print(f"  Mean: {ratio.mean():.3f}%")
    print(f"  Max: {ratio.max():.3f}% ({ratio.idxmax().strftime('%Y-%m')})")

    # Compute transforms
    # 1. Raw ratio
    raw_ratio = ratio

    # 2. Percentile (10Y rolling, expanding from start)
    pctl_10y = ratio.expanding(min_periods=60).apply(
        lambda x: (x.iloc[:-1] < x.iloc[-1]).sum() / len(x.iloc[:-1]) * 100 if len(x) > 1 else 50
    )

    # 3. Z-score (10Y)
    rolling_mean = ratio.rolling(120, min_periods=60).mean()
    rolling_std = ratio.rolling(120, min_periods=60).std()
    zscore_10y = (ratio - rolling_mean) / rolling_std

    # 4. 12-month change
    delta_12m = ratio.diff(12)

    # 5. Delta percentile
    delta_pctl = delta_12m.rolling(120, min_periods=60).apply(
        lambda x: (x.iloc[:-1] < x.iloc[-1]).sum() / len(x.iloc[:-1]) * 100 if len(x) > 1 else 50
    )

    return pd.DataFrame({
        'margin_debt': margin_aligned,
        'market_cap': mcap_in_millions,
        'ratio': raw_ratio,
        'pctl': pctl_10y,
        'zscore': zscore_10y,
        'delta_12m': delta_12m,
        'delta_pctl': delta_pctl,
    })


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


def run_production_validation(factor: pd.Series, spx: pd.Series, transform_name: str):
    """Run production validation tests"""

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

        # Top 20% as danger zone
        danger_mask = df['factor'] > df['factor'].quantile(0.8)

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

        danger_mask = df['factor'] > df['factor'].quantile(0.8)

        if danger_mask.sum() > 0:
            cr = df.loc[danger_mask, 'is_crash'].mean() * 100
            print(f"  {horizon_name}: Danger zone crash rate = {cr:.1f}%")
            results[f'test2_{horizon_name}'] = {'crash_rate': cr}

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

    danger_mask = full_df['factor'] > full_df['factor'].quantile(0.8)
    full_cr = full_df.loc[danger_mask, 'is_crash'].mean() * 100 if danger_mask.sum() > 0 else 0
    print(f"  Full Sample: {full_cr:.1f}%")

    loco_crs = [full_cr]
    for crisis_name, (start, end) in CRISIS_PERIODS.items():
        exclude_mask = (full_df.index >= start) & (full_df.index <= end)
        loco_df = full_df[~exclude_mask]

        loco_danger = loco_df['factor'] > loco_df['factor'].quantile(0.8)

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
            in_danger = (pre_crisis_factor > factor_lagged.quantile(0.8)).mean() * 100
            print(f"  {crisis_name}: {in_danger:.0f}% in danger zone")

    return results


def generate_validation_report(all_results: list, production_results: dict, ratio_df: pd.DataFrame):
    """Generate comprehensive validation report"""

    best = max(all_results, key=lambda x: x['gates_passed'])

    report = f"""# V8 Margin Debt / Wilshire 5000 - Complete Validation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

使用 **Margin Debt / Wilshire 5000 Market Cap** 作为新的标准化方式，验证杠杆率因子。

这比 YoY 增速更直观，反映的是相对于市场规模的杠杆水平。

---

## Data Overview

| Metric | Value |
|--------|-------|
| Current Margin Debt | ${ratio_df['margin_debt'].iloc[-1]/1e6:.2f} Trillion |
| Current Market Cap | ${ratio_df['market_cap'].iloc[-1]/1e6:.2f} Trillion |
| **Current Ratio** | **{ratio_df['ratio'].iloc[-1]:.3f}%** |
| Historical Mean | {ratio_df['ratio'].mean():.3f}% |
| Historical Max | {ratio_df['ratio'].max():.3f}% ({ratio_df['ratio'].idxmax().strftime('%Y-%m')}) |

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
| Current Ratio | {ratio_df['ratio'].iloc[-1]:.3f}% |
| Percentile | {ratio_df['pctl'].iloc[-1]:.1f}% |
| Z-score | {ratio_df['zscore'].iloc[-1]:.2f} |
| Δ(12M) | {ratio_df['delta_12m'].iloc[-1]:.4f}% |

---

## Conclusion

| Status | Details |
|--------|---------|
| **Validation** | {best['gates_passed']}/5 Gates ({'CONDITIONAL' if best['gates_passed'] >= 3 else 'REJECTED'}) |
| **Best Transform** | {best['transform']} |
| **Direction** | high_is_danger |
| **Optimal Zone** | {best['best_zone']} |

---

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    filepath = os.path.join(OUTPUT_DIR, 'V8_RATIO_VALIDATION_REPORT.md')
    with open(filepath, 'w') as f:
        f.write(report)
    print(f"\nReport saved: {filepath}")

    return report


def main():
    print("=" * 70)
    print("V8 Margin Debt / Wilshire 5000 - Complete Validation")
    print("=" * 70)

    # Load data
    margin = load_margin_debt()
    market_cap = load_wilshire5000()
    spx = load_spx()

    # Compute ratio and transforms
    print("\n[Computing Ratio and Transforms]")
    ratio_df = compute_margin_ratio(margin, market_cap)

    # Validate each transform
    all_results = []

    # 1. Raw Ratio
    result = validate_transform(
        'Ratio (Raw)',
        ratio_df['ratio'],
        spx,
        direction='high_is_danger'
    )
    all_results.append(result)

    # 2. Percentile
    result = validate_transform(
        'Percentile',
        ratio_df['pctl'],
        spx,
        direction='high_is_danger'
    )
    all_results.append(result)

    # 3. Z-score
    result = validate_transform(
        'Z-score(10Y)',
        ratio_df['zscore'],
        spx,
        direction='high_is_danger'
    )
    all_results.append(result)

    # 4. Δ(12M)
    result = validate_transform(
        'Δ(12M)',
        ratio_df['delta_12m'],
        spx,
        direction='high_is_danger'
    )
    all_results.append(result)

    # 5. Δ(12M) Percentile
    result = validate_transform(
        'Δ(12M)_Pctl',
        ratio_df['delta_pctl'],
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
    if best['transform'] == 'Ratio (Raw)':
        best_factor = ratio_df['ratio']
    elif best['transform'] == 'Percentile':
        best_factor = ratio_df['pctl']
    elif best['transform'] == 'Z-score(10Y)':
        best_factor = ratio_df['zscore']
    elif best['transform'] == 'Δ(12M)':
        best_factor = ratio_df['delta_12m']
    else:
        best_factor = ratio_df['delta_pctl']

    production_results = run_production_validation(best_factor, spx, best['transform'])

    # Generate report
    generate_validation_report(all_results, production_results, ratio_df)

    # Save data
    ratio_df.to_csv(os.path.join(OUTPUT_DIR, 'margin_ratio_data.csv'))

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    # Current status summary
    print(f"\n[Current Status]")
    print(f"  Margin Debt / Market Cap: {ratio_df['ratio'].iloc[-1]:.3f}%")
    print(f"  Percentile: {ratio_df['pctl'].iloc[-1]:.1f}%")
    print(f"  Z-score: {ratio_df['zscore'].iloc[-1]:.2f}")


if __name__ == '__main__':
    main()
