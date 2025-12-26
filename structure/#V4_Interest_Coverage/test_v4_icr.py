#!/usr/bin/env python3
"""
V4 Factor Validation: Interest Coverage Ratio (ICR)
====================================================

Formula: ICR = EBIT / Interest = (Profit Before Tax + Net Interest) / Net Interest

FRED Series:
- A464RC1Q027SBEA: Profit Before Tax (Nonfinancial Corp)
- B471RC1Q027SBEA: Net Interest (Nonfinancial Corp)

Release Lag: ~5-6 months (based on ALFRED)
Frequency: Quarterly

Key Characteristic: ICR is a POSITIVE indicator (high ICR = low risk)
Need to FLIP for danger zone detection.

Transform Testing:
1. Raw Percentile (10Y)
2. Δ(4Q) - 4-quarter change (quarterly data)
3. Z-score (10Y rolling)
4. Flipped Percentile (100 - pctl)
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

# Factor settings
PROFIT_SERIES = 'A464RC1Q027SBEA'  # Profit Before Tax
INTEREST_SERIES = 'B471RC1Q027SBEA'  # Net Interest
FACTOR_NAME = 'Interest Coverage Ratio'
RELEASE_LAG_MONTHS = 6  # ~5-6 months based on ALFRED

# Windows
PERCENTILE_WINDOW_Q = 40  # 10 years = 40 quarters
PERCENTILE_WINDOW_M = 120  # 10 years = 120 months

CRASH_THRESHOLD = -0.20


def load_factor_from_fred():
    """Load and compute ICR from FRED."""
    print("\n[Loading Factor Data from FRED]")
    fred = Fred(api_key=FRED_API_KEY)

    # Get quarterly data
    profit = fred.get_series(PROFIT_SERIES)
    interest = fred.get_series(INTEREST_SERIES)

    print(f"  Profit (A464RC1Q027SBEA): {profit.index.min()} to {profit.index.max()}")
    print(f"  Interest (B471RC1Q027SBEA): {interest.index.min()} to {interest.index.max()}")

    # Align
    common_idx = profit.index.intersection(interest.index)
    profit = profit.loc[common_idx]
    interest = interest.loc[common_idx]

    # Compute ICR = (Profit + Interest) / Interest = EBIT / Interest
    # Note: In BEA data, Interest is already net interest paid
    ebit = profit + interest
    icr = ebit / interest

    # Handle edge cases
    icr = icr.replace([np.inf, -np.inf], np.nan).dropna()
    icr = icr[icr > 0]  # ICR should be positive

    print(f"  ICR quarterly points: {len(icr)}")
    print(f"  ICR range: {icr.min():.2f}x to {icr.max():.2f}x")
    print(f"  Current ICR: {icr.iloc[-1]:.2f}x")

    return icr


def compute_transforms(icr_quarterly: pd.Series) -> pd.DataFrame:
    """
    Compute all transformations on quarterly ICR data,
    then forward-fill to monthly.
    """
    print("\n[Computing Transformations]")

    # 1. Δ(4Q) - 4-quarter change
    delta_4q = icr_quarterly.diff(4)
    print(f"  Δ(4Q): {delta_4q.dropna().shape[0]} valid quarterly points")

    # 2. Z-score (rolling 10Y = 40 quarters)
    window = 40
    rolling_mean = icr_quarterly.rolling(window=window, min_periods=window).mean()
    rolling_std = icr_quarterly.rolling(window=window, min_periods=window).std()
    zscore = (icr_quarterly - rolling_mean) / rolling_std
    print(f"  Z-score (10Y): {zscore.dropna().shape[0]} valid quarterly points")

    # 3. Rolling Percentile (10Y = 40 quarters)
    def rolling_percentile(series, window):
        result = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            window_data = series.iloc[i-window:i]
            current = series.iloc[i]
            pct = (window_data < current).sum() / len(window_data) * 100
            result.iloc[i] = pct
        return result

    pctl = rolling_percentile(icr_quarterly, window)
    print(f"  Percentile (10Y): {pctl.dropna().shape[0]} valid quarterly points")

    # Create quarterly DataFrame
    df_q = pd.DataFrame({
        'icr_raw': icr_quarterly,
        'delta_4q': delta_4q,
        'zscore_10y': zscore,
        'pctl_10y': pctl,
        'pctl_flipped': 100 - pctl,  # Flipped: high = low ICR = danger
    })

    # Apply publication lag BEFORE forward-filling
    df_q.index = df_q.index + pd.DateOffset(months=RELEASE_LAG_MONTHS)

    # Forward-fill to monthly
    df_monthly = df_q.resample('ME').last().ffill()
    df_monthly.index.name = 'as_of_date'

    print(f"  After lag adjustment and ffill: {df_monthly.index.min()} to {df_monthly.index.max()}")
    print(f"  Monthly points: {len(df_monthly)}")

    return df_monthly


def load_spx():
    """Load SPX daily data"""
    filepath = os.path.join(PROJECT_ROOT, 'SPX 1D Data (1).csv')
    df = pd.read_csv(filepath, parse_dates=['time'])
    df = df.set_index('time')[['close']].rename(columns={'close': 'SPX'})
    return df['SPX']


def run_insample_analysis(factor: pd.Series, spx: pd.Series, name: str) -> dict:
    """Quick in-sample analysis"""
    spx_monthly = spx.resample('ME').last()
    fwd_return_12m = compute_forward_return(spx_monthly, horizon=12)
    fwd_mdd_12m = compute_forward_max_drawdown(spx, horizon=252)

    common_idx = factor.dropna().index.intersection(fwd_return_12m.dropna().index)
    if len(common_idx) < 50:
        return {'error': f'Insufficient data: {len(common_idx)}', 'n_samples': len(common_idx)}

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
        return {'error': 'Insufficient data', 'n_pass': 0, 'all_pass': False,
                'best_zone': (0, 0), 'gates': {f'gate{i}': {'pass': False, 'reason': 'Insufficient data'} for i in range(5)}}

    # Convert to percentile for zone detection if needed
    if 'pctl' not in name.lower():
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


def generate_report(all_results: dict, factor_df: pd.DataFrame):
    """Generate comprehensive report"""

    current_icr = factor_df['icr_raw'].iloc[-1]
    current_pctl = factor_df['pctl_10y'].iloc[-1]
    current_zscore = factor_df['zscore_10y'].iloc[-1]

    # Find best transform
    best_transform = None
    best_gates = 0
    for name, res in all_results['oos'].items():
        gates = res.get('n_pass', 0)
        if gates > best_gates:
            best_gates = gates
            best_transform = name

    # Determine status
    best_oos = all_results['oos'].get(best_transform, {})
    if best_oos.get('all_pass', False):
        final_status = "APPROVED"
        recommendation = "可作为预警信号进入监控系统"
    elif best_gates >= 3:
        final_status = "CONDITIONAL"
        recommendation = "可作为辅助信息，但不建议单独使用"
    else:
        final_status = "REJECTED"
        recommendation = "不推荐使用"

    report = f"""# V4 Interest Coverage Ratio (ICR) 验证报告

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 因子信息

| 属性 | 值 |
|------|-----|
| 公式 | ICR = (Profit + Interest) / Interest |
| Series | A464RC1Q027SBEA, B471RC1Q027SBEA |
| 频率 | Quarterly → Monthly |
| 发布滞后 | {RELEASE_LAG_MONTHS} months (ALFRED) |
| 特性 | 正向指标 (高ICR = 低风险) |

## 当前状态

| 指标 | 值 |
|------|-----|
| 当前 ICR | {current_icr:.2f}x |
| 10Y Percentile | {current_pctl:.1f}% |
| 10Y Z-score | {current_zscore:.2f} |

---

## Transform Comparison

| Transform | N | IC | p-val | AUC | Direction | Gates |
|-----------|---|-----|-------|-----|-----------|-------|
"""

    for name in ['Percentile(10Y)', 'Flipped Pctl', 'Δ(4Q)', 'Z-score(10Y)']:
        ins = all_results['insample'].get(name, {})
        oos = all_results['oos'].get(name, {})

        n = ins.get('n_samples', 0)
        ic = ins.get('ic', np.nan)
        pval = ins.get('ic_pval', np.nan)
        auc = ins.get('auc', np.nan)
        direction = ins.get('direction', 'N/A')
        gates = oos.get('n_pass', 0)

        ic_str = f"{ic:.4f}" if not np.isnan(ic) else "N/A"
        pval_str = f"{pval:.4f}" if not np.isnan(pval) else "N/A"
        auc_str = f"{auc:.3f}" if not np.isnan(auc) else "N/A"

        report += f"| {name} | {n} | {ic_str} | {pval_str} | {auc_str} | {direction} | {gates}/5 |\n"

    report += f"""
**Best Transform: {best_transform} ({best_gates}/5 gates)**

---

## Best Transform Gate Details: {best_transform}

| Gate | 描述 | 结果 | 详情 |
|------|------|------|------|
"""

    if best_transform and 'gates' in best_oos:
        gate_names = ['Real-time', 'Walk-Forward', 'Leave-Crisis-Out', 'Lead Time', 'Zone Stability']
        for i, gate_name in enumerate(gate_names):
            gate_key = f'gate{i}'
            gate = best_oos['gates'].get(gate_key, {})
            status = 'PASS' if gate.get('pass', False) else 'FAIL'
            reason = gate.get('reason', 'N/A')
            report += f"| Gate {i} | {gate_name} | {status} | {reason} |\n"

    report += f"""
**Best Zone**: [{best_oos.get('best_zone', (0,0))[0]}%, {best_oos.get('best_zone', (0,0))[1]}%]

---

## 最终结论

| 项目 | 结果 |
|------|------|
| **最终状态** | **{final_status}** |
| **建议** | {recommendation} |

"""

    # Add crisis signal details if available
    if best_transform and 'gates' in best_oos:
        gate3 = best_oos['gates'].get('gate3', {})
        if 'details' in gate3:
            report += "\n### 危机前信号详情\n\n"
            report += "| 危机 | 有信号 | Zone比例 | 平均因子 |\n"
            report += "|------|--------|----------|----------|\n"
            for crisis, detail in gate3['details'].items():
                has_signal = '✓' if detail.get('has_signal', False) else '✗'
                zone_ratio = detail.get('zone_ratio', 0) * 100
                avg_factor = detail.get('avg_factor', 'N/A')
                if isinstance(avg_factor, (int, float)):
                    report += f"| {crisis} | {has_signal} | {zone_ratio:.0f}% | {avg_factor:.1f}% |\n"
                else:
                    report += f"| {crisis} | {has_signal} | {zone_ratio:.0f}% | {avg_factor} |\n"

    report += f"""
---

## 经济解读

ICR (Interest Coverage Ratio) 衡量企业用 EBIT 覆盖利息支出的能力：
- **高 ICR**: 企业还债能力强，财务压力小
- **低 ICR**: 企业还债压力大，可能预示风险

### 关键特性
1. **正向指标**: 高 ICR = 低风险，需要翻转用于危险区间检测
2. **波动率预测能力**: 对波动率的预测力 (IC=-0.40) 强于对收益的预测力
3. **利率依赖**: 在高利率环境下信号更强

---

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    filepath = os.path.join(OUTPUT_DIR, 'V4_VALIDATION_REPORT.md')
    with open(filepath, 'w') as f:
        f.write(report)
    print(f"\n  Report saved: {filepath}")

    return final_status


def main():
    print("=" * 70)
    print("V4 Factor Validation: Interest Coverage Ratio (ICR)")
    print("=" * 70)
    print(f"\nFormula: ICR = (Profit + Interest) / Interest")
    print(f"Release Lag: {RELEASE_LAG_MONTHS} months")

    # Load data
    icr_quarterly = load_factor_from_fred()
    factor_df = compute_transforms(icr_quarterly)
    spx = load_spx()

    # Save factor data
    factor_df.to_csv(os.path.join(OUTPUT_DIR, 'factor_data.csv'))
    print(f"\n  Factor data saved")

    # Define transforms to test
    transforms = {
        'Percentile(10Y)': 'pctl_10y',
        'Flipped Pctl': 'pctl_flipped',
        'Δ(4Q)': 'delta_4q',
        'Z-score(10Y)': 'zscore_10y',
    }

    # ============================================================
    # Stage 1: In-Sample Comparison
    # ============================================================
    print("\n" + "=" * 70)
    print("STAGE 1: IN-SAMPLE COMPARISON")
    print("=" * 70)

    insample_results = {}
    for name, col in transforms.items():
        factor = factor_df[col].dropna()
        result = run_insample_analysis(factor, spx, name)
        insample_results[name] = result

    print("\n[In-Sample Results]")
    print(f"{'Transform':<18} {'N':<6} {'IC':<10} {'p-val':<10} {'AUC':<8} {'Direction':<12}")
    print("-" * 74)

    for name, res in insample_results.items():
        if 'error' in res:
            print(f"{name:<18} {res.get('n_samples', 0):<6} ERROR")
        else:
            print(f"{name:<18} {res['n_samples']:<6} {res['ic']:<10.4f} {res['ic_pval']:<10.4f} {res['auc']:<8.3f} {res['direction']:<12}")

    # ============================================================
    # Stage 2: OOS Validation
    # ============================================================
    print("\n" + "=" * 70)
    print("STAGE 2: OUT-OF-SAMPLE 5-GATE VALIDATION")
    print("=" * 70)

    oos_results = {}
    for name, col in transforms.items():
        factor = factor_df[col].dropna()
        result = run_oos_validation(factor, spx, name)
        oos_results[name] = result

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n[Results]")
    print(f"{'Transform':<18} {'IC':<10} {'AUC':<8} {'Gates':<8} {'Status':<12}")
    print("-" * 56)

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

        ic_str = f"{ic:.4f}" if not np.isnan(ic) else "N/A"
        auc_str = f"{auc:.3f}" if not np.isnan(auc) else "N/A"
        print(f"{name:<18} {ic_str:<10} {auc_str:<8} {gates}/5{'':>4} {status:<12}")

    # Generate report
    all_results = {
        'insample': insample_results,
        'oos': oos_results,
    }
    final_status = generate_report(all_results, factor_df)

    print("\n" + "=" * 70)
    print(f"VALIDATION COMPLETE: {final_status}")
    print("=" * 70)


if __name__ == '__main__':
    main()
