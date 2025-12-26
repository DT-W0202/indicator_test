#!/usr/bin/env python3
"""
V2 Factor Validation: Unstable Deposits Ratio
==============================================

Formula: WDDNS / DPSACBW027SBOG * 100
- WDDNS: Demand Deposits (活期存款)
- DPSACBW027SBOG: Total Deposits, All Commercial Banks (总存款)

Hypothesis: High demand deposit ratio indicates:
- More "hot money" that can flee quickly
- Less sticky funding for banks
- Potential liquidity stress indicator

Source: Federal Reserve H.8 Release
Frequency: Weekly
Release Lag: ~1-3 weeks
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from fredapi import Fred
from scipy import stats

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from lib import (
    # In-Sample analysis
    run_quintile_analysis,
    # OOS validation
    validate_factor,
    STANDARD_CRISIS_PERIODS,
    STANDARD_WALKFORWARD_WINDOWS,
    # Utils
    compute_forward_max_drawdown,
    compute_forward_return,
)

# ============== Configuration ==============
FRED_API_KEY = 'b37a95dcefcfcc0f98ddfb87daca2e34'
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Factor settings
NUMERATOR_SERIES = 'WDDNS'  # Demand Deposits
DENOMINATOR_SERIES = 'DPSACBW027SBOG'  # Total Deposits
FACTOR_NAME = 'Unstable Deposits Ratio'
RELEASE_LAG_MONTHS = 1  # ~1-3 weeks, use 1 month to be conservative

# Percentile windows (in months)
PERCENTILE_WINDOWS = {
    '5Y': 60,   # 5 years = 60 months
    '10Y': 120,  # 10 years = 120 months
}

# Crash definition
CRASH_THRESHOLD = -0.20  # MDD < -20%


def load_factor_from_fred():
    """Load and compute Unstable Deposits Ratio from FRED."""
    print("\n[Loading Factor Data from FRED]")
    fred = Fred(api_key=FRED_API_KEY)

    # Get both series (weekly)
    demand_deposits = fred.get_series(NUMERATOR_SERIES)
    total_deposits = fred.get_series(DENOMINATOR_SERIES)

    print(f"  {NUMERATOR_SERIES} (Demand Deposits):")
    print(f"    Range: {demand_deposits.index.min()} to {demand_deposits.index.max()}")
    print(f"    Points: {len(demand_deposits)}")

    print(f"  {DENOMINATOR_SERIES} (Total Deposits):")
    print(f"    Range: {total_deposits.index.min()} to {total_deposits.index.max()}")
    print(f"    Points: {len(total_deposits)}")

    # Resample to monthly (end of month)
    demand_monthly = demand_deposits.resample('ME').last()
    total_monthly = total_deposits.resample('ME').last()

    # Align indices
    common_idx = demand_monthly.index.intersection(total_monthly.index)
    demand_monthly = demand_monthly.loc[common_idx]
    total_monthly = total_monthly.loc[common_idx]

    # Compute ratio
    factor_raw = (demand_monthly / total_monthly * 100).dropna()

    print(f"\n  Unstable Deposits Ratio:")
    print(f"    Range: {factor_raw.index.min()} to {factor_raw.index.max()}")
    print(f"    Monthly points: {len(factor_raw)}")
    print(f"    Current value: {factor_raw.iloc[-1]:.2f}%")
    print(f"    Min: {factor_raw.min():.2f}%, Max: {factor_raw.max():.2f}%")

    # Rolling percentile function
    def rolling_percentile(series, window):
        """Compute rolling percentile rank (0-100)"""
        result = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            window_data = series.iloc[i-window:i]
            current = series.iloc[i]
            pct = (window_data < current).sum() / len(window_data) * 100
            result.iloc[i] = pct
        return result

    # Compute percentiles
    percentiles = {}
    for name, window in PERCENTILE_WINDOWS.items():
        pctl = rolling_percentile(factor_raw, window)
        percentiles[name] = pctl
        valid_count = pctl.notna().sum()
        print(f"  {name} window ({window}M): {valid_count} valid points")

    # Create dataframe
    df = pd.DataFrame({
        'factor_raw': factor_raw,
        'factor_pctl_5Y': percentiles['5Y'],
        'factor_pctl_10Y': percentiles['10Y'],
    })

    # Apply publication lag
    df.index = df.index + pd.DateOffset(months=RELEASE_LAG_MONTHS)
    df.index.name = 'as_of_date'
    df = df.dropna()

    print(f"  After lag adjustment: {df.index.min()} to {df.index.max()}")

    return df


def load_spx():
    """Load SPX daily data"""
    filepath = os.path.join(PROJECT_ROOT, 'SPX 1D Data (1).csv')
    df = pd.read_csv(filepath, parse_dates=['time'])
    df = df.set_index('time')[['close']].rename(columns={'close': 'SPX'})
    return df['SPX']


def run_insample_analysis(factor: pd.Series, spx: pd.Series, window_name: str) -> dict:
    """Stage 1: In-Sample Analysis"""
    print(f"\n{'='*60}")
    print(f"STAGE 1: IN-SAMPLE ANALYSIS ({window_name})")
    print(f"{'='*60}")

    results = {}

    # Compute forward returns and MDD
    spx_monthly = spx.resample('ME').last()
    fwd_return_12m = compute_forward_return(spx_monthly, horizon=12)
    fwd_mdd_12m = compute_forward_max_drawdown(spx, horizon=252)

    # Align data
    common_idx = factor.index.intersection(fwd_return_12m.dropna().index)
    factor_aligned = factor.loc[common_idx]
    return_aligned = fwd_return_12m.loc[common_idx]

    print(f"\n  Aligned samples: {len(common_idx)}")
    print(f"  Date range: {common_idx.min()} to {common_idx.max()}")

    # 1. IC Analysis
    print("\n[1. IC Analysis]")
    ic, p_val = stats.spearmanr(factor_aligned, return_aligned)

    results['ic'] = {
        'ic': ic,
        'p_value': p_val,
    }
    print(f"  Spearman IC: {ic:.4f}")
    print(f"  p-value: {p_val:.4f}")

    ic_pass = abs(ic) > 0.03 and p_val < 0.05
    print(f"  Status: {'PASS' if ic_pass else 'FAIL'} (|IC|>0.03 & p<0.05)")

    # 2. AUC Analysis
    print("\n[2. AUC Analysis (Crash Prediction)]")
    mdd_common = fwd_mdd_12m.reindex(common_idx).dropna()
    factor_mdd = factor_aligned.loc[mdd_common.index]

    is_crash = (mdd_common < CRASH_THRESHOLD).astype(int)
    crash_rate = is_crash.mean()
    print(f"  Crash rate (MDD<{CRASH_THRESHOLD*100:.0f}%): {crash_rate*100:.1f}%")

    if crash_rate > 0 and crash_rate < 1:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(is_crash, factor_mdd)
        auc_effective = max(auc, 1 - auc)
        direction = "high→crash" if auc >= 0.5 else "low→crash"
    else:
        auc = np.nan
        auc_effective = np.nan
        direction = "N/A"

    results['auc'] = {
        'auc': auc,
        'auc_effective': auc_effective,
        'direction': direction,
        'crash_rate': crash_rate,
    }
    print(f"  AUC: {auc:.3f}" if not np.isnan(auc) else "  AUC: N/A")
    print(f"  Effective AUC: {auc_effective:.3f} ({direction})" if not np.isnan(auc_effective) else "")

    auc_pass = auc_effective > 0.55 if not np.isnan(auc_effective) else False
    print(f"  Status: {'PASS' if auc_pass else 'FAIL'} (AUC>0.55)")

    # 3. Quintile Analysis
    print("\n[3. Quintile Analysis]")
    try:
        df_quintile = pd.DataFrame({
            'factor': factor_aligned.values,
            'return': return_aligned.values
        }, index=factor_aligned.index)
        quintile_result = run_quintile_analysis(df_quintile, 'factor', 'return', n_quantiles=5)
        results['quintile'] = quintile_result

        mono = quintile_result.get('monotonicity', {})
        spearman_q = mono.get('spearman_corr', np.nan)
        q5_q1 = mono.get('q5_minus_q1', np.nan)

        print(f"  Quintile Spearman: {spearman_q:.3f}" if not np.isnan(spearman_q) else "  Quintile Spearman: N/A")
        print(f"  Q5-Q1 spread: {q5_q1*100:.2f}%" if not np.isnan(q5_q1) else "  Q5-Q1 spread: N/A")

        quintile_pass = abs(spearman_q) > 0.7 if not np.isnan(spearman_q) else False
        print(f"  Status: {'PASS' if quintile_pass else 'FAIL'} (|Spearman|>0.7)")
    except Exception as e:
        print(f"  Error: {e}")
        results['quintile'] = {'error': str(e)}
        quintile_pass = False

    # Summary
    print("\n[In-Sample Summary]")
    n_pass = sum([ic_pass, auc_pass, quintile_pass])
    print(f"  Tests passed: {n_pass}/3")
    print(f"  Proceed to OOS: {'YES' if n_pass >= 1 else 'NO'}")

    results['insample_summary'] = {
        'ic_pass': ic_pass,
        'auc_pass': auc_pass,
        'quintile_pass': quintile_pass,
        'n_pass': n_pass,
        'proceed_to_oos': n_pass >= 1
    }

    return results


def run_oos_validation(factor: pd.Series, spx: pd.Series, window_name: str) -> dict:
    """Stage 2: Out-of-Sample 5-Gate Validation"""
    print(f"\n{'='*60}")
    print(f"STAGE 2: OUT-OF-SAMPLE 5-GATE VALIDATION ({window_name})")
    print(f"{'='*60}")

    # Prepare crash labels
    fwd_mdd_12m = compute_forward_max_drawdown(spx, horizon=252)

    # Align
    common_idx = factor.index.intersection(fwd_mdd_12m.dropna().index)

    df = pd.DataFrame({
        'percentile': factor.loc[common_idx],
        'is_crash': (fwd_mdd_12m.loc[common_idx] < CRASH_THRESHOLD).astype(int)
    })
    df = df.dropna()

    print(f"\n  Samples for validation: {len(df)}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Crash rate: {df['is_crash'].mean()*100:.1f}%")

    # Run 5-Gate validation
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


def generate_report(insample_results: dict, oos_results: dict,
                    factor_df: pd.DataFrame, window_name: str):
    """Generate validation report"""

    current_raw = factor_df['factor_raw'].iloc[-1]
    current_pctl = factor_df[f'factor_pctl_{window_name}'].iloc[-1]

    # Determine final status
    oos_pass = oos_results['all_pass']

    if oos_pass:
        final_status = "APPROVED"
        recommendation = "可作为预警信号进入监控系统"
    elif oos_results['n_pass'] >= 3:
        final_status = "CONDITIONAL"
        recommendation = "可作为辅助信息，但不建议单独使用"
    else:
        final_status = "REJECTED"
        recommendation = "不推荐使用"

    report = f"""# V2 Unstable Deposits Ratio ({window_name}) 验证报告

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 因子信息

| 属性 | 值 |
|------|-----|
| 公式 | WDDNS / DPSACBW027SBOG * 100 |
| 名称 | {FACTOR_NAME} |
| 含义 | 活期存款 / 总存款 (不稳定资金比例) |
| 频率 | Weekly → Monthly |
| 发布滞后 | {RELEASE_LAG_MONTHS} month |
| Percentile Window | {window_name} |

## 当前状态

| 指标 | 值 |
|------|-----|
| 当前比率 | {current_raw:.2f}% |
| {window_name} Percentile | {current_pctl:.1f}% |

---

## Stage 1: In-Sample Analysis

| 检验 | 结果 | 状态 |
|------|------|------|
| IC (Spearman) | {insample_results['ic']['ic']:.4f} | {'PASS' if insample_results['insample_summary']['ic_pass'] else 'FAIL'} |
| AUC (Crash) | {insample_results['auc'].get('auc_effective', 0):.3f} ({insample_results['auc'].get('direction', 'N/A')}) | {'PASS' if insample_results['insample_summary']['auc_pass'] else 'FAIL'} |
| Quintile Monotonicity | {insample_results['quintile'].get('monotonicity', {}).get('spearman_corr', 'N/A')} | {'PASS' if insample_results['insample_summary']['quintile_pass'] else 'FAIL'} |

**In-Sample 通过: {insample_results['insample_summary']['n_pass']}/3**

---

## Stage 2: Out-of-Sample 5-Gate Validation

| Gate | 描述 | 结果 | 详情 |
|------|------|------|------|
| Gate 0 | Real-time Availability | {'PASS' if oos_results['gates']['gate0']['pass'] else 'FAIL'} | {oos_results['gates']['gate0']['reason']} |
| Gate 1 | Walk-Forward OOS Lift | {'PASS' if oos_results['gates']['gate1']['pass'] else 'FAIL'} | {oos_results['gates']['gate1']['reason']} |
| Gate 2 | Leave-One-Crisis-Out | {'PASS' if oos_results['gates']['gate2']['pass'] else 'FAIL'} | {oos_results['gates']['gate2']['reason']} |
| Gate 3 | Lead Time | {'PASS' if oos_results['gates']['gate3']['pass'] else 'FAIL'} | {oos_results['gates']['gate3']['reason']} |
| Gate 4 | Zone Stability | {'PASS' if oos_results['gates']['gate4']['pass'] else 'FAIL'} | {oos_results['gates']['gate4']['reason']} |

**OOS Gates 通过: {oos_results['n_pass']}/5**
**Best Zone: [{oos_results['best_zone'][0]}%, {oos_results['best_zone'][1]}%]**

---

## 最终结论

| 项目 | 结果 |
|------|------|
| **最终状态** | **{final_status}** |
| **建议** | {recommendation} |

"""

    # Add crisis detail
    if 'details' in oos_results['gates']['gate3']:
        report += "\n### 危机前信号详情\n\n"
        report += "| 危机 | 有信号 | Zone比例 | 平均因子 |\n"
        report += "|------|--------|----------|----------|\n"
        for crisis, detail in oos_results['gates']['gate3']['details'].items():
            has_signal = '✓' if detail.get('has_signal', False) else '✗'
            zone_ratio = detail.get('zone_ratio', 0) * 100
            avg_factor = detail.get('avg_factor', 'N/A')
            if isinstance(avg_factor, (int, float)):
                report += f"| {crisis} | {has_signal} | {zone_ratio:.0f}% | {avg_factor:.1f}% |\n"
            else:
                report += f"| {crisis} | {has_signal} | {zone_ratio:.0f}% | {avg_factor} |\n"

    report += f"\n---\n\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"

    filepath = os.path.join(OUTPUT_DIR, f'V2_VALIDATION_{window_name}.md')
    with open(filepath, 'w') as f:
        f.write(report)
    print(f"\n  Report saved: {filepath}")

    return final_status


def main():
    """Main validation workflow"""
    print("=" * 70)
    print("V2 Factor Validation: Unstable Deposits Ratio")
    print("=" * 70)
    print(f"\nFormula: {NUMERATOR_SERIES} / {DENOMINATOR_SERIES} * 100")
    print("Meaning: Demand Deposits / Total Deposits (不稳定资金比例)")

    # Load data
    factor_df = load_factor_from_fred()
    spx = load_spx()

    # Save factor data
    factor_df.to_csv(os.path.join(OUTPUT_DIR, 'factor_data.csv'))
    print(f"\n  Factor data saved to {OUTPUT_DIR}/factor_data.csv")

    # Validate 10Y window
    window_name = '10Y'
    factor_col = f'factor_pctl_{window_name}'

    if factor_col not in factor_df.columns:
        print(f"ERROR: {factor_col} not found")
        return

    factor = factor_df[factor_col].dropna()

    # Stage 1: In-Sample
    insample_results = run_insample_analysis(factor, spx, window_name)

    # Stage 2: OOS
    if insample_results['insample_summary']['proceed_to_oos']:
        oos_results = run_oos_validation(factor, spx, window_name)
    else:
        print("\n[SKIPPING OOS] In-sample analysis shows insufficient signal")
        oos_results = {
            'all_pass': False,
            'n_pass': 0,
            'best_zone': (0, 0),
            'gates': {
                'gate0': {'pass': False, 'reason': 'Skipped'},
                'gate1': {'pass': False, 'reason': 'Skipped'},
                'gate2': {'pass': False, 'reason': 'Skipped'},
                'gate3': {'pass': False, 'reason': 'Skipped'},
                'gate4': {'pass': False, 'reason': 'Skipped'},
            }
        }

    # Generate report
    final_status = generate_report(insample_results, oos_results, factor_df, window_name)

    print("\n" + "=" * 70)
    print(f"VALIDATION COMPLETE: {final_status}")
    print("=" * 70)


if __name__ == '__main__':
    main()
