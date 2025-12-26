#!/usr/bin/env python3
"""
V5 Factor Validation: TDSP (Household Debt Service Ratio)
==========================================================

Series: TDSP - Household Debt Service Payments as a Percent of Disposable Personal Income
Source: Federal Reserve Board
Frequency: Quarterly
Release Lag: ~3 months (based on ALFRED analysis)

Hypothesis: High household debt burden (high TDSP) may precede market stress
- High debt service = less disposable income = vulnerable to shocks
- May be a leading indicator for consumer-driven recessions
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
    ICAnalyzer,
    run_quintile_analysis,
    compute_drawdown_event_auc,
    ols_with_hac,
    block_bootstrap_regression,
    # OOS validation
    validate_factor,
    find_best_zone,
    evaluate_zone,
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
FACTOR_SERIES = 'TDSP'
FACTOR_NAME = 'Household Debt Service Ratio'
RELEASE_LAG_MONTHS = 3  # ~2.7-3 months based on ALFRED analysis

# Percentile windows (in quarters)
PERCENTILE_WINDOWS = {
    '5Y': 20,   # 5 years = 20 quarters
    '10Y': 40,  # 10 years = 40 quarters
}

# Crash definition
CRASH_THRESHOLD = -0.20  # MDD < -20%

# HAC lag for 12M overlapping returns
HAC_LAG = 11


def load_factor_from_fred():
    """Load TDSP data from FRED with proper lag adjustment."""
    print("\n[Loading Factor Data from FRED]")
    fred = Fred(api_key=FRED_API_KEY)

    # Get factor series (quarterly)
    factor_raw = fred.get_series(FACTOR_SERIES)
    factor_raw = factor_raw.dropna()

    print(f"  Series: {FACTOR_SERIES} ({FACTOR_NAME})")
    print(f"  Data range: {factor_raw.index.min()} to {factor_raw.index.max()}")
    print(f"  Quarterly data points: {len(factor_raw)}")
    print(f"  Release lag: {RELEASE_LAG_MONTHS} months (ALFRED)")
    print(f"  Current value: {factor_raw.iloc[-1]:.2f}%")

    # Rolling percentile function (on quarterly data)
    def rolling_percentile(series, window):
        """Compute rolling percentile rank (0-100)"""
        result = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            window_data = series.iloc[i-window:i]
            current = series.iloc[i]
            pct = (window_data < current).sum() / len(window_data) * 100
            result.iloc[i] = pct
        return result

    # Compute percentiles on quarterly data
    percentiles = {}
    for name, window in PERCENTILE_WINDOWS.items():
        pctl = rolling_percentile(factor_raw, window)
        percentiles[name] = pctl
        valid_count = pctl.notna().sum()
        print(f"  {name} window ({window}Q): {valid_count} valid points")

    # Create quarterly dataframe
    df_quarterly = pd.DataFrame({
        'factor_raw': factor_raw,
        'factor_pctl_5Y': percentiles['5Y'],
        'factor_pctl_10Y': percentiles['10Y'],
    })

    # Apply publication lag
    df_quarterly.index = df_quarterly.index + pd.DateOffset(months=RELEASE_LAG_MONTHS)
    print(f"  After lag adjustment: {df_quarterly.index.min()} to {df_quarterly.index.max()}")

    # Forward-fill to monthly
    df_monthly = df_quarterly.resample('ME').last().ffill()
    df_monthly.index.name = 'as_of_date'
    df_monthly = df_monthly.dropna()

    print(f"  Monthly data points: {len(df_monthly)}")

    return df_monthly


def load_spx():
    """Load SPX daily data"""
    filepath = os.path.join(PROJECT_ROOT, 'SPX 1D Data (1).csv')
    df = pd.read_csv(filepath, parse_dates=['time'])
    df = df.set_index('time')[['close']].rename(columns={'close': 'SPX'})
    return df['SPX']


def run_insample_analysis(factor: pd.Series, spx: pd.Series, window_name: str) -> dict:
    """
    Stage 1: In-Sample Analysis
    - IC with HAC
    - AUC for crash prediction
    - Quintile analysis
    - Bootstrap significance
    """
    print(f"\n{'='*60}")
    print(f"STAGE 1: IN-SAMPLE ANALYSIS ({window_name})")
    print(f"{'='*60}")

    results = {}

    # Compute forward returns and MDD
    spx_monthly = spx.resample('ME').last()
    fwd_return_12m = compute_forward_return(spx_monthly, horizon=12)  # 12 months for monthly data
    fwd_mdd_12m = compute_forward_max_drawdown(spx, horizon=252)  # 252 days = 12 months

    # Align data
    common_idx = factor.index.intersection(fwd_return_12m.dropna().index)
    factor_aligned = factor.loc[common_idx]
    return_aligned = fwd_return_12m.loc[common_idx]

    print(f"\n  Aligned samples: {len(common_idx)}")
    print(f"  Date range: {common_idx.min()} to {common_idx.max()}")

    # 1. IC Analysis
    print("\n[1. IC Analysis (HAC)]")
    ic, p_val = stats.spearmanr(factor_aligned, return_aligned)

    # HAC regression for t-stat
    try:
        hac_result = ols_with_hac(return_aligned.values, factor_aligned.values, lag=HAC_LAG)
        t_stat = hac_result['t_stat']
    except Exception as e:
        print(f"  HAC error: {e}")
        t_stat = np.nan

    results['ic'] = {
        'ic': ic,
        'p_value': p_val,
        't_stat_hac': t_stat,
    }
    print(f"  Spearman IC: {ic:.4f}")
    print(f"  p-value: {p_val:.4f}")
    print(f"  HAC t-stat: {t_stat:.2f}" if not np.isnan(t_stat) else "  HAC t-stat: N/A")

    # Use p-value if HAC t-stat not available
    if not np.isnan(t_stat):
        ic_pass = abs(ic) > 0.03 and abs(t_stat) > 2.0
    else:
        ic_pass = abs(ic) > 0.03 and p_val < 0.05
    print(f"  Status: {'PASS' if ic_pass else 'FAIL'} (|IC|>0.03 & significant)")

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
        # If AUC < 0.5, flip interpretation (high factor = low crash)
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
    print(f"  Effective AUC: {auc_effective:.3f} ({direction})" if not np.isnan(auc_effective) else "  Effective AUC: N/A")

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

    # 4. Bootstrap Significance
    print("\n[4. Bootstrap Significance]")
    try:
        boot_result = block_bootstrap_regression(
            return_aligned.values, factor_aligned.values,
            n_bootstrap=1000, block_size=12
        )
        results['bootstrap'] = boot_result

        beta = boot_result['original_beta']
        ci_lower = boot_result['ci_lower']
        ci_upper = boot_result['ci_upper']
        p_val_boot = boot_result['p_value']

        print(f"  Beta: {beta:.4f}")
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  p-value: {p_val_boot:.4f}")

        boot_pass = ci_lower * ci_upper > 0  # CI doesn't contain 0
        print(f"  Status: {'PASS' if boot_pass else 'FAIL'} (CI excludes 0)")
    except Exception as e:
        print(f"  Error: {e}")
        results['bootstrap'] = {'error': str(e)}
        boot_pass = False

    # Summary
    print("\n[In-Sample Summary]")
    n_pass = sum([ic_pass, auc_pass, quintile_pass, boot_pass])
    print(f"  Tests passed: {n_pass}/4")
    print(f"  Proceed to OOS: {'YES' if n_pass >= 2 else 'NO (insufficient signal)'}")

    results['insample_summary'] = {
        'ic_pass': ic_pass,
        'auc_pass': auc_pass,
        'quintile_pass': quintile_pass,
        'bootstrap_pass': boot_pass,
        'n_pass': n_pass,
        'proceed_to_oos': n_pass >= 2
    }

    return results


def run_oos_validation(factor: pd.Series, spx: pd.Series, window_name: str) -> dict:
    """
    Stage 2: Out-of-Sample 5-Gate Validation
    """
    print(f"\n{'='*60}")
    print(f"STAGE 2: OUT-OF-SAMPLE 5-GATE VALIDATION ({window_name})")
    print(f"{'='*60}")

    # Prepare crash labels
    spx_monthly = spx.resample('ME').last()
    fwd_mdd_12m = compute_forward_max_drawdown(spx, horizon=252)  # 252 days = 12 months

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
    insample_pass = insample_results['insample_summary']['n_pass'] >= 2
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

    report = f"""# V5 TDSP ({window_name}) 因子验证报告

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 因子信息

| 属性 | 值 |
|------|-----|
| Series | TDSP |
| 名称 | {FACTOR_NAME} |
| 频率 | Quarterly |
| 发布滞后 | {RELEASE_LAG_MONTHS} months |
| Percentile Window | {window_name} |

## 当前状态

| 指标 | 值 |
|------|-----|
| 当前 TDSP | {current_raw:.2f}% |
| {window_name} Percentile | {current_pctl:.1f}% |

---

## Stage 1: In-Sample Analysis

| 检验 | 结果 | 状态 |
|------|------|------|
| IC (Spearman) | {insample_results['ic']['ic']:.4f} | {'PASS' if insample_results['insample_summary']['ic_pass'] else 'FAIL'} |
| AUC (Crash) | {insample_results['auc']['auc_effective']:.3f} | {'PASS' if insample_results['insample_summary']['auc_pass'] else 'FAIL'} |
| Quintile Monotonicity | {insample_results['quintile'].get('monotonicity', {}).get('spearman_corr', 'N/A')} | {'PASS' if insample_results['insample_summary']['quintile_pass'] else 'FAIL'} |
| Bootstrap 95% CI | {insample_results['bootstrap'].get('ci_lower', 0):.4f} to {insample_results['bootstrap'].get('ci_upper', 0):.4f} | {'PASS' if insample_results['insample_summary']['bootstrap_pass'] else 'FAIL'} |

**In-Sample 通过: {insample_results['insample_summary']['n_pass']}/4**

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

    # Add crisis detail if available
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

    filepath = os.path.join(OUTPUT_DIR, f'V5_VALIDATION_{window_name}.md')
    with open(filepath, 'w') as f:
        f.write(report)
    print(f"\n  Report saved: {filepath}")

    return final_status


def main():
    """Main validation workflow"""
    print("=" * 70)
    print("V5 Factor Validation: TDSP (Household Debt Service Ratio)")
    print("=" * 70)

    # Load data
    factor_df = load_factor_from_fred()
    spx = load_spx()

    # Save factor data
    factor_df.to_csv(os.path.join(OUTPUT_DIR, 'factor_data.csv'))
    print(f"\n  Factor data saved to {OUTPUT_DIR}/factor_data.csv")

    # Validate 10Y window (more stable)
    window_name = '10Y'
    factor_col = f'factor_pctl_{window_name}'

    if factor_col not in factor_df.columns:
        print(f"ERROR: {factor_col} not found")
        return

    factor = factor_df[factor_col].dropna()

    # Stage 1: In-Sample
    insample_results = run_insample_analysis(factor, spx, window_name)

    # Stage 2: OOS (only if in-sample shows signal)
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
