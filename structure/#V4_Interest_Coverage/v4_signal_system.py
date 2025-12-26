#!/usr/bin/env python3
"""
V4 Interest Coverage Ratio - 3-Level Signal System
====================================================

Signal Levels:
- GREEN (安全): ΔICR(4Q) > 0 (ICR 上升)
- YELLOW (早期压力): ΔICR(4Q) < 0 但 > -1σ (ICR 下降但未触发危险)
- RED (现金流裂缝): ΔICR(4Q) < -1σ (ICR 大幅下降)

Trigger Combination:
- 只有当信用条件收紧时，YELLOW/RED 才升级为"系统风险"
- 触发器: HY OAS 走阔 或 FCI 收紧

这样可以显著降低误报，并与其他指标形成互补。
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

from lib import compute_forward_max_drawdown

# ============== Configuration ==============
FRED_API_KEY = 'b37a95dcefcfcc0f98ddfb87daca2e34'
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Factor settings
PROFIT_SERIES = 'A464RC1Q027SBEA'
INTEREST_SERIES = 'B471RC1Q027SBEA'
RELEASE_LAG_MONTHS = 6

# Trigger series
HY_OAS_SERIES = 'BAMLH0A0HYM2'  # ICE BofA US High Yield OAS
NFCI_SERIES = 'NFCI'  # Chicago Fed National Financial Conditions Index

# Signal thresholds
RED_THRESHOLD_ZSCORE = -1.0  # Δ < -1σ = RED
CRASH_THRESHOLD = -0.20


def load_icr_data():
    """Load ICR and compute Δ(4Q) with rolling stats"""
    print("\n[Loading ICR Data]")
    fred = Fred(api_key=FRED_API_KEY)

    profit = fred.get_series(PROFIT_SERIES)
    interest = fred.get_series(INTEREST_SERIES)

    common_idx = profit.index.intersection(interest.index)
    ebit = profit.loc[common_idx] + interest.loc[common_idx]
    icr = (ebit / interest.loc[common_idx]).replace([np.inf, -np.inf], np.nan).dropna()
    icr = icr[icr > 0]

    # Compute Δ(4Q)
    delta_4q = icr.diff(4)

    # Rolling mean and std for Z-score (10Y = 40 quarters)
    window = 40
    rolling_mean = delta_4q.rolling(window=window, min_periods=20).mean()
    rolling_std = delta_4q.rolling(window=window, min_periods=20).std()
    delta_zscore = (delta_4q - rolling_mean) / rolling_std

    print(f"  ICR range: {icr.index.min()} to {icr.index.max()}")
    print(f"  Current ICR: {icr.iloc[-1]:.2f}x")
    print(f"  Current Δ(4Q): {delta_4q.iloc[-1]:.2f}")
    print(f"  Current Δ Z-score: {delta_zscore.iloc[-1]:.2f}")

    return pd.DataFrame({
        'icr': icr,
        'delta_4q': delta_4q,
        'delta_zscore': delta_zscore,
        'rolling_mean': rolling_mean,
        'rolling_std': rolling_std,
    })


def load_trigger_data():
    """Load credit spread and FCI data"""
    print("\n[Loading Trigger Data]")
    fred = Fred(api_key=FRED_API_KEY)

    # HY OAS (daily → monthly)
    try:
        hy_oas = fred.get_series(HY_OAS_SERIES)
        hy_oas_monthly = hy_oas.resample('ME').last()
        # Compute 6-month change
        hy_oas_change = hy_oas_monthly.diff(6)
        # Compute percentile
        hy_oas_pctl = hy_oas_monthly.rolling(120).apply(
            lambda x: (x.iloc[:-1] < x.iloc[-1]).sum() / len(x.iloc[:-1]) * 100 if len(x) > 1 else 50
        )
        print(f"  HY OAS: {hy_oas_monthly.index.min()} to {hy_oas_monthly.index.max()}")
        print(f"  Current HY OAS: {hy_oas_monthly.iloc[-1]:.0f} bps")
    except Exception as e:
        print(f"  HY OAS error: {e}")
        hy_oas_monthly = None
        hy_oas_change = None
        hy_oas_pctl = None

    # NFCI (weekly → monthly)
    try:
        nfci = fred.get_series(NFCI_SERIES)
        nfci_monthly = nfci.resample('ME').last()
        print(f"  NFCI: {nfci_monthly.index.min()} to {nfci_monthly.index.max()}")
        print(f"  Current NFCI: {nfci_monthly.iloc[-1]:.2f}")
    except Exception as e:
        print(f"  NFCI error: {e}")
        nfci_monthly = None

    return {
        'hy_oas': hy_oas_monthly,
        'hy_oas_change': hy_oas_change,
        'hy_oas_pctl': hy_oas_pctl,
        'nfci': nfci_monthly,
    }


def compute_signal_levels(icr_df: pd.DataFrame) -> pd.Series:
    """
    Compute 3-level signal based on Δ(4Q) Z-score:
    - GREEN (0): Δ > 0
    - YELLOW (1): -1σ < Δ < 0
    - RED (2): Δ < -1σ
    """
    delta_zscore = icr_df['delta_zscore']
    delta_4q = icr_df['delta_4q']

    signal = pd.Series(index=delta_zscore.index, dtype=str)

    # GREEN: ICR rising
    signal[delta_4q > 0] = 'GREEN'

    # YELLOW: ICR falling but not severely
    yellow_mask = (delta_4q <= 0) & (delta_zscore > RED_THRESHOLD_ZSCORE)
    signal[yellow_mask] = 'YELLOW'

    # RED: ICR falling severely
    red_mask = delta_zscore <= RED_THRESHOLD_ZSCORE
    signal[red_mask] = 'RED'

    return signal


def compute_triggered_signal(icr_signal: pd.Series, triggers: dict) -> pd.Series:
    """
    Combine ICR signal with credit triggers:
    - GREEN stays GREEN
    - YELLOW + trigger → YELLOW_TRIGGERED
    - RED + trigger → RED_TRIGGERED (系统风险)

    Trigger conditions:
    1. HY OAS > 80th percentile (credit stress)
    2. OR NFCI > 0 (tight financial conditions)
    """
    triggered_signal = icr_signal.copy()

    hy_oas_pctl = triggers.get('hy_oas_pctl')
    nfci = triggers.get('nfci')

    # Create trigger mask
    trigger_mask = pd.Series(False, index=icr_signal.index)

    if hy_oas_pctl is not None:
        # HY OAS > 80th percentile = credit stress
        hy_aligned = hy_oas_pctl.reindex(icr_signal.index, method='ffill')
        trigger_mask = trigger_mask | (hy_aligned > 80)

    if nfci is not None:
        # NFCI > 0 = tight conditions
        nfci_aligned = nfci.reindex(icr_signal.index, method='ffill')
        trigger_mask = trigger_mask | (nfci_aligned > 0)

    # Apply triggers
    yellow_triggered = (icr_signal == 'YELLOW') & trigger_mask
    red_triggered = (icr_signal == 'RED') & trigger_mask

    triggered_signal[yellow_triggered] = 'YELLOW_TRIGGERED'
    triggered_signal[red_triggered] = 'RED_TRIGGERED'

    return triggered_signal


def load_spx():
    """Load SPX data"""
    filepath = os.path.join(PROJECT_ROOT, 'SPX 1D Data (1).csv')
    df = pd.read_csv(filepath, parse_dates=['time'])
    return df.set_index('time')['close']


def backtest_signal_system(icr_df: pd.DataFrame, triggers: dict, spx: pd.Series):
    """Backtest the 3-level signal system"""
    print("\n[Backtesting Signal System]")

    # Compute signals
    icr_signal = compute_signal_levels(icr_df)
    triggered_signal = compute_triggered_signal(icr_signal, triggers)

    # Forward-fill quarterly to monthly and apply lag
    icr_signal_monthly = icr_signal.resample('ME').last().ffill()
    icr_signal_monthly.index = icr_signal_monthly.index + pd.DateOffset(months=RELEASE_LAG_MONTHS)

    triggered_signal_monthly = triggered_signal.resample('ME').last().ffill()
    triggered_signal_monthly.index = triggered_signal_monthly.index + pd.DateOffset(months=RELEASE_LAG_MONTHS)

    # Compute forward MDD
    fwd_mdd = compute_forward_max_drawdown(spx, horizon=252)
    fwd_mdd_monthly = fwd_mdd.resample('ME').last()

    # Align
    common_idx = icr_signal_monthly.dropna().index.intersection(fwd_mdd_monthly.dropna().index)

    results = pd.DataFrame({
        'icr_signal': icr_signal_monthly.loc[common_idx],
        'triggered_signal': triggered_signal_monthly.loc[common_idx],
        'fwd_mdd': fwd_mdd_monthly.loc[common_idx],
        'is_crash': (fwd_mdd_monthly.loc[common_idx] < CRASH_THRESHOLD).astype(int),
    })

    print(f"\n  Backtest period: {common_idx.min()} to {common_idx.max()}")
    print(f"  Total samples: {len(results)}")

    # Analyze ICR signal alone
    print("\n[ICR Signal Alone]")
    print(f"{'Signal':<10} {'N':<6} {'Crash Rate':<12} {'Avg MDD':<10}")
    print("-" * 40)
    for signal in ['GREEN', 'YELLOW', 'RED']:
        mask = results['icr_signal'] == signal
        n = mask.sum()
        if n > 0:
            cr = results.loc[mask, 'is_crash'].mean() * 100
            avg_mdd = results.loc[mask, 'fwd_mdd'].mean() * 100
            print(f"{signal:<10} {n:<6} {cr:<12.1f}% {avg_mdd:<10.1f}%")

    # Analyze triggered signal
    print("\n[With Credit Trigger]")
    print(f"{'Signal':<20} {'N':<6} {'Crash Rate':<12} {'Avg MDD':<10}")
    print("-" * 50)
    for signal in ['GREEN', 'YELLOW', 'YELLOW_TRIGGERED', 'RED', 'RED_TRIGGERED']:
        mask = results['triggered_signal'] == signal
        n = mask.sum()
        if n > 0:
            cr = results.loc[mask, 'is_crash'].mean() * 100
            avg_mdd = results.loc[mask, 'fwd_mdd'].mean() * 100
            print(f"{signal:<20} {n:<6} {cr:<12.1f}% {avg_mdd:<10.1f}%")

    # False positive analysis
    print("\n[False Positive Analysis]")
    # RED without crash
    red_no_crash = ((results['icr_signal'] == 'RED') & (results['is_crash'] == 0)).sum()
    red_total = (results['icr_signal'] == 'RED').sum()
    red_fp_rate = red_no_crash / red_total * 100 if red_total > 0 else 0

    # RED_TRIGGERED without crash
    red_trig_no_crash = ((results['triggered_signal'] == 'RED_TRIGGERED') & (results['is_crash'] == 0)).sum()
    red_trig_total = (results['triggered_signal'] == 'RED_TRIGGERED').sum()
    red_trig_fp_rate = red_trig_no_crash / red_trig_total * 100 if red_trig_total > 0 else 0

    print(f"  RED alone: {red_no_crash}/{red_total} false positives ({red_fp_rate:.1f}%)")
    print(f"  RED + Trigger: {red_trig_no_crash}/{red_trig_total} false positives ({red_trig_fp_rate:.1f}%)")

    return results


def analyze_crisis_periods(icr_df: pd.DataFrame, triggers: dict):
    """Analyze signal behavior during crisis periods"""
    print("\n[Crisis Period Analysis]")

    icr_signal = compute_signal_levels(icr_df)
    triggered_signal = compute_triggered_signal(icr_signal, triggers)

    crisis_periods = {
        'Dot-com': ('1999-06-01', '2000-03-01'),
        'GFC': ('2007-01-01', '2007-10-01'),
        'COVID': ('2019-06-01', '2020-02-01'),
        '2022 Rate Hike': ('2021-06-01', '2022-01-01'),
    }

    print(f"\n{'Crisis':<20} {'ICR Signal':<12} {'Triggered':<18} {'Δ Z-score':<12}")
    print("-" * 64)

    for name, (start, end) in crisis_periods.items():
        mask = (icr_df.index >= start) & (icr_df.index <= end)
        if mask.sum() > 0:
            # Get dominant signal
            signals = icr_signal.loc[mask].dropna()
            triggered = triggered_signal.loc[mask].dropna()
            zscores = icr_df.loc[mask, 'delta_zscore'].dropna()

            if len(signals) > 0:
                dominant_signal = signals.mode().iloc[0] if len(signals.mode()) > 0 else 'N/A'
                dominant_triggered = triggered.mode().iloc[0] if len(triggered.mode()) > 0 else 'N/A'
                avg_zscore = zscores.mean() if len(zscores) > 0 else np.nan

                print(f"{name:<20} {dominant_signal:<12} {dominant_triggered:<18} {avg_zscore:<12.2f}")


def generate_signal_report(backtest_results: pd.DataFrame, icr_df: pd.DataFrame):
    """Generate signal system documentation"""

    current_icr = icr_df['icr'].iloc[-1]
    current_delta = icr_df['delta_4q'].iloc[-1]
    current_zscore = icr_df['delta_zscore'].iloc[-1]

    # Determine current signal
    if current_delta > 0:
        current_signal = 'GREEN'
    elif current_zscore > RED_THRESHOLD_ZSCORE:
        current_signal = 'YELLOW'
    else:
        current_signal = 'RED'

    report = f"""# V4 Interest Coverage Ratio - 3-Level Signal System

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Signal Definition

| Level | Condition | 含义 |
|-------|-----------|------|
| **GREEN** | Δ(4Q) > 0 | ICR 上升，企业盈利覆盖改善 |
| **YELLOW** | Δ(4Q) < 0 且 Z > -1σ | ICR 下降但未严重恶化 |
| **RED** | Δ(4Q) Z-score < -1σ | ICR 大幅下降，现金流裂缝 |

## Trigger Combination

**原则**: 只有当信用条件收紧时，YELLOW/RED 才升级为"系统风险"

| ICR Signal | + Credit Trigger | → Final Signal |
|------------|-----------------|----------------|
| GREEN | Any | GREEN (安全) |
| YELLOW | No trigger | YELLOW (观察) |
| YELLOW | Triggered | **YELLOW_TRIGGERED** (警惕) |
| RED | No trigger | RED (企业压力) |
| RED | Triggered | **RED_TRIGGERED** (系统风险) |

**触发条件**:
- HY OAS > 80th percentile (信用利差走阔)
- OR NFCI > 0 (金融条件收紧)

---

## 当前状态

| 指标 | 值 |
|------|-----|
| 当前 ICR | {current_icr:.2f}x |
| Δ(4Q) | {current_delta:+.2f} |
| Δ Z-score | {current_zscore:+.2f} |
| **当前信号** | **{current_signal}** |

---

## Backtest Results

### ICR Signal Alone

| Signal | N | Crash Rate | Avg MDD |
|--------|---|------------|---------|
"""

    for signal in ['GREEN', 'YELLOW', 'RED']:
        mask = backtest_results['icr_signal'] == signal
        n = mask.sum()
        if n > 0:
            cr = backtest_results.loc[mask, 'is_crash'].mean() * 100
            avg_mdd = backtest_results.loc[mask, 'fwd_mdd'].mean() * 100
            report += f"| {signal} | {n} | {cr:.1f}% | {avg_mdd:.1f}% |\n"

    report += """
### With Credit Trigger

| Signal | N | Crash Rate | Avg MDD |
|--------|---|------------|---------|
"""

    for signal in ['GREEN', 'YELLOW', 'YELLOW_TRIGGERED', 'RED', 'RED_TRIGGERED']:
        mask = backtest_results['triggered_signal'] == signal
        n = mask.sum()
        if n > 0:
            cr = backtest_results.loc[mask, 'is_crash'].mean() * 100
            avg_mdd = backtest_results.loc[mask, 'fwd_mdd'].mean() * 100
            report += f"| {signal} | {n} | {cr:.1f}% | {avg_mdd:.1f}% |\n"

    # Calculate improvement
    red_mask = backtest_results['icr_signal'] == 'RED'
    red_trig_mask = backtest_results['triggered_signal'] == 'RED_TRIGGERED'

    red_cr = backtest_results.loc[red_mask, 'is_crash'].mean() * 100 if red_mask.sum() > 0 else 0
    red_trig_cr = backtest_results.loc[red_trig_mask, 'is_crash'].mean() * 100 if red_trig_mask.sum() > 0 else 0

    report += f"""
---

## Key Findings

### False Positive Reduction

| Metric | RED Alone | RED + Trigger |
|--------|-----------|---------------|
| Crash Rate | {red_cr:.1f}% | {red_trig_cr:.1f}% |
| Improvement | - | +{red_trig_cr - red_cr:.1f}pp |

### Signal Interpretation

1. **GREEN**: 安全期，正常配置
2. **YELLOW**: 早期观察，关注信用条件变化
3. **YELLOW_TRIGGERED**: 信用条件已收紧，考虑降低风险敞口
4. **RED**: 企业盈利压力，但可能是孤立事件
5. **RED_TRIGGERED**: **系统风险信号**，建议显著降低风险敞口

---

## Implementation

```python
def get_v4_signal(icr_delta_4q, icr_delta_zscore, hy_oas_pctl, nfci):
    '''
    Get V4 ICR signal with trigger combination
    '''
    # Step 1: ICR signal
    if icr_delta_4q > 0:
        icr_signal = 'GREEN'
    elif icr_delta_zscore > -1.0:
        icr_signal = 'YELLOW'
    else:
        icr_signal = 'RED'

    # Step 2: Check trigger
    trigger_active = (hy_oas_pctl > 80) or (nfci > 0)

    # Step 3: Combine
    if icr_signal == 'GREEN':
        return 'GREEN'
    elif trigger_active:
        return f'{{icr_signal}}_TRIGGERED'
    else:
        return icr_signal
```

---

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    filepath = os.path.join(OUTPUT_DIR, 'V4_SIGNAL_SYSTEM.md')
    with open(filepath, 'w') as f:
        f.write(report)
    print(f"\n  Signal system report saved: {filepath}")


def main():
    print("=" * 70)
    print("V4 Interest Coverage Ratio - 3-Level Signal System")
    print("=" * 70)

    # Load data
    icr_df = load_icr_data()
    triggers = load_trigger_data()
    spx = load_spx()

    # Analyze crisis periods
    analyze_crisis_periods(icr_df, triggers)

    # Backtest
    backtest_results = backtest_signal_system(icr_df, triggers, spx)

    # Generate report
    generate_signal_report(backtest_results, icr_df)

    print("\n" + "=" * 70)
    print("SIGNAL SYSTEM ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
