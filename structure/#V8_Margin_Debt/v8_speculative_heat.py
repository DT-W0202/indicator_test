#!/usr/bin/env python3
"""
V8 Speculative Leverage Heat Module
====================================

将 Margin Debt 从失败的 ratio 指标升级为"投机热度温度计"。

三个子指标:
1. Margin YoY: 杠杆扩张速度 (原始)
2. ΔYoY (6m): 加速度 - 扩张是否见顶
3. Margin YoY - SPX YoY: 杠杆是否跑赢市场

核心洞察:
- Ratio 失败是因为"分母污染 + 滞后数据 → 同步指标"
- YoY/加速度更能捕捉投机周期的前瞻信号
- 最佳用法: 增强 CAPE 的泡沫末期解释力

Failure Taxonomy:
- Type: Denominator-driven mechanical reversal
- Cause: slow numerator / fast denominator
- Fix: Use rate of change instead of ratio
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

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CRASH_THRESHOLD = -0.20


def load_margin_debt():
    """Load FINRA Margin Debt data"""
    filepath = os.path.join(OUTPUT_DIR, 'all_methods_data.csv')
    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    return df


def load_spx():
    """Load SPX data"""
    filepath = os.path.join(PROJECT_ROOT, 'SPX 1D Data (1).csv')
    df = pd.read_csv(filepath, parse_dates=['time'])
    return df.set_index('time')['close']


def compute_speculative_heat_indicators(margin_df: pd.DataFrame, spx: pd.Series) -> pd.DataFrame:
    """
    Compute Speculative Leverage Heat indicators

    Returns DataFrame with:
    1. margin_yoy: 12-month YoY growth (original)
    2. delta_yoy_6m: 6-month change in YoY (acceleration)
    3. margin_minus_spx: Margin YoY - SPX YoY (excess leverage)
    4. yoy_zscore: Z-score of YoY (10Y window)
    5. yoy_peak_signal: Peak detection (YoY starting to decline from high)
    """

    margin = margin_df['margin_debt']
    margin_yoy = margin_df['margin_yoy']

    # 1. ΔYoY (6m) - Acceleration
    # Positive = accelerating, Negative = decelerating
    delta_yoy_6m = margin_yoy.diff(6)

    # 2. Margin YoY - SPX YoY (Excess Leverage Growth)
    spx_monthly = spx.resample('ME').last()
    spx_yoy = spx_monthly.pct_change(12) * 100  # As percentage

    # Align
    common_idx = margin_yoy.dropna().index.intersection(spx_yoy.dropna().index)
    margin_minus_spx = margin_yoy.loc[common_idx] - spx_yoy.loc[common_idx]

    # 3. YoY Z-score (10Y rolling)
    yoy_mean = margin_yoy.rolling(120, min_periods=60).mean()
    yoy_std = margin_yoy.rolling(120, min_periods=60).std()
    yoy_zscore = (margin_yoy - yoy_mean) / yoy_std

    # 4. Peak Detection Signal
    # Signal = 1 when YoY was in top 20% but has declined for 3+ months
    yoy_pctl = margin_yoy.rolling(120, min_periods=60).apply(
        lambda x: (x.iloc[:-1] < x.iloc[-1]).sum() / len(x.iloc[:-1]) * 100 if len(x) > 1 else 50
    )

    # Was in top 20% within last 6 months
    was_high = yoy_pctl.rolling(6).max() >= 80
    # Currently declining (3-month negative trend)
    is_declining = margin_yoy.diff(1).rolling(3).sum() < 0
    # Peak signal
    peak_signal = (was_high & is_declining).astype(int)

    # Combine into DataFrame
    result = pd.DataFrame({
        'margin_debt': margin,
        'margin_yoy': margin_yoy,
        'delta_yoy_6m': delta_yoy_6m,
        'yoy_zscore': yoy_zscore,
        'yoy_pctl': yoy_pctl,
        'peak_signal': peak_signal,
    })

    # Add margin_minus_spx with proper alignment
    result['spx_yoy'] = spx_yoy
    result['margin_minus_spx'] = margin_minus_spx

    return result


def analyze_peak_signal(heat_df: pd.DataFrame, spx: pd.Series):
    """
    Experiment 1: Peak Detection Signal Analysis

    Test if "YoY declining from high" is more predictive than "YoY is high"
    """

    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Peak Detection Signal")
    print("=" * 70)
    print("\nHypothesis: Leverage peaks (YoY declining from high) predict crashes")
    print("better than 'YoY is currently high'")

    # Compute forward MDD
    fwd_mdd = compute_forward_max_drawdown(spx, horizon=252)
    fwd_mdd_monthly = fwd_mdd.resample('ME').last()

    # Align
    common_idx = heat_df.dropna(subset=['peak_signal', 'yoy_pctl']).index.intersection(
        fwd_mdd_monthly.dropna().index
    )

    df = pd.DataFrame({
        'peak_signal': heat_df.loc[common_idx, 'peak_signal'],
        'yoy_pctl': heat_df.loc[common_idx, 'yoy_pctl'],
        'margin_yoy': heat_df.loc[common_idx, 'margin_yoy'],
        'fwd_mdd': fwd_mdd_monthly.loc[common_idx],
    })
    df['is_crash'] = (df['fwd_mdd'] < CRASH_THRESHOLD).astype(int)

    print(f"\nSample: {len(df)} observations")
    print(f"Base crash rate: {df['is_crash'].mean()*100:.1f}%")

    # Compare signals
    print("\n[Signal Comparison]")

    # 1. YoY in top quintile (original)
    high_yoy = df['yoy_pctl'] >= 80
    if high_yoy.sum() > 0:
        cr = df.loc[high_yoy, 'is_crash'].mean() * 100
        print(f"  YoY >= 80th pctl: {high_yoy.sum()} obs, Crash rate = {cr:.1f}%")

    # 2. Peak signal
    peak = df['peak_signal'] == 1
    if peak.sum() > 0:
        cr = df.loc[peak, 'is_crash'].mean() * 100
        print(f"  Peak Signal (YoY declining from high): {peak.sum()} obs, Crash rate = {cr:.1f}%")

    # 3. Combination: High YoY AND declining
    combo = high_yoy & peak
    if combo.sum() > 0:
        cr = df.loc[combo, 'is_crash'].mean() * 100
        print(f"  High YoY + Declining: {combo.sum()} obs, Crash rate = {cr:.1f}%")

    # Check crisis coverage
    print("\n[Crisis Coverage - Peak Signal]")
    crisis_periods = {
        'Dot-com': ('1999-06-01', '2000-02-01'),
        'GFC': ('2007-01-01', '2007-09-01'),
        'COVID': ('2019-06-01', '2020-01-01'),
        '2022': ('2021-06-01', '2021-12-01'),
    }

    for crisis, (start, end) in crisis_periods.items():
        mask = (df.index >= start) & (df.index <= end)
        if mask.sum() > 0:
            has_peak = (df.loc[mask, 'peak_signal'] == 1).any()
            peak_pct = (df.loc[mask, 'peak_signal'] == 1).mean() * 100
            print(f"  {crisis}: {'✓' if has_peak else '✗'} ({peak_pct:.0f}% with signal)")

    return df


def analyze_cape_margin_interaction(heat_df: pd.DataFrame, spx: pd.Series):
    """
    Experiment 2: CAPE × Margin Interaction

    Test if combining CAPE danger + Margin high creates a stronger bubble signal
    """

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: CAPE × Margin YoY Interaction")
    print("=" * 70)
    print("\nHypothesis: CAPE Danger + Margin High = Stronger bubble signal")

    # Load CAPE data
    cape_path = os.path.join(PROJECT_ROOT, 'structure/V7_Shiller_PE_CAPE/all_methods_data.csv')
    cape_df = pd.read_csv(cape_path, parse_dates=['Date'], index_col='Date')
    cape = cape_df['cape_raw']

    # Compute CAPE percentile
    cape_pctl = cape.expanding(min_periods=60).apply(
        lambda x: (x.iloc[:-1] < x.iloc[-1]).sum() / len(x.iloc[:-1]) * 100 if len(x) > 1 else 50
    )

    # Compute forward MDD
    fwd_mdd = compute_forward_max_drawdown(spx, horizon=252)
    fwd_mdd_monthly = fwd_mdd.resample('ME').last()

    # Align all data
    common_idx = (
        heat_df.dropna(subset=['yoy_pctl']).index
        .intersection(cape_pctl.dropna().index)
        .intersection(fwd_mdd_monthly.dropna().index)
    )

    df = pd.DataFrame({
        'cape_pctl': cape_pctl.loc[common_idx],
        'margin_yoy_pctl': heat_df.loc[common_idx, 'yoy_pctl'],
        'margin_yoy': heat_df.loc[common_idx, 'margin_yoy'],
        'fwd_mdd': fwd_mdd_monthly.loc[common_idx],
    })
    df['is_crash'] = (df['fwd_mdd'] < CRASH_THRESHOLD).astype(int)

    # Create interaction signals
    df['cape_danger'] = df['cape_pctl'] >= 90  # Top decile
    df['margin_high'] = df['margin_yoy_pctl'] >= 80  # Top quintile
    df['bubble_score'] = (df['cape_danger'] & df['margin_high']).astype(int)

    print(f"\nSample: {len(df)} observations")
    print(f"Base crash rate: {df['is_crash'].mean()*100:.1f}%")

    # Compare signals
    print("\n[Signal Comparison]")

    # 1. CAPE only
    cape_danger = df['cape_danger']
    if cape_danger.sum() > 0:
        cr = df.loc[cape_danger, 'is_crash'].mean() * 100
        print(f"  CAPE >= 90th pctl only: {cape_danger.sum()} obs, Crash rate = {cr:.1f}%")

    # 2. Margin only
    margin_high = df['margin_high']
    if margin_high.sum() > 0:
        cr = df.loc[margin_high, 'is_crash'].mean() * 100
        print(f"  Margin YoY >= 80th pctl only: {margin_high.sum()} obs, Crash rate = {cr:.1f}%")

    # 3. Bubble Score (interaction)
    bubble = df['bubble_score'] == 1
    if bubble.sum() > 0:
        cr = df.loc[bubble, 'is_crash'].mean() * 100
        print(f"  Bubble Score (CAPE + Margin): {bubble.sum()} obs, Crash rate = {cr:.1f}%")

    # 4. CAPE danger but NOT margin high (clean overvaluation)
    clean_cape = cape_danger & (~margin_high)
    if clean_cape.sum() > 0:
        cr = df.loc[clean_cape, 'is_crash'].mean() * 100
        print(f"  CAPE danger, Margin normal: {clean_cape.sum()} obs, Crash rate = {cr:.1f}%")

    # Crisis coverage
    print("\n[Bubble Score Crisis Coverage]")
    crisis_periods = {
        'Dot-com': ('1999-06-01', '2000-03-01'),
        'GFC': ('2007-01-01', '2007-10-01'),
        '2022': ('2021-06-01', '2022-01-01'),
    }

    for crisis, (start, end) in crisis_periods.items():
        mask = (df.index >= start) & (df.index <= end)
        if mask.sum() > 0:
            has_bubble = (df.loc[mask, 'bubble_score'] == 1).any()
            bubble_pct = (df.loc[mask, 'bubble_score'] == 1).mean() * 100
            print(f"  {crisis}: {'✓' if has_bubble else '✗'} ({bubble_pct:.0f}% with Bubble Score)")

    # Current status
    print("\n[Current Status]")
    print(f"  Current CAPE pctl: {df['cape_pctl'].iloc[-1]:.1f}%")
    print(f"  Current Margin YoY pctl: {df['margin_yoy_pctl'].iloc[-1]:.1f}%")
    print(f"  Current Margin YoY: {df['margin_yoy'].iloc[-1]:.1f}%")
    print(f"  Bubble Score: {df['bubble_score'].iloc[-1]} {'(ACTIVE!)' if df['bubble_score'].iloc[-1] else ''}")

    return df


def analyze_excess_leverage(heat_df: pd.DataFrame, spx: pd.Series):
    """
    Analyze Margin YoY - SPX YoY (Excess Leverage Growth)

    Tests if leverage growing faster than market is a risk signal
    """

    print("\n" + "=" * 70)
    print("ANALYSIS: Excess Leverage Growth (Margin YoY - SPX YoY)")
    print("=" * 70)

    # Compute forward MDD
    fwd_mdd = compute_forward_max_drawdown(spx, horizon=252)
    fwd_mdd_monthly = fwd_mdd.resample('ME').last()

    # Align
    common_idx = heat_df.dropna(subset=['margin_minus_spx']).index.intersection(
        fwd_mdd_monthly.dropna().index
    )

    df = pd.DataFrame({
        'margin_minus_spx': heat_df.loc[common_idx, 'margin_minus_spx'],
        'margin_yoy': heat_df.loc[common_idx, 'margin_yoy'],
        'spx_yoy': heat_df.loc[common_idx, 'spx_yoy'],
        'fwd_mdd': fwd_mdd_monthly.loc[common_idx],
    })
    df['is_crash'] = (df['fwd_mdd'] < CRASH_THRESHOLD).astype(int)

    # Compute IC
    ic, pval = stats.spearmanr(df['margin_minus_spx'].dropna(),
                               df.loc[df['margin_minus_spx'].notna(), 'fwd_mdd'])

    print(f"\nSample: {len(df)} observations")
    print(f"IC (Excess Leverage vs MDD): {ic:.3f} (p={pval:.4f})")

    # Quintile analysis
    print("\n[Quintile Analysis]")
    df['quintile'] = pd.qcut(df['margin_minus_spx'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        mask = df['quintile'] == q
        if mask.sum() > 0:
            avg_spread = df.loc[mask, 'margin_minus_spx'].mean()
            cr = df.loc[mask, 'is_crash'].mean() * 100
            print(f"  {q}: Avg spread = {avg_spread:+.1f}pp, Crash rate = {cr:.1f}%")

    # Current status
    print("\n[Current Status]")
    current_spread = df['margin_minus_spx'].iloc[-1]
    current_margin = df['margin_yoy'].iloc[-1]
    current_spx = df['spx_yoy'].iloc[-1]
    print(f"  Current Margin YoY: {current_margin:.1f}%")
    print(f"  Current SPX YoY: {current_spx:.1f}%")
    print(f"  Excess Leverage: {current_spread:+.1f}pp")

    if current_spread > 20:
        print(f"  ⚠️ Margin growing MUCH faster than market!")
    elif current_spread > 10:
        print(f"  ⚠️ Margin growing faster than market")
    elif current_spread < -10:
        print(f"  Margin lagging market (deleveraging?)")

    return df


def generate_heat_report(heat_df: pd.DataFrame, peak_df: pd.DataFrame,
                         cape_df: pd.DataFrame, excess_df: pd.DataFrame):
    """Generate Speculative Leverage Heat report"""

    report = f"""# V8 Speculative Leverage Heat Module

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Module Overview

将 Margin Debt 从失败的 ratio 指标升级为"投机热度温度计"。

### 为什么 Ratio 失败了？

**Failure Type: Denominator-driven mechanical reversal（分母驱动机械反向）**

| 阶段 | Market Cap | Margin Debt | Ratio | 解读 |
|------|------------|-------------|-------|------|
| 牛市 | ↑↑↑ (快) | ↑ (慢) | ↓ | 看起来"安全" |
| 崩盘 | ↓↓↓ (快) | → (滞后) | ↑↑ | 看起来"危险" |

**结论**: Ratio 是同步/滞后指标，不适合作为领先风险因子。

---

## 三个替代指标

### 1. Margin YoY (原始)
- **定义**: 12个月同比增速
- **当前值**: {heat_df['margin_yoy'].iloc[-1]:.1f}%
- **10Y 分位**: {heat_df['yoy_pctl'].iloc[-1]:.1f}%
- **评估**: {'⚠️ 高位警示' if heat_df['yoy_pctl'].iloc[-1] >= 80 else '正常'}

### 2. ΔYoY (6m) - 加速度
- **定义**: YoY 的 6 个月变化
- **当前值**: {heat_df['delta_yoy_6m'].iloc[-1]:+.1f}pp
- **解读**: {'杠杆扩张加速' if heat_df['delta_yoy_6m'].iloc[-1] > 0 else '杠杆扩张放缓'}

### 3. Margin YoY - SPX YoY (超额杠杆)
- **定义**: 杠杆增速 vs 市场增速
- **当前值**: {excess_df['margin_minus_spx'].iloc[-1]:+.1f}pp
- **解读**: {'⚠️ 杠杆跑赢市场!' if excess_df['margin_minus_spx'].iloc[-1] > 10 else '正常'}

---

## 实验结果

### Experiment 1: Peak Detection Signal

测试"YoY 从高位回落"是否比"YoY 很高"更有预警性。

| 信号 | N | Crash Rate |
|------|---|------------|
"""

    # Add peak signal results
    high_yoy = peak_df['yoy_pctl'] >= 80
    peak = peak_df['peak_signal'] == 1

    if high_yoy.sum() > 0:
        cr = peak_df.loc[high_yoy, 'is_crash'].mean() * 100
        report += f"| YoY >= 80th pctl | {high_yoy.sum()} | {cr:.1f}% |\n"

    if peak.sum() > 0:
        cr = peak_df.loc[peak, 'is_crash'].mean() * 100
        report += f"| Peak Signal (高位回落) | {peak.sum()} | {cr:.1f}% |\n"

    report += f"""
### Experiment 2: CAPE × Margin Interaction

测试"高估值 + 高杠杆"组合信号。

| 信号 | N | Crash Rate |
|------|---|------------|
"""

    # Add bubble score results
    cape_danger = cape_df['cape_danger']
    margin_high = cape_df['margin_high']
    bubble = cape_df['bubble_score'] == 1

    if cape_danger.sum() > 0:
        cr = cape_df.loc[cape_danger, 'is_crash'].mean() * 100
        report += f"| CAPE >= 90th pctl only | {cape_danger.sum()} | {cr:.1f}% |\n"

    if margin_high.sum() > 0:
        cr = cape_df.loc[margin_high, 'is_crash'].mean() * 100
        report += f"| Margin YoY >= 80th pctl only | {margin_high.sum()} | {cr:.1f}% |\n"

    if bubble.sum() > 0:
        cr = cape_df.loc[bubble, 'is_crash'].mean() * 100
        report += f"| **Bubble Score (CAPE + Margin)** | {bubble.sum()} | **{cr:.1f}%** |\n"

    # Current status
    bubble_active = cape_df['bubble_score'].iloc[-1] == 1

    report += f"""
---

## 当前状态

| 指标 | 值 | 状态 |
|------|-----|------|
| Margin Debt | ${heat_df['margin_debt'].iloc[-1]/1e6:.2f}T | 历史新高 |
| Margin YoY | {heat_df['margin_yoy'].iloc[-1]:.1f}% | {'⚠️ Q5' if heat_df['yoy_pctl'].iloc[-1] >= 80 else 'Normal'} |
| CAPE Pctl | {cape_df['cape_pctl'].iloc[-1]:.1f}% | {'⚠️ Extreme' if cape_df['cape_pctl'].iloc[-1] >= 90 else 'Normal'} |
| **Bubble Score** | {'**ACTIVE**' if bubble_active else 'Inactive'} | {'🔴' if bubble_active else '🟢'} |

---

## 系统集成建议

### 在 V4 ICR × V7 CAPE 框架中的位置

```
                        ICR (现金流裂缝)
                    ┌─────────┴─────────┐
                    │                   │
                Normal             Danger (Q5)
                    │                   │
        ┌───────────┴───────────┐       │
        │                       │       │
    CAPE Normal             CAPE High   │
        │                       │       │
        │                   ┌───┴───┐   │
        │                   │       │   │
        │               Margin    Margin
        │               Normal    High
        │                 │         │
        │             "干净"     "杠杆"
        │             高估值    高估值
        │                ↓         ↓
        │            中等风险   高风险
                     (估值型)   (泡沫型)
```

### Margin Debt 的正确用法

1. **不要**: 用 Margin/MarketCap ratio 作为领先指标
2. **不要**: 期望它独立预测 crash（Bootstrap 不显著）
3. **要做**: 在 CAPE Danger 状态下，用 Margin YoY 区分:
   - 干净高估值 (Margin 正常): 中等风险
   - 杠杆推动高估值 (Margin 高): 高风险（泡沫末期）

---

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    filepath = os.path.join(OUTPUT_DIR, 'V8_SPECULATIVE_HEAT_REPORT.md')
    with open(filepath, 'w') as f:
        f.write(report)
    print(f"\nReport saved: {filepath}")

    return report


def main():
    print("=" * 70)
    print("V8 Speculative Leverage Heat Module")
    print("=" * 70)

    # Load data
    margin_df = load_margin_debt()
    spx = load_spx()

    # Compute heat indicators
    print("\n[Computing Speculative Heat Indicators]")
    heat_df = compute_speculative_heat_indicators(margin_df, spx)

    print(f"  Observations: {len(heat_df)}")
    print(f"  Current YoY: {heat_df['margin_yoy'].iloc[-1]:.1f}%")
    print(f"  Current ΔYoY (6m): {heat_df['delta_yoy_6m'].iloc[-1]:+.1f}pp")
    print(f"  Current Z-score: {heat_df['yoy_zscore'].iloc[-1]:.2f}")

    # Run experiments
    peak_df = analyze_peak_signal(heat_df, spx)
    cape_df = analyze_cape_margin_interaction(heat_df, spx)
    excess_df = analyze_excess_leverage(heat_df, spx)

    # Generate report
    generate_heat_report(heat_df, peak_df, cape_df, excess_df)

    # Save data
    heat_df.to_csv(os.path.join(OUTPUT_DIR, 'speculative_heat_data.csv'))

    print("\n" + "=" * 70)
    print("SPECULATIVE HEAT ANALYSIS COMPLETE")
    print("=" * 70)

    # Summary
    print("\n[Summary]")
    print(f"  Margin YoY: {heat_df['margin_yoy'].iloc[-1]:.1f}% (pctl: {heat_df['yoy_pctl'].iloc[-1]:.1f}%)")
    print(f"  CAPE pctl: {cape_df['cape_pctl'].iloc[-1]:.1f}%")
    print(f"  Bubble Score: {'ACTIVE' if cape_df['bubble_score'].iloc[-1] else 'Inactive'}")


if __name__ == '__main__':
    main()
