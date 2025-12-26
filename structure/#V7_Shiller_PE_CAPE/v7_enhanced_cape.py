#!/usr/bin/env python3
"""
V7 Enhanced CAPE - Real Rate Conditioning & Breadth Trigger
============================================================

增强方案:
(A) CAPE × Real Rate Conditioning (贴现率条件化)
    - 高 CAPE + 高实际利率 = 更危险 (双杀)
    - 高 CAPE + 负实际利率 = 相对安全 (TINA)

(B) CAPE + Market Breadth Trigger (泡沫末期确认)
    - CAPE 危险区 + 市场宽度恶化 = 确认信号

数据来源:
- FRED: DGS10 (10年国债收益率), T10YIE (10年通胀预期)
- 实际利率 = DGS10 - T10YIE (TIPS implied)
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from lib import compute_forward_max_drawdown

# Try to import pandas_datareader for FRED data
try:
    import pandas_datareader.data as web
    HAS_DATAREADER = True
except ImportError:
    HAS_DATAREADER = False
    print("Warning: pandas_datareader not installed. Using fallback data.")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CRASH_THRESHOLD = -0.20

CRISIS_PERIODS = {
    'Dot-com': ('2000-03-01', '2002-10-01'),
    'GFC': ('2007-10-01', '2009-03-01'),
    'COVID': ('2020-02-01', '2020-03-31'),
    '2022': ('2022-01-01', '2022-10-01'),
}


def load_cape_data():
    """Load CAPE data from existing CSV"""
    filepath = os.path.join(OUTPUT_DIR, 'all_methods_data.csv')
    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    cape = df['cape_raw'].dropna()

    print(f"[CAPE Data]")
    print(f"  Range: {cape.index.min().strftime('%Y-%m')} to {cape.index.max().strftime('%Y-%m')}")
    print(f"  Current: {cape.iloc[-1]:.1f}")

    return cape


def load_spx():
    """Load SPX data"""
    filepath = os.path.join(PROJECT_ROOT, 'SPX 1D Data (1).csv')
    df = pd.read_csv(filepath, parse_dates=['time'])
    return df.set_index('time')['close']


def load_real_rate_from_fred():
    """
    Load real interest rate data from FRED

    Real Rate = 10Y Nominal - 10Y Breakeven Inflation

    Series:
    - DGS10: 10-Year Treasury Constant Maturity Rate
    - T10YIE: 10-Year Breakeven Inflation Rate (from TIPS)
    """
    if not HAS_DATAREADER:
        return None

    print("\n[Loading Real Rate Data from FRED]")

    try:
        # TIPS breakeven data starts from 2003
        start = '2003-01-01'
        end = datetime.now().strftime('%Y-%m-%d')

        # Nominal 10Y yield
        dgs10 = web.DataReader('DGS10', 'fred', start, end)

        # 10Y Breakeven inflation (TIPS implied)
        t10yie = web.DataReader('T10YIE', 'fred', start, end)

        # Calculate real rate
        real_rate = dgs10['DGS10'] - t10yie['T10YIE']
        real_rate = real_rate.dropna()

        # Resample to monthly
        real_rate_monthly = real_rate.resample('ME').last()

        print(f"  Range: {real_rate_monthly.index.min().strftime('%Y-%m')} to {real_rate_monthly.index.max().strftime('%Y-%m')}")
        print(f"  Current Real Rate: {real_rate_monthly.iloc[-1]:.2f}%")

        return real_rate_monthly

    except Exception as e:
        print(f"  Error loading FRED data: {e}")
        return None


def load_ffr_from_ewi():
    """Load FFR from ewi_risk_index_data.csv as fallback"""
    filepath = os.path.join(PROJECT_ROOT, 'ewi_risk_index_data.csv')
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        if 'FFR' in df.columns:
            ffr = df['FFR'].dropna()
            print(f"\n[Using FFR from EWI data]")
            print(f"  Range: {ffr.index.min().strftime('%Y-%m')} to {ffr.index.max().strftime('%Y-%m')}")
            return ffr
    return None


def compute_cape_percentile(cape: pd.Series, window: int = 120) -> pd.Series:
    """Compute rolling percentile of CAPE using expanding window from 1990"""
    # Use expanding window from 1990 onwards for more stable percentiles
    cape_from_1990 = cape[cape.index >= '1990-01-01']

    # Expanding percentile (use all history up to each point)
    pctl = cape_from_1990.expanding(min_periods=60).apply(
        lambda x: (x.iloc[:-1] < x.iloc[-1]).sum() / len(x.iloc[:-1]) * 100 if len(x) > 1 else 50
    )

    return pctl


def compute_real_rate_regime(real_rate: pd.Series) -> pd.Series:
    """
    Classify real rate regime:
    - Negative (< 0): Low discount rate, TINA environment
    - Low (0-2%): Neutral
    - High (> 2%): High discount rate, risk for high valuations
    """
    regime = pd.Series(index=real_rate.index, dtype='object')
    regime[real_rate < 0] = 'negative'
    regime[(real_rate >= 0) & (real_rate < 2)] = 'low'
    regime[real_rate >= 2] = 'high'
    return regime


def compute_enhanced_cape_signal(cape_pctl: pd.Series, real_rate: pd.Series) -> pd.DataFrame:
    """
    Compute enhanced CAPE signal with real rate conditioning

    Signal Matrix (V2 - More Aggressive):
    |              | Negative Real Rate | Low Real Rate (0-1.5%) | High Real Rate (>1.5%) |
    |--------------|-------------------|------------------------|------------------------|
    | Low CAPE (<60th) | GREEN         | GREEN                  | GREEN                  |
    | Medium CAPE (60-80th) | YELLOW   | YELLOW                 | ORANGE                 |
    | High CAPE (80-90th) | YELLOW     | ORANGE                 | RED                    |
    | Extreme CAPE (>90th) | ORANGE    | RED                    | RED                    |
    """

    # Align data
    common_idx = cape_pctl.dropna().index.intersection(real_rate.dropna().index)

    df = pd.DataFrame({
        'cape_pctl': cape_pctl.loc[common_idx],
        'real_rate': real_rate.loc[common_idx],
    })

    # CAPE zones (adjusted thresholds)
    df['cape_zone'] = 'low'
    df.loc[df['cape_pctl'] >= 60, 'cape_zone'] = 'medium'
    df.loc[df['cape_pctl'] >= 80, 'cape_zone'] = 'high'
    df.loc[df['cape_pctl'] >= 90, 'cape_zone'] = 'extreme'

    # Real rate regime (adjusted thresholds - lower threshold for "high")
    df['rate_regime'] = 'low'
    df.loc[df['real_rate'] < 0, 'rate_regime'] = 'negative'
    df.loc[df['real_rate'] >= 1.5, 'rate_regime'] = 'high'

    # Signal mapping (more aggressive)
    signal_map = {
        ('low', 'negative'): 'GREEN',
        ('low', 'low'): 'GREEN',
        ('low', 'high'): 'GREEN',
        ('medium', 'negative'): 'YELLOW',
        ('medium', 'low'): 'YELLOW',
        ('medium', 'high'): 'ORANGE',
        ('high', 'negative'): 'YELLOW',
        ('high', 'low'): 'ORANGE',
        ('high', 'high'): 'RED',
        ('extreme', 'negative'): 'ORANGE',
        ('extreme', 'low'): 'RED',
        ('extreme', 'high'): 'RED',
    }

    df['signal'] = df.apply(
        lambda row: signal_map.get((row['cape_zone'], row['rate_regime']), 'YELLOW'),
        axis=1
    )

    # Numeric signal for analysis
    signal_numeric = {'GREEN': 0, 'YELLOW': 1, 'ORANGE': 2, 'RED': 3}
    df['signal_numeric'] = df['signal'].map(signal_numeric)

    return df


def backtest_enhanced_signal(signal_df: pd.DataFrame, spx: pd.Series):
    """Backtest enhanced CAPE signal"""

    print("\n" + "=" * 70)
    print("ENHANCED CAPE SIGNAL BACKTEST")
    print("=" * 70)

    # Compute forward MDD
    fwd_mdd = compute_forward_max_drawdown(spx, horizon=252)
    fwd_mdd_monthly = fwd_mdd.resample('ME').last()

    # Align
    common_idx = signal_df.index.intersection(fwd_mdd_monthly.dropna().index)

    df = signal_df.loc[common_idx].copy()
    df['fwd_mdd'] = fwd_mdd_monthly.loc[common_idx]
    df['is_crash'] = (df['fwd_mdd'] < CRASH_THRESHOLD).astype(int)

    print(f"\n[Sample: {len(df)} observations]")
    print(f"  Date range: {df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')}")
    print(f"  Overall crash rate: {df['is_crash'].mean()*100:.1f}%")

    # Signal distribution
    print(f"\n[Signal Distribution]")
    for signal in ['GREEN', 'YELLOW', 'ORANGE', 'RED']:
        mask = df['signal'] == signal
        n = mask.sum()
        pct = n / len(df) * 100
        if n > 0:
            cr = df.loc[mask, 'is_crash'].mean() * 100
            avg_mdd = df.loc[mask, 'fwd_mdd'].mean() * 100
            print(f"  {signal:8s}: {n:4d} ({pct:5.1f}%)  Crash Rate: {cr:5.1f}%  Avg MDD: {avg_mdd:6.1f}%")

    # Lift calculation
    base_rate = df['is_crash'].mean()

    print(f"\n[Lift Analysis]")
    for signal in ['GREEN', 'YELLOW', 'ORANGE', 'RED']:
        mask = df['signal'] == signal
        if mask.sum() > 0:
            signal_rate = df.loc[mask, 'is_crash'].mean()
            lift = signal_rate / base_rate if base_rate > 0 else 0
            print(f"  {signal:8s}: {lift:.2f}x base rate")

    # RED signal analysis
    print(f"\n[RED Signal Detail]")
    red_mask = df['signal'] == 'RED'
    if red_mask.sum() > 0:
        red_df = df[red_mask]

        # Crash rate by MDD threshold
        for thresh in [-0.10, -0.15, -0.20, -0.25]:
            cr = (red_df['fwd_mdd'] < thresh).mean() * 100
            print(f"  MDD < {thresh*100:.0f}%: {cr:.1f}%")

        # Crisis coverage
        print(f"\n  Crisis Coverage:")
        for crisis_name, (start, end) in CRISIS_PERIODS.items():
            # Check if any RED signal in pre-crisis period
            pre_start = pd.Timestamp(start) - pd.DateOffset(months=6)
            mask = (red_df.index >= pre_start) & (red_df.index <= start)
            if mask.sum() > 0:
                print(f"    {crisis_name}: ✓ ({mask.sum()} RED signals before crisis)")
            else:
                print(f"    {crisis_name}: ✗ (no RED signal)")

    return df


def compare_original_vs_enhanced(cape_pctl: pd.Series, enhanced_df: pd.DataFrame, spx: pd.Series):
    """Compare original CAPE signal vs enhanced signal"""

    print("\n" + "=" * 70)
    print("ORIGINAL VS ENHANCED COMPARISON")
    print("=" * 70)

    # Compute forward MDD
    fwd_mdd = compute_forward_max_drawdown(spx, horizon=252)
    fwd_mdd_monthly = fwd_mdd.resample('ME').last()

    # Original signal: High CAPE (top 20%) = danger
    original_df = pd.DataFrame({
        'cape_pctl': cape_pctl,
        'fwd_mdd': fwd_mdd_monthly,
    }).dropna()

    original_df['is_crash'] = (original_df['fwd_mdd'] < CRASH_THRESHOLD).astype(int)
    original_df['original_danger'] = original_df['cape_pctl'] >= 80

    # Enhanced signal: RED = danger
    common_idx = original_df.index.intersection(enhanced_df.index)

    comparison_df = pd.DataFrame({
        'cape_pctl': original_df.loc[common_idx, 'cape_pctl'],
        'fwd_mdd': original_df.loc[common_idx, 'fwd_mdd'],
        'is_crash': original_df.loc[common_idx, 'is_crash'],
        'original_danger': original_df.loc[common_idx, 'original_danger'],
        'enhanced_signal': enhanced_df.loc[common_idx, 'signal'],
    })

    comparison_df['enhanced_danger'] = comparison_df['enhanced_signal'].isin(['RED', 'ORANGE'])
    comparison_df['enhanced_red'] = comparison_df['enhanced_signal'] == 'RED'

    # Compare metrics
    print(f"\n[Sample: {len(comparison_df)} observations]")

    # Original CAPE (top 20%)
    orig_danger = comparison_df['original_danger']
    if orig_danger.sum() > 0:
        orig_cr = comparison_df.loc[orig_danger, 'is_crash'].mean() * 100
        orig_n = orig_danger.sum()
        print(f"\n  Original (CAPE >= 80th pctl):")
        print(f"    N in danger zone: {orig_n} ({orig_n/len(comparison_df)*100:.1f}%)")
        print(f"    Crash rate: {orig_cr:.1f}%")

    # Enhanced (RED only)
    enh_red = comparison_df['enhanced_red']
    if enh_red.sum() > 0:
        enh_cr = comparison_df.loc[enh_red, 'is_crash'].mean() * 100
        enh_n = enh_red.sum()
        print(f"\n  Enhanced (RED signal only):")
        print(f"    N in danger zone: {enh_n} ({enh_n/len(comparison_df)*100:.1f}%)")
        print(f"    Crash rate: {enh_cr:.1f}%")

    # Enhanced (RED + ORANGE)
    enh_danger = comparison_df['enhanced_danger']
    if enh_danger.sum() > 0:
        enh2_cr = comparison_df.loc[enh_danger, 'is_crash'].mean() * 100
        enh2_n = enh_danger.sum()
        print(f"\n  Enhanced (RED + ORANGE):")
        print(f"    N in danger zone: {enh2_n} ({enh2_n/len(comparison_df)*100:.1f}%)")
        print(f"    Crash rate: {enh2_cr:.1f}%")

    # Signal transition analysis
    print(f"\n[Signal Transition Analysis]")

    # Cases where original would say danger but enhanced says safe
    orig_danger_enh_safe = orig_danger & (~comparison_df['enhanced_danger'])
    if orig_danger_enh_safe.sum() > 0:
        cr = comparison_df.loc[orig_danger_enh_safe, 'is_crash'].mean() * 100
        print(f"  Original=DANGER, Enhanced=SAFE: {orig_danger_enh_safe.sum()} cases, Crash rate: {cr:.1f}%")
        print(f"    → These are 'TINA' situations (high CAPE but negative real rates)")

    # Cases where original would say safe but enhanced says danger
    orig_safe_enh_danger = (~orig_danger) & comparison_df['enhanced_red']
    if orig_safe_enh_danger.sum() > 0:
        cr = comparison_df.loc[orig_safe_enh_danger, 'is_crash'].mean() * 100
        print(f"  Original=SAFE, Enhanced=RED: {orig_safe_enh_danger.sum()} cases, Crash rate: {cr:.1f}%")

    return comparison_df


def generate_enhanced_report(enhanced_df: pd.DataFrame, backtest_df: pd.DataFrame,
                              comparison_df: pd.DataFrame = None):
    """Generate enhanced CAPE report"""

    report = f"""# V7 Enhanced CAPE - Real Rate Conditioning Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Enhancement Overview

### (A) CAPE × Real Rate Conditioning (贴现率条件化)

**Rationale:**
- 高 CAPE + 高实际利率 = 更危险 (估值压力 + 贴现率上升 = 双杀)
- 高 CAPE + 负实际利率 = 相对安全 (TINA: There Is No Alternative)

**Signal Matrix:**

|              | 负实际利率 (<0%) | 低实际利率 (0-2%) | 高实际利率 (>2%) |
|--------------|-----------------|------------------|-----------------|
| Low CAPE (<50th) | GREEN | GREEN | GREEN |
| Medium CAPE (50-80th) | GREEN | YELLOW | YELLOW |
| High CAPE (80-95th) | YELLOW | ORANGE | RED |
| Extreme CAPE (>95th) | ORANGE | RED | RED |

---

## Backtest Results

### Signal Distribution

| Signal | Count | % | Crash Rate (MDD<-20%) | Avg MDD |
|--------|-------|---|----------------------|---------|
"""

    for signal in ['GREEN', 'YELLOW', 'ORANGE', 'RED']:
        mask = backtest_df['signal'] == signal
        if mask.sum() > 0:
            n = mask.sum()
            pct = n / len(backtest_df) * 100
            cr = backtest_df.loc[mask, 'is_crash'].mean() * 100
            avg_mdd = backtest_df.loc[mask, 'fwd_mdd'].mean() * 100
            report += f"| {signal} | {n} | {pct:.1f}% | {cr:.1f}% | {avg_mdd:.1f}% |\n"

    # RED signal detail
    red_mask = backtest_df['signal'] == 'RED'
    if red_mask.sum() > 0:
        red_df = backtest_df[red_mask]

        report += f"""
### RED Signal Performance

| MDD Threshold | Crash Rate |
|---------------|------------|
"""
        for thresh in [-0.10, -0.15, -0.20, -0.25]:
            cr = (red_df['fwd_mdd'] < thresh).mean() * 100
            report += f"| MDD < {thresh*100:.0f}% | {cr:.1f}% |\n"

    # Comparison if available
    if comparison_df is not None:
        orig_danger = comparison_df['original_danger']
        enh_red = comparison_df['enhanced_red']

        if orig_danger.sum() > 0 and enh_red.sum() > 0:
            orig_cr = comparison_df.loc[orig_danger, 'is_crash'].mean() * 100
            enh_cr = comparison_df.loc[enh_red, 'is_crash'].mean() * 100

            report += f"""
---

## Original vs Enhanced Comparison

| Method | Danger Zone Size | Crash Rate |
|--------|-----------------|------------|
| Original (CAPE ≥80th pctl) | {orig_danger.sum()} ({orig_danger.sum()/len(comparison_df)*100:.1f}%) | {orig_cr:.1f}% |
| Enhanced (RED only) | {enh_red.sum()} ({enh_red.sum()/len(comparison_df)*100:.1f}%) | {enh_cr:.1f}% |

**Improvement:** Enhanced signal has {'higher' if enh_cr > orig_cr else 'lower'} crash rate with {'larger' if enh_red.sum() > orig_danger.sum() else 'smaller'} danger zone.
"""

    # Current status
    current_cape_pctl = enhanced_df['cape_pctl'].iloc[-1]
    current_real_rate = enhanced_df['real_rate'].iloc[-1]
    current_signal = enhanced_df['signal'].iloc[-1]

    report += f"""
---

## Current Status

| Metric | Value |
|--------|-------|
| CAPE Percentile (10Y) | {current_cape_pctl:.1f}% |
| Real Rate (10Y TIPS) | {current_real_rate:.2f}% |
| **Current Signal** | **{current_signal}** |

---

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    filepath = os.path.join(OUTPUT_DIR, 'V7_ENHANCED_REPORT.md')
    with open(filepath, 'w') as f:
        f.write(report)
    print(f"\nReport saved: {filepath}")

    return report


def main():
    print("=" * 70)
    print("V7 Enhanced CAPE - Real Rate Conditioning")
    print("=" * 70)

    # Load data
    cape = load_cape_data()
    spx = load_spx()

    # Compute CAPE percentile
    cape_pctl = compute_cape_percentile(cape)

    # Try to load real rate data
    real_rate = load_real_rate_from_fred()

    if real_rate is None:
        print("\nFRED data not available. Using FFR as proxy.")
        ffr = load_ffr_from_ewi()

        if ffr is None:
            print("No rate data available. Exiting.")
            return

        # Use FFR - 2% as crude real rate proxy
        real_rate = ffr - 2.0
        print(f"  Using FFR - 2% as real rate proxy")

    # Compute enhanced signal
    print("\n[Computing Enhanced Signal]")
    enhanced_df = compute_enhanced_cape_signal(cape_pctl, real_rate)

    print(f"  Observations: {len(enhanced_df)}")
    print(f"  Signal distribution:")
    for signal in ['GREEN', 'YELLOW', 'ORANGE', 'RED']:
        n = (enhanced_df['signal'] == signal).sum()
        pct = n / len(enhanced_df) * 100
        print(f"    {signal}: {n} ({pct:.1f}%)")

    # Backtest
    backtest_df = backtest_enhanced_signal(enhanced_df, spx)

    # Compare with original
    comparison_df = compare_original_vs_enhanced(cape_pctl, enhanced_df, spx)

    # Generate report
    generate_enhanced_report(enhanced_df, backtest_df, comparison_df)

    # Save data
    enhanced_df.to_csv(os.path.join(OUTPUT_DIR, 'enhanced_cape_data.csv'))
    backtest_df.to_csv(os.path.join(OUTPUT_DIR, 'enhanced_cape_backtest.csv'))

    print("\n" + "=" * 70)
    print("ENHANCED CAPE ANALYSIS COMPLETE")
    print("=" * 70)

    # Current status
    print(f"\n[Current Status]")
    print(f"  CAPE Percentile: {enhanced_df['cape_pctl'].iloc[-1]:.1f}%")
    print(f"  Real Rate: {enhanced_df['real_rate'].iloc[-1]:.2f}%")
    print(f"  Signal: {enhanced_df['signal'].iloc[-1]}")


if __name__ == '__main__':
    main()
