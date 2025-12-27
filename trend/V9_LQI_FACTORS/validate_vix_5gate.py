#!/usr/bin/env python3
"""
VIX 5-Gate Validation Script

使用 structure 层的验证框架对 VIX 进行完整验证
"""

import sys
import os

# 直接导入 factor_validation_gates 避免加载整个 lib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime

# 尝试从 FRED 获取数据
try:
    import pandas_datareader.data as web
    HAS_DATAREADER = True
except ImportError:
    HAS_DATAREADER = False
    print("Warning: pandas_datareader not available, using synthetic data")

# 直接执行模块文件避免 __init__.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "factor_validation_gates",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "lib", "factor_validation_gates.py")
)
fvg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fvg)

validate_factor = fvg.validate_factor
find_best_zone = fvg.find_best_zone
evaluate_zone = fvg.evaluate_zone
STANDARD_CRISIS_PERIODS = fvg.STANDARD_CRISIS_PERIODS
STANDARD_WALKFORWARD_WINDOWS = fvg.STANDARD_WALKFORWARD_WINDOWS


def generate_synthetic_data(start='1990-01-01', end='2024-12-31'):
    """
    生成合成 VIX 和 SPX 数据用于演示

    基于真实市场特征：
    - VIX 均值约 20，危机时飙升至 40-80
    - SPX 长期上涨，危机期间下跌 20-50%
    """
    np.random.seed(42)

    dates = pd.date_range(start, end, freq='ME')
    n = len(dates)

    # 定义危机期 (月份索引)
    crisis_periods = {
        'Dot-com': (120, 150),      # 2000-03 to 2002-06
        'GFC': (213, 230),          # 2007-10 to 2009-03
        'COVID': (362, 364),        # 2020-02 to 2020-04
        '2022': (384, 394),         # 2022-01 to 2022-10
    }

    # 生成 VIX
    vix = np.ones(n) * 18  # 基线
    vix += np.random.randn(n) * 3  # 噪声

    # 危机期间 VIX 飙升
    for name, (start_idx, end_idx) in crisis_periods.items():
        if end_idx < n:
            # 危机期间升高
            vix[start_idx:end_idx] += np.random.uniform(15, 40, end_idx - start_idx)
            # 危机前 3-6 个月开始升高
            pre_start = max(0, start_idx - 6)
            vix[pre_start:start_idx] += np.linspace(0, 10, start_idx - pre_start)

    vix = np.clip(vix, 10, 80)

    # 生成 SPX
    # 年化收益约 8%，波动率约 15%
    monthly_return = 0.08 / 12
    monthly_vol = 0.15 / np.sqrt(12)

    spx = np.zeros(n)
    spx[0] = 350  # 1990 年初

    for i in range(1, n):
        # 基本收益
        ret = monthly_return + np.random.randn() * monthly_vol

        # 危机期间大跌
        in_crisis = False
        for name, (start_idx, end_idx) in crisis_periods.items():
            if start_idx <= i < end_idx:
                ret = -0.05 + np.random.randn() * 0.08  # 危机期间平均 -5%
                in_crisis = True
                break

        spx[i] = spx[i-1] * (1 + ret)

    # 创建 DataFrame
    vix_df = pd.DataFrame({'VIX': vix}, index=dates)
    spx_df = pd.DataFrame({'SPX': spx}, index=dates)

    return vix_df, spx_df


def fetch_vix_data(start='1990-01-01', end='2024-12-31'):
    """从 FRED 获取 VIX 数据，失败则返回合成数据"""
    if not HAS_DATAREADER:
        return None

    try:
        vix = web.DataReader('VIXCLS', 'fred', start, end)
        vix = vix.resample('ME').last()  # 月末数据
        vix.columns = ['VIX']
        return vix
    except Exception as e:
        print(f"Warning: Could not fetch VIX from FRED: {e}")
        return None


def fetch_spx_data(start='1990-01-01', end='2024-12-31'):
    """从 FRED 获取 SPX 数据，失败则返回合成数据"""
    if not HAS_DATAREADER:
        return None

    try:
        spx = web.DataReader('SP500', 'fred', start, end)
        spx = spx.resample('ME').last()
        spx.columns = ['SPX']
        return spx
    except Exception as e:
        print(f"Warning: Could not fetch SPX from FRED: {e}")
        return None


def compute_forward_max_drawdown(prices: pd.Series, horizon: int = 12) -> pd.Series:
    """
    计算前向最大回撤 (Forward Maximum Drawdown)

    Parameters:
    -----------
    prices : 价格序列
    horizon : 前向月份数

    Returns:
    --------
    fwd_mdd : 每个时点未来 horizon 个月内的最大回撤
    """
    fwd_mdd = pd.Series(index=prices.index, dtype=float)

    for i in range(len(prices) - 1):
        end_idx = min(i + horizon, len(prices) - 1)
        future_prices = prices.iloc[i:end_idx + 1]

        if len(future_prices) < 2:
            fwd_mdd.iloc[i] = 0
            continue

        # 从当前点开始计算最大回撤
        running_max = future_prices.iloc[0]
        max_dd = 0

        for price in future_prices:
            if price > running_max:
                running_max = price
            dd = (price - running_max) / running_max
            if dd < max_dd:
                max_dd = dd

        fwd_mdd.iloc[i] = max_dd

    return fwd_mdd


def compute_rolling_percentile(series: pd.Series, window: int = 60) -> pd.Series:
    """
    计算滚动百分位数 (0-100)

    Parameters:
    -----------
    series : 数据序列
    window : 滚动窗口 (月)

    Returns:
    --------
    percentile : 滚动百分位数
    """
    def pctl(x):
        return (x.rank().iloc[-1] / len(x)) * 100

    return series.rolling(window, min_periods=window//2).apply(pctl, raw=False)


def prepare_validation_data(vix: pd.DataFrame, spx: pd.DataFrame,
                            crash_threshold: float = -0.20,
                            pctl_window: int = 60) -> pd.DataFrame:
    """
    准备验证数据

    Parameters:
    -----------
    vix : VIX 数据
    spx : SPX 数据
    crash_threshold : 崩盘阈值 (MDD < threshold)
    pctl_window : 百分位计算窗口

    Returns:
    --------
    df : 包含因子和崩盘标签的数据
    """
    # 合并数据
    df = vix.join(spx, how='inner')

    # 计算前向最大回撤
    df['FWD_MDD'] = compute_forward_max_drawdown(df['SPX'], horizon=12)

    # 崩盘标签
    df['crash'] = (df['FWD_MDD'] < crash_threshold).astype(int)

    # VIX 百分位
    df['VIX_PCTL'] = compute_rolling_percentile(df['VIX'], window=pctl_window)

    # 清理
    df = df.dropna()

    return df


def run_vix_validation():
    """运行 VIX 5-Gate 验证"""

    print("=" * 70)
    print("VIX 5-Gate Validation")
    print("=" * 70)
    print(f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 获取数据
    print("\n[1] Fetching data...")
    vix = fetch_vix_data()
    spx = fetch_spx_data()

    if vix is None or spx is None:
        print("    FRED unavailable, using synthetic data for demonstration")
        vix, spx = generate_synthetic_data()
        print("    (Note: Results are based on synthetic data mimicking real VIX behavior)")

    print(f"    VIX: {vix.index.min().date()} to {vix.index.max().date()} ({len(vix)} months)")
    print(f"    SPX: {spx.index.min().date()} to {spx.index.max().date()} ({len(spx)} months)")

    # 准备数据
    print("\n[2] Preparing validation data...")
    df = prepare_validation_data(vix, spx)
    print(f"    Combined: {df.index.min().date()} to {df.index.max().date()} ({len(df)} months)")
    print(f"    Crash rate: {df['crash'].mean()*100:.1f}%")
    print(f"    VIX range: {df['VIX'].min():.1f} to {df['VIX'].max():.1f}")
    print(f"    VIX current: {df['VIX'].iloc[-1]:.1f} ({df['VIX_PCTL'].iloc[-1]:.0f}th percentile)")

    # 调整危机期和滚动窗口（适应 VIX 数据起始时间）
    crisis_periods = {
        'Dot-com (2000-02)': ('2000-03', '2002-10'),
        'GFC (2007-09)': ('2007-10', '2009-03'),
        'COVID (2020)': ('2020-02', '2020-03'),
        '2022 Rate Hike': ('2022-01', '2022-10'),
    }

    # VIX 从 1990 年开始，调整窗口
    walkforward_windows = [
        ('1990-01', '1999-12', '2000-01', '2007-12'),
        ('1990-01', '2007-12', '2008-01', '2014-12'),
        ('1990-01', '2014-12', '2015-01', '2019-12'),
        ('1990-01', '2019-12', '2020-01', '2024-12'),
    ]

    # 运行验证
    print("\n[3] Running 5-Gate Validation...")
    print("-" * 70)

    results = validate_factor(
        df=df,
        factor_col='VIX_PCTL',  # 使用百分位
        crash_col='crash',
        release_lag_months=0,  # VIX 是实时的
        crisis_periods=crisis_periods,
        walkforward_windows=walkforward_windows,
        horizon_months=12
    )

    # 额外分析
    print("\n" + "=" * 70)
    print("ADDITIONAL ANALYSIS")
    print("=" * 70)

    # Quintile 分析
    print("\n[Quintile Analysis]")
    df['quintile'] = pd.qcut(df['VIX_PCTL'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

    quintile_stats = df.groupby('quintile').agg({
        'crash': ['count', 'mean'],
        'VIX': 'mean',
        'FWD_MDD': 'mean'
    }).round(3)

    print("\n| Quintile | N | Crash Rate | Avg VIX | Avg MDD |")
    print("|----------|---|------------|---------|---------|")
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        n = int(quintile_stats.loc[q, ('crash', 'count')])
        cr = quintile_stats.loc[q, ('crash', 'mean')] * 100
        vix_avg = quintile_stats.loc[q, ('VIX', 'mean')]
        mdd_avg = quintile_stats.loc[q, ('FWD_MDD', 'mean')] * 100
        print(f"| {q} | {n} | {cr:.1f}% | {vix_avg:.1f} | {mdd_avg:.1f}% |")

    # Zone 性能
    print("\n[Best Zone Performance]")
    best_zone = results['best_zone']
    zone_perf = evaluate_zone(df, best_zone, 'VIX_PCTL', 'crash')

    print(f"  Zone: [{best_zone[0]:.0f}%, {best_zone[1]:.0f}%]")
    print(f"  Baseline CR: {zone_perf['baseline']*100:.1f}%")
    print(f"  Zone CR: {zone_perf['zone_cr']*100:.1f}%")
    print(f"  Non-Zone CR: {zone_perf['non_zone_cr']*100:.1f}%")
    print(f"  Lift: {zone_perf['lift']:.2f}x")
    print(f"  Recall: {zone_perf['recall']*100:.1f}%")
    print(f"  Precision: {zone_perf['precision']*100:.1f}%")

    # 当前状态
    print("\n[Current Status]")
    current_vix = df['VIX'].iloc[-1]
    current_pctl = df['VIX_PCTL'].iloc[-1]
    in_danger = best_zone[0] <= current_pctl <= best_zone[1]

    print(f"  Current VIX: {current_vix:.1f}")
    print(f"  Current Percentile: {current_pctl:.0f}%")
    print(f"  In Danger Zone: {'YES' if in_danger else 'NO'}")

    if current_pctl < 20:
        status = "LOW RISK (VIX very low)"
    elif current_pctl < 50:
        status = "NORMAL"
    elif current_pctl < 80:
        status = "ELEVATED"
    else:
        status = "HIGH RISK (VIX elevated)"

    print(f"  Risk Status: {status}")

    return results


if __name__ == '__main__':
    results = run_vix_validation()
