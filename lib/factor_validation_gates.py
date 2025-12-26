#!/usr/bin/env python3
"""
Factor Validation Gates
========================

标准化的因子验证框架，所有结构因子必须通过 5 个 Gate 才能进入监控系统。

Gates:
- Gate 0: 实时性可用 (Release Lag < Horizon/2)
- Gate 1: OOS Walk-Forward Lift > 1 且稳定
- Gate 2: Leave-One-Crisis-Out 不崩
- Gate 3: 危机前 6-12 个月有提前量
- Gate 4: 阈值稳定 (Zone 不漂移)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime


def find_best_zone(df: pd.DataFrame, factor_col: str, crash_col: str,
                   step: int = 10, min_samples: int = 10) -> Tuple[float, float]:
    """
    在训练数据上找最优危险区间

    Parameters:
    -----------
    df : 训练数据
    factor_col : 因子列名
    crash_col : crash 标签列名
    step : 搜索步长 (百分点)
    min_samples : zone 内最小样本数

    Returns:
    --------
    (lower, upper) : 最优危险区间
    """
    baseline = df[crash_col].mean()
    if baseline == 0 or baseline == 1:
        return (30, 100)

    best_score = 0
    best_zone = (30, 100)

    for lower in np.arange(0, 91, step):
        for upper in np.arange(lower + step, 101, step):
            in_zone = (df[factor_col] >= lower) & (df[factor_col] <= upper)
            if in_zone.sum() < min_samples:
                continue

            zone_cr = df.loc[in_zone, crash_col].mean()
            lift = zone_cr / baseline if baseline > 0 else 0
            coverage = in_zone.mean()

            # Score = Lift × sqrt(Coverage)
            score = lift * np.sqrt(coverage) if zone_cr > baseline else 0

            if score > best_score:
                best_score = score
                best_zone = (lower, upper)

    return best_zone


def evaluate_zone(df: pd.DataFrame, zone: Tuple[float, float],
                  factor_col: str, crash_col: str) -> Dict:
    """
    在数据上评估一个危险区间的性能

    Returns:
    --------
    dict with: baseline, zone_cr, non_zone_cr, lift, recall, precision, n_zone
    """
    lower, upper = zone
    in_zone = (df[factor_col] >= lower) & (df[factor_col] <= upper)

    baseline = df[crash_col].mean()
    if baseline == 0:
        return {'lift': 0, 'recall': 0, 'precision': 0, 'n_zone': 0}

    if in_zone.sum() == 0:
        return {'lift': 0, 'recall': 0, 'precision': 0, 'n_zone': 0}

    zone_cr = df.loc[in_zone, crash_col].mean()
    non_zone_cr = df.loc[~in_zone, crash_col].mean() if (~in_zone).sum() > 0 else 0
    lift = zone_cr / baseline if baseline > 0 else 0

    # Precision and Recall
    tp = ((in_zone) & (df[crash_col] == 1)).sum()
    fp = ((in_zone) & (df[crash_col] == 0)).sum()
    fn = ((~in_zone) & (df[crash_col] == 1)).sum()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    return {
        'baseline': baseline,
        'zone_cr': zone_cr,
        'non_zone_cr': non_zone_cr,
        'lift': lift,
        'recall': recall,
        'precision': precision,
        'n_zone': in_zone.sum(),
        'n_total': len(df)
    }


# =============================================================================
# Gate 0: Real-time Availability
# =============================================================================

def check_gate0_realtime(release_lag_months: float, horizon_months: int = 12) -> Dict:
    """
    Gate 0: 检查发布滞后是否可接受

    Parameters:
    -----------
    release_lag_months : 平均发布滞后（月）
    horizon_months : 预测 horizon（月）

    Returns:
    --------
    dict with: pass, release_lag, max_acceptable, reason
    """
    max_acceptable = horizon_months / 2

    return {
        'pass': release_lag_months <= max_acceptable,
        'release_lag_months': release_lag_months,
        'max_acceptable': max_acceptable,
        'reason': f"滞后 {release_lag_months:.1f} 月 {'<=' if release_lag_months <= max_acceptable else '>'} {max_acceptable} 月"
    }


# =============================================================================
# Gate 1: Walk-Forward OOS Lift
# =============================================================================

def check_gate1_walkforward(df: pd.DataFrame, factor_col: str, crash_col: str,
                            windows: List[Tuple[str, str, str, str]],
                            min_avg_lift: float = 1.0,
                            max_std_lift: float = 0.5) -> Dict:
    """
    Gate 1: Walk-forward OOS Lift 检验

    Parameters:
    -----------
    df : 完整数据
    factor_col : 因子列名
    crash_col : crash 标签列名
    windows : list of (train_start, train_end, test_start, test_end)

    Returns:
    --------
    dict with: pass, avg_lift, std_lift, min_lift, all_lifts, reason
    """
    lifts = []
    details = []

    for train_start, train_end, test_start, test_end in windows:
        train_mask = (df.index >= train_start) & (df.index <= train_end)
        test_mask = (df.index >= test_start) & (df.index <= test_end)

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(train_df) < 30 or len(test_df) < 10:
            continue

        # Find best zone on training data
        best_zone = find_best_zone(train_df, factor_col, crash_col)

        # Evaluate on test data
        test_results = evaluate_zone(test_df, best_zone, factor_col, crash_col)

        lifts.append(test_results['lift'])
        details.append({
            'train_period': f"{train_start} to {train_end}",
            'test_period': f"{test_start} to {test_end}",
            'train_zone': best_zone,
            'test_lift': test_results['lift']
        })

    if len(lifts) == 0:
        return {
            'pass': False,
            'reason': 'Insufficient data for walk-forward validation'
        }

    avg_lift = np.mean(lifts)
    std_lift = np.std(lifts)
    min_lift = min(lifts)

    passed = avg_lift > min_avg_lift and std_lift < max_std_lift and min_lift > 0

    return {
        'pass': passed,
        'avg_lift': avg_lift,
        'std_lift': std_lift,
        'min_lift': min_lift,
        'all_lifts': lifts,
        'details': details,
        'reason': f"Avg={avg_lift:.2f}x, Std={std_lift:.2f}, Min={min_lift:.2f}x"
    }


# =============================================================================
# Gate 2: Leave-One-Crisis-Out
# =============================================================================

def check_gate2_leave_crisis_out(df: pd.DataFrame, factor_col: str, crash_col: str,
                                  crisis_periods: Dict[str, Tuple[str, str]],
                                  min_test_lift: float = 0.8,
                                  max_zone_drift: float = 20) -> Dict:
    """
    Gate 2: Leave-one-crisis-out 稳健性检验

    Parameters:
    -----------
    df : 完整数据
    factor_col : 因子列名
    crash_col : crash 标签列名
    crisis_periods : dict of {'name': (start, end)}

    Returns:
    --------
    dict with: pass, min_test_lift, zone_drift, details, reason
    """
    results = {}
    zones = []

    for crisis_name, (crisis_start, crisis_end) in crisis_periods.items():
        # Exclude this crisis from training
        exclude_mask = (df.index >= crisis_start) & (df.index <= crisis_end)
        train_df = df[~exclude_mask]
        test_df = df[exclude_mask]

        if len(train_df) < 50 or len(test_df) < 5:
            results[crisis_name] = {'error': 'Insufficient data'}
            continue

        # Find best zone on training data
        best_zone = find_best_zone(train_df, factor_col, crash_col)
        zones.append(best_zone)

        # Evaluate on excluded crisis
        test_results = evaluate_zone(test_df, best_zone, factor_col, crash_col)

        results[crisis_name] = {
            'train_zone': best_zone,
            'test_lift': test_results['lift'],
            'test_recall': test_results['recall'],
            'n_test': len(test_df)
        }

    # Calculate zone stability
    if len(zones) >= 2:
        zone_widths = [z[1] - z[0] for z in zones]
        zone_centers = [(z[0] + z[1]) / 2 for z in zones]
        zone_drift = max(zone_centers) - min(zone_centers)
    else:
        zone_drift = 0

    # Get minimum test lift
    test_lifts = [r['test_lift'] for r in results.values() if 'test_lift' in r]
    actual_min_lift = min(test_lifts) if test_lifts else 0

    passed = actual_min_lift > min_test_lift and zone_drift < max_zone_drift

    return {
        'pass': passed,
        'min_test_lift': actual_min_lift,
        'zone_drift': zone_drift,
        'all_zones': zones,
        'details': results,
        'reason': f"Min Lift={actual_min_lift:.2f}x, Zone Drift={zone_drift:.0f}%"
    }


# =============================================================================
# Gate 3: Lead Time (危机前信号)
# =============================================================================

def check_gate3_lead_time(df: pd.DataFrame, factor_col: str,
                          zone: Tuple[float, float],
                          crisis_periods: Dict[str, Tuple[str, str]],
                          lead_months: int = 6,
                          min_signal_rate: float = 0.5) -> Dict:
    """
    Gate 3: 检查危机前是否有提前信号

    Parameters:
    -----------
    df : 完整数据
    factor_col : 因子列名
    zone : (lower, upper) 危险区间
    crisis_periods : dict of {'name': (start, end)}
    lead_months : 检查危机前多少个月
    min_signal_rate : 最小要求的信号比例 (50%的危机有信号)

    Returns:
    --------
    dict with: pass, signal_rate, n_with_signal, n_total, details, reason
    """
    signals = {}
    lower, upper = zone

    for crisis_name, (crisis_start, _) in crisis_periods.items():
        # Look at lead_months BEFORE crisis start
        lead_start = pd.to_datetime(crisis_start) - pd.DateOffset(months=lead_months)
        lead_end = pd.to_datetime(crisis_start) - pd.DateOffset(months=1)

        lead_mask = (df.index >= lead_start) & (df.index <= lead_end)
        lead_df = df[lead_mask]

        if len(lead_df) == 0:
            signals[crisis_name] = {'has_signal': False, 'reason': 'No pre-crisis data'}
            continue

        # Calculate time in danger zone
        in_zone = (lead_df[factor_col] >= lower) & (lead_df[factor_col] <= upper)
        zone_ratio = in_zone.mean()

        # At least 50% of time in danger zone counts as signal
        has_signal = zone_ratio >= 0.5

        signals[crisis_name] = {
            'has_signal': has_signal,
            'zone_ratio': zone_ratio,
            'avg_factor': lead_df[factor_col].mean(),
            'n_months': len(lead_df)
        }

    n_with_signal = sum([1 for s in signals.values() if s.get('has_signal', False)])
    n_total = len(signals)
    signal_rate = n_with_signal / n_total if n_total > 0 else 0

    return {
        'pass': signal_rate >= min_signal_rate,
        'signal_rate': signal_rate,
        'n_with_signal': n_with_signal,
        'n_total': n_total,
        'details': signals,
        'reason': f"{n_with_signal}/{n_total} 危机有提前信号 ({signal_rate*100:.0f}%)"
    }


# =============================================================================
# Gate 4: Zone Stability
# =============================================================================

def check_gate4_zone_stability(df: pd.DataFrame, factor_col: str, crash_col: str,
                                n_splits: int = 5,
                                max_boundary_range: float = 20,
                                max_center_range: float = 15) -> Dict:
    """
    Gate 4: Zone 稳定性检验

    Parameters:
    -----------
    df : 完整数据
    factor_col : 因子列名
    crash_col : crash 标签列名
    n_splits : 时间切分数

    Returns:
    --------
    dict with: pass, lower_range, upper_range, center_range, all_zones, reason
    """
    # Time series split
    split_points = pd.date_range(df.index.min(), df.index.max(), periods=n_splits+1)

    zones = []
    for i in range(n_splits):
        train_df = df[df.index < split_points[i+1]]
        if len(train_df) < 50:
            continue

        best_zone = find_best_zone(train_df, factor_col, crash_col)
        zones.append({
            'cutoff': split_points[i+1],
            'zone': best_zone
        })

    if len(zones) < 2:
        return {
            'pass': False,
            'reason': 'Insufficient data for stability check'
        }

    # Calculate zone characteristics
    lowers = [z['zone'][0] for z in zones]
    uppers = [z['zone'][1] for z in zones]
    centers = [(z['zone'][0] + z['zone'][1]) / 2 for z in zones]

    lower_range = max(lowers) - min(lowers)
    upper_range = max(uppers) - min(uppers)
    center_range = max(centers) - min(centers)

    # Boundary change < max_boundary_range, center change < max_center_range
    boundary_stable = lower_range <= max_boundary_range and upper_range <= max_boundary_range
    center_stable = center_range <= max_center_range

    return {
        'pass': boundary_stable and center_stable,
        'lower_range': lower_range,
        'upper_range': upper_range,
        'center_range': center_range,
        'all_zones': zones,
        'reason': f"Lower range={lower_range:.0f}%, Upper range={upper_range:.0f}%, Center range={center_range:.0f}%"
    }


# =============================================================================
# Complete Validation Pipeline
# =============================================================================

def validate_factor(df: pd.DataFrame, factor_col: str, crash_col: str,
                    release_lag_months: float,
                    crisis_periods: Dict[str, Tuple[str, str]],
                    walkforward_windows: List[Tuple[str, str, str, str]],
                    horizon_months: int = 12) -> Dict:
    """
    完整的因子验证流程

    Parameters:
    -----------
    df : 完整数据 (需包含 factor_col 和 crash_col)
    factor_col : 因子列名
    crash_col : crash 标签列名
    release_lag_months : 发布滞后（月）
    crisis_periods : dict of {'name': (start, end)}
    walkforward_windows : list of (train_start, train_end, test_start, test_end)
    horizon_months : 预测 horizon

    Returns:
    --------
    dict with all gate results and final recommendation
    """
    results = {}

    print("=" * 70)
    print("FACTOR VALIDATION GATES")
    print("=" * 70)

    # Gate 0: Real-time availability
    print("\n[Gate 0: Real-time Availability]")
    results['gate0'] = check_gate0_realtime(release_lag_months, horizon_months)
    status = "PASS" if results['gate0']['pass'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  {results['gate0']['reason']}")

    # Gate 1: Walk-forward
    print("\n[Gate 1: Walk-Forward OOS Lift]")
    results['gate1'] = check_gate1_walkforward(df, factor_col, crash_col, walkforward_windows)
    status = "PASS" if results['gate1']['pass'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  {results['gate1']['reason']}")

    # Gate 2: Leave-one-crisis-out
    print("\n[Gate 2: Leave-One-Crisis-Out]")
    results['gate2'] = check_gate2_leave_crisis_out(df, factor_col, crash_col, crisis_periods)
    status = "PASS" if results['gate2']['pass'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  {results['gate2']['reason']}")

    # Get full-sample best zone for Gates 3-4
    best_zone = find_best_zone(df, factor_col, crash_col)
    print(f"\n[Full-sample Best Zone: [{best_zone[0]}%, {best_zone[1]}%]]")

    # Gate 3: Lead time
    print("\n[Gate 3: Lead Time]")
    results['gate3'] = check_gate3_lead_time(df, factor_col, best_zone, crisis_periods)
    status = "PASS" if results['gate3']['pass'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  {results['gate3']['reason']}")

    # Gate 4: Zone stability
    print("\n[Gate 4: Zone Stability]")
    results['gate4'] = check_gate4_zone_stability(df, factor_col, crash_col)
    status = "PASS" if results['gate4']['pass'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  {results['gate4']['reason']}")

    # Summary
    all_pass = all([r['pass'] for r in results.values()])
    n_pass = sum([1 for r in results.values() if r['pass']])

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\nGates Passed: {n_pass}/5")
    print(f"Final Status: {'APPROVED' if all_pass else 'REJECTED'}")

    if all_pass:
        recommendation = "Use as warning signal in monitoring system"
    elif n_pass >= 3:
        recommendation = "Use as context indicator only (not for alerts)"
    else:
        recommendation = "Do not use - insufficient predictive power"

    print(f"Recommendation: {recommendation}")

    return {
        'all_pass': all_pass,
        'n_pass': n_pass,
        'gates': results,
        'best_zone': best_zone,
        'recommendation': recommendation
    }


def generate_validation_report(validation_results: Dict, factor_name: str,
                               output_path: Optional[str] = None) -> str:
    """
    生成验证报告

    Parameters:
    -----------
    validation_results : validate_factor() 返回的结果
    factor_name : 因子名称
    output_path : 可选，保存路径

    Returns:
    --------
    report : 报告文本
    """
    report = f"""# {factor_name} Validation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Item | Value |
|------|-------|
| Gates Passed | {validation_results['n_pass']}/5 |
| Final Status | {'APPROVED' if validation_results['all_pass'] else 'REJECTED'} |
| Best Zone | [{validation_results['best_zone'][0]}%, {validation_results['best_zone'][1]}%] |
| Recommendation | {validation_results['recommendation']} |

## Gate Results

| Gate | Description | Result | Details |
|------|-------------|--------|---------|
"""

    gate_names = {
        'gate0': 'Real-time Availability',
        'gate1': 'Walk-Forward OOS Lift',
        'gate2': 'Leave-One-Crisis-Out',
        'gate3': 'Lead Time',
        'gate4': 'Zone Stability'
    }

    for gate_id, gate_name in gate_names.items():
        if gate_id in validation_results['gates']:
            gate = validation_results['gates'][gate_id]
            status = 'PASS' if gate['pass'] else 'FAIL'
            reason = gate.get('reason', 'N/A')
            report += f"| {gate_id.upper()} | {gate_name} | {status} | {reason} |\n"

    report += f"""
## Conclusion

{'This factor has passed all validation gates and is approved for use in the monitoring system.' if validation_results['all_pass'] else 'This factor has FAILED validation and should NOT be used as a warning signal.'}

{validation_results['recommendation']}
"""

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")

    return report


# =============================================================================
# Standard Crisis Periods and Walk-Forward Windows
# =============================================================================

STANDARD_CRISIS_PERIODS = {
    'Dot-com (2000-02)': ('2000-03', '2002-10'),
    'GFC (2007-09)': ('2007-10', '2009-03'),
    'COVID (2020)': ('2020-02', '2020-03'),
    '2022 Rate Hike': ('2022-01', '2022-10'),
}

STANDARD_WALKFORWARD_WINDOWS = [
    ('1960-01', '1999-12', '2000-01', '2007-12'),
    ('1960-01', '2007-12', '2008-01', '2014-12'),
    ('1960-01', '2014-12', '2015-01', '2019-12'),
    ('1960-01', '2019-12', '2020-01', '2024-12'),
]
