# 结构因子验证 Gate Checklist

所有结构因子在进入监控系统前，必须通过以下 5 个 Gate 检验。

---

## Gate 0: 实时性可用 (Real-time Availability)

**标准**: 发布滞后 < 1/2 预测 horizon

| 预测 Horizon | 最大可接受滞后 |
|--------------|----------------|
| 12M | 6 months |
| 6M | 3 months |
| 3M | 1.5 months |

**检验方法**:
1. 查 ALFRED vintage dates
2. 计算 `实际发布日期 - 数据期末日期`
3. 如果滞后 > horizon/2，该因子天然不适合做预警

**代码示例**:
```python
def check_gate0_realtime(factor_series, alfred_vintages, horizon_months=12):
    """
    Gate 0: 检查发布滞后是否可接受
    """
    # 计算平均发布滞后
    lags = []
    for vintage_date, data_date in alfred_vintages:
        lag_days = (vintage_date - data_date).days
        lags.append(lag_days)

    avg_lag_months = np.mean(lags) / 30
    max_acceptable = horizon_months / 2

    return {
        'pass': avg_lag_months <= max_acceptable,
        'avg_lag_months': avg_lag_months,
        'max_acceptable': max_acceptable,
        'reason': f"滞后 {avg_lag_months:.1f} 月 {'<=' if avg_lag_months <= max_acceptable else '>'} {max_acceptable} 月"
    }
```

---

## Gate 1: OOS Walk-Forward Lift > 1 且稳定

**标准**:
- 平均 OOS Lift > 1.0
- 各窗口 Lift 标准差 < 0.5
- 没有负 Lift 窗口

**检验方法**:
1. 使用滚动训练窗口（如 10 年）
2. 在每个测试窗口计算 Lift = Zone CR / Baseline CR
3. 汇总所有窗口的 Lift

**代码示例**:
```python
def check_gate1_walkforward(df, factor_col, crash_col, windows):
    """
    Gate 1: Walk-forward OOS Lift 检验

    windows: list of (train_start, train_end, test_start, test_end)
    """
    lifts = []

    for train_start, train_end, test_start, test_end in windows:
        train_df = df[(df.index >= train_start) & (df.index <= train_end)]
        test_df = df[(df.index >= test_start) & (df.index <= test_end)]

        # 在训练集找最优 zone
        best_zone = find_best_zone(train_df, factor_col, crash_col)

        # 在测试集计算 lift
        lift = evaluate_zone(test_df, best_zone, factor_col, crash_col)['lift']
        lifts.append(lift)

    avg_lift = np.mean(lifts)
    std_lift = np.std(lifts)
    min_lift = min(lifts)

    return {
        'pass': avg_lift > 1.0 and std_lift < 0.5 and min_lift > 0,
        'avg_lift': avg_lift,
        'std_lift': std_lift,
        'min_lift': min_lift,
        'all_lifts': lifts,
        'reason': f"Avg={avg_lift:.2f}x, Std={std_lift:.2f}, Min={min_lift:.2f}x"
    }
```

---

## Gate 2: Leave-One-Crisis-Out 不崩

**标准**:
- 排除任一危机后，OOS Lift 仍 > 0.8
- 排除最大危机后，Zone 定义变化 < 20 个百分点

**检验方法**:
1. 定义主要危机期（如 2000-02, 2007-09, 2020, 2022）
2. 依次排除每个危机，重新训练
3. 检验在被排除危机上的表现

**代码示例**:
```python
def check_gate2_leave_crisis_out(df, factor_col, crash_col, crisis_periods):
    """
    Gate 2: Leave-one-crisis-out 稳健性

    crisis_periods: dict of {'name': (start, end)}
    """
    results = {}

    for crisis_name, (crisis_start, crisis_end) in crisis_periods.items():
        # 排除该危机
        train_df = df[~((df.index >= crisis_start) & (df.index <= crisis_end))]
        test_df = df[(df.index >= crisis_start) & (df.index <= crisis_end)]

        # 在训练集找最优 zone
        best_zone = find_best_zone(train_df, factor_col, crash_col)

        # 在被排除的危机上测试
        test_lift = evaluate_zone(test_df, best_zone, factor_col, crash_col)['lift']

        results[crisis_name] = {
            'train_zone': best_zone,
            'test_lift': test_lift
        }

    # 检查 zone 稳定性
    zones = [r['train_zone'] for r in results.values()]
    zone_ranges = [(z[1] - z[0]) for z in zones]
    zone_drift = max(zone_ranges) - min(zone_ranges)

    min_test_lift = min([r['test_lift'] for r in results.values()])

    return {
        'pass': min_test_lift > 0.8 and zone_drift < 20,
        'min_test_lift': min_test_lift,
        'zone_drift': zone_drift,
        'details': results,
        'reason': f"Min Lift={min_test_lift:.2f}x, Zone Drift={zone_drift:.0f}%"
    }
```

---

## Gate 3: 危机前 6-12 个月有提前量

**标准**:
- 至少 50% 的危机在前 6 个月有可见信号
- 信号定义：因子进入危险区间的时间 > 3 个月

**检验方法**:
1. 对每个历史危机，检查前 6-12 个月的因子状态
2. 计算因子在危险区间的时间占比
3. 统计有提前信号的危机数量

**代码示例**:
```python
def check_gate3_lead_time(df, factor_col, zone, crisis_periods, lead_months=6):
    """
    Gate 3: 检查危机前是否有提前信号

    zone: (lower, upper) 危险区间定义
    """
    signals = {}

    for crisis_name, (crisis_start, _) in crisis_periods.items():
        # 看危机前 lead_months 个月
        lead_start = pd.to_datetime(crisis_start) - pd.DateOffset(months=lead_months)
        lead_end = pd.to_datetime(crisis_start) - pd.DateOffset(months=1)

        lead_df = df[(df.index >= lead_start) & (df.index <= lead_end)]

        if len(lead_df) == 0:
            signals[crisis_name] = {'has_signal': False, 'reason': 'No data'}
            continue

        # 计算在危险区间的比例
        in_zone = (lead_df[factor_col] >= zone[0]) & (lead_df[factor_col] <= zone[1])
        zone_ratio = in_zone.mean()

        # 至少 50% 时间在危险区间才算有信号
        has_signal = zone_ratio >= 0.5

        signals[crisis_name] = {
            'has_signal': has_signal,
            'zone_ratio': zone_ratio,
            'avg_factor': lead_df[factor_col].mean()
        }

    n_with_signal = sum([1 for s in signals.values() if s['has_signal']])
    n_total = len(signals)
    signal_rate = n_with_signal / n_total if n_total > 0 else 0

    return {
        'pass': signal_rate >= 0.5,
        'signal_rate': signal_rate,
        'n_with_signal': n_with_signal,
        'n_total': n_total,
        'details': signals,
        'reason': f"{n_with_signal}/{n_total} 危机有提前信号 ({signal_rate*100:.0f}%)"
    }
```

---

## Gate 4: 阈值稳定 (Zone 不漂移)

**标准**:
- 不同训练样本的最优 Zone 范围变化 < 20 个百分点
- Zone 中心点变化 < 15 个百分点

**检验方法**:
1. 使用不同的训练窗口（扩展窗口、滚动窗口）
2. 记录每个窗口的最优 Zone
3. 计算 Zone 的稳定性

**代码示例**:
```python
def check_gate4_zone_stability(df, factor_col, crash_col, n_splits=5):
    """
    Gate 4: Zone 稳定性检验
    """
    # 时间序列切分
    split_points = pd.date_range(df.index.min(), df.index.max(), periods=n_splits+1)

    zones = []
    for i in range(n_splits):
        train_df = df[df.index < split_points[i+1]]
        if len(train_df) < 50:
            continue

        best_zone = find_best_zone(train_df, factor_col, crash_col)
        zones.append(best_zone)

    if len(zones) < 2:
        return {'pass': False, 'reason': 'Insufficient data for stability check'}

    # 计算 zone 特征
    lowers = [z[0] for z in zones]
    uppers = [z[1] for z in zones]
    centers = [(z[0] + z[1]) / 2 for z in zones]
    widths = [z[1] - z[0] for z in zones]

    lower_range = max(lowers) - min(lowers)
    upper_range = max(uppers) - min(uppers)
    center_range = max(centers) - min(centers)
    width_range = max(widths) - min(widths)

    # 边界变化 < 20, 中心变化 < 15
    boundary_stable = lower_range < 20 and upper_range < 20
    center_stable = center_range < 15

    return {
        'pass': boundary_stable and center_stable,
        'lower_range': lower_range,
        'upper_range': upper_range,
        'center_range': center_range,
        'width_range': width_range,
        'all_zones': zones,
        'reason': f"Lower range={lower_range:.0f}%, Upper range={upper_range:.0f}%, Center range={center_range:.0f}%"
    }
```

---

## 完整验证流程

```python
def validate_factor(df, factor_col, crash_col,
                    alfred_vintages, crisis_periods,
                    walkforward_windows, horizon_months=12):
    """
    完整的因子验证流程
    """
    results = {}

    # Gate 0: 实时性
    results['gate0'] = check_gate0_realtime(df[factor_col], alfred_vintages, horizon_months)

    # Gate 1: Walk-forward
    results['gate1'] = check_gate1_walkforward(df, factor_col, crash_col, walkforward_windows)

    # Gate 2: Leave-one-crisis-out
    results['gate2'] = check_gate2_leave_crisis_out(df, factor_col, crash_col, crisis_periods)

    # 获取全样本最优 zone
    best_zone = find_best_zone(df, factor_col, crash_col)

    # Gate 3: 提前量
    results['gate3'] = check_gate3_lead_time(df, factor_col, best_zone, crisis_periods)

    # Gate 4: Zone 稳定性
    results['gate4'] = check_gate4_zone_stability(df, factor_col, crash_col)

    # 汇总
    all_pass = all([r['pass'] for r in results.values()])

    return {
        'all_pass': all_pass,
        'gates': results,
        'best_zone': best_zone,
        'recommendation': 'APPROVED for monitoring' if all_pass else 'REJECTED'
    }
```

---

## 验证报告模板

```
=== FACTOR VALIDATION REPORT ===

Factor: [Name]
Data Source: [FRED/ALFRED Series]
Horizon: 12M
Crash Definition: MDD < -20%

--- Gate Results ---

Gate 0 (Real-time): [PASS/FAIL]
  - Release Lag: X months
  - Acceptable: <= Y months

Gate 1 (Walk-Forward): [PASS/FAIL]
  - Avg OOS Lift: X.XXx
  - Lift Std: X.XX
  - Min Lift: X.XXx

Gate 2 (Leave-Crisis-Out): [PASS/FAIL]
  - Min Test Lift: X.XXx
  - Zone Drift: XX%

Gate 3 (Lead Time): [PASS/FAIL]
  - Crises with Signal: X/Y (ZZ%)

Gate 4 (Zone Stability): [PASS/FAIL]
  - Boundary Range: XX%
  - Center Range: XX%

--- FINAL RESULT ---

Status: [APPROVED/REJECTED]
Best Zone: [XX%, YY%]
Recommendation: [Use as warning signal / Use as context only / Do not use]
```

---

## V1 ST Debt Ratio 验证结果

| Gate | 结果 | 原因 |
|------|------|------|
| Gate 0 | PASS | 滞后 5 月 < 6 月 |
| **Gate 1** | **FAIL** | Avg Lift = 0.78x < 1.0 |
| **Gate 2** | **FAIL** | Zone 在不同危机间变化剧烈 |
| **Gate 3** | **FAIL** | 只有 1/4 危机有提前信号 |
| **Gate 4** | **FAIL** | Zone 从 [20,100] 到 [40,100] 漂移 |

**最终结果: REJECTED**

---

*Version: 1.0*
*Created: 2025-12-25*
