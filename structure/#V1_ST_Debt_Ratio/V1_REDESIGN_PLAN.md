# V1 ST Debt Ratio 重新设计计划

## 问题诊断

### 当前方案的核心缺陷

1. **Zone Crash Rate 过高 (89%)** → 样本稀疏/定义偏差风险
   - MDD<-10% 作为 crash 定义太宽松
   - Non-zone 也有 52% crash rate，说明"事件太常见"
   - 这不是"预警"，而是"风险常态识别"

2. **Bin 样本数不透明**
   - D6/D7 的 100%/90% 很可能是小样本极值
   - 需要展示每个 decile 的样本量

3. **">80% 接近底部"叙事危险**
   - 高位不是"安全"，而是"风险已释放/价格已调整"
   - 实时监控中可能导致过早解除警戒

4. **缺乏 Out-of-Sample 验证**
   - 三次历史捕获 (2000, 2007-08, 2021) 样本太少
   - 可能是事后选择最符合直觉的样例

5. **单因子承担过重**
   - 52% Non-zone crash rate 说明需要触发器配合

---

## 重新设计方案

### 1. Sanity Check (立即执行)

#### 1.1 双档 Crash 定义对比
```
| Crash Definition | Zone CR | Non-Zone CR | Lift |
|------------------|---------|-------------|------|
| MDD < -10%       | ?       | ?           | ?    |
| MDD < -20%       | ?       | ?           | ?    |
```

#### 1.2 每个 Decile 样本量
```
| Decile | N | Crash Rate | 是否可信 (N>20?) |
```

#### 1.3 Precision / Recall / FPR
对监控系统，比 AUC 更重要：
- Precision: 危险区间内实际 crash 比例
- Recall: 所有 crash 中被捕获的比例
- FPR: 非 crash 月份被错误标记为危险的比例

### 2. 从"因子"改为"Phase 状态机"

#### 2.1 四阶段定义
```
Phase 1: 稳健期 (Percentile < 30%)
  - 企业短期债务占比低
  - 再融资压力小
  - 预期：低风险，正常配置

Phase 2: 积累期 (30% ≤ Percentile < 60%)
  - 短期债务依赖开始上升
  - 风险正在积累但尚未显现
  - 预期：提高警惕

Phase 3: 临界期 (60% ≤ Percentile < 80%)
  - 杠杆结构恶化
  - 市场可能尚未完全定价
  - 预期：高风险，减少敞口

Phase 4: 释放/修复期 (Percentile ≥ 80%)
  - 结构压力已暴露
  - 可能正在出清或已出清
  - 预期：需配合市场压力指标判断
```

#### 2.2 每个 Phase 统计
```
| Phase | N | Avg Return | Crash Rate | Future Vol |
|       |   | 3M/6M/12M  | -10%/-20%  | 12M        |
```

### 3. 稳健性测试

#### 3.1 Leave-One-Crisis-Out
1. 排除 2007-08 → 重新找最优阈值 → 测试能否抓到 2000, 2021
2. 排除 2000 → 重新找最优阈值 → 测试能否抓到 2007-08, 2021
3. 排除 2021 → 重新找最优阈值 → 测试能否抓到 2000, 2007-08

#### 3.2 Walk-Forward 验证
```
Training Window    Test Window    Optimal Zone    Test Performance
1960-1995         1996-2005      ?               ?
1960-2005         2006-2015      ?               ?
1960-2015         2016-2024      ?               ?
```

### 4. 触发器组合设计

V1 提供结构层信号，需要配合市场压力触发器：

```python
# 单独 V1 信号 (结构风险状态)
structure_risk = get_v1_phase(current_pctl)

# 市场压力触发器 (任选其一)
pressure_signals = [
    hy_oas > hy_oas_6m_median * 1.2,  # HY 信用利差走阔
    vix > vix_6m_median * 1.3,         # 波动率上升
    move > move_6m_median * 1.2,       # 利率波动上升
    # 或金融条件指数收紧
]

# 组合规则
if structure_risk in ['临界期', '释放期'] and any(pressure_signals):
    final_signal = "HIGH RISK"
elif structure_risk == '积累期':
    final_signal = "ELEVATED"
else:
    final_signal = "NORMAL"
```

### 5. 输出格式重新设计

从：
```
RiskScore = 0/1
AUC = 0.628
```

改为：
```
{
  "phase": "临界期",
  "percentile": 77.5,
  "phase_stats": {
    "avg_12m_return": -3.2%,
    "crash_prob_10": 75%,
    "crash_prob_20": 35%,
    "future_vol": 22%
  },
  "interpretation": "结构压力已进入高位，需配合市场压力指标确认",
  "requires_trigger": true
}
```

---

## 修正后的机制解释 (可写入报告)

> V1 并非线性风险因子，而是**结构风险周期状态变量**。
>
> 风险并不集中在极端高位，而集中在 40–80% 的"结构转折区间"，对应杠杆/短债偏好上升但市场尚未完全价格化的阶段。
>
> 当指标进入 >80% 的极端区间时，往往意味着结构压力已经暴露并开始出清，风险形态从"积累"转向"释放/修复"，需与市场压力指标联立判断。
>
> **关键：V1 是状态识别器，不是独立预警器。它回答"结构处于什么阶段"，而非"明天会不会跌"。**

---

## 执行顺序

1. [ ] Sanity Check: 双档 crash、样本量、P/R/FPR
2. [ ] 重新定义四阶段并计算每阶段统计
3. [ ] Leave-One-Crisis-Out 测试
4. [ ] Walk-Forward 验证
5. [ ] 设计触发器组合规则
6. [ ] 更新 V1_SUMMARY.md

---

*Plan Created: 2025-12-25*
