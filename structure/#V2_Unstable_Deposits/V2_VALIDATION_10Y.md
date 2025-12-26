# V2 Unstable Deposits Ratio (10Y) 验证报告

Generated: 2025-12-25 23:43:58

## 因子信息

| 属性 | 值 |
|------|-----|
| 公式 | WDDNS / DPSACBW027SBOG * 100 |
| 名称 | Unstable Deposits Ratio |
| 含义 | 活期存款 / 总存款 (不稳定资金比例) |
| 频率 | Weekly → Monthly |
| 发布滞后 | 1 month |
| Percentile Window | 10Y |

## 当前状态

| 指标 | 值 |
|------|-----|
| 当前比率 | 35.96% |
| 10Y Percentile | 100.0% |

---

## Stage 1: In-Sample Analysis

| 检验 | 结果 | 状态 |
|------|------|------|
| IC (Spearman) | 0.4162 | PASS |
| AUC (Crash) | 0.671 (low→crash) | PASS |
| Quintile Monotonicity | N/A | FAIL |

**In-Sample 通过: 2/3**

---

## Stage 2: Out-of-Sample 5-Gate Validation

| Gate | 描述 | 结果 | 详情 |
|------|------|------|------|
| Gate 0 | Real-time Availability | PASS | 滞后 1.0 月 <= 6.0 月 |
| Gate 1 | Walk-Forward OOS Lift | FAIL | Avg=2.15x, Std=3.04, Min=0.00x |
| Gate 2 | Leave-One-Crisis-Out | FAIL | Min Lift=0.00x, Zone Drift=0% |
| Gate 3 | Lead Time | PASS | 3/4 危机有提前信号 (75%) |
| Gate 4 | Zone Stability | FAIL | Lower range=0%, Upper range=40%, Center range=20% |

**OOS Gates 通过: 2/5**
**Best Zone: [0%, 70%]**

---

## 最终结论

| 项目 | 结果 |
|------|------|
| **最终状态** | **REJECTED** |
| **建议** | 不推荐使用 |


### 危机前信号详情

| 危机 | 有信号 | Zone比例 | 平均因子 |
|------|--------|----------|----------|
| Dot-com (2000-02) | ✓ | 100% | 3.3% |
| GFC (2007-09) | ✓ | 100% | 1.0% |
| COVID (2020) | ✓ | 67% | 71.7% |
| 2022 Rate Hike | ✗ | 0% | 99.0% |

---

*Generated: 2025-12-25 23:43:58*
