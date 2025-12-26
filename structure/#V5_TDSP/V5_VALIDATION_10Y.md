# V5 TDSP (10Y) 因子验证报告

Generated: 2025-12-25 23:31:52

## 因子信息

| 属性 | 值 |
|------|-----|
| Series | TDSP |
| 名称 | Household Debt Service Ratio |
| 频率 | Quarterly |
| 发布滞后 | 3 months |
| Percentile Window | 10Y |

## 当前状态

| 指标 | 值 |
|------|-----|
| 当前 TDSP | 11.25% |
| 10Y Percentile | 42.5% |

---

## Stage 1: In-Sample Analysis

| 检验 | 结果 | 状态 |
|------|------|------|
| IC (Spearman) | -0.3422 | PASS |
| AUC (Crash) | 0.722 | PASS |
| Quintile Monotonicity | N/A | FAIL |
| Bootstrap 95% CI | 0.0000 to 0.0000 | FAIL |

**In-Sample 通过: 2/4**

---

## Stage 2: Out-of-Sample 5-Gate Validation

| Gate | 描述 | 结果 | 详情 |
|------|------|------|------|
| Gate 0 | Real-time Availability | PASS | 滞后 3.0 月 <= 6.0 月 |
| Gate 1 | Walk-Forward OOS Lift | FAIL | Avg=1.88x, Std=2.66, Min=0.00x |
| Gate 2 | Leave-One-Crisis-Out | FAIL | Min Lift=0.00x, Zone Drift=0% |
| Gate 3 | Lead Time | FAIL | 1/4 危机有提前信号 (25%) |
| Gate 4 | Zone Stability | PASS | Lower range=0%, Upper range=0%, Center range=0% |

**OOS Gates 通过: 2/5**
**Best Zone: [90%, 100%]**

---

## 最终结论

| 项目 | 结果 |
|------|------|
| **最终状态** | **REJECTED** |
| **建议** | 不推荐使用 |


### 危机前信号详情

| 危机 | 有信号 | Zone比例 | 平均因子 |
|------|--------|----------|----------|
| Dot-com (2000-02) | ✗ | 25% | 86.9% |
| GFC (2007-09) | ✓ | 100% | 100.0% |
| COVID (2020) | ✗ | 0% | 22.5% |
| 2022 Rate Hike | ✗ | 0% | 5.8% |

---

*Generated: 2025-12-25 23:31:52*
