# V4 Interest Coverage Ratio (ICR) 验证报告

Generated: 2025-12-25 23:59:33

## 因子信息

| 属性 | 值 |
|------|-----|
| 公式 | ICR = (Profit + Interest) / Interest |
| Series | A464RC1Q027SBEA, B471RC1Q027SBEA |
| 频率 | Quarterly → Monthly |
| 发布滞后 | 6 months (ALFRED) |
| 特性 | 正向指标 (高ICR = 低风险) |

## 当前状态

| 指标 | 值 |
|------|-----|
| 当前 ICR | 15.23x |
| 10Y Percentile | 90.0% |
| 10Y Z-score | 1.64 |

---

## Transform Comparison

| Transform | N | IC | p-val | AUC | Direction | Gates |
|-----------|---|-----|-------|-----|-----------|-------|
| Percentile(10Y) | 305 | 0.0421 | 0.4636 | 0.687 | low→crash | 2/5 |
| Flipped Pctl | 305 | -0.0421 | 0.4636 | 0.687 | high→crash | 2/5 |
| Δ(4Q) | 305 | 0.2730 | 0.0000 | 0.799 | low→crash | 3/5 |
| Z-score(10Y) | 305 | -0.0057 | 0.9213 | 0.655 | low→crash | 3/5 |

**Best Transform: Δ(4Q) (3/5 gates)**

---

## Best Transform Gate Details: Δ(4Q)

| Gate | 描述 | 结果 | 详情 |
|------|------|------|------|
| Gate 0 | Real-time | PASS | 滞后 6.0 月 <= 6.0 月 |
| Gate 1 | Walk-Forward | FAIL | Avg=1.66x, Std=1.32, Min=0.00x |
| Gate 2 | Leave-Crisis-Out | FAIL | Min Lift=0.00x, Zone Drift=10% |
| Gate 3 | Lead Time | PASS | 2/4 危机有提前信号 (50%) |
| Gate 4 | Zone Stability | PASS | Lower range=0%, Upper range=10%, Center range=5% |

**Best Zone**: [0%, 40%]

---

## 最终结论

| 项目 | 结果 |
|------|------|
| **最终状态** | **CONDITIONAL** |
| **建议** | 可作为辅助信息，但不建议单独使用 |


### 危机前信号详情

| 危机 | 有信号 | Zone比例 | 平均因子 |
|------|--------|----------|----------|
| Dot-com (2000-02) | ✗ | 25% | 46.2% |
| GFC (2007-09) | ✓ | 100% | 22.6% |
| COVID (2020) | ✓ | 100% | 33.7% |
| 2022 Rate Hike | ✗ | 0% | 94.5% |

---

## 经济解读

ICR (Interest Coverage Ratio) 衡量企业用 EBIT 覆盖利息支出的能力：
- **高 ICR**: 企业还债能力强，财务压力小
- **低 ICR**: 企业还债压力大，可能预示风险

### 关键特性
1. **正向指标**: 高 ICR = 低风险，需要翻转用于危险区间检测
2. **波动率预测能力**: 对波动率的预测力 (IC=-0.40) 强于对收益的预测力
3. **利率依赖**: 在高利率环境下信号更强

---

*Generated: 2025-12-25 23:59:33*
