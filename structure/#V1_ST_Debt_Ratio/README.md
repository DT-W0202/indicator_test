# V1 ST Debt Ratio

**Status: REJECTED**

企业短期债务占总债务比例 (FRED: BOGZ1FL104140006Q)

## 验证结果

| Gate | 标准 | 结果 |
|------|------|------|
| Gate 0 | 发布滞后 < 6 月 | PASS (5月) |
| Gate 1 | OOS Lift > 1.0 | **FAIL** (0.78x) |
| Gate 2 | Leave-Crisis-Out 稳定 | **FAIL** |
| Gate 3 | 危机前有信号 | **FAIL** (1/4) |
| Gate 4 | Zone 稳定 | **FAIL** |

**结论**: 该因子是同步/滞后指标，不适合作为预警信号。

## 文件说明

| 文件 | 说明 |
|------|------|
| `V1_SUMMARY.md` | 完整验证报告 |
| `V1_REDESIGN_PLAN.md` | 重新设计计划（历史记录） |
| `all_methods_data.csv` | 因子数据 |
| `test_validation.py` | 验证脚本 |
| `04_structural_break.png` | 结构断点分析图 |
| `05_rate_regime_ic.png` | 利率regime分析图 |
| `08_risk_target_ic.png` | 风险目标分析图 |
| `09_quintile_analysis.png` | Quintile分析图 |
| `10_danger_zone_analysis.png` | 危险区间分析图 |

## 数据源

- **Series**: BOGZ1FL104140006Q
- **Source**: FRED / Federal Reserve Z.1
- **Frequency**: Quarterly
- **Release Lag**: ~5 months
- **History**: 1945-present
