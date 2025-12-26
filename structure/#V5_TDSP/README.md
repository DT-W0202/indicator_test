# V5 TDSP - Household Debt Service Ratio

**Status: REJECTED**

家庭债务偿还占可支配收入比例 (FRED: TDSP)

## 验证结果

| Gate | 标准 | 结果 |
|------|------|------|
| Gate 0 | 发布滞后 < 6 月 | PASS (3月) |
| Gate 1 | OOS Lift > 1.0 & Stable | **FAIL** (Std=2.66) |
| Gate 2 | Leave-Crisis-Out 稳定 | **FAIL** (Min Lift=0) |
| Gate 3 | 危机前有信号 | **FAIL** (1/4) |
| Gate 4 | Zone 稳定 | PASS |

**结论**: 尽管 In-Sample IC (-0.34) 和 AUC (0.72) 表现不错，但 OOS 验证失败。

## 问题诊断

### 1. 危机信号不一致
- **GFC (2007-09)**: 因子在危机前处于 100% 高位 ✓
- **Dot-com (2000)**: 因子在 87% 高位，但 zone (90-100%) 未覆盖
- **COVID (2020)**: 因子在 22.5% 低位 ✗
- **2022**: 因子在 5.8% 低位 ✗

### 2. 根本原因
TDSP 在 2008 年达到历史高峰 (13.2%) 后，由于：
1. 低利率环境降低债务成本
2. 家庭去杠杆化
3. 消费者债务结构变化

因子从 2010 年后持续下降，目前处于历史低位 (11.25%)。这意味着：
- 2020/2022 危机时家庭债务负担处于低位
- 因子无法预警这些危机

### 3. 适用范围有限
TDSP 可能只对**债务驱动型危机** (如 GFC) 有效，对于外生冲击 (COVID) 或资产价格调整 (2022) 无预警能力。

## 数据源

- **Series**: TDSP
- **Source**: Federal Reserve Board
- **Frequency**: Quarterly
- **Release Lag**: ~3 months
- **History**: 1980-present

## 文件说明

| 文件 | 说明 |
|------|------|
| `V5_VALIDATION_10Y.md` | 完整验证报告 |
| `test_v5_tdsp.py` | 验证脚本 |
| `factor_data.csv` | 因子数据 |

---

*Version: 1.0*
*Created: 2025-12-25*
