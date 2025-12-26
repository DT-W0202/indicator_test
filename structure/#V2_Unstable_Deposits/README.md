# V2 Unstable Deposits Ratio

**Status: REJECTED**

不稳定存款比率 = 活期存款 / 总存款 (FRED: WDDNS / DPSACBW027SBOG)

## 验证结果

| Gate | 标准 | 结果 |
|------|------|------|
| Gate 0 | 发布滞后 < 6 月 | PASS (1月) |
| Gate 1 | OOS Lift > 1.0 & Stable | **FAIL** (Std=3.04) |
| Gate 2 | Leave-Crisis-Out 稳定 | **FAIL** (Min Lift=0) |
| Gate 3 | 危机前有信号 | PASS (3/4) |
| Gate 4 | Zone 稳定 | **FAIL** (range=40%) |

## 变换方法对比

尝试了三种变换方法来处理结构性漂移：

| Transform | IC | AUC | Direction | Gates |
|-----------|-----|-----|-----------|-------|
| Δ(12m) | 0.35 | 0.64 | low→crash | 2/5 |
| Z-score(10Y) | 0.42 | 0.70 | low→crash | 2/5 |
| Percentile(10Y) | 0.42 | 0.67 | low→crash | 2/5 |
| **Flipped Percentile** | **-0.42** | 0.67 | high→low ret | 2/5 |

**结论**: 所有变换方法都是 2/5 gates，REJECTED。

## 危机时的 Z-score 分析

| 危机 | 原始比率 | Z-score | 解读 |
|------|----------|---------|------|
| Dot-com (2000) | ~10% | **-1.79** | 历史低位 → 崩盘 |
| GFC (2008) | ~5% | **-1.53** | 历史低位 → 崩盘 |
| COVID (2020) | ~13% | +0.71 | 中高位 |
| 2022 | ~27% | **+3.53** | 极高位 → 无预警 |

## 核心问题

### 1. 方向不一致
- 2000, 2008: **低比率**时发生崩盘 (Z < -1.5)
- 2022: **高比率**时没有预警 (Z > +3)

这说明因子与市场风险的关系**不稳定**，不是简单的单调关系。

### 2. 结构性断裂
```
1980s: Mean=16.1%, Range=[12.2%, 23.5%]
1990s: Mean=13.0%, Range=[10.1%, 16.1%]
2000s: Mean=7.0%,  Range=[4.6%, 9.8%]   ← 历史低位
2010s: Mean=11.1%, Range=[5.6%, 14.0%]
2020s: Mean=26.9%, Range=[11.7%, 36.0%] ← COVID后暴涨
```

COVID 后的结构性断裂导致历史关系失效。

### 3. 可能的解释
- **低比率 = 资金紧张**: 2000/2008 时，低活期存款可能反映企业/家庭流动性不足
- **高比率 = 流动性充裕**: 2020 后 COVID 刺激导致存款激增，提供缓冲
- **当前状态**: 100% percentile 可能是安全信号，也可能是"新常态"

## 当前状态

| 指标 | 值 |
|------|-----|
| 当前比率 | 35.96% |
| 10Y Percentile | 100% |
| Z-score | +1.82 |

**解读**: 处于历史高位，按历史模式应该是"安全"信号，但结构已变。

## 数据源

- **WDDNS**: Demand Deposits (活期存款)
- **DPSACBW027SBOG**: Total Deposits (总存款)
- **Source**: Federal Reserve H.8
- **Frequency**: Weekly
- **Release Lag**: ~1-3 weeks

## 文件说明

| 文件 | 说明 |
|------|------|
| `V2_VALIDATION_10Y.md` | 原始验证报告 |
| `V2_TRANSFORM_COMPARISON.md` | 变换方法对比 |
| `test_v2_unstable_deposits.py` | 原始验证脚本 |
| `test_v2_transforms.py` | 变换方法测试脚本 |
| `factor_data.csv` | 因子数据 |
| `factor_transforms.csv` | 变换后数据 |

---

*Version: 2.0*
*Updated: 2025-12-25*
*Transforms tested: Δ(12m), Z-score, Percentile, Flipped*
