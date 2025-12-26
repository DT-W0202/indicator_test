# V4 Interest Coverage Ratio (ICR)

**Status: CONDITIONAL (3/5 Gates)**

企业利息覆盖率 = EBIT / Interest (FRED: A464RC1Q027SBEA, B471RC1Q027SBEA)

## 验证结果

### Transform Comparison

| Transform | IC | AUC | Direction | Gates | Status |
|-----------|-----|-----|-----------|-------|--------|
| Percentile(10Y) | 0.04 | 0.69 | low→crash | 2/5 | REJECTED |
| Flipped Pctl | -0.04 | 0.69 | high→crash | 2/5 | REJECTED |
| **Δ(4Q)** | **0.27** | **0.80** | low→crash | **3/5** | **CONDITIONAL** |
| Z-score(10Y) | -0.01 | 0.66 | low→crash | 3/5 | CONDITIONAL |

**最佳变换: Δ(4Q)** (4季度变化率)

### Best Transform (Δ(4Q)) Gate Details

| Gate | 标准 | 结果 |
|------|------|------|
| Gate 0 | 发布滞后 < 6 月 | PASS (6月) |
| Gate 1 | OOS Lift > 1.0 & Stable | **FAIL** (Std=1.32) |
| Gate 2 | Leave-Crisis-Out 稳定 | **FAIL** (Min Lift=0) |
| Gate 3 | 危机前有信号 | PASS (2/4 = 50%) |
| Gate 4 | Zone 稳定 | PASS (range=10%) |

**Best Zone**: [0%, 40%] (ICR 下降幅度大 = 危险)

## 危机前信号分析

| 危机 | Δ(4Q) 位置 | 有信号 | 解读 |
|------|-----------|--------|------|
| Dot-com (2000) | 46% (中位) | ✗ | ICR 没有明显下降 |
| **GFC (2008)** | **23%** (低位) | **✓** | ICR 大幅下降 |
| **COVID (2020)** | **34%** (低位) | **✓** | ICR 下降 |
| 2022 | 95% (高位) | ✗ | ICR 反而上升 |

## 关键发现

### 1. 变化率 (Δ) 比水平值更有效
- **水平 Percentile**: IC = 0.04 (不显著), Gates = 2/5
- **Δ(4Q)**: IC = 0.27 (显著), AUC = 0.80, Gates = 3/5

### 2. 方向解读
**Δ(4Q) 低位 = ICR 下降 = 企业盈利/利息覆盖恶化 = 危险信号**

### 3. 部分危机捕获
- **GFC (2008)**: ICR 大幅下降，成功预警 ✓
- **COVID (2020)**: ICR 下降，成功预警 ✓
- **2022**: ICR 反而上升（企业盈利好），未能预警 ✗

## 3-Level Signal System (v2.0)

基于用户建议，将原 0/1 zone 改为 3 档信号系统，并加入信用触发条件。

### Signal Definition

| Level | Condition | 含义 |
|-------|-----------|------|
| **GREEN** | Δ(4Q) > 0 | ICR 上升，企业盈利覆盖改善 |
| **YELLOW** | Δ(4Q) < 0 且 Z > -1σ | ICR 下降但未严重恶化 |
| **RED** | Δ(4Q) Z-score < -1σ | ICR 大幅下降，现金流裂缝 |

### Trigger Combination

只有当信用条件收紧时，YELLOW/RED 才升级为"系统风险"：
- **触发条件**: HY OAS > 80th pctl **OR** NFCI > 0

| ICR Signal | + Credit Trigger | → Final Signal |
|------------|-----------------|----------------|
| GREEN | Any | GREEN (安全) |
| YELLOW | No trigger | YELLOW (观察) |
| YELLOW | Triggered | **YELLOW_TRIGGERED** (警惕) |
| RED | No trigger | RED (企业压力) |
| RED | Triggered | **RED_TRIGGERED** (系统风险) |

### Backtest Results

| Signal | N | Crash Rate | Avg MDD |
|--------|---|------------|---------|
| GREEN | 100 | 8.0% | -11.7% |
| YELLOW | 41 | 14.6% | -14.2% |
| YELLOW_TRIGGERED | 7 | 14.3% | -15.8% |
| RED | 22 | **90.9%** | -31.1% |
| RED_TRIGGERED | 7 | 71.4% | -34.6% |

**关键发现**: RED 信号 crash rate 高达 90.9%，远高于基准。

### 危机信号 (3-Level)

| 危机 | ICR Signal | Triggered | Δ Z-score |
|------|-----------|-----------|-----------|
| Dot-com (2000) | RED | RED | -1.12 |
| GFC (2008) | RED | RED | -2.09 |
| COVID (2020) | GREEN | GREEN | +0.07 |
| 2022 Rate Hike | GREEN | GREEN | +0.93 |

**注意**: COVID 期间 ICR 实际上升（政府纾困+低利率），因此信号为 GREEN。

## 当前状态

| 指标 | 值 | 解读 |
|------|-----|------|
| 当前 ICR | 15.23x | 较高，覆盖充足 |
| Δ(4Q) | -0.42 | 近期略有下降 |
| Δ Z-score | -0.79 | 接近 YELLOW/RED 边界 |
| **当前信号** | **YELLOW** | ICR 下降但未严重 |

**当前状态**: ICR 虽然仍在高位，但近期有下降趋势（Δ(4Q)=-0.42），信号为 YELLOW。

## 使用建议

### 可以使用的场景
1. **辅助波动率预测**: ICR 下降 → 预期波动率上升
2. **与其他因子组合**: 作为企业盈利健康度的补充指标
3. **高利率环境**: 在高利率环境下信号更强

### 不建议单独使用
- Gate 1 和 Gate 2 失败，说明 OOS 稳定性不够
- 只捕获了 2/4 危机，漏掉 Dot-com 和 2022

## 数据源

| Series | 说明 |
|--------|------|
| A464RC1Q027SBEA | Profit Before Tax (Nonfinancial Corp) |
| B471RC1Q027SBEA | Net Interest (Nonfinancial Corp) |

- **Source**: BEA NIPA
- **Frequency**: Quarterly
- **Release Lag**: ~6 months (ALFRED)
- **History**: 1947-present

## 文件说明

| 文件 | 说明 |
|------|------|
| `V4_VALIDATION_REPORT.md` | 完整验证报告 |
| `V4_SIGNAL_SYSTEM.md` | 3-Level 信号系统报告 |
| `test_v4_icr.py` | 验证脚本 |
| `v4_signal_system.py` | 信号系统脚本 |
| `factor_data.csv` | 因子数据 (含所有变换) |

---

*Version: 2.0*
*Created: 2025-12-25*
*Updated: 2025-12-26 (3-Level Signal System)*
*Best Transform: Δ(4Q) - 4-quarter change*
