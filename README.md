# Indicator Test - 结构层风险因子验证框架

基于 5-Gate OOS Validation Framework 的金融风险因子研究系统。

## 项目概述

本项目包含多个结构层风险因子的完整验证流程，使用严格的样本外验证方法评估因子有效性。

## 5-Gate OOS Validation Framework

| Gate | 名称 | 描述 | 通过标准 |
|------|------|------|----------|
| Gate 0 | Real-time | 数据发布滞后检验 | Lag < 6 months |
| Gate 1 | Walk-Forward | 滚动样本外验证 | Avg Lift > 1.0x, Std < 0.5 |
| Gate 2 | Leave-Crisis-Out | 排除危机验证 | Min Lift > 0.8x, Drift < 20% |
| Gate 3 | Lead Time | 危机预警能力 | ≥ 2/4 crises |
| Gate 4 | Zone Stability | 最优区间稳定性 | Range < 20% |

## 因子清单

### 已验证因子 (CONDITIONAL/APPROVED)

| 因子 | Gates | 状态 | 当前信号 |
|------|-------|------|----------|
| V4 ICR Δ(4Q) | 3/5 | CONDITIONAL | YELLOW |
| V7 CAPE Pctl | 3/5 | CONDITIONAL | RED |

### 辅助因子

| 因子 | 用途 | 当前状态 |
|------|------|----------|
| V8 Margin Debt | 投机热度温度计 | Bubble Score ACTIVE |

## 目录结构

```
indicator_test/
├── lib/                           # 核心验证库
│   ├── __init__.py               # 主入口
│   ├── factor_validation_gates.py # 5-Gate 验证框架
│   ├── alfred_data.py            # FRED/ALFRED 数据接口
│   ├── ic_analysis.py            # IC 分析
│   ├── hac_inference.py          # HAC 稳健推断
│   ├── regime_analysis.py        # 利率 Regime 分析
│   ├── structural_break.py       # 结构断点检验
│   └── transform_layers.py       # 因子变换层
│
├── structure/                     # 因子研究文件夹
│   ├── FACTOR_VALIDATION_GATES.md # 验证框架文档
│   ├── #V1_ST_Debt_Ratio/        # 短期债务比率
│   ├── #V2_Unstable_Deposits/    # 不稳定存款
│   ├── #V4_Interest_Coverage/    # 利息覆盖率 ⭐
│   ├── #V5_TDSP/                 # 债务偿还压力
│   ├── #V7_Shiller_PE_CAPE/      # Shiller PE (CAPE) ⭐
│   └── #V8_Margin_Debt/          # 融资融券余额
```

## 核心发现

### V4 Interest Coverage Ratio (ICR)
- **定义**: 企业利息支付能力
- **最佳 Transform**: Δ(4Q) - 4季度变化
- **Gates**: 3/5 (CONDITIONAL)
- **Danger Zone Crash Rate**: 86.2%
- **覆盖危机**: 盈利/现金流型

### V7 Shiller PE (CAPE)
- **定义**: 周期调整市盈率
- **最佳 Transform**: Percentile (10Y)
- **Gates**: 3/5 (CONDITIONAL)
- **Enhanced**: CAPE × Real Rate Conditioning
- **覆盖危机**: 估值泡沫型

### V8 Margin Debt (Speculative Heat)
- **原始 Ratio**: REJECTED (分母驱动机械反向)
- **替代指标**: YoY 增速、Peak Signal、Bubble Score
- **Bubble Score**: CAPE (>90th) × Margin (>80th)
- **当前**: Bubble Score ACTIVE

## 使用方法

```python
from lib import compute_forward_max_drawdown
from lib.factor_validation_gates import (
    find_best_zone,
    evaluate_zone,
    check_gate0_realtime,
    check_gate1_walkforward,
    check_gate2_leave_crisis_out,
    check_gate3_lead_time,
    check_gate4_zone_stability,
)

# 加载因子数据
factor = load_your_factor()
spx = load_spx_data()

# 计算前向最大回撤
fwd_mdd = compute_forward_max_drawdown(spx, horizon=252)

# 运行 5-Gate 验证
gate0 = check_gate0_realtime(release_lag_months=1)
gate1 = check_gate1_walkforward(df, 'factor', 'crash', windows)
# ...
```

## 依赖

- pandas
- numpy
- scipy
- pandas_datareader (可选，用于 FRED 数据)

## 许可

MIT License

---

*Generated: 2025-12-26*
