"""
ALFRED Point-in-Time Backtest Library
=====================================

结构因子验证框架 - 完整流程

==============================================================================
验证流程概览
==============================================================================

因子验证分为两个阶段：

┌─────────────────────────────────────────────────────────────────────────────┐
│ 第一阶段: In-Sample 分析 (探索性)                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ 目的: 理解因子特性，初步筛选                                                │
│                                                                             │
│ 1. IC (Information Coefficient)                                             │
│    - Spearman rank correlation: 因子 vs 未来收益                            │
│    - HAC t-stat (Newey-West): 考虑自相关                                    │
│    - 期望: |IC| > 0.03, t-stat > 2.0                                        │
│                                                                             │
│ 2. AUC (Area Under ROC Curve)                                               │
│    - 二分类: 因子 vs Crash (MDD < -20%)                                     │
│    - 期望: AUC > 0.55 (偏离 0.5 越多越好)                                   │
│                                                                             │
│ 3. Quintile Analysis                                                        │
│    - 分组收益: Q1 vs Q5 差异                                                │
│    - 单调性: 收益/风险随因子分位单调变化                                    │
│                                                                             │
│ 4. Bootstrap Significance                                                   │
│    - Block Bootstrap (考虑时序相关)                                         │
│    - 95% CI 不含 0                                                          │
│                                                                             │
│ ⚠️  通过 In-Sample 只说明"值得继续研究"，不代表因子有效                    │
└─────────────────────────────────────────────────────────────────────────────┘

                                    │
                                    ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│ 第二阶段: Out-of-Sample 验证 (决定性)                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ 目的: 确认因子是否真正有预测能力，避免 overfitting                          │
│                                                                             │
│ 必须通过全部 5 个 Gate 才能进入监控系统:                                    │
│                                                                             │
│ Gate 0: 实时性 (Real-time Availability)                                     │
│    - 标准: 发布滞后 < 预测 horizon / 2                                      │
│    - 12M horizon → 滞后 < 6 月                                              │
│    - 检验: ALFRED vintage dates                                             │
│                                                                             │
│ Gate 1: Walk-Forward OOS Lift                                               │
│    - 标准: 平均 OOS Lift > 1.0, Std < 0.5, 无负 Lift                        │
│    - 方法: 滚动训练→测试，在测试期计算 Zone Crash Rate / Baseline           │
│                                                                             │
│ Gate 2: Leave-One-Crisis-Out                                                │
│    - 标准: 排除任一危机后，OOS Lift > 0.8, Zone 漂移 < 20%                  │
│    - 方法: 依次排除 2000, 2008, 2020, 2022 危机，在被排除危机上测试         │
│                                                                             │
│ Gate 3: 提前量 (Lead Time)                                                  │
│    - 标准: ≥50% 的危机在前 6 个月有信号                                     │
│    - 方法: 检查每个危机前 6 个月因子是否在危险区间                          │
│                                                                             │
│ Gate 4: Zone 稳定性                                                         │
│    - 标准: 不同训练窗口的最优 Zone 边界变化 < 20%, 中心变化 < 15%           │
│    - 方法: 时序切分，比较各期最优 Zone                                      │
│                                                                             │
│ ✅ 全部通过 → APPROVED: 可作为预警信号                                      │
│ ⚠️  3-4 个通过 → CONDITIONAL: 仅作为辅助信息                                │
│ ❌ <3 个通过 → REJECTED: 不推荐使用                                         │
└─────────────────────────────────────────────────────────────────────────────┘

==============================================================================
使用方法
==============================================================================

1. In-Sample 分析:

   from lib import ICAnalyzer, run_quintile_analysis, block_bootstrap_regression

   analyzer = ICAnalyzer(df, factor_col='factor', ret_col='fwd_return')
   ic_result = analyzer.compute_ic_with_hac()
   quintile_result = run_quintile_analysis(df, 'factor', 'fwd_return')

2. Out-of-Sample 验证:

   from lib import (
       validate_factor,
       STANDARD_CRISIS_PERIODS,
       STANDARD_WALKFORWARD_WINDOWS
   )

   results = validate_factor(
       df,
       factor_col='percentile',
       crash_col='is_crash',
       release_lag_months=5,
       crisis_periods=STANDARD_CRISIS_PERIODS,
       walkforward_windows=STANDARD_WALKFORWARD_WINDOWS
   )

   if results['all_pass']:
       print("因子验证通过，可用于监控系统")

==============================================================================
Modules
==============================================================================

- alfred_data: ALFRED API data loading and PIT reconstruction
- transform_layers: Factor transformation pipeline
- ic_analysis: Information Coefficient analysis
- structural_break: Structural break detection and analysis
- regime_analysis: Regime-conditional factor analysis with HAC inference
- hac_inference: HAC standard errors and block bootstrap
- factor_validation_gates: 5-Gate OOS validation framework (NEW)
"""

from .alfred_data import (
    ALFREDDataLoader,
    HybridPITLoader,
    build_pit_factor_series,
    build_simulated_pit_series,
    RELEASE_LAG_DAYS,
)
from .transform_layers import TransformPipeline, compute_factor_change
from .ic_analysis import ICAnalyzer

from .structural_break import (
    chow_test,
    detect_changepoints_cusum,
    rolling_ic_with_ci,
    rolling_beta_ols,
    compute_subsample_ic,
    compute_subsample_beta,
    analyze_structural_break,
)

from .regime_analysis import (
    compute_current_drawdown,
    compute_forward_max_drawdown,
    compute_forward_realized_vol,
    compute_forward_return,
    classify_drawdown_regime,
    classify_volatility_regime,
    compute_regime_ic,
    compute_risk_target_ic,
    compute_drawdown_event_auc,
    run_conditional_regression,
    run_interaction_regression,
    run_quintile_analysis,
)

from .hac_inference import (
    newey_west_se,
    ols_with_hac,
    block_bootstrap_regression,
    rolling_beta_with_hac,
    quantile_regression,
    compute_tail_quantile_ic,
)

from .factor_validation_gates import (
    # Core functions
    find_best_zone,
    evaluate_zone,
    # Gate checks
    check_gate0_realtime,
    check_gate1_walkforward,
    check_gate2_leave_crisis_out,
    check_gate3_lead_time,
    check_gate4_zone_stability,
    # Complete validation
    validate_factor,
    generate_validation_report,
    # Standard configurations
    STANDARD_CRISIS_PERIODS,
    STANDARD_WALKFORWARD_WINDOWS,
)

__all__ = [
    # alfred_data
    'ALFREDDataLoader',
    'HybridPITLoader',
    'build_pit_factor_series',
    'build_simulated_pit_series',
    'RELEASE_LAG_DAYS',
    # transform_layers
    'TransformPipeline',
    'compute_factor_change',
    # ic_analysis
    'ICAnalyzer',
    # structural_break
    'chow_test',
    'detect_changepoints_cusum',
    'rolling_ic_with_ci',
    'rolling_beta_ols',
    'compute_subsample_ic',
    'compute_subsample_beta',
    'analyze_structural_break',
    # regime_analysis
    'compute_current_drawdown',
    'compute_forward_max_drawdown',
    'compute_forward_realized_vol',
    'compute_forward_return',
    'classify_drawdown_regime',
    'classify_volatility_regime',
    'compute_regime_ic',
    'compute_risk_target_ic',
    'compute_drawdown_event_auc',
    'run_conditional_regression',
    'run_interaction_regression',
    'run_quintile_analysis',
    # hac_inference
    'newey_west_se',
    'ols_with_hac',
    'block_bootstrap_regression',
    'rolling_beta_with_hac',
    'quantile_regression',
    'compute_tail_quantile_ic',
    # factor_validation_gates
    'find_best_zone',
    'evaluate_zone',
    'check_gate0_realtime',
    'check_gate1_walkforward',
    'check_gate2_leave_crisis_out',
    'check_gate3_lead_time',
    'check_gate4_zone_stability',
    'validate_factor',
    'generate_validation_report',
    'STANDARD_CRISIS_PERIODS',
    'STANDARD_WALKFORWARD_WINDOWS',
]


# =============================================================================
# Quick Reference: Validation Decision Tree
# =============================================================================
#
# 新因子验证决策树:
#
#     ┌─────────────────────┐
#     │ 1. In-Sample 分析   │
#     │    IC, AUC, Quintile │
#     └─────────┬───────────┘
#               │
#               ▼
#     ┌─────────────────────┐
#     │ IC 显著且方向正确?  │───No──→ 检查非线性关系
#     └─────────┬───────────┘         (Danger Zone 方法)
#               │ Yes                        │
#               ▼                            ▼
#     ┌─────────────────────┐      ┌─────────────────────┐
#     │ 2. OOS 5-Gate 验证  │      │ 用 find_best_zone() │
#     │    validate_factor() │      │ 找最优危险区间      │
#     └─────────┬───────────┘      └─────────┬───────────┘
#               │                            │
#               ▼                            ▼
#     ┌─────────────────────┐      ┌─────────────────────┐
#     │ 全部 5 Gate 通过?   │      │ 2. OOS 5-Gate 验证  │
#     └─────────┬───────────┘      └─────────┬───────────┘
#               │                            │
#        ┌──────┴──────┐              ┌──────┴──────┐
#        ▼             ▼              ▼             ▼
#   ┌─────────┐  ┌─────────┐    ┌─────────┐  ┌─────────┐
#   │ APPROVED │  │ REJECTED │    │ APPROVED │  │ REJECTED │
#   │ 可用于  │  │ 不推荐   │    │ 可用于  │  │ 不推荐   │
#   │ 监控系统 │  │ 使用     │    │ 监控系统 │  │ 使用     │
#   └─────────┘  └─────────┘    └─────────┘  └─────────┘
#
