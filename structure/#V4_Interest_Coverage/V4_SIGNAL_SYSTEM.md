# V4 Interest Coverage Ratio - 3-Level Signal System

Generated: 2025-12-26 00:05:38

## Signal Definition

| Level | Condition | 含义 |
|-------|-----------|------|
| **GREEN** | Δ(4Q) > 0 | ICR 上升，企业盈利覆盖改善 |
| **YELLOW** | Δ(4Q) < 0 且 Z > -1σ | ICR 下降但未严重恶化 |
| **RED** | Δ(4Q) Z-score < -1σ | ICR 大幅下降，现金流裂缝 |

## Trigger Combination

**原则**: 只有当信用条件收紧时，YELLOW/RED 才升级为"系统风险"

| ICR Signal | + Credit Trigger | → Final Signal |
|------------|-----------------|----------------|
| GREEN | Any | GREEN (安全) |
| YELLOW | No trigger | YELLOW (观察) |
| YELLOW | Triggered | **YELLOW_TRIGGERED** (警惕) |
| RED | No trigger | RED (企业压力) |
| RED | Triggered | **RED_TRIGGERED** (系统风险) |

**触发条件**:
- HY OAS > 80th percentile (信用利差走阔)
- OR NFCI > 0 (金融条件收紧)

---

## 当前状态

| 指标 | 值 |
|------|-----|
| 当前 ICR | 15.23x |
| Δ(4Q) | -0.42 |
| Δ Z-score | -0.79 |
| **当前信号** | **YELLOW** |

---

## Backtest Results

### ICR Signal Alone

| Signal | N | Crash Rate | Avg MDD |
|--------|---|------------|---------|
| GREEN | 100 | 8.0% | -11.7% |
| YELLOW | 48 | 14.6% | -14.4% |
| RED | 29 | 86.2% | -32.0% |

### With Credit Trigger

| Signal | N | Crash Rate | Avg MDD |
|--------|---|------------|---------|
| GREEN | 100 | 8.0% | -11.7% |
| YELLOW | 41 | 14.6% | -14.2% |
| YELLOW_TRIGGERED | 7 | 14.3% | -15.8% |
| RED | 22 | 90.9% | -31.1% |
| RED_TRIGGERED | 7 | 71.4% | -34.6% |

---

## Key Findings

### False Positive Reduction

| Metric | RED Alone | RED + Trigger |
|--------|-----------|---------------|
| Crash Rate | 86.2% | 71.4% |
| Improvement | - | +-14.8pp |

### Signal Interpretation

1. **GREEN**: 安全期，正常配置
2. **YELLOW**: 早期观察，关注信用条件变化
3. **YELLOW_TRIGGERED**: 信用条件已收紧，考虑降低风险敞口
4. **RED**: 企业盈利压力，但可能是孤立事件
5. **RED_TRIGGERED**: **系统风险信号**，建议显著降低风险敞口

---

## Implementation

```python
def get_v4_signal(icr_delta_4q, icr_delta_zscore, hy_oas_pctl, nfci):
    '''
    Get V4 ICR signal with trigger combination
    '''
    # Step 1: ICR signal
    if icr_delta_4q > 0:
        icr_signal = 'GREEN'
    elif icr_delta_zscore > -1.0:
        icr_signal = 'YELLOW'
    else:
        icr_signal = 'RED'

    # Step 2: Check trigger
    trigger_active = (hy_oas_pctl > 80) or (nfci > 0)

    # Step 3: Combine
    if icr_signal == 'GREEN':
        return 'GREEN'
    elif trigger_active:
        return f'{icr_signal}_TRIGGERED'
    else:
        return icr_signal
```

---

*Generated: 2025-12-26 00:05:38*
