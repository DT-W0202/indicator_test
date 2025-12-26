# V8 Margin Debt / Wilshire 5000 - Complete Validation Report

Generated: 2025-12-26 17:08:22

## Executive Summary

使用 **Margin Debt / Wilshire 5000 Market Cap** 作为新的标准化方式，验证杠杆率因子。

这比 YoY 增速更直观，反映的是相对于市场规模的杠杆水平。

---

## Data Overview

| Metric | Value |
|--------|-------|
| Current Margin Debt | $1.21 Trillion |
| Current Market Cap | $68.34 Trillion |
| **Current Ratio** | **1.777%** |
| Historical Mean | 1.988% |
| Historical Max | 2.911% (2008-09) |

---

## Transform Comparison

| Transform | N | IC | p-val | Best Zone | Lift | Gates |
|-----------|---|-----|-------|-----------|------|-------|
| Ratio (Raw) | 178 | 0.039 | 0.6066 | (30, 100) | 0.00x | 2/5 (REJECTED) |
| Percentile | 161 | 0.080 | 0.3126 | (20, 50) | 2.88x | 2/5 (REJECTED) |
| Z-score(10Y) | 161 | 0.073 | 0.3558 | (30, 100) | 0.00x | 2/5 (REJECTED) |
| Δ(12M) | 178 | 0.112 | 0.1375 | (30, 100) | 0.00x | 2/5 (REJECTED) |
| Δ(12M)_Pctl | 154 | 0.034 | 0.6761 | (80, 100) | 2.85x | 1/5 (REJECTED) |

**Best Transform: Ratio (Raw) (2/5 gates)**

---

## Best Transform Gate Details: Ratio (Raw)

| Gate | Description | Result | Details |
|------|-------------|--------|---------|
| Gate 0 | Real-time | PASS | Lag = 1m |
| Gate 1 | Walk-Forward | FAIL | Avg=0.00x, Std=0.00 |
| Gate 2 | Leave-Crisis-Out | FAIL | Min=0.00x, Drift=0.0% |
| Gate 3 | Lead Time | FAIL | 0/4 crises |
| Gate 4 | Zone Stability | PASS | Range=0-0% |

---

## Production Validation Results

### Test 1: Strict Event Definition

| Threshold | Danger Zone Crash Rate | Avg MDD |
|-----------|------------------------|---------|
| MDD<-10% | 61.1% | -20.0% |
| MDD<-20% | 27.8% | -20.0% |
| MDD<-25% | 25.0% | -20.0% |

### Test 2: Multi-Horizon

| Horizon | Danger Zone Crash Rate |
|---------|------------------------|
| 3m | 8.1% |
| 6m | 13.5% |
| 12m | 27.8% |

### Test 3: LOCO Stability

**LOCO Stability: 25.4pp range**

---

## Current Status

| Metric | Value |
|--------|-------|
| Current Ratio | 1.777% |
| Percentile | 34.7% |
| Z-score | -0.41 |
| Δ(12M) | 0.3137% |

---

## Conclusion

| Status | Details |
|--------|---------|
| **Validation** | 2/5 Gates (REJECTED) |
| **Best Transform** | Ratio (Raw) |
| **Direction** | high_is_danger |
| **Optimal Zone** | (30, 100) |

---

*Generated: 2025-12-26 17:08:22*
