# V7 Shiller PE (CAPE) - Complete Validation Report

Generated: 2025-12-26 16:41:08

## Executive Summary

使用 5-Gate OOS Validation Framework 重新验证 Shiller PE (CAPE)。

---

## Transform Comparison

| Transform | N | IC | p-val | Best Zone | Lift | Gates |
|-----------|---|-----|-------|-----------|------|-------|
| Percentile(10Y) | 305 | -0.143 | 0.0124 | (40, 100) | 1.21x | 3/5 (CONDITIONAL) |
| Flipped_Pctl | 305 | 0.143 | 0.0124 | (0, 60) | 1.21x | 3/5 (CONDITIONAL) |
| Δ(12M)_Pctl | 305 | 0.166 | 0.0036 | (0, 30) | 2.23x | 2/5 (REJECTED) |
| Z-score(10Y) | 305 | -0.179 | 0.0017 | (0, 10) | 1.24x | 3/5 (CONDITIONAL) |
| Δ(12M)_Zscore | 305 | 0.179 | 0.0017 | (30, 100) | 0.00x | 2/5 (REJECTED) |

**Best Transform: Percentile(10Y) (3/5 gates)**

---

## Best Transform Gate Details: Percentile(10Y)

| Gate | Description | Result | Details |
|------|-------------|--------|---------|
| Gate 0 | Real-time | PASS | Lag = 0m |
| Gate 1 | Walk-Forward | FAIL | Avg=0.35x, Std=0.41 |
| Gate 2 | Leave-Crisis-Out | FAIL | Min=0.00x, Drift=70.0% |
| Gate 3 | Lead Time | PASS | 2/4 crises |
| Gate 4 | Zone Stability | PASS | Range=10-10% |

---

## Production Validation Results

### Test 1: Strict Event Definition

| Threshold | Danger Zone Crash Rate | Avg MDD |
|-----------|------------------------|---------|
| MDD<-10% | 87.3% | -16.2% |
| MDD<-20% | 23.6% | -16.2% |
| MDD<-25% | 16.4% | -16.2% |

### Test 2: Multi-Horizon

| Horizon | Danger Zone Crash Rate |
|---------|------------------------|
| 3m | 0.0% |
| 6m | 3.5% |
| 12m | 23.6% |

### Test 3: LOCO Stability

**LOCO Stability: 7.6pp range**

---

## Current Status

| Metric | Value |
|--------|-------|
| Current CAPE | 39.8 |
| 10Y Percentile | 100.0% |
| 10Y Z-score | 2.15 |
| Δ(12M) | 2.1 |

---

## Conclusion

| Status | Details |
|--------|---------|
| **Validation** | 3/5 Gates (CONDITIONAL) |
| **Best Transform** | Percentile(10Y) |
| **Direction** | high_is_danger |
| **Optimal Zone** | (40, 100) |

---

*Generated: 2025-12-26 16:41:08*
