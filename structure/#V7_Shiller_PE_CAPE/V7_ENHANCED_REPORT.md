# V7 Enhanced CAPE - Real Rate Conditioning Report

Generated: 2025-12-26 17:02:35

## Enhancement Overview

### (A) CAPE × Real Rate Conditioning (贴现率条件化)

**Rationale:**
- 高 CAPE + 高实际利率 = 更危险 (估值压力 + 贴现率上升 = 双杀)
- 高 CAPE + 负实际利率 = 相对安全 (TINA: There Is No Alternative)

**Signal Matrix:**

|              | 负实际利率 (<0%) | 低实际利率 (0-2%) | 高实际利率 (>2%) |
|--------------|-----------------|------------------|-----------------|
| Low CAPE (<50th) | GREEN | GREEN | GREEN |
| Medium CAPE (50-80th) | GREEN | YELLOW | YELLOW |
| High CAPE (80-95th) | YELLOW | ORANGE | RED |
| Extreme CAPE (>95th) | ORANGE | RED | RED |

---

## Backtest Results

### Signal Distribution

| Signal | Count | % | Crash Rate (MDD<-20%) | Avg MDD |
|--------|-------|---|----------------------|---------|
| GREEN | 134 | 50.8% | 11.9% | -13.5% |
| YELLOW | 52 | 19.7% | 23.1% | -15.0% |
| ORANGE | 65 | 24.6% | 20.0% | -15.7% |
| RED | 13 | 4.9% | 0.0% | -15.8% |

### RED Signal Performance

| MDD Threshold | Crash Rate |
|---------------|------------|
| MDD < -10% | 76.9% |
| MDD < -15% | 69.2% |
| MDD < -20% | 0.0% |
| MDD < -25% | 0.0% |

---

## Original vs Enhanced Comparison

| Method | Danger Zone Size | Crash Rate |
|--------|-----------------|------------|
| Original (CAPE ≥80th pctl) | 62 (23.5%) | 22.6% |
| Enhanced (RED only) | 13 (4.9%) | 0.0% |

**Improvement:** Enhanced signal has lower crash rate with smaller danger zone.

---

## Current Status

| Metric | Value |
|--------|-------|
| CAPE Percentile (10Y) | 95.1% |
| Real Rate (10Y TIPS) | 1.94% |
| **Current Signal** | **RED** |

---

*Generated: 2025-12-26 17:02:35*
