# V7 Shiller PE (CAPE) - Validation Results

Generated: 2025-12-24 19:49

## Factor Definition

**Formula**: `CAPE = S&P 500 Price / 10-Year Average Inflation-Adjusted Earnings`

**Hypothesis**: High CAPE → Market overvalued → Lower future returns

**Current CAPE**: 39.8

**Historical Mean**: 17.7

---

## Step 0: Data Summary

| Statistic | Value |
|-----------|-------|
| Mean | 17.7 |
| Median | 16.6 |
| Current | 39.8 |

---

## Step 1: Structural Break Analysis

### Chow Test Results

| Breakpoint | F-stat | p-value | Significant |
|------------|--------|---------|-------------|
| 2008-01-01 | 18.14 | 0.0000 | ✓ |
| 2015-01-01 | 34.79 | 0.0000 | ✓ |
| 2020-01-01 | 35.88 | 0.0000 | ✓ |

### Subsample β Estimates

| Period | β | HAC t-stat | p-value |
|--------|---|------------|----------|
| 1999-2007 | -0.0132 | -5.47 | 0.0000 |
| 2008-2014 | -0.0246 | -3.84 | 0.0002 |
| 2015-2019 | -0.0094 | -2.29 | 0.0254 |
| 2020-2024 | -0.0304 | -8.36 | 0.0000 |

---

## Step 2: Interest Rate Interaction

**Full Sample IC**: -0.3056 (p = 0.0000)

### Interaction Regression

Model: R_{t→12M} = α + β·CAPE + γ·Rate + δ·(CAPE×Rate) + ε

| Coefficient | Value | HAC t-stat | p-value |
|-------------|-------|------------|----------|
| const | 0.320299 | 3.40 | 0.0008 |
| factor | -0.008480 | -2.08 | 0.0388 |
| condition | -0.042944 | -0.72 | 0.4712 |
| interaction | 0.000946 | 0.56 | 0.5735 |

R² = 0.1289

### Regime-Split IC

| Regime | N | IC | p-value |
|--------|---|----|---------|
| High Rate | 147 | -0.0306 | 0.7133 |
| Low Rate | 158 | -0.3693 | 0.0000 |

---

## Step 3: Risk Target Variables

### IC for Different Targets

| Target | IC | p-value | N |
|--------|----|---------|---|
| Forward Return 6M | -0.1962 | 0.0005 | 311 |
| Forward Return 12M | -0.3056 | 0.0000 | 305 |
| Forward Max Drawdown | -0.3332 | 0.0000 | 215 |
| Forward Realized Vol | 0.2507 | 0.0000 | 305 |

### Drawdown Event AUC

| Threshold | AUC | Event Rate | N |
|-----------|-----|------------|---|
| -10% | 0.666 | 57.7% | 317 |
| -15% | 0.631 | 40.1% | 317 |
| -20% | 0.675 | 21.8% | 317 |

---

## Step 4: Tail Quantile Analysis

| Metric | Value |
|--------|-------|
| Mean β | -0.008976 |
| 5% Quantile β | -0.002406 |
| Tail Ratio | 0.27 |

---

## Step 5: Quintile Analysis

| Quintile | Avg CAPE | Avg Return | Crash Rate | N |
|----------|----------|------------|------------|---|
| Q1 | 20.3 | 11.70% | 41.9% | 43 |
| Q2 | 24.6 | 6.34% | 20.9% | 43 |
| Q3 | 26.8 | 3.63% | 20.9% | 43 |
| Q4 | 29.7 | 10.21% | 44.2% | 43 |
| Q5 | 36.9 | -4.46% | 76.7% | 43 |

**Monotonicity**: Spearman = -0.700 (p = 0.1881)

**Q5-Q1 Spread**: -16.16%

---

## Step 6: Bootstrap Robustness

### Full Sample

| Metric | Value |
|--------|-------|
| Original β | -0.008976 |
| 95% CI | [-0.016620, -0.001608] |
| Bootstrap p-value | 0.0240 |

### High Rate Regime

| Metric | Value |
|--------|-------|
| Original β | -0.001421 |
| 95% CI | [-0.019541, 0.028251] |
| Bootstrap p-value | 0.7680 |

---

## 结论

待补充...
