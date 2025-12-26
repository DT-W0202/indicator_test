# V8 Margin Debt - Validation Results

Generated: 2025-12-24 19:49

## Factor Definition

**Formula**: `Margin Debt YoY = (Margin Debt / Margin Debt_12M) - 1`

**Data Source**: FINRA Margin Statistics (Debit Balances in Customers' Securities Margin Accounts)

**Hypothesis**: High Margin Debt Growth → Excessive speculation → Lower future returns / Higher crash risk

**Current Level**: $1,214,321 Million

**Current YoY Growth**: 36.3%

---

## Step 0: Data Summary

| Statistic | Value |
|-----------|-------|
| Current Level | $1,214,321M |
| Current YoY | 36.3% |
| Mean Level | $417,793M |
| Mean YoY | 10.5% |

---

## Step 1: Structural Break Analysis

### Chow Test Results

| Breakpoint | F-stat | p-value | Significant |
|------------|--------|---------|-------------|
| 2008-01-01 | 17.22 | 0.0000 | ✓ |
| 2015-01-01 | 10.85 | 0.0000 | ✓ |
| 2020-01-01 | 9.50 | 0.0001 | ✓ |

### Subsample β Estimates

| Period | β | HAC t-stat | p-value |
|--------|---|------------|----------|
| 1999-2007 | -0.0002 | -0.37 | 0.7109 |
| 2008-2014 | -0.0006 | -0.67 | 0.5032 |
| 2015-2019 | -0.0017 | -1.90 | 0.0627 |
| 2020-2024 | -0.0023 | -3.17 | 0.0025 |

---

## Step 2: Interest Rate Interaction

**Full Sample IC**: -0.1928 (p = 0.0007)

### Interaction Regression

Model: R_{t→12M} = α + β·Margin_YoY + γ·Rate + δ·(Margin×Rate) + ε

| Coefficient | Value | HAC t-stat | p-value |
|-------------|-------|------------|----------|
| const | 0.109993 | 3.94 | 0.0001 |
| factor | -0.000739 | -0.77 | 0.4416 |
| condition | -0.021751 | -1.34 | 0.1801 |
| interaction | -0.000063 | -0.18 | 0.8553 |

R² = 0.1008

### Regime-Split IC

| Regime | N | IC | p-value |
|--------|---|----|---------|
| High Rate | 147 | -0.1343 | 0.1048 |
| Low Rate | 158 | -0.3092 | 0.0001 |

---

## Step 3: Risk Target Variables

### IC for Different Targets

| Target | IC | p-value | N |
|--------|----|---------|---|
| Forward Return 6M | -0.0285 | 0.6169 | 311 |
| Forward Return 12M | -0.1928 | 0.0007 | 305 |
| Forward Max Drawdown | 0.0792 | 0.2474 | 215 |
| Forward Realized Vol | -0.2071 | 0.0003 | 305 |

### Drawdown Event AUC

| Threshold | AUC | Event Rate | N |
|-----------|-----|------------|---|
| -10% | 0.449 | 57.9% | 316 |
| -15% | 0.474 | 40.2% | 316 |
| -20% | 0.372 | 21.8% | 316 |

---

## Step 4: Tail Quantile Analysis

| Metric | Value |
|--------|-------|
| Mean β | -0.001112 |
| 5% Quantile β | -0.000273 |
| Tail Ratio | 0.25 |

---

## Step 5: Quintile Analysis

| Quintile | Avg YoY% | Avg Return | Crash Rate | N |
|----------|----------|------------|------------|---|
| Q1 | -23.8% | 4.94% | 58.1% | 43 |
| Q2 | -3.1% | 11.82% | 27.9% | 43 |
| Q3 | 11.9% | 8.14% | 27.9% | 43 |
| Q4 | 20.4% | 5.49% | 30.2% | 43 |
| Q5 | 40.0% | -2.97% | 60.5% | 43 |

**Monotonicity**: Spearman = -0.400 (p = 0.5046)

**Q5-Q1 Spread**: -7.91%

---

## Step 6: Bootstrap Robustness

### Full Sample

| Metric | Value |
|--------|-------|
| Original β | -0.001112 |
| 95% CI | [-0.003046, 0.001216] |
| Bootstrap p-value | 0.3520 |

### High Rate Regime

| Metric | Value |
|--------|-------|
| Original β | -0.000848 |
| 95% CI | [-0.003860, 0.002879] |
| Bootstrap p-value | 0.5440 |

---

## 结论

待补充...
