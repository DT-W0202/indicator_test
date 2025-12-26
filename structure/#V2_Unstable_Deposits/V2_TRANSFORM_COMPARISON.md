# V2 Unstable Deposits Ratio - Transform Comparison Report

Generated: 2025-12-25 23:49:24

## Transform Comparison

| Transform | IC | AUC | Direction | Gates Passed |
|-----------|-----|-----|-----------|--------------|
| Δ(12m) | 0.3519 | 0.643 | low→crash | 2/5 |
| Z-score(10Y) | 0.4208 | 0.699 | low→crash | 2/5 |
| Percentile(10Y) | 0.4162 | 0.671 | low→crash | 2/5 |

## Best Transform: Δ(12m)

### Gate Details
- **gate0**: PASS - 滞后 1.0 月 <= 6.0 月
- **gate1**: FAIL - Avg=0.00x, Std=0.00, Min=0.00x
- **gate2**: FAIL - Min Lift=0.00x, Zone Drift=0%
- **gate3**: FAIL - 0/4 危机有提前信号 (0%)
- **gate4**: PASS - Lower range=0%, Upper range=0%, Center range=0%

## Conclusion

**Best Zone**: [30%, 100%]
**Final Status**: REJECTED

### Key Insights

1. **Δ(12m)** (12-month change): Captures momentum/trend
2. **Z-score(10Y)**: Measures deviation from long-term mean
3. **Percentile(10Y)**: Relative position in historical distribution

For structural drift variables, change-based transforms often outperform level-based measures
because they capture the "shock" rather than the absolute position.

---

*Generated: 2025-12-25 23:49:24*
