# Momentum Optimization Report
Generated: 2025-12-15 05:39:14Z
Sources: `docs/tables/multiwindow_performance_matrix.csv` (performance matrix) and `docs/tables/multiwindow_regression_summary.csv` (regression summary).

**Recommendation:** MOM_24M wins with a score of 82.5, delivering 3.22 pp average ΔR² improvement and 41.8% MOM significance.

## Ranking
| Rank | Model | ΔR² (pp) | MOM Sig. (%) | Simplicity | Economic | Score |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | MOM_24M | 3.22 | 41.8 | 100 | 100 | 82.5 |
| 2 | MOM_4M | 2.86 | 38.2 | 100 | 100 | 72.7 |
| 3 | MOM_EQ | 2.05 | 32.7 | 70 | 80 | 43.3 |
| 4 | MOM_8M | 1.89 | 18.2 | 100 | 100 | 42.8 |
| 5 | MOM_12M | 1.59 | 34.5 | 100 | 100 | 40.4 |

## Methodology
Weights → ΔR²: 40%, MOM sig.: 30%, Simplicity: 20%, Economic: 10%.

The scorecard rewards models that improve FF5 ΔR², retain significant MOM coefficients, and stay simple for implementation.