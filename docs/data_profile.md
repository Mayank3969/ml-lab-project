# Data Profile (Pre-Model Selection)
**Generated:** 2026-09-20 21:08:50

## Dimensionality Analysis
- **Raw Features Kept:** 11
  - Nominal (Frequency Encoded): 8
  - Ordinal / Numeric: 3
- **Post-Encoding Dimensionality:** 11 dense numerical columns

## Class Imbalance
| Target Class (casualty_severity) | Samples (Train) | Percentage |
|---|---|---|
| 1 (Fatal) | 1,230 | 1.20% |
| 2 (Serious) | 22,206 | 21.71% |
| 3 (Slight) | 78,869 | 77.09% |

## Mathematical Suitability for Models
Based on this profile:
1. **Decision Tree / Ensembles**: Highly suitable. Splits natively on non-linear ordinal and frequency-encoded continuous distributions.
2. **Logistic Regression**: Suitable as a calibrated linear model when standardized (`StandardScaler`).
3. **K-Nearest Neighbors (KNN)**: **Unblocked & Suitable**. By utilizing Frequency Encoding instead of one-hot expansion, the feature space remains compact (11 dense dimensions). Fast spatial index trees (`kd_tree`) compute neighbors in sub-second time with zero OOM bottlenecks.
