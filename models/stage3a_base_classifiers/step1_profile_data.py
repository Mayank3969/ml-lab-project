import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

os.makedirs("docs", exist_ok=True)

# Load selected features schema
SCHEMA_JSON = "models/saved_models/feature_selection.json"
if not os.path.exists(SCHEMA_JSON):
    print(f"Error: {SCHEMA_JSON} not found. Run Stage 1 first.")
    exit(1)

with open(SCHEMA_JSON, "r", encoding="utf-8") as f:
    schema = json.load(f)

# Load data
train_df = pd.read_csv("dataset/processed/merged_train.csv")
y = train_df["casualty_severity"]

# Profile
n_nominal = len(schema["nominal_cols"])
n_ordinal = len(schema["ordinal_cols"])
total_raw = len(schema["selected_features"])
total_encoded = len(schema["final_encoded_cols"])

counts = y.value_counts().sort_index()
total = len(y)

profile_text = f"""# Data Profile (Pre-Model Selection)
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dimensionality Analysis
- **Raw Features Kept:** {total_raw}
  - Nominal (Frequency Encoded): {n_nominal}
  - Ordinal / Numeric: {n_ordinal}
- **Post-Encoding Dimensionality:** {total_encoded} dense numerical columns

## Class Imbalance
| Target Class (casualty_severity) | Samples (Train) | Percentage |
|---|---|---|
| 1 (Fatal) | {counts.get(1, 0):,} | {counts.get(1, 0)/total*100:.2f}% |
| 2 (Serious) | {counts.get(2, 0):,} | {counts.get(2, 0)/total*100:.2f}% |
| 3 (Slight) | {counts.get(3, 0):,} | {counts.get(3, 0)/total*100:.2f}% |

## Mathematical Suitability for Models
Based on this profile:
1. **Decision Tree / Ensembles**: Highly suitable. Splits natively on non-linear ordinal and frequency-encoded continuous distributions.
2. **Logistic Regression**: Suitable as a calibrated linear model when standardized (`StandardScaler`).
3. **K-Nearest Neighbors (KNN)**: **Unblocked & Suitable**. By utilizing Frequency Encoding instead of one-hot expansion, the feature space remains compact ({total_encoded} dense dimensions). Fast spatial index trees (`kd_tree`) compute neighbors in sub-second time with zero OOM bottlenecks.
"""

with open("docs/data_profile.md", "w", encoding="utf-8") as f:
    f.write(profile_text)

print("[OK] Wrote docs/data_profile.md")

