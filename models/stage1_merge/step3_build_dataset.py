import os
import json
import hashlib
import pandas as pd
import numpy as np
import functools
import argparse
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

print = functools.partial(print, flush=True)

# Parse arguments for --refresh-features
parser = argparse.ArgumentParser()
parser.add_argument('--refresh-features', action='store_true', help='Force re-run of ablation study')
args, unknown = parser.parse_known_args()

def hash_file(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Paths
CAS_PATH = "dataset/raw/casualty.csv"
COL_PATH = "dataset/raw/collision.csv"
VEH_PATH = "dataset/raw/vehicle.csv"
SELECTION_JSON = "models/saved_models/feature_selection.json"

os.makedirs("dataset/processed", exist_ok=True)
os.makedirs("models/saved_models", exist_ok=True)
os.makedirs("docs", exist_ok=True)

print("=" * 70)
print("STEP 3: Merge -> Clean -> Train/Test Split -> Data-Driven Ablation")
print("=" * 70)

# Calculate hashes
print("[1] Verifying input hashes...")
current_hashes = {
    "casualty": hash_file(CAS_PATH),
    "collision": hash_file(COL_PATH),
    "vehicle": hash_file(VEH_PATH)
}

cached_features = None
if os.path.exists(SELECTION_JSON) and not args.refresh_features:
    with open(SELECTION_JSON, "r", encoding="utf-8") as f:
        cache_data = json.load(f)
    if cache_data.get("input_hashes") == current_hashes:
        cached_features = cache_data
        print("  => MATCH: Loading feature selection from cache.")
    else:
        print("  => MISMATCH: Data changed. Forcing re-run of ablation.")
elif args.refresh_features:
    print("  => --refresh-features flag detected. Forcing re-run.")
else:
    print("  => Cache not found. Running ablation.")

# Load Data
print("\n[2] Loading raw data...")
cas = pd.read_csv(CAS_PATH, converters={"collision_index": str, "vehicle_reference": str}, low_memory=False)
col = pd.read_csv(COL_PATH, converters={"collision_index": str}, low_memory=False)
veh = pd.read_csv(VEH_PATH, converters={"collision_index": str, "vehicle_reference": str}, low_memory=False)

print("\n[3] Merging: casualty <- collision <- vehicle ...")
df = cas.merge(col, on="collision_index", how="inner", suffixes=("", "_col"))
df = df.merge(veh, on=["collision_index", "vehicle_reference"], how="left", suffixes=("", "_veh"))

JUNK = ["collision_year_col", "collision_ref_no_col", "collision_year_veh", "collision_ref_no_veh"]
df.drop(columns=[c for c in JUNK if c in df.columns], inplace=True)
df = df.dropna(subset=["casualty_severity"])

feature_cols = [c for c in df.columns if c not in ["collision_index", "casualty_severity"]]
wrong_dupes = df.duplicated(subset=feature_cols).sum()
correct_dupes = df.duplicated(subset=feature_cols + ["collision_index", "casualty_severity"]).sum()
df = df.drop_duplicates(subset=feature_cols + ["collision_index", "casualty_severity"])

print("\n[4] Pre-Filter: Dropping Leakage, IDs, Historic, and Zero-Variance columns...")
prefilter_dropped = []

leakage_and_ids = [
    "collision_severity", "enhanced_severity_collision", "collision_injury_based",
    "collision_adjusted_severity_serious", "collision_adjusted_severity_slight",
    "enhanced_casualty_severity", "casualty_injury_based",
    "casualty_adjusted_severity_serious", "casualty_adjusted_severity_slight",
    "location_easting_osgr", "location_northing_osgr", "longitude", "latitude",
    "lsoa_of_accident_location", "lsoa_of_casualty", "lsoa_of_driver",
    "local_authority_district", "local_authority_ons_district", "local_authority_highway", "local_authority_highway_current",
    "police_force", "collision_ref_no", "vehicle_reference", "casualty_reference",
    "first_road_number", "second_road_number", "generic_make_model",
    "date", "time", "collision_year"
]
historic_cols = [c for c in df.columns if c.endswith("_historic")]

drop_list = set(leakage_and_ids + historic_cols)
for c in drop_list:
    if c in df.columns:
        df.drop(columns=[c], inplace=True)
        prefilter_dropped.append({"col": c, "reason": "Historic/Leakage/ID"})

for c in df.columns:
    if c != "collision_index":
        df[c] = pd.to_numeric(df[c], errors="coerce")
num_cols = [c for c in df.columns if c != "collision_index"]
df[num_cols] = df[num_cols].fillna(-1).astype(int)

# Zero variance check
for c in [x for x in df.columns if x not in ["collision_index", "casualty_severity"]]:
    if df[c].nunique() <= 1:
        df.drop(columns=[c], inplace=True)
        prefilter_dropped.append({"col": c, "reason": "Zero variance"})

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(sgkf.split(df, df["casualty_severity"], df["collision_index"]))

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

if cached_features:
    kept_cols = cached_features["selected_features"]
    ablation_log_text = cached_features.get("ablation_log_text", "Loaded from cache.")
else:
    print("\n[5] Feature Importance Profiling (with Shadow Feature) on TRAIN...")
    
    candidate_cols = [c for c in train_df.columns if c not in ["casualty_severity", "number_of_casualties", "collision_index"]]
    X = train_df[candidate_cols].copy()
    y = train_df["casualty_severity"]
    groups = train_df["collision_index"]
    
    # Add shadow column
    np.random.seed(42)
    X["_SHADOW_"] = np.random.permutation(X[candidate_cols[0]].values)
    
    clf = DecisionTreeClassifier(random_state=42)
    
    cv_splits = list(sgkf.split(X, y, groups))
    importances_matrix = []
    
    # 5-fold CV to average feature importances
    for tr_idx, val_idx in cv_splits:
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        clf.fit(X_tr, y_tr)
        importances_matrix.append(clf.feature_importances_)
        
    imp_mean = np.mean(importances_matrix, axis=0)
    imp_std = np.std(importances_matrix, axis=0)
    
    # Map back to columns
    feature_imp = {}
    for i, col in enumerate(X.columns):
        feature_imp[col] = {"mean": imp_mean[i], "std": imp_std[i]}
        
    shadow_mean = feature_imp["_SHADOW_"]["mean"]
    print(f"  Shadow feature baseline importance: {shadow_mean:.5f}")
    
    # Sort candidates (excluding shadow) by mean importance descending
    candidates_sorted = sorted(candidate_cols, key=lambda c: feature_imp[c]["mean"], reverse=True)
    
    hard_drop = []
    hard_keep = []
    ambiguous = []
    
    total_importance = sum(feature_imp[c]["mean"] for c in candidate_cols if feature_imp[c]["mean"] > shadow_mean)
    cumulative_importance = 0
    
    for c in candidates_sorted:
        mean_imp = feature_imp[c]["mean"]
        std_imp = feature_imp[c]["std"]
        
        # Hard drop rule
        if mean_imp <= shadow_mean:
            hard_drop.append({"col": c, "reason": f"<= Shadow (Mean: {mean_imp:.5f})"})
            continue
            
        # Add to cumulative
        cumulative_importance += mean_imp
        cum_ratio = cumulative_importance / total_importance if total_importance > 0 else 1.0
        
        # Routing logic
        # High variance (std > mean * 0.5) pushes it to ambiguous if it's not the absolute top feature
        if cum_ratio <= 0.95 and (std_imp < mean_imp * 0.5):
            hard_keep.append({"col": c, "reason": f"Top 95% (Mean: {mean_imp:.5f})"})
        elif cum_ratio <= 0.99 or (cum_ratio <= 0.95 and std_imp >= mean_imp * 0.5):
            ambiguous.append(c)
        else:
            hard_drop.append({"col": c, "reason": f">99% Tail (Mean: {mean_imp:.5f})"})
            
    print(f"  Hard Keep: {len(hard_keep)} | Ambiguous: {len(ambiguous)} | Hard Drop: {len(hard_drop)}")
    
    kept_cols = [x["col"] for x in hard_keep]
    ambiguous_log = []
    
    if ambiguous:
        print("\n[6] Targeted Leave-One-Out CV for Ambiguous Features...")
        X_base = train_df[kept_cols + ambiguous]
        
        def eval_subset(feats):
            scores = cross_val_score(DecisionTreeClassifier(random_state=42), X_base[feats], y, groups=groups, cv=cv_splits, scoring='f1_macro', n_jobs=-1)
            return np.mean(scores)
            
        baseline_f1 = eval_subset(kept_cols + ambiguous)
        print(f"  Ambiguous Baseline Macro-F1: {baseline_f1:.4f}")
        
        for c in ambiguous:
            feats_without = [f for f in kept_cols + ambiguous if f != c]
            f1_without = eval_subset(feats_without)
            delta = baseline_f1 - f1_without
            if delta >= 0:
                kept_cols.append(c)
                ambiguous_log.append({"col": c, "action": "KEEP", "delta": delta, "reason": "Positive/Neutral delta"})
                print(f"  [KEEP] {c:35s} | delta: {delta:+.5f}")
            else:
                ambiguous_log.append({"col": c, "action": "DROP", "delta": delta, "reason": "Negative delta"})
                print(f"  [DROP] {c:35s} | delta: {delta:+.5f}")
                
    else:
        print("\n[6] No ambiguous features to check.")
        baseline_f1 = 0
        
    print(f"\n  Final Kept Features: {len(kept_cols)}")
    
    # Build ablation log
    log_lines = [
        "# Stage 1 Feature Selection (Shadow + Cumulative Importance)",
        "",
        f"**Duplicate Row Check:**",
        f"- Features ONLY: {wrong_dupes} (wrongly dropped)",
        f"- Features + keys: {correct_dupes} (safely dropped)",
        "",
        "## 1. Pre-filtered (Historic / ID / Zero Variance)",
        "| Feature | Reason |",
        "|---|---|",
        *([f"| `{x['col']}` | {x['reason']} |" for x in prefilter_dropped]),
        "",
        "## 2. Hard Keep (Top 95% Cumulative Importance)",
        "| Feature | Reason |",
        "|---|---|",
        *([f"| `{x['col']}` | {x['reason']} |" for x in hard_keep]),
        "",
        "## 3. Targeted Leave-One-Out (Ambiguous 95-99% or High StdDev)",
        f"**Baseline Macro-F1:** `{baseline_f1:.4f}`\n",
        "| Feature | Delta | Action |",
        "|---|---|---|",
        *([f"| `{x['col']}` | {x['delta']:+.5f} | {x['action']} |" for x in ambiguous_log]),
        "",
        "## 4. Hard Drop (<= Shadow Mean or >99% Tail)",
        "| Feature | Reason |",
        "|---|---|",
        *([f"| `{x['col']}` | {x['reason']} |" for x in hard_drop])
    ]
    ablation_log_text = "\n".join(log_lines)
    
    with open("docs/ablation_log.md", "w", encoding="utf-8") as f:
        f.write(ablation_log_text)
        
print("\n[7] Applying Frequency Encoding for Nominal Columns...")
ORDINAL_COLS = ["speed_limit", "age_of_casualty", "age_of_driver", "age_of_vehicle", "engine_capacity_cc", "age_band_of_casualty", "age_band_of_driver"]
nominal_cols = [c for c in kept_cols if c not in ORDINAL_COLS]

always_keep = ["casualty_severity", "number_of_casualties", "collision_index"]
train_df = train_df[kept_cols + always_keep].copy()
test_df = test_df[kept_cols + always_keep].copy()

# Frequency encoding avoids the curse of dimensionality & OOM for KNN/distance models
freq_maps = {}
for c in nominal_cols:
    freq = train_df[c].value_counts(normalize=True).to_dict()
    freq_maps[c] = freq
    train_df[c] = train_df[c].map(freq).fillna(0)
    test_df[c] = test_df[c].map(freq).fillna(0)

# Log specific demographic decisions explicitly
print("\n  [Validation] Demographic Features Keep/Drop Status:")
for demo in ["age_of_driver", "sex_of_driver", "driver_imd_decile", "casualty_imd_decile", "escooter_flag"]:
    status = "KEPT" if demo in kept_cols else "DROPPED"
    print(f"    - {demo}: {status}")

print(f"\n  Final merged_train.csv shape: {train_df.shape}")
print(f"  Feature columns ({len(kept_cols)}): {kept_cols}")

# Save Cache
cache_data = {
    "input_hashes": current_hashes,
    "selected_features": kept_cols,
    "nominal_cols": nominal_cols,
    "ordinal_cols": [c for c in kept_cols if c in ORDINAL_COLS],
    "final_encoded_cols": kept_cols,
    "frequency_encodings": freq_maps,
    "ablation_log_text": ablation_log_text
}
with open(SELECTION_JSON, "w", encoding="utf-8") as f:
    json.dump(cache_data, f, indent=4)

print("\n[8] Saving Data...")
train_df.to_csv("dataset/processed/merged_train.csv", index=False)
test_df.to_csv("dataset/processed/merged_test.csv", index=False)
print("[OK] Saved merged datasets.")