import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

os.makedirs("dataset/processed", exist_ok=True)
os.makedirs("docs", exist_ok=True)

print("=" * 70)
print("STEP 3: Merge -> Cleanup -> Deduplicate -> Split 80/20")
print("=" * 70)

# ══════════════════════════════════════════════════════════════
# [1] LOAD RAW DATA (IDs forced to text so they don't break)
# ══════════════════════════════════════════════════════════════
print("\n[1] Loading raw data...")
cas = pd.read_csv("dataset/raw/casualty.csv",
                  converters={"collision_index": str, "vehicle_reference": str},
                  low_memory=False)
col = pd.read_csv("dataset/raw/collision.csv",
                  converters={"collision_index": str},
                  low_memory=False)
veh = pd.read_csv("dataset/raw/vehicle.csv",
                  converters={"collision_index": str, "vehicle_reference": str},
                  low_memory=False)

print(f"  casualty : {cas.shape[0]:>8,} rows × {cas.shape[1]} cols")
print(f"  collision: {col.shape[0]:>8,} rows × {col.shape[1]} cols")
print(f"  vehicle  : {veh.shape[0]:>8,} rows × {veh.shape[1]} cols")

# ══════════════════════════════════════════════════════════════
# [2] PRE-MERGE DUPLICATE KEY CHECK (cause of row multiplication)
# ══════════════════════════════════════════════════════════════
print("\n[2] Checking join keys for duplicates BEFORE merge...")
col_dupes = col["collision_index"].duplicated().sum()
veh_dupes = veh.duplicated(subset=["collision_index", "vehicle_reference"]).sum()
print(f"  duplicate collision_index in collision.csv          : {col_dupes:,}")
print(f"  duplicate (collision+vehicle) keys in vehicle.csv   : {veh_dupes:,}")
if col_dupes == 0 and veh_dupes == 0:
    print("  ✓ Join keys are unique — merge will NOT multiply rows")
else:
    print("  ⚠️  Duplicates found — merge may inflate rows (will check after)")

# ══════════════════════════════════════════════════════════════
# [3] MERGE
# ══════════════════════════════════════════════════════════════
print("\n[3] Merging: casualty ← collision ← vehicle ...")
df = cas.merge(col, on="collision_index", how="inner", suffixes=("", "_col"))
df = df.merge(veh, on=["collision_index", "vehicle_reference"],
              how="left", suffixes=("", "_veh"))

if df.shape[0] > cas.shape[0]:
    print(f"  ⚠️  JOIN INFLATION: {cas.shape[0]:,} → {df.shape[0]:,} rows "
          f"(+{df.shape[0]-cas.shape[0]:,})")
else:
    print(f"  ✓ No inflation. Merged shape: {df.shape[0]:,} rows × {df.shape[1]} cols")

# ══════════════════════════════════════════════════════════════
# [4] DROP LEAKAGE COLUMNS (9) — these encode the answer
# ══════════════════════════════════════════════════════════════
print("\n[4] Dropping 9 LEAKAGE columns...")
leakage_cols = {
    "collision_severity": "Directly encodes target severity",
    "enhanced_severity_collision": "Adjusted severity, encodes target",
    "collision_injury_based": "Injury-based severity, encodes target",
    "collision_adjusted_severity_serious": "Binary encoding of target",
    "collision_adjusted_severity_slight": "Binary encoding of target",
    "enhanced_casualty_severity": "Adjusted casualty severity, encodes target",
    "casualty_injury_based": "Injury-based severity, encodes target",
    "casualty_adjusted_severity_serious": "Binary encoding of target",
    "casualty_adjusted_severity_slight": "Binary encoding of target",
}
for c in leakage_cols:
    if c in df.columns:
        df.drop(columns=[c], inplace=True)
        print(f"  ✗ {c:42s} → {leakage_cols[c]}")

# ══════════════════════════════════════════════════════════════
# [5] DROP IRRELEVANT COLUMNS — not environmental/road conditions
# ══════════════════════════════════════════════════════════════
print("\n[5] Dropping IRRELEVANT columns...")
irrelevant_cols = {
    # Geographic / location
    "location_easting_osgr": "Grid coordinate, not a road condition",
    "location_northing_osgr": "Grid coordinate, not a road condition",
    "longitude": "GPS coordinate, causes location overfitting",
    "latitude": "GPS coordinate, causes location overfitting",
    "lsoa_of_accident_location": "Area code, not a road condition",
    "local_authority_district": "Admin area, not a road condition",
    "local_authority_ons_district": "Admin area code",
    "local_authority_highway": "Admin area code",
    "local_authority_highway_current": "Admin area code",
    "police_force": "Police jurisdiction, not a road condition",
    "lsoa_of_casualty": "Casualty home area, irrelevant",
    "lsoa_of_driver": "Driver home area, irrelevant",
    # IDs / references
    "collision_year": "All values are 2025, zero variance",
    "collision_ref_no": "Unique ID, no predictive value",
    "vehicle_reference": "Join key only, not a feature",
    "casualty_reference": "Unique ID within collision",
    # Date / time (too specific)
    "date": "Specific date doesn't generalize",
    "time": "Too granular (day_of_week captures the pattern)",
    # Specific road identifiers (cause overfitting)
    "first_road_number": "Identifies a specific road, overfitting",
    "second_road_class": "Secondary road, low relevance",
    "second_road_number": "Identifies a specific road, overfitting",
    # Pedestrian crossing (granular / mostly empty)
    "pedestrian_crossing_human_control_historic": "Historic, mostly empty",
    "pedestrian_crossing_physical_facilities_historic": "Historic, mostly empty",
    "pedestrian_crossing": "Too granular for this use case",
    # Administrative / post-accident
    "did_police_officer_attend_scene_of_accident": "Post-accident response, not a condition",
    "trunk_road_flag": "Already captured by road_type",
    # Historic / superseded
    "junction_detail_historic": "Superseded by junction_detail",
    "carriageway_hazards_historic": "Superseded by carriageway_hazards",
    "vehicle_manoeuvre_historic": "Superseded by vehicle_manoeuvre",
    "journey_purpose_of_driver_historic": "Superseded",
    "vehicle_location_restricted_lane_historic": "Superseded",
    # Casualty demographics (not environmental/road)
    "sex_of_casualty": "Demographic, not a road condition",
    "age_of_casualty": "Demographic, not a road condition",
    "age_band_of_casualty": "Demographic, not a road condition",
    "pedestrian_location": "Granular position detail",
    "pedestrian_movement": "Granular position detail",
    "car_passenger": "Position detail, not road condition",
    "bus_or_coach_passenger": "Position detail, not road condition",
    "pedestrian_road_maintenance_worker": "Too specific",
    "casualty_imd_decile": "Socioeconomic status, not road condition",
    "casualty_distance_banding": "Distance metric, not road condition",
    # Vehicle details (not environmental/road)
    "vehicle_type": "Vehicle characteristic, not road condition",
    "towing_and_articulation": "Vehicle configuration",
    "vehicle_manoeuvre": "Driver action, not road condition",
    "vehicle_direction_from": "Direction, not road condition",
    "vehicle_direction_to": "Direction, not road condition",
    "vehicle_location_restricted_lane": "Lane position, not road condition",
    "junction_location": "Position in junction, not road condition",
    "skidding_and_overturning": "Accident dynamics, not road condition",
    "hit_object_in_carriageway": "Accident dynamics, not road condition",
    "vehicle_leaving_carriageway": "Accident dynamics, not road condition",
    "hit_object_off_carriageway": "Accident dynamics, not road condition",
    "first_point_of_impact": "Accident dynamics, not road condition",
    "vehicle_left_hand_drive": "Vehicle configuration",
    "journey_purpose_of_driver": "Driver context, not road condition",
    "sex_of_driver": "Driver demographic, not road condition",
    "age_of_driver": "Driver demographic, not road condition",
    "age_band_of_driver": "Driver demographic, not road condition",
    "engine_capacity_cc": "Vehicle specification",
    "propulsion_code": "Vehicle specification",
    "age_of_vehicle": "Vehicle specification",
    "generic_make_model": "Vehicle identity, causes overfitting",
    "driver_imd_decile": "Socioeconomic status, not road condition",
    "escooter_flag": "Vehicle type flag",
    "driver_distance_banding": "Distance metric, not road condition",
}
removed = []
for c in irrelevant_cols:
    if c in df.columns:
        df.drop(columns=[c], inplace=True)
        removed.append(c)
        print(f"  ✗ {c:42s} → {irrelevant_cols[c]}")
print(f"\n  Removed {len(removed)} irrelevant columns")

# ══════════════════════════════════════════════════════════════
# [6] WHAT'S LEFT
# ══════════════════════════════════════════════════════════════
print("\n[6] FINAL FEATURE SET:")
roles = {"casualty_severity": "← TARGET",
         "number_of_casualties": "← Stage 1 target",
         "collision_index": "← Grouping key (dropped before modelling)"}
for i, c in enumerate(df.columns, 1):
    print(f"  {i:2d}. {c:32s} {roles.get(c, '')}")

# ══════════════════════════════════════════════════════════════
# [7] FINAL DUPLICATE CHECK
# ══════════════════════════════════════════════════════════════
print("\n[7] Final duplicate-row check...")
dupes = df.duplicated().sum()
print(f"  Exact duplicate rows: {dupes:,}")
if dupes > 0:
    df = df.drop_duplicates()
    print(f"  Removed them → {df.shape[0]:,} rows remain")

# ══════════════════════════════════════════════════════════════
# [8] CLEAN + CONVERT
# ══════════════════════════════════════════════════════════════
print("\n[8] Cleaning & converting dtypes...")
df = df.dropna(subset=["casualty_severity"])
for c in df.columns:
    if c != "collision_index":
        df[c] = pd.to_numeric(df[c], errors="coerce")
num_cols = [c for c in df.columns if c != "collision_index"]
df[num_cols] = df[num_cols].fillna(-1).astype(int)

# ══════════════════════════════════════════════════════════════
# [9] TARGET DISTRIBUTION
# ══════════════════════════════════════════════════════════════
print("\n[9] Target distribution:")
labels = {1: "Fatal", 2: "Serious", 3: "Slight"}
counts = df["casualty_severity"].value_counts().sort_index()
for sev, n in counts.items():
    print(f"  {sev} ({labels[sev]:8s}): {n:>7,}  ({n/len(df)*100:5.2f}%)")

# ══════════════════════════════════════════════════════════════
# [10] SPLIT 80/20 (group-aware)
# ══════════════════════════════════════════════════════════════
print("\n[10] Splitting 80/20 (same collision never in both sets)...")
groups = df["collision_index"].values
y = df["casualty_severity"].values
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, y, groups))
train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

overlap = set(train_df["collision_index"]) & set(test_df["collision_index"])
print(f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
print(f"  Collisions in BOTH sets: {len(overlap)}  (must be 0)")

# ══════════════════════════════════════════════════════════════
# [11] SAVE
# ══════════════════════════════════════════════════════════════
train_df.to_csv("dataset/processed/merged_train.csv", index=False)
test_df.to_csv("dataset/processed/merged_test.csv", index=False)

# ══════════════════════════════════════════════════════════════
# [12] AUTO-GENERATE DOCUMENTATION
# ══════════════════════════════════════════════════════════════
with open("docs/column_removal_log.md", "w") as f:
    f.write("# Column Removal Log\n\n")
    f.write("## Dropped: Leakage columns (encode the target)\n\n")
    f.write("| Column | Reason |\n|---|---|\n")
    for c, r in leakage_cols.items():
        f.write(f"| `{c}` | {r} |\n")
    f.write("\n## Dropped: Irrelevant columns (not environmental/road conditions)\n\n")
    f.write("| Column | Reason |\n|---|---|\n")
    for c, r in irrelevant_cols.items():
        f.write(f"| `{c}` | {r} |\n")
    f.write("\n## Kept: Final feature set\n\n")
    for c in df.columns:
        f.write(f"- `{c}` {roles.get(c, '')}\n")
    f.write(f"\n**Final shape:** {df.shape[0]:,} rows × {df.shape[1]} columns\n")

print("\n✅ Saved merged_train.csv + merged_test.csv → dataset/processed/")
print("✅ Saved column_removal_log.md → docs/")
print("\nReady for Stage 1 (regression).")