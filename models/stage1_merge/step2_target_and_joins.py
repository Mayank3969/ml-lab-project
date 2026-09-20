import pandas as pd

pd.set_option('display.width', 200)

print("=" * 70)
print("STEP 2: Load safely (IDs as text) -> target, joins, leakage")
print("=" * 70)

# FIX: force join keys to text so big IDs don't become floats
cas = pd.read_csv("dataset/raw/casualty.csv",  dtype={"collision_index": str})
veh = pd.read_csv("dataset/raw/vehicle.csv",   dtype={"collision_index": str})
col = pd.read_csv("dataset/raw/collision.csv", dtype={"collision_index": str})

print(f"casualty  : {len(cas):>8,} rows")
print(f"vehicle   : {len(veh):>8,} rows")
print(f"collision : {len(col):>8,} rows")

# ---------- A) THE TARGET ----------
print("\n" + "=" * 70)
print("A) TARGET: casualty_severity  (1=Fatal, 2=Serious, 3=Slight)")
print("=" * 70)
counts = cas["casualty_severity"].value_counts().sort_index()
pct    = cas["casualty_severity"].value_counts(normalize=True).sort_index() * 100
for sev in counts.index:
    print(f"  {sev}: {counts[sev]:>7,}  ({pct[sev]:5.2f}%)")

# ---------- B) JOIN INTEGRITY ----------
print("\n" + "=" * 70)
print("B) JOIN KEYS: do the tables actually connect?")
print("=" * 70)
print("casualty collision_index examples :", cas["collision_index"].head(3).tolist())
print("collision collision_index examples:", col["collision_index"].head(3).tolist())

cas_ids, col_ids, veh_ids = (set(cas["collision_index"]),
                             set(col["collision_index"]),
                             set(veh["collision_index"]))
print(f"\nunique collision_index -> casualty:{len(cas_ids):,}  collision:{len(col_ids):,}  vehicle:{len(veh_ids):,}")
print(f"casualty ids found in collision   : {len(cas_ids & col_ids):,} / {len(cas_ids):,}")
print(f"vehicle  ids found in collision   : {len(veh_ids & col_ids):,} / {len(veh_ids):,}")

# ---------- C) REQUIRED FEATURE COLUMNS ----------
print("\n" + "=" * 70)
print("C) REQUIRED PIPELINE FEATURES present?")
print("=" * 70)
def check(df, name, cols):
    print(f"\n{name}:")
    for c in cols:
        print(f"  {'[OK]     ' if c in df.columns else '[MISSING]'} {c}")

check(col, "collision.csv", ["road_type","speed_limit","light_conditions",
                             "weather_conditions","junction_detail","number_of_casualties"])
check(cas, "casualty.csv",  ["casualty_severity","vehicle_reference","collision_index"])
check(veh, "vehicle.csv",   ["vehicle_reference","collision_index"])

# ---------- D) LEAKAGE COLUMNS ----------
print("\n" + "=" * 70)
print("D) LEAKAGE columns (encode the target -> MUST be dropped)")
print("=" * 70)
keys = ["enhanced", "adjusted", "injury_based"]
print("casualty.csv:")
for c in cas.columns:
    if any(k in c for k in keys): print(f"  - {c}")
print("collision.csv:")
for c in col.columns:
    if any(k in c for k in keys) or c == "collision_severity": print(f"  - {c}")

print("\nDone.")