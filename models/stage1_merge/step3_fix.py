import pandas as pd
import os

os.makedirs("docs", exist_ok=True)

print("=" * 70)
print("STEP 3 FIX: Remove 4 junk columns + regenerate docs")
print("=" * 70)

# The 4 columns that survived due to merge suffixes
JUNK = ["collision_year_col", "collision_ref_no_col",
        "collision_year_veh", "collision_ref_no_veh"]

for name in ["merged_train", "merged_test"]:
    path = f"dataset/processed/{name}.csv"
    df = pd.read_csv(path, converters={"collision_index": str})
    before = df.shape[1]
    df.drop(columns=[c for c in JUNK if c in df.columns], inplace=True)
    df.to_csv(path, index=False)
    print(f"  {name}: {before} cols -> {df.shape[1]} cols ({len(df):,} rows)")

# Regenerate the doc (with utf-8 this time)
train = pd.read_csv("dataset/processed/merged_train.csv", nrows=0)
cols = list(train.columns)

roles = {"casualty_severity": "TARGET",
         "number_of_casualties": "Stage 1 target",
         "collision_index": "Grouping key (dropped before modelling)"}

with open("docs/column_removal_log.md", "w", encoding="utf-8") as f:
    f.write("# Column Removal Log\n\n")
    f.write("## Kept: Final feature set\n\n")
    for c in cols:
        tag = f"  <-- {roles[c]}" if c in roles else ""
        f.write(f"- `{c}`{tag}\n")
    f.write(f"\n**Total columns:** {len(cols)}\n")
    f.write(f"**Junk columns removed in fix:** {JUNK}\n")

print(f"\n  Final columns ({len(cols)}):")
for i, c in enumerate(cols, 1):
    tag = roles.get(c, "")
    print(f"  {i:2d}. {c:32s} {tag}")

print("\n  Docs saved -> docs/column_removal_log.md")
print("\n  DONE. Ready for Stage 1.")