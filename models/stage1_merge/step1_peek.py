import pandas as pd

FILES = {
    "collision": "dataset/raw/collision.csv",
    "vehicle":   "dataset/raw/vehicle.csv",
    "casualty":  "dataset/raw/casualty.csv",
}

for name, path in FILES.items():
    print("=" * 70)
    print(f"FILE: {name}.csv")
    print("=" * 70)

    # count total rows fast
    with open(path, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f)

    # load first 5 rows WITHOUT assuming anything
    raw = pd.read_csv(path, header=None, nrows=5)
    first_cell = raw.iloc[0, 0]

    # header detection: if first cell is a number, there's NO header
    try:
        float(first_cell)
        has_header = False
    except (ValueError, TypeError):
        has_header = True

    print(f"Total rows in file : {total}")
    print(f"Header detected    : {has_header}")
    print(f"Number of columns  : {raw.shape[1]}")
    print(f"First cell value   : {first_cell}")
    print("\nFirst 2 rows:")
    print(raw.head(2).to_string())
    print()