"""Rebuild data/nutrition/food_lookup.csv from Health Canada's Canadian Nutrient File (CNF).

1. Download the CNF bulk CSV files (FOOD NAME.csv, NUTRIENT AMOUNT.csv) from
   https://www.canada.ca/en/health-canada/services/food-nutrition/healthy-eating/nutrient-data/canadian-nutrient-file-2015-download-files.html
   and unzip into data/raw/cnf/
2. Edit FOOD101_TO_CNF below (Food-101 label -> CNF FoodID) after searching FOOD NAME.csv.
3. python scripts/fetch_cnf.py
Energy (kcal) is CNF NutrientID 208 (verify in NUTRIENT NAME.csv). Values are per 100 g.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CNF = ROOT / "data/raw/cnf"
FOOD101_TO_CNF = {  # fill with verified FoodIDs, e.g. "greek_salad": 1234
}


def main():
    if not FOOD101_TO_CNF:
        raise SystemExit("Fill FOOD101_TO_CNF with verified CNF FoodIDs first (see docstring).")
    kcal = {}
    with open(CNF / "NUTRIENT AMOUNT.csv", encoding="latin-1") as f:
        for r in csv.DictReader(f):
            if r["NutrientID"] == "208":
                kcal[r["FoodID"]] = float(r["NutrientValue"])
    old = {r["label"]: r for r in csv.DictReader(open(ROOT / "data/nutrition/food_lookup.csv"))}
    for label, fid in FOOD101_TO_CNF.items():
        if str(fid) in kcal and label in old:
            old[label]["kcal_per_100g"] = round(kcal[str(fid)])
    with open(ROOT / "data/nutrition/food_lookup.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(old.values())).keys()))
        w.writeheader()
        w.writerows(old.values())
    print(f"updated {len(FOOD101_TO_CNF)} rows from CNF")


if __name__ == "__main__":
    main()
