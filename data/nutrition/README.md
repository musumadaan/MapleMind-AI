`food_lookup.csv` holds ILLUSTRATIVE approximate values for a subset of Food-101 labels so the demo runs offline.
For any real use, regenerate it from Health Canada's Canadian Nutrient File (CNF):
`python scripts/fetch_cnf.py` (downloads the CNF bulk CSV from canada.ca and maps Food-101 labels to CNF food codes).
