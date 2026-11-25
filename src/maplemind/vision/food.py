"""Meal-photo analysis (computer vision skill).

Pipeline
  1. Dish classification - ViT fine-tuned on Food-101 (HF `nateraw/food`, 101 classes) when backend=hf.
     Offline backend=mock uses the filename as the label (for tests/demo) - clearly labelled as such.
  2. Plate-balance estimate - HSV colour segmentation: share of plate pixels that are green/colourful
     (vegetable/fruit proxy) vs beige/brown (grain/fried proxy). Cheap, explainable, runs on CPU in ms.
  3. Map label -> food group + approximate energy using data/nutrition/food_lookup.csv
     (regenerate from Health Canada's Canadian Nutrient File for real values).
  4. Compare with Canada's Food Guide plate (1/2 vegetables & fruits, 1/4 whole grains, 1/4 protein).

Uncertainty handling: if classifier confidence < vision.min_confidence the agent ASKS the user to confirm the dish
instead of guessing. Portion size from a single 2-D photo is unreliable (Nutrition5k shows even nutritionists
struggle), so energy is reported as a wide range, never a precise number.

Responsible-AI decision: no face / emotion recognition. Inferring mood from faces is scientifically weak and
high-risk; MapleMind only analyses food photos the user chooses to share.
"""
from __future__ import annotations

import csv
import colorsys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..config import resolve


@dataclass
class MealAnalysis:
    label: str
    confidence: float
    food_group: str
    kcal_range: tuple[int, int] | None
    veg_fraction: float
    beige_fraction: float
    processed_level: str
    needs_confirmation: bool
    backend: str

    def feedback(self) -> list[str]:
        tips = []
        if self.veg_fraction < 0.35:
            tips.append(f"Vegetables/fruit look like roughly {self.veg_fraction:.0%} of the plate; Canada's Food Guide suggests about half.")
        else:
            tips.append(f"Nice - colourful vegetables/fruit cover roughly {self.veg_fraction:.0%} of the plate.")
        if self.processed_level == "high":
            tips.append("This dish is often highly processed; the Food Guide suggests limiting highly processed foods.")
        if self.beige_fraction > 0.5:
            tips.append("Much of the plate is beige/fried foods; swapping part of it for vegetables or whole grains could help balance it.")
        return tips


def _lookup() -> dict:
    with open(resolve("data/nutrition/food_lookup.csv")) as f:
        return {r["label"]: r for r in csv.DictReader(f)}


def colour_fractions(img: Image.Image) -> tuple[float, float]:
    import numpy as np
    small = np.asarray(img.convert("RGB").resize((96, 96))).reshape(-1, 3)
    veg = beige = plate = 0
    for r, g, b in small.tolist():
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        if v > 0.92 and s < 0.08:          # white plate / background
            continue
        plate += 1
        if s > 0.35 and (0.17 < h < 0.45 or h < 0.03 or h > 0.92):     # greens, reds (veg/fruit)
            veg += 1
        elif 0.05 < h < 0.17 and s > 0.2:                                 # beige / brown / golden
            beige += 1
    plate = max(plate, 1)
    return veg / plate, beige / plate


class FoodAnalyzer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.backend = cfg.get("backend", "mock")
        self.table = _lookup()
        self.clf = None
        if self.backend == "hf":
            from transformers import pipeline
            self.clf = pipeline("image-classification", model=cfg["hf_model_id"])

    def classify(self, path: Path, img: Image.Image) -> tuple[str, float]:
        if self.clf is not None:
            top = self.clf(img, top_k=1)[0]
            return top["label"], float(top["score"])
        stem = path.stem.lower().split("__")[0]
        return (stem, 0.9) if stem in self.table else (stem, 0.3)

    def analyze(self, image_path: str | Path) -> MealAnalysis:
        path = Path(image_path)
        img = Image.open(path)
        label, conf = self.classify(path, img)
        veg, beige = colour_fractions(img)
        row = self.table.get(label)
        if row:
            kcal = float(row["kcal_per_100g"]) * float(row["typical_portion_g"]) / 100
            rng = (int(kcal * 0.6 // 10 * 10), int(kcal * 1.5 // 10 * 10))      # wide on purpose
            group, proc = row["food_group"], row["processed_level"]
        else:
            rng, group, proc = None, "unknown", "unknown"
        return MealAnalysis(label, round(conf, 3), group, rng, round(veg, 3), round(beige, 3), proc,
                            conf < self.cfg.get("min_confidence", 0.55), self.backend)
