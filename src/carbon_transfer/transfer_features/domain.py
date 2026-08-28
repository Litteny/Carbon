from __future__ import annotations

import hashlib
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _fold(cell_id: str, folds: int) -> int:
    digest = hashlib.sha256(str(cell_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def domain_classification(
    panel: pd.DataFrame,
    feature: str,
    *,
    folds: int = 5,
    max_samples_per_city: int = 5000,
    seed: int = 42,
) -> Dict:
    samples = []
    for city, city_frame in panel.groupby("city_id", sort=True, observed=True):
        if len(city_frame) > max_samples_per_city:
            city_frame = city_frame.sample(max_samples_per_city, random_state=seed)
        samples.append(city_frame[["city_id", "cell_id", feature]])
    data = pd.concat(samples, ignore_index=True)
    data["fold"] = data["cell_id"].map(lambda value: _fold(str(value), folds))
    predictions = np.empty(len(data), dtype=object)
    valid = np.zeros(len(data), dtype=bool)
    for fold in range(folds):
        test = data["fold"].to_numpy() == fold
        train = ~test
        if not test.any() or data.loc[train, "city_id"].nunique() < 2:
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=300, class_weight="balanced", random_state=seed),
        )
        model.fit(data.loc[train, [feature]], data.loc[train, "city_id"].astype(str))
        predictions[test] = model.predict(data.loc[test, [feature]])
        valid[test] = True
    accuracy = float(balanced_accuracy_score(
        data.loc[valid, "city_id"].astype(str), predictions[valid],
    ))
    chance = 1.0 / data["city_id"].nunique()
    invariance = float(np.clip(1.0 - (accuracy - chance) / (1.0 - chance), 0.0, 1.0))
    return {
        "feature": feature,
        "domain_balanced_accuracy": accuracy,
        "domain_chance_accuracy": chance,
        "domain_invariance_score": invariance,
        "domain_samples": int(valid.sum()),
        "domain_folds": folds,
    }
