from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42


def normalized_feature_names() -> list[str]:
    data = load_breast_cancer()
    return [name.replace(" ", "_") for name in data.feature_names]


def generate_artifacts(model_path: Path, reference_path: Path) -> None:
    data = load_breast_cancer()
    feature_names = normalized_feature_names()
    reference_df = pd.DataFrame(data.data, columns=feature_names)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(data.data, data.target)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    reference_df.to_csv(reference_path, index=False)


def ensure_artifacts_exist(model_path: Path, reference_path: Path) -> None:
    if not model_path.exists() or not reference_path.exists():
        generate_artifacts(model_path, reference_path)


def load_model(model_path: Path):
    return joblib.load(model_path)
