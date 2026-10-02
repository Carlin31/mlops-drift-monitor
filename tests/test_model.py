import numpy as np

from app.core.model import (
    ensure_artifacts_exist,
    generate_artifacts,
    load_model,
    normalized_feature_names,
)


def test_normalized_feature_names_have_no_spaces():
    names = normalized_feature_names()

    assert len(names) == 30
    assert all(" " not in name for name in names)


def test_generate_artifacts_creates_model_and_reference_files(tmp_path):
    model_path = tmp_path / "model.pkl"
    reference_path = tmp_path / "reference.csv"

    generate_artifacts(model_path, reference_path)

    assert model_path.exists()
    assert reference_path.exists()

    model = load_model(model_path)
    prediction = model.predict(np.zeros((1, 30)))
    assert prediction.shape == (1,)


def test_ensure_artifacts_exist_skips_generation_when_present(tmp_path):
    model_path = tmp_path / "model.pkl"
    reference_path = tmp_path / "reference.csv"
    generate_artifacts(model_path, reference_path)
    original_mtime = model_path.stat().st_mtime

    ensure_artifacts_exist(model_path, reference_path)

    assert model_path.stat().st_mtime == original_mtime
