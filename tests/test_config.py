from app.core.config import Settings


def test_default_settings_values():
    settings = Settings()

    assert settings.window_size == 100
    assert settings.min_samples_for_drift == 30
    assert settings.psi_buckets == 10
    assert settings.psi_threshold == 0.2
    assert settings.ks_pvalue_threshold == 0.05
    assert settings.model_path == "models/model.pkl"
    assert settings.reference_data_path == "data/reference.csv"
