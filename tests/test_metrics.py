import numpy as np

from app.observability.metrics import build_drift_report, compute_ks, compute_psi


def test_compute_psi_identical_distributions_is_near_zero():
    rng = np.random.default_rng(42)
    reference = rng.normal(loc=0, scale=1, size=1000)
    current = rng.normal(loc=0, scale=1, size=200)

    psi = compute_psi(reference, current, buckets=10)

    assert psi < 0.1


def test_compute_psi_shifted_distribution_exceeds_threshold():
    rng = np.random.default_rng(42)
    reference = rng.normal(loc=0, scale=1, size=1000)
    current = rng.normal(loc=5, scale=1, size=200)

    psi = compute_psi(reference, current, buckets=10)

    assert psi > 0.2


def test_compute_ks_identical_distributions_has_high_pvalue():
    rng = np.random.default_rng(42)
    reference = rng.normal(loc=0, scale=1, size=1000)
    current = rng.normal(loc=0, scale=1, size=200)

    _, pvalue = compute_ks(reference, current)

    assert pvalue > 0.05


def test_compute_ks_shifted_distribution_has_low_pvalue():
    rng = np.random.default_rng(42)
    reference = rng.normal(loc=0, scale=1, size=1000)
    current = rng.normal(loc=5, scale=1, size=200)

    _, pvalue = compute_ks(reference, current)

    assert pvalue < 0.05


def test_build_drift_report_contains_all_reference_columns():
    import pandas as pd

    reference_df = pd.DataFrame({"f1": np.linspace(0, 1, 100), "f2": np.linspace(10, 20, 100)})
    current_samples = [{"f1": 0.5, "f2": 15.0} for _ in range(40)]

    report = build_drift_report(
        reference_df,
        current_samples,
        psi_threshold=0.2,
        ks_pvalue_threshold=0.05,
        psi_buckets=10,
    )

    assert set(report["features"].keys()) == {"f1", "f2"}
    assert isinstance(report["overall_drift_detected"], bool)
    for feature_report in report["features"].values():
        assert {"psi", "ks_statistic", "ks_pvalue", "drift"} <= feature_report.keys()
