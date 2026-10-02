import numpy as np
from scipy.stats import ks_2samp

EPSILON = 1e-6


def compute_psi(reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
    quantiles = np.linspace(0, 1, buckets + 1)
    breakpoints = np.unique(np.quantile(reference, quantiles))
    if len(breakpoints) < 2:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_props = ref_counts / len(reference) + EPSILON
    cur_props = cur_counts / len(current) + EPSILON

    return float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))


def compute_ks(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    result = ks_2samp(reference, current)
    return float(result.statistic), float(result.pvalue)


def build_drift_report(
    reference_df,
    current_samples: list[dict[str, float]],
    psi_threshold: float,
    ks_pvalue_threshold: float,
    psi_buckets: int,
) -> dict:
    features_report = {}
    overall_drift = False

    for feature in reference_df.columns:
        reference_values = reference_df[feature].to_numpy()
        current_values = np.array([sample[feature] for sample in current_samples])

        psi = compute_psi(reference_values, current_values, buckets=psi_buckets)
        ks_statistic, ks_pvalue = compute_ks(reference_values, current_values)
        drift = psi > psi_threshold or ks_pvalue < ks_pvalue_threshold

        features_report[feature] = {
            "psi": round(psi, 4),
            "ks_statistic": round(ks_statistic, 4),
            "ks_pvalue": round(ks_pvalue, 4),
            "drift": drift,
        }
        overall_drift = overall_drift or drift

    return {"overall_drift_detected": overall_drift, "features": features_report}
