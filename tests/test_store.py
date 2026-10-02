from app.observability.store import MetricsStore


def test_store_starts_empty():
    store = MetricsStore()

    report, last_updated = store.get()

    assert report is None
    assert last_updated is None


def test_set_and_get_returns_report_with_timestamp():
    store = MetricsStore()

    store.set({"overall_drift_detected": False, "features": {}})
    report, last_updated = store.get()

    assert report == {"overall_drift_detected": False, "features": {}}
    assert last_updated is not None


def test_reset_clears_report():
    store = MetricsStore()
    store.set({"overall_drift_detected": True, "features": {}})

    store.reset()
    report, last_updated = store.get()

    assert report is None
    assert last_updated is None
