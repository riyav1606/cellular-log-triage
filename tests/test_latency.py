"""
Pytest performance test validating sub-millisecond per-log inference latency budget.
"""

import time
import pytest
from src.feature_extractor import CellularLogFeatureExtractor
from src.classifier import CellularLogClassifier

def test_sub_millisecond_inference_latency():
    extractor = CellularLogFeatureExtractor()
    sample_corpus = [
        "[PHY_LAYER][FAIL] PSS correlation peak below threshold at 2100 MHz Band n7.",
        "[RF_VAL][ERROR] Transmit EVM measurement 8.4% exceeds 3GPP limit.",
        "[L2_HARQ][CRITICAL] Maximum HARQ retransmissions exceeded on PDSCH process 2.",
        "[MODEM_CTRL][ERROR] AT+CFUN=1 command response timeout > 30000ms."
    ] * 20
    labels = ["PHY_SYNC_LOSS", "RF_EVM_DEGRADATION", "HARQ_RETX_EXHAUSTION", "AT_CMD_TIMEOUT"] * 20

    X = extractor.fit_transform(sample_corpus)
    clf = CellularLogClassifier(model_type="LinearSVC")
    clf.train(X, labels)

    test_log = "[PHY_LAYER][FAIL] PSS/SSS correlation lost at sample offset 64."
    
    # Warmup
    for _ in range(10):
        _ = clf.predict(extractor.transform([test_log]))

    N_ITER = 500
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        vec = extractor.transform([test_log])
        _ = clf.predict(vec)
    t1 = time.perf_counter()

    per_log_ms = ((t1 - t0) / N_ITER) * 1000
    print(f"Measured per-log inference latency: {per_log_ms:.3f} ms")
    assert per_log_ms < 1.0, f"Per-log inference latency must be < 1.0 ms (got {per_log_ms:.3f} ms)"
