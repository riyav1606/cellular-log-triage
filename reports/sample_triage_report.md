# 📊 Cellular Validation Test Failure Triage Diagnostic Summary

**Generated:** 2026-10-26 18:03:13 UTC  
**Validation Suite:** 3GPP Rel-16/17 RF & Baseband Conformance  

---

## 🚀 Executive Benchmark Summary

| Evaluation Metric | Rule-Based Regex Baseline | ML LinearSVC (NLP Pipeline) | Gain |
| :--- | :---: | :---: | :---: |
| **Overall Classification Accuracy** | **84.29%** | **100.00%** | **+15.71%** |
| **Weighted F1-Score** | 0.8429 | **1.0000** | **+0.1571** |
| **5-Fold Cross-Validation F1** | N/A | **1.0000 (±0.0000)** | Robust |
| **Per-Log Inference Latency** | ~0.08 ms | **<0.37 ms** | Real-time |

---

## 🔍 Root-Cause Diagnostic Reference

- **`PHY_SYNC_LOSS`**: Doppler shift exceeding AFC loop limits or low SNR below sensitivity floor.
- **`RACH_PREAMBLE_FAIL`**: Modem transmit power ramp limit reached without gNB preamble acknowledgement (RAR).
- **`RRC_SETUP_REJECT`**: Radio Resource Control establishment rejected due to PLMN cell congestion or cell barring.
- **`RF_EVM_DEGRADATION`**: Power Amplifier (PA) compression/non-linearity causing constellation distortion & spectral regrowth.
- **`HARQ_RETX_EXHAUSTION`**: Severe channel fading inducing continuous NACK responses on PDSCH/PUSCH transport blocks.
- **`AT_CMD_TIMEOUT`**: Baseband modem control interface deadlock or serial UART buffer overrun.
- **`VALIDATION_PASS`**: Test case completed with zero assertions. BLER, EVM, and throughput within 3GPP margin.
