"""
Pytest unit tests for regex baseline and telemetry extraction.
"""

from src.parser import RegexTriageBaseline

def test_regex_baseline_matching():
    assert RegexTriageBaseline.predict("[PHY_LAYER][FAIL] PSS detection lost") == "PHY_SYNC_LOSS"
    assert RegexTriageBaseline.predict("[RF_VAL][ERROR] EVM 8.5% exceeds limit") == "RF_EVM_DEGRADATION"
    assert RegexTriageBaseline.predict("[TEST][INFO] Completed successfully") == "VALIDATION_PASS"

def test_telemetry_extraction():
    log = "[PHY_LAYER] SNR=-8.4 dB, EVM=4.20%, Band n78 carrier lock."
    telemetry = RegexTriageBaseline.extract_telemetry(log)
    assert telemetry.get("snr_db") == -8.4
    assert telemetry.get("evm_pct") == 4.20
    assert telemetry.get("band") == 78
