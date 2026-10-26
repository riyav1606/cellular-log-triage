"""
Rule-based baseline regex parser and metadata extractor for cellular validation test logs.
"""

import re
from typing import Dict, Any, Optional

REGEX_RULES = {
    "PHY_SYNC_LOSS": r"(pss|sss|frame sync|cell search|doppler)",
    "RACH_PREAMBLE_FAIL": r"(rach|prach|preamble|rar|contention resolution)",
    "RRC_SETUP_REJECT": r"(rrc|rrcsetup|cell barring|earfcn|srb0)",
    "RF_EVM_DEGRADATION": r"(evm|aclr|power amplifier|constellation error|iq imbalance)",
    "HARQ_RETX_EXHAUSTION": r"(harq|nack|turbo|ldpc|crc error|transport block)",
    "AT_CMD_TIMEOUT": r"(at\+|uart|modem daemon|ttyusb|urc)",
    "VALIDATION_PASS": r"(completed successfully|pass|zero assertions)"
}

class RegexTriageBaseline:
    """Traditional rule-based regex parser used as an evaluation comparator."""

    @staticmethod
    def predict(log_message: str) -> str:
        lower_msg = log_message.lower()
        for label, pattern in REGEX_RULES.items():
            if re.search(pattern, lower_msg):
                return label
        return "UNKNOWN_FAILURE"

    @staticmethod
    def extract_telemetry(log_message: str) -> Dict[str, Any]:
        """Extracts RF and physical layer parameters from semi-structured log strings."""
        telemetry = {}
        snr_match = re.search(r"SNR=([-+]?\d+\.\d+)", log_message)
        if snr_match:
            telemetry["snr_db"] = float(snr_match.group(1))

        evm_match = re.search(r"EVM=(\d+\.\d+)%", log_message)
        if evm_match:
            telemetry["evm_pct"] = float(evm_match.group(1))

        band_match = re.search(r"Band\s+n?(\d+)", log_message, re.IGNORECASE)
        if band_match:
            telemetry["band"] = int(band_match.group(1))

        return telemetry
