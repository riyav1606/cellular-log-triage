"""
Feature extraction & text preprocessing pipeline for 3GPP cellular validation test logs.
"""

import re
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse

class CellularLogFeatureExtractor:
    """Extracts n-gram TF-IDF representations while preserving key protocol tokens."""

    def __init__(self, max_features: int = 1200, ngram_range: Tuple[int, int] = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
            token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z0-9_]*\b"
        )
        self.is_fitted = False

    @staticmethod
    def preprocess_log(text: str) -> str:
        """Normalizes hexadecimal registers, timestamps, and floating-point measurements."""
        text = re.sub(r"0x[0-9a-fA-F]+", "HEX_ADDR", text)
        text = re.sub(r"[-+]?\d+\.\d+", "NUM_FLOAT", text)
        text = re.sub(r"\b\d+\b", "NUM_INT", text)
        return text.lower().strip()

    def fit_transform(self, raw_logs: List[str]) -> scipy.sparse.csr_matrix:
        cleaned = [self.preprocess_log(log) for log in raw_logs]
        X = self.vectorizer.fit_transform(cleaned)
        self.is_fitted = True
        return X

    def transform(self, raw_logs: List[str]) -> scipy.sparse.csr_matrix:
        if not self.is_fitted:
            raise RuntimeError("Feature extractor must be fitted before transform.")
        cleaned = [self.preprocess_log(log) for log in raw_logs]
        return self.vectorizer.transform(cleaned)
