import re
from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """Base class for all evaluation metrics."""

    @staticmethod
    def tokenize(text: str):
        """Tokenize text: strip punctuation and convert to lowercase. Shared for BLEU & ROUGE."""
        tokens = re.sub(r'[,;]', ' ', str(text)).split()
        return [t.strip('.,;:').lower() for t in tokens if t.strip('.,;:')]

    @staticmethod
    def split_symptoms(text: str):
        """Split multi-label text into a list of symptoms (each symptom corresponds to one class)."""
        return [s.strip() for s in str(text).split(',') if s.strip()]

    @abstractmethod
    def compute_sentence(self, ground_truth: str, pred: str):
        """Compute metric for a single pair of (ground_truth, pred). Returns a dictionary."""
        raise NotImplementedError

    @abstractmethod
    def compute_for_type(self, df_type, dtype: str = None):
        """Compute average metric for a given disease type. Returns a dictionary."""
        raise NotImplementedError

    def name(self) -> str:
        """Metric name (used for logging and reporting)."""
        return self.__class__.__name__
