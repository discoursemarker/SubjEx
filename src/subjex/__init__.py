"""SubjEx: rule-based English subject extraction and classification."""

from .classifier import classify_subjects, get_pipeline

__all__ = ["classify_subjects", "get_pipeline"]
__version__ = "0.1.0"
