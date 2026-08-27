"""Task Relevance baseline module for CWRU 4-class bearing fault diagnosis."""

from src.relevance.relevance import relevance_from_class, relevance_from_probabilities

__all__ = ["relevance_from_class", "relevance_from_probabilities"]
