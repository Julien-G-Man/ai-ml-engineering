"""Metrics used by the evaluation harness.

These are deterministic heuristics. They are useful as regression signals, but
they are not a substitute for human review or a model-based factuality judge.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


class EvaluationMetrics:
    @staticmethod
    def accuracy(results: list[dict[str, Any]]) -> float:
        return sum(bool(result.get("passed")) for result in results) / len(results) if results else 0.0

    @staticmethod
    def pass_rate(results: list[dict[str, Any]], category: str | None = None) -> float:
        selected = [result for result in results if category is None or result.get("category") == category]
        return EvaluationMetrics.accuracy(selected)

    @staticmethod
    def severity_failures(results: list[dict[str, Any]], severity: str) -> int:
        return sum(
            1 for result in results
            if not result.get("passed") and result.get("severity") == severity
        )

    @staticmethod
    def token_overlap(output: str, context: str) -> float:
        output_words = set(re.findall(r"\b\w+\b", output.casefold()))
        context_words = set(re.findall(r"\b\w+\b", context.casefold()))
        return len(output_words & context_words) / len(output_words) if output_words else 1.0

    @staticmethod
    def check_hallucination(output: str, context: str, threshold: float = 0.7) -> bool:
        """Return whether output has unusually low token overlap with context.

        This is a cheap regression signal, not a claim that the response is
        factually false. Use a claim-level or human review for factuality.
        """
        return EvaluationMetrics.token_overlap(output, context) < (1.0 - threshold)

    @staticmethod
    def relevance_score(query: str, response: str) -> float:
        query_words = set(re.findall(r"\b\w+\b", query.casefold()))
        response_words = set(re.findall(r"\b\w+\b", response.casefold()))
        union = query_words | response_words
        return len(query_words & response_words) / len(union) if union else 0.0

    @staticmethod
    def word_frequencies(text: str) -> Counter[str]:
        return Counter(re.findall(r"\b\w+\b", text.casefold()))

    @staticmethod
    def quality_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
        categories = sorted({result.get("category", "uncategorized") for result in results})
        return {
            "total": len(results),
            "passed": sum(bool(result.get("passed")) for result in results),
            "failed": sum(not bool(result.get("passed")) for result in results),
            "accuracy": EvaluationMetrics.accuracy(results),
            "by_category": {
                category: EvaluationMetrics.pass_rate(results, category)
                for category in categories
            },
            "critical_failures": EvaluationMetrics.severity_failures(results, "critical"),
            "high_failures": EvaluationMetrics.severity_failures(results, "high"),
        }
