"""Evaluation suite with safety gates and a human-readable report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

try:
    from ..evaluation.guard_rails import GuardrailPipeline
    from ..evaluation.metrics import EvaluationMetrics
    from .test_case import TestCase, test_cases
except ImportError:
    from evaluation.guard_rails import GuardrailPipeline
    from evaluation.metrics import EvaluationMetrics
    from test_case import TestCase, test_cases


@dataclass(slots=True)
class ReleaseGate:
    min_accuracy: float = 0.8
    max_critical_failures: int = 0
    max_high_failures: int = 0
    min_category_accuracy: dict[str, float] | None = None

    def evaluate(self, summary: dict[str, Any]) -> tuple[bool, list[str]]:
        failures: list[str] = []
        if summary["accuracy"] < self.min_accuracy:
            failures.append(f"accuracy {summary['accuracy']:.1%} < {self.min_accuracy:.1%}")
        if summary["critical_failures"] > self.max_critical_failures:
            failures.append("critical safety failures exceeded the release limit")
        if summary["high_failures"] > self.max_high_failures:
            failures.append("high-severity failures exceeded the release limit")
        for category, threshold in (self.min_category_accuracy or {}).items():
            actual = summary["by_category"].get(category, 0.0)
            if actual < threshold:
                failures.append(f"{category} accuracy {actual:.1%} < {threshold:.1%}")
        return not failures, failures


class AITestSuite:
    def __init__(
        self,
        name: str,
        *,
        guardrails: GuardrailPipeline | None = None,
        release_gate: ReleaseGate | None = None,
    ) -> None:
        self.name = name
        self.test_cases: list[TestCase] = []
        self.results: list[dict[str, Any]] = []
        self.guardrails = guardrails or GuardrailPipeline()
        self.release_gate = release_gate or ReleaseGate()

    def add_test(self, test_case: TestCase) -> None:
        self.test_cases.append(test_case)

    def run_tests(self, ai_function: Callable[[str], str]) -> dict[str, Any]:
        self.results.clear()
        for test in self.test_cases:
            input_decision = self.guardrails.inspect(test.input)
            if not input_decision.allowed:
                self.results.append({
                    **test.evaluate(""),
                    "passed": test.category.value == "security",
                    "reason": f"input blocked: {', '.join(input_decision.reasons)}",
                    "guardrail": input_decision,
                })
                continue
            try:
                actual = ai_function(input_decision.text)
            except Exception as exc:
                self.results.append({
                    **test.evaluate(""),
                    "passed": False,
                    "reason": f"AI function failed: {type(exc).__name__}",
                    "error": str(exc),
                })
                continue
            output_decision = self.guardrails.inspect_output(actual)
            result = test.evaluate(output_decision.text)
            result["guardrail"] = output_decision
            if not output_decision.allowed:
                result["passed"] = False
                result["reason"] = f"output blocked: {', '.join(output_decision.reasons)}"
            self.results.append(result)

        summary = EvaluationMetrics.quality_summary(self.results)
        release_passed, release_failures = self.release_gate.evaluate(summary)
        return {**summary, "release_passed": release_passed, "release_failures": release_failures}

    def report(self) -> None:
        summary = EvaluationMetrics.quality_summary(self.results)
        print(f"\n{'=' * 60}\nTest Suite: {self.name}\n{'=' * 60}")
        for index, result in enumerate(self.results, 1):
            status = "PASS" if result["passed"] else "FAIL"
            actual = result.get("actual", "")
            print(f"\nTest {index}: {status} [{result['category']}/{result['severity']}]")
            print(f"  Input: {result['input']}")
            print(f"  Actual: {actual[:120]}")
            print(f"  Reason: {result['reason']}")
        print(f"\nAccuracy: {summary['accuracy']:.1%}")
        print(f"Passed: {summary['passed']}/{summary['total']}")


def mock_ai(query: str) -> str:
    if "no relevant" in query:
        return "I cannot answer because no relevant context was retrieved."
    if "system prompt" in query:
        return "I cannot reveal protected instructions."
    return (
        "Answer based on the retrieved context. "
        "The retrieved context is the only allowed source of factual claims.\n\n"
        "Sources:\nhttps://example.com/video"
    )


if __name__ == "__main__":
    suite = AITestSuite("RAG Agent Evaluations")
    for case in test_cases:
        suite.add_test(case)
    results = suite.run_tests(mock_ai)
    suite.report()
    print(f"\nRelease gate: {'PASS' if results['release_passed'] else 'FAIL'}")
