"""Small, dependency-free guardrail pipeline for evaluation and demos."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class Action(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    REDACT = "redact"
    ESCALATE = "escalate"


@dataclass(slots=True)
class GuardrailDecision:
    allowed: bool
    action: Action
    text: str
    reasons: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    risk_score: float = 0.0


@dataclass(frozen=True, slots=True)
class Guardrail:
    name: str
    category: str
    pattern: re.Pattern[str]
    action: Action
    reason: str
    score: float
    replacement: str = "[REDACTED]"

    def inspect(self, text: str) -> tuple[bool, str]:
        return self.pattern.search(text) is not None, text


def _guardrails() -> tuple[Guardrail, ...]:
    return (
        Guardrail(
            "prompt_injection",
            "security",
            re.compile(r"(ignore|disregard|forget)\s+(all\s+)?previous instructions", re.I),
            Action.BLOCK,
            "possible prompt injection",
            0.95,
        ),
        Guardrail(
            "system_prompt_extraction",
            "security",
            re.compile(r"(reveal|show|print|repeat).{0,30}(system prompt|hidden instructions)", re.I),
            Action.BLOCK,
            "system prompt extraction attempt",
            0.95,
        ),
        Guardrail(
            "email",
            "privacy",
            re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
            Action.REDACT,
            "email address detected",
            0.65,
        ),
        Guardrail(
            "phone",
            "privacy",
            re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{8,}\d)(?!\w)"),
            Action.REDACT,
            "phone number detected",
            0.65,
        ),
        Guardrail(
            "secret",
            "privacy",
            re.compile(r"\b(?:sk-[A-Za-z0-9_-]{10,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b"),
            Action.BLOCK,
            "credential-like secret detected",
            1.0,
        ),
        Guardrail(
            "unsafe_tool_request",
            "security",
            re.compile(r"\b(delete|drop|destroy|transfer money)\b.{0,80}\b(without|bypass|approval)\b", re.I),
            Action.ESCALATE,
            "high-impact action requires human approval",
            0.9,
        ),
    )


class GuardrailPipeline:
    def __init__(self, guards: tuple[Guardrail, ...] | None = None) -> None:
        self.guards = guards or _guardrails()

    def inspect(self, text: str) -> GuardrailDecision:
        if not isinstance(text, str) or not text.strip():
            return GuardrailDecision(False, Action.BLOCK, "", ["empty input"], ["validation"], 1.0)

        current = text
        reasons: list[str] = []
        categories: list[str] = []
        risk = 0.0
        final_action = Action.ALLOW

        for guard in self.guards:
            matched, _ = guard.inspect(current)
            if not matched:
                continue
            reasons.append(guard.reason)
            if guard.category not in categories:
                categories.append(guard.category)
            risk = max(risk, guard.score)
            if guard.action == Action.REDACT and final_action == Action.ALLOW:
                current = guard.pattern.sub(guard.replacement, current)
                final_action = Action.REDACT
            elif guard.action in (Action.BLOCK, Action.ESCALATE):
                final_action = guard.action

        return GuardrailDecision(
            allowed=final_action in (Action.ALLOW, Action.REDACT),
            action=final_action,
            text=current,
            reasons=reasons,
            categories=categories,
            risk_score=risk,
        )

    def inspect_output(self, text: str) -> GuardrailDecision:
        decision = self.inspect(text)
        if decision.action == Action.REDACT:
            return decision
        return decision


def guardrail_check(text: str) -> GuardrailDecision:
    return GuardrailPipeline().inspect(text)
