"""labeling: etiquetado de modo (reglas + juez) y control de calidad."""

from routerpolicy.labeling.agreement import AgreementReport, agreement_report
from routerpolicy.labeling.judge import (
    JUDGE_RUBRIC,
    FakeJudge,
    JudgeError,
    LlmModeJudge,
    ModeJudge,
    build_judge_prompt,
    parse_judge_output,
)
from routerpolicy.labeling.mode_rules import (
    is_short_factual,
    rule_mode_for_chat_task,
    rule_mode_for_tool_task,
)

__all__ = [
    "JUDGE_RUBRIC",
    "AgreementReport",
    "FakeJudge",
    "JudgeError",
    "LlmModeJudge",
    "ModeJudge",
    "agreement_report",
    "build_judge_prompt",
    "is_short_factual",
    "parse_judge_output",
    "rule_mode_for_chat_task",
    "rule_mode_for_tool_task",
]
