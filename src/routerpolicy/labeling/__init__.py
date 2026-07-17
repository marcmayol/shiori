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
from routerpolicy.labeling.pipeline import (
    LabelingReport,
    build_report,
    done_task_ids,
    label_mode,
    run_mode_labeling,
    run_sufficiency_labeling,
)
from routerpolicy.labeling.records import ModeRecord, SufficiencyRecord

__all__ = [
    "JUDGE_RUBRIC",
    "AgreementReport",
    "FakeJudge",
    "JudgeError",
    "LabelingReport",
    "LlmModeJudge",
    "ModeJudge",
    "ModeRecord",
    "SufficiencyRecord",
    "agreement_report",
    "build_judge_prompt",
    "build_report",
    "done_task_ids",
    "is_short_factual",
    "label_mode",
    "parse_judge_output",
    "rule_mode_for_chat_task",
    "rule_mode_for_tool_task",
    "run_mode_labeling",
    "run_sufficiency_labeling",
]
