"""The gate, as a pure function. The candidate passes only if EVERY check passes.

    1. candidate >= floor                      an absolute minimum, whatever runs today
    2. candidate >= champion - tolerance       not noticeably worse than the model in production, measured on the
                                               SAME test days (skipped when there is no champion yet)

Two places use it: the Evaluate job computes `champion_floor` for evaluation.json (which the pipeline's Condition step
compares with), and `mlops.py run` prints the table at the end of a pipeline execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Check:
    name: str
    needs: str            # the rule in words, e.g. ">= 0.7000"
    passed: bool
    skipped: bool = False


def champion_floor(champion_pr_auc: float | None, tolerance: float) -> float:
    """champion - tolerance, rounded DOWN to 6 decimals (so 0.78 passes against 0.80 - 0.02 despite float noise);
    0.0 when there is no champion (the Condition step then compares with a number every score clears)."""
    if champion_pr_auc is None:
        return 0.0
    return max(0.0, math.floor((champion_pr_auc - tolerance) * 1e6) / 1e6)


def gate_checks(candidate: float, floor: float, tolerance: float, champion: float | None,
                champion_label: str = "champion") -> list[Check]:
    eps = 1e-9
    checks = [Check("floor", f">= {floor:.4f}", candidate + eps >= floor)]
    if champion is None:
        checks.append(Check("champion", "no champion yet", True, skipped=True))
    else:
        needed = champion - tolerance
        checks.append(Check(f"{champion_label} - {tolerance:g}", f">= {needed:.4f}", candidate + eps >= needed))
    return checks


def gate_table(job: str, candidate: float, checks: list[Check]) -> list[str]:
    lines = [f"  gate for {job}  (candidate PR-AUC {candidate:.4f})",
             f"  {'check':<28} {'candidate':>9}   {'needs':<18} result"]
    for c in checks:
        shown = "-" if c.skipped else f"{candidate:.4f}"
        result = "skipped" if c.skipped else ("PASS" if c.passed else "FAIL")
        lines.append(f"  {c.name:<28} {shown:>9}   {c.needs:<18} {result}")
    return lines


def checks_from_evaluation(evaluation: dict) -> list[Check]:
    """The gate re-read from an evaluation.json (what the Evaluate step wrote)."""
    gate = evaluation["gate"]
    champion = evaluation.get("champion")
    label = f"champion v{champion['version']} ({champion['pr_auc']:.4f})" if champion else "champion"
    return gate_checks(float(evaluation["candidate"]["pr_auc"]), float(gate["floor"]), float(gate["tolerance"]),
                       float(champion["pr_auc"]) if champion else None, champion_label=label)
