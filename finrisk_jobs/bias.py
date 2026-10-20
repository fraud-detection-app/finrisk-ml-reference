"""Bias metrics: the standard SageMaker Clarify formulas, computed in pandas.

SageMaker Clarify (bias and explainability jobs) closed to new customers on 30 Jul 2026, so there is no service to call.
The formulas are public and short; this file is all of them. The vocabulary:

    facet a  the reference group (here: the LARGEST segment, the group the model has seen most)
    facet d  the group being compared with it (every other segment)
    q        the share of a facet whose LABEL is positive (real fraud)          q'  the share the MODEL flags

    pre-training (on the labels only)
      CI    class imbalance                 (n_a - n_d) / (n_a + n_d)
      DPL   difference in proportions of labels   q_a - q_d
    post-training (needs the model's decisions)
      DPPL  difference in positive proportions in predicted labels   q'_a - q'_d
      DI    disparate impact                q'_d / q'_a        (0.8-1.25 is the "four-fifths rule" band)
      dTPR  recall difference (equal opportunity)   TPR_a - TPR_d
      dFPR  specificity difference                  FPR_a - FPR_d   (= Clarify's SD, since specificity = 1 - FPR)

PaySim has NO protected attribute (no age, gender, region ...). So the class sees the MECHANICS on business segments:
the transaction type and the amount quartile. The numbers say "the model treats these groups differently", not "it is unfair".
Nothing here decides anything; a person reads it (and it travels with the model in the registry).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

NOTE = ("PaySim has no protected attribute (no age, gender or region). The segments below are business segments, chosen to "
        "show how the Clarify formulas work. A large gap means 'the model treats these groups differently', not 'it is unfair'.")
DEFINITIONS = {
    "CI": "(n_a - n_d) / (n_a + n_d)  - class imbalance between the reference facet a and the facet d",
    "DPL": "q_a - q_d  - difference in the proportion of positive LABELS (pre-training)",
    "DPPL": "q'_a - q'_d  - difference in the proportion of positive PREDICTIONS (post-training)",
    "DI": "q'_d / q'_a  - disparate impact; 0.8 to 1.25 is the usual four-fifths band",
    "delta_tpr": "TPR_a - TPR_d  - recall difference (equal opportunity)",
    "delta_fpr": "FPR_a - FPR_d  - specificity difference",
}


def _num(value: Any) -> float | None:
    """A JSON-safe float: NaN and inf become None."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _ratio(a: float, b: float) -> float | None:
    return _num(a / b) if b else None


def amount_band_edges(train_amounts: Sequence[float] | pd.Series) -> list[float]:
    """The 25 / 50 / 75 % quantiles of the TRAINING amounts: the band limits stay fixed when later traffic changes."""
    values = np.asarray(pd.Series(train_amounts).dropna(), dtype=float)
    if values.size == 0:
        raise ValueError("no training amounts to take quartiles from")
    edges = [float(x) for x in np.quantile(values, [0.25, 0.5, 0.75])]
    return edges


def band_labels(edges: Sequence[float]) -> list[str]:
    e = list(edges)
    return [f"Q1 (<= {e[0]:,.0f})", f"Q2 ({e[0]:,.0f} - {e[1]:,.0f})", f"Q3 ({e[1]:,.0f} - {e[2]:,.0f})", f"Q4 (> {e[2]:,.0f})"]


def assign_bands(amounts: pd.Series, edges: Sequence[float]) -> pd.Series:
    labels = band_labels(edges)
    codes = np.searchsorted(np.asarray(edges, dtype=float), amounts.to_numpy(dtype=float), side="left")
    return pd.Series([labels[min(int(c), 3)] for c in codes], index=amounts.index, name="amount_band")


def segment_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    n = int(len(y_true))
    pos, flagged = int(y_true.sum()), int(y_pred.sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    neg = n - pos
    return {"rows": n, "frauds": pos, "prevalence": _ratio(pos, n), "flagged": flagged, "positive_rate": _ratio(flagged, n),
            "tpr": _ratio(tp, pos), "fpr": _ratio(fp, neg), "precision": _ratio(tp, flagged)}


def clarify_vs_reference(d: dict[str, Any], a: dict[str, Any]) -> dict[str, float | None]:
    """The six numbers for facet d against the reference facet a (both from segment_stats)."""
    def diff(x: float | None, y: float | None) -> float | None:
        return None if x is None or y is None else _num(x - y)

    return {
        "ci": _ratio(a["rows"] - d["rows"], a["rows"] + d["rows"]),
        "dpl": diff(a["prevalence"], d["prevalence"]),
        "dppl": diff(a["positive_rate"], d["positive_rate"]),
        "di": _ratio(d["positive_rate"], a["positive_rate"]) if d["positive_rate"] is not None and a["positive_rate"] else None,
        "delta_tpr": diff(a["tpr"], d["tpr"]),
        "delta_fpr": diff(a["fpr"], d["fpr"]),
    }


def segment_report(groups: pd.Series, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    stats = {str(name): segment_stats(y_true[idx], y_pred[idx]) for name, idx in groups.groupby(groups).indices.items()}
    if not stats:
        return {"reference": None, "groups": {}}
    reference = max(stats, key=lambda k: stats[k]["rows"])
    out = {}
    for name, s in sorted(stats.items(), key=lambda kv: -kv[1]["rows"]):
        out[name] = {**s, **({} if name == reference else clarify_vs_reference(s, stats[reference]))}
    return {"reference": reference, "groups": out}


def bias_report(frame: pd.DataFrame, y_score: np.ndarray, threshold: float, train_amounts: Sequence[float] | pd.Series,
                target: str = "is_fraud", segments: Sequence[str] = ("transaction_type", "amount_band")) -> dict[str, Any]:
    """The whole bias.json: per segment the group table and the six Clarify numbers against the largest group."""
    frame = frame.reset_index(drop=True)
    edges = amount_band_edges(train_amounts)
    frame["amount_band"] = assign_bands(frame["amount"], edges)
    y_true = frame[target].to_numpy(dtype=int)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    report: dict[str, Any] = {
        "note": NOTE, "definitions": DEFINITIONS, "reference_rule": "the largest segment (by rows) is facet a",
        "threshold": float(threshold), "rows": int(len(frame)), "frauds": int(y_true.sum()),
        "amount_band_edges": edges, "amount_bands": band_labels(edges), "segments": {},
    }
    for name in segments:
        report["segments"][name] = segment_report(frame[name], y_true, y_pred)
    report["highlights"] = highlights(report)
    return report


def highlights(report: dict[str, Any], top: int = 3) -> list[str]:
    """Plain sentences for the model card: the biggest gaps in recall (dTPR), the thing a fraud team cares about."""
    found = []
    for seg, body in report["segments"].items():
        for group, row in body["groups"].items():
            if group != body["reference"] and row.get("delta_tpr") is not None:
                found.append((abs(row["delta_tpr"]), seg, group, body["reference"], row))
    found.sort(key=lambda t: -t[0])
    sentences = []
    for _, seg, group, ref, row in found[:top]:
        direction = "lower" if row["delta_tpr"] > 0 else "higher"
        text = (f"{seg}={group}: recall {_pct(row['tpr'])} is {abs(row['delta_tpr']) * 100:.1f} points {direction} "
                f"than the reference {ref}")
        if row.get("di") is not None:
            text += f" (disparate impact {row['di']:.2f})"
        sentences.append(text)
    return sentences


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f} %"
