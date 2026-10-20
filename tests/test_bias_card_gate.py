"""The Clarify formulas on a table small enough to check by hand; the Model Card; the gate arithmetic."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from finrisk_jobs import bias, card, gate


def frame():
    """Facet A (reference, 8 rows): 4 frauds, flagged 4 (3 true). Facet B (4 rows): 1 fraud, flagged 1 (1 true)... built row by row."""
    rows = []
    for fraud, flagged in [(1, 1), (1, 1), (1, 1), (1, 0), (0, 1), (0, 0), (0, 0), (0, 0)]:      # A
        rows.append(("A", fraud, flagged))
    for fraud, flagged in [(1, 1), (0, 0), (0, 0), (0, 1)]:                                          # B
        rows.append(("B", fraud, flagged))
    df = pd.DataFrame(rows, columns=["transaction_type", "is_fraud", "flag"])
    df["amount"] = np.arange(len(df)) * 10.0
    return df


def test_clarify_numbers_by_hand():
    df = frame()
    report = bias.segment_report(df["transaction_type"], df["is_fraud"].to_numpy(), df["flag"].to_numpy())
    assert report["reference"] == "A"
    a, b = report["groups"]["A"], report["groups"]["B"]
    assert (a["rows"], a["frauds"], a["flagged"]) == (8, 4, 4) and a["prevalence"] == 0.5 and a["tpr"] == 0.75 and a["fpr"] == 0.25 and a["precision"] == 0.75
    assert (b["rows"], b["prevalence"], b["positive_rate"], b["tpr"], b["fpr"]) == (4, 0.25, 0.5, 1.0, 0.3333333333333333)
    assert b["ci"] == pytest.approx((8 - 4) / (8 + 4))                       # (n_a - n_d) / (n_a + n_d)
    assert b["dpl"] == pytest.approx(0.5 - 0.25)                             # q_a - q_d
    assert b["dppl"] == pytest.approx(0.5 - 0.5)                             # q'_a - q'_d
    assert b["di"] == pytest.approx(0.5 / 0.5)                               # q'_d / q'_a
    assert b["delta_tpr"] == pytest.approx(0.75 - 1.0) and b["delta_fpr"] == pytest.approx(0.25 - 1 / 3)


def test_undefined_values_are_none_not_nan():
    only_negatives = pd.DataFrame({"transaction_type": ["A"] * 3 + ["B"] * 2, "is_fraud": [0] * 5, "flag": [0, 0, 1, 0, 0]})
    report = bias.segment_report(only_negatives["transaction_type"], only_negatives["is_fraud"].to_numpy(), only_negatives["flag"].to_numpy())
    assert report["groups"]["B"]["tpr"] is None and report["groups"]["B"]["delta_tpr"] is None and report["groups"]["B"]["precision"] is None
    assert report["groups"]["B"]["di"] == 0.0                                  # B flags nothing, A flags a third: DI is a real 0, not undefined
    nothing_flagged = bias.segment_report(only_negatives["transaction_type"], only_negatives["is_fraud"].to_numpy(), np.zeros(5, dtype=int))
    assert nothing_flagged["groups"]["B"]["di"] is None                        # 0 / 0 is undefined
    json.dumps(report, allow_nan=False)                                       # strict JSON: no NaN anywhere


def test_amount_bands_are_the_training_quartiles():
    edges = bias.amount_band_edges(range(1, 101))
    assert edges == pytest.approx([25.75, 50.5, 75.25])
    labels = bias.assign_bands(pd.Series([1, 25.75, 26, 50.6, 99, 1e9]), edges)
    assert [l[:2] for l in labels] == ["Q1", "Q1", "Q2", "Q3", "Q4", "Q4"]


def test_bias_report_covers_both_segments_and_says_there_is_no_protected_attribute():
    df = frame()
    report = bias.bias_report(df.drop(columns="flag"), df["flag"].astype(float).to_numpy(), 0.5, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110])
    assert set(report["segments"]) == {"transaction_type", "amount_band"} and "no protected attribute" in report["note"]
    assert report["rows"] == 12 and report["frauds"] == 5 and set(report["definitions"]) == {"CI", "DPL", "DPPL", "DI", "delta_tpr", "delta_fpr"}
    assert report["highlights"] and "recall" in report["highlights"][0]
    json.dumps(report, allow_nan=False)


# ---------------------------------------------------------------- the card


def card_args(**over):
    candidate = {"pr_auc": 0.9, "roc_auc": 0.99, "precision": 0.5, "recall": 0.8, "threshold": 0.4, "validation_pr_auc": 0.88}
    base = dict(model_type="optimized", training_job="job-1", image="repo@sha256:" + "a" * 64, model_artifact="s3://b/m.tar.gz", hyperparameters={"max_depth": 5},
                cutoff="2026-09-19T23:00:00Z", dataset_uri="s3://b/datasets/e", splits={"train": {"rows": 100, "frauds": 2, "prevalence": 0.02, "days": [1, 13]},
                "validation": {"rows": 20, "days": [14, 15]}, "test": {"rows": 20, "frauds": 1, "days": [16, 17]}}, candidate=candidate, champion=None,
                gate={"floor": 0.7, "tolerance": 0.02, "champion_floor": 0.0}, shap_top=[{"feature": "amount", "mean_abs_shap": 1.2}], bias_highlights=["x"],
                test_window={"from": "a", "to": "b", "rows": 20, "frauds": 1})
    return {**base, **over}


def test_the_card_has_the_schema_sections_and_the_facts():
    c = card.build_model_card(**card_args())
    assert set(c) == {"model_overview", "intended_uses", "business_details", "training_details", "evaluation_details", "additional_information"}
    assert c["model_overview"]["problem_type"] == "Binary Classification" and "XGBoost" in c["model_overview"]["algorithm_type"]
    assert c["training_details"]["training_job_details"]["hyper_parameters"] == [{"name": "max_depth", "value": "5"}]
    assert c["additional_information"]["custom_details"]["cutoff"] == "2026-09-19T23:00:00Z" and "educational" in json.dumps(c).lower()
    assert "no champion" in c["evaluation_details"][0]["evaluation_observation"]
    names = [m["name"] for m in c["evaluation_details"][0]["metric_groups"][0]["metric_data"]]
    assert "PR-AUC (test)" in names and not any("Champion" in n for n in names)


def test_the_card_names_the_champion_when_there_is_one_and_stays_under_the_limit():
    c = card.build_model_card(**card_args(champion={"version": 3, "pr_auc": 0.85}))
    assert "version 3" in c["evaluation_details"][0]["evaluation_observation"]
    text = card.card_content(c)
    assert json.loads(text) == c and len(text) < card.MAX_CARD_CHARS and "\n" not in text
    with pytest.raises(ValueError, match="limit"):
        card.card_content({"x": "y" * 100_001})


# ---------------------------------------------------------------- the gate


def test_gate_passes_above_the_floor_with_no_champion():
    checks = gate.gate_checks(0.75, 0.70, 0.02, None)
    assert all(c.passed for c in checks) and [c.name for c in checks] == ["floor", "champion"] and checks[1].skipped


def test_gate_fails_below_the_floor_even_with_no_champion():
    assert not gate.gate_checks(0.69, 0.70, 0.02, None)[0].passed


def test_the_floor_itself_passes_and_so_does_champion_minus_tolerance():
    assert gate.gate_checks(0.70, 0.70, 0.02, None)[0].passed
    assert all(c.passed for c in gate.gate_checks(0.78, 0.70, 0.02, 0.80))               # exactly champion - tolerance, despite float noise
    assert not all(c.passed for c in gate.gate_checks(0.77, 0.70, 0.02, 0.80))


def test_champion_floor_is_rounded_down_so_float_noise_cannot_fail_the_condition_step():
    assert gate.champion_floor(None, 0.02) == 0.0
    assert gate.champion_floor(0.80, 0.02) == 0.78 and 0.78 >= gate.champion_floor(0.80, 0.02)
    assert gate.champion_floor(0.123456789, 0.02) <= 0.123456789 - 0.02 and gate.champion_floor(0.01, 0.02) == 0.0


def test_the_table_reads_back_from_an_evaluation_json():
    ev = {"candidate": {"pr_auc": 0.77}, "champion": {"version": 3, "pr_auc": 0.80}, "gate": {"floor": 0.7, "tolerance": 0.02, "champion_floor": 0.78}}
    lines = gate.gate_table("job", 0.77, gate.checks_from_evaluation(ev))
    assert "champion v3 (0.8000) - 0.02" in lines[2] + (lines[3] if len(lines) > 3 else "") and "FAIL" in "\n".join(lines) and "PASS" in "\n".join(lines)
