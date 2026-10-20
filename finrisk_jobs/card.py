"""The Model Card: what a model is, for whom, how it was made and how good it is -- one JSON document.

The content follows the SageMaker Model Card schema (model_overview, intended_uses, business_details, training_details,
evaluation_details, additional_information). It is written to model_card.json by the Evaluate step, and ALSO stored in the
registry with the model version (CreateModelPackage `ModelCard`, status Draft). approve-model.yml moves the status:
Approved when a person approves the version, Archived when they reject it.

The card is honest about what it is: synthetic data, a teaching project.
"""

from __future__ import annotations

import json
from typing import Any

MAX_CARD_CHARS = 100_000        # ModelCardContent limit of CreateModelPackage / UpdateModelPackage


def _metric(name: str, value: float | None) -> dict[str, Any]:
    return {"name": name, "type": "number", "value": None if value is None else round(float(value), 6)}


def build_model_card(*, model_type: str, training_job: str, image: str, model_artifact: str, hyperparameters: dict[str, Any],
                     cutoff: str, dataset_uri: str, splits: dict[str, dict], candidate: dict[str, Any],
                     champion: dict[str, Any] | None, gate: dict[str, Any], shap_top: list[dict[str, Any]],
                     bias_highlights: list[str], test_window: dict[str, Any], mlflow_run_id: str | None = None,
                     execution_arn: str | None = None) -> dict[str, Any]:
    algorithm = "XGBoost (gradient-boosted trees)" if model_type == "optimized" else "Logistic regression"
    train, valid, test = (splits.get(k, {}) for k in ("train", "validation", "test"))
    top_features = ", ".join(f"{s['feature']} ({s['mean_abs_shap']:.3f})" for s in shap_top[:5]) or "not computed"
    verdict = (f"PR-AUC {candidate['pr_auc']:.4f} on the test days"
               + (f" against {champion['pr_auc']:.4f} for the champion (version {champion['version']}) on the same days." if champion
                  else "; there was no champion to compare with."))
    metrics = [_metric("PR-AUC (test)", candidate["pr_auc"]), _metric("ROC-AUC (test)", candidate.get("roc_auc")),
               _metric("Precision at threshold (test)", candidate.get("precision")), _metric("Recall at threshold (test)", candidate.get("recall")),
               _metric("Decision threshold", candidate.get("threshold"))]
    if champion:
        metrics.append(_metric(f"Champion v{champion['version']} PR-AUC (same test days)", champion["pr_auc"]))
    return {
        "model_overview": {
            "model_description": f"A fraud-risk classifier for mobile-money transactions: {algorithm}. It scores one transaction "
                                 "and says how likely it is to be fraud; an analyst decides what to do with a flagged one.",
            "model_creator": "finrisk-ml pipeline (Zhitongguigu VIP course)",
            "model_artifact": [model_artifact],
            "algorithm_type": algorithm, "problem_type": "Binary Classification",
            "model_owner": "Fraud analytics team (course project)",
        },
        "intended_uses": {
            "purpose_of_model": "Rank incoming transactions by fraud risk so analysts review the riskiest first.",
            "intended_uses": "Decision support on PaySim-like transactions inside this course project. Not a decision-maker: "
                             "flags go to a person.",
            "factors_affecting_model_efficiency": "The fraud rate in live traffic (prior shift) and the hour-of-day pattern of "
                                                  "the traffic. Both moved in the simulated go-live; the monitor watches them.",
            "risk_rating": "Medium",
            "explanations_for_risk_rating": "Synthetic data and a teaching setting, but the model gates a (simulated) money flow.",
        },
        "business_details": {
            "business_problem": "Catch fraudulent transfers and cash-outs early without flooding analysts with false alarms.",
            "business_stakeholders": "Fraud analysts, risk managers, the engineering team that runs the service.",
            "line_of_business": "Payments / mobile money (simulated)",
        },
        "training_details": {
            "objective_function": {"function": {"function": "Minimize", "facet": "Loss", "condition": "binary log-loss with scale_pos_weight = negatives / positives"},
                                   "notes": "Hyperparameters chosen by SageMaker Automatic Model Tuning to MAXIMISE PR-AUC on the validation days."},
            "training_observations": f"Point-in-time dataset as of {cutoff}. Train {train.get('rows', 0):,} rows (days {_days(train)}, "
                                     f"{train.get('frauds', 0):,} frauds), validation {valid.get('rows', 0):,} rows (days {_days(valid)}). "
                                     f"Fraud prevalence in train {100 * train.get('prevalence', 0):.3f} %.",
            "training_job_details": {
                "training_arn": training_job, "training_datasets": [dataset_uri],
                "training_environment": {"container_image": [image]},
                "training_metrics": [_metric("PR-AUC (validation)", candidate.get("validation_pr_auc"))],
                "user_provided_training_metrics": [],
                "hyper_parameters": [{"name": str(k), "value": str(v)} for k, v in hyperparameters.items()],
            },
        },
        "evaluation_details": [{
            "name": "Held-out test days",
            "evaluation_observation": f"{verdict} Gate: floor {gate['floor']}, tolerance {gate['tolerance']}.",
            "datasets": [f"{dataset_uri}/test/ ({test.get('rows', 0):,} rows, {test.get('frauds', 0):,} frauds, days {_days(test)})"],
            "metric_groups": [{"name": "Classification", "metric_data": metrics}],
        }],
        "additional_information": {
            "ethical_considerations": "PaySim has no protected attributes, so fairness cannot be judged here; bias.json shows the "
                                      "Clarify formulas on business segments. " + ("; ".join(bias_highlights) if bias_highlights else ""),
            "caveats_and_recommendations": "Educational use, synthetic data. Re-evaluate whenever the monitor reports drift or the fraud "
                                           "rate changes; never promote without the gate and a human approval.",
            "custom_details": {
                "top_features_by_mean_abs_shap": top_features,
                "cutoff": cutoff, "test_window": json.dumps(test_window), "mlflow_run_id": mlflow_run_id or "",
                "pipeline_execution_arn": execution_arn or "",
                "data": "synthetic (PaySim); educational",
            },
        },
    }


def _days(split: dict[str, Any]) -> str:
    d = split.get("days") or []
    return f"{d[0]}-{d[1]}" if len(d) == 2 else "?"


def card_content(card: dict[str, Any]) -> str:
    """The card as the string CreateModelPackage / UpdateModelPackage take (compact JSON, <= 100 000 characters)."""
    text = json.dumps(card, separators=(",", ":"), ensure_ascii=True)
    if len(text) > MAX_CARD_CHARS:
        raise ValueError(f"the model card is {len(text)} characters; the limit is {MAX_CARD_CHARS}")
    return text
