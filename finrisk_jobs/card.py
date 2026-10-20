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


# The limits of the official schema (CloudFormation AWS::SageMaker::ModelCard, vendored in tests/fixtures/model_card_cfn_schema.json).
# A card that breaks one is refused by CreateModelPackage ("The ModelCardContent JSON isn't valid") -- a pipeline run that got all the
# way to Register would fail on a comma. So every free-text field is clipped to its limit (with an ellipsis) instead of failing.
LIMITS = {"model_description": 1024, "model_owner": 1024, "model_creator": 1024, "algorithm_type": 1024, "problem_type": 1024, "artifact": 1024,
          "purpose_of_model": 2048, "intended_uses": 2048, "factors": 2048, "risk_explanation": 2048, "business": 2048,
          "condition": 63, "facet": 63, "objective_notes": 1024, "training_observations": 1024, "training_arn": 1024, "dataset": 1024,
          "container_image": 1024, "hyper_name": 255, "hyper_value": 255, "metric_name": 255, "notes": 1024, "evaluation_name": 63,
          "evaluation_observation": 2096, "metric_group_name": 63, "ethical": 2048, "caveats": 2048, "custom": 1024}
MAX_ITEMS = {"model_artifact": 15, "container_image": 15, "training_datasets": 15, "datasets": 10}


def clip(text: Any, limit: int) -> str:
    """text as a string of at most `limit` characters; a longer one ends with an ellipsis."""
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"


def _metric(name: str, value: float | None, kind: bool = True) -> dict[str, Any] | None:
    """An EVALUATION metric {name, type, value}; with kind=False a TRAINING metric {name, value}: the schema allows only
    name / notes / value there, and `type` only on evaluation metric_data. A missing value makes no metric (value must be a number)."""
    if value is None:
        return None
    metric: dict[str, Any] = {"name": clip(name, LIMITS["metric_name"]) or "metric"}
    if kind:
        metric["type"] = "number"
    metric["value"] = round(float(value), 6)
    return metric


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
    metrics = [m for m in metrics if m]
    training_metrics = [m for m in [_metric("PR-AUC (validation)", candidate.get("validation_pr_auc"), kind=False)] if m]
    L = LIMITS
    ethical = "PaySim has no protected attributes, so fairness cannot be judged here; bias.json shows the Clarify formulas on business segments. " \
        + "; ".join(bias_highlights)
    return {
        "model_overview": {
            "model_description": clip(f"A fraud-risk classifier for mobile-money transactions: {algorithm}. It scores one transaction "
                                      "and says how likely it is to be fraud; an analyst decides what to do with a flagged one.", L["model_description"]),
            "model_creator": clip("finrisk-ml pipeline (Zhitongguigu VIP course)", L["model_creator"]),
            "model_artifact": [clip(model_artifact, L["artifact"])][: MAX_ITEMS["model_artifact"]],
            "algorithm_type": clip(algorithm, L["algorithm_type"]), "problem_type": "Binary Classification",
            "model_owner": clip("Fraud analytics team (course project)", L["model_owner"]),
        },
        "intended_uses": {
            "purpose_of_model": clip("Rank incoming transactions by fraud risk so analysts review the riskiest first.", L["purpose_of_model"]),
            "intended_uses": clip("Decision support on PaySim-like transactions inside this course project. Not a decision-maker: "
                                  "flags go to a person.", L["intended_uses"]),
            "factors_affecting_model_efficiency": clip("The fraud rate in live traffic (prior shift) and the hour-of-day pattern of "
                                                       "the traffic. Both moved in the simulated go-live; the monitor watches them.", L["factors"]),
            "risk_rating": "Medium",
            "explanations_for_risk_rating": clip("Synthetic data and a teaching setting, but the model gates a (simulated) money flow.", L["risk_explanation"]),
        },
        "business_details": {
            "business_problem": clip("Catch fraudulent transfers and cash-outs early without flooding analysts with false alarms.", L["business"]),
            "business_stakeholders": clip("Fraud analysts, risk managers, the engineering team that runs the service.", L["business"]),
            "line_of_business": clip("Payments / mobile money (simulated)", L["business"]),
        },
        "training_details": {
            "objective_function": {"function": {"function": "Minimize", "facet": clip("Loss", L["facet"]),
                                                "condition": clip("log-loss, scale_pos_weight = negatives / positives", L["condition"])},
                                   "notes": clip("Hyperparameters chosen by SageMaker Automatic Model Tuning to MAXIMISE PR-AUC on the validation days.", L["objective_notes"])},
            "training_observations": clip(f"Point-in-time dataset as of {cutoff}. Train {train.get('rows', 0):,} rows (days {_days(train)}, "
                                          f"{train.get('frauds', 0):,} frauds), validation {valid.get('rows', 0):,} rows (days {_days(valid)}). "
                                          f"Fraud prevalence in train {100 * train.get('prevalence', 0):.3f} %.", L["training_observations"]),
            "training_job_details": {
                "training_arn": clip(training_job, L["training_arn"]), "training_datasets": [clip(dataset_uri, L["dataset"])][: MAX_ITEMS["training_datasets"]],
                "training_environment": {"container_image": [clip(image, L["container_image"])][: MAX_ITEMS["container_image"]]},
                "training_metrics": training_metrics,
                "user_provided_training_metrics": [],
                "hyper_parameters": [{"name": clip(k, L["hyper_name"]) or "?", "value": clip(v, L["hyper_value"]) or "-"} for k, v in hyperparameters.items()],
            },
        },
        "evaluation_details": [{
            "name": clip("Held-out test days", L["evaluation_name"]),
            "evaluation_observation": clip(f"{verdict} Gate: floor {gate['floor']}, tolerance {gate['tolerance']}.", L["evaluation_observation"]),
            "datasets": [clip(f"{dataset_uri}/test/ ({test.get('rows', 0):,} rows, {test.get('frauds', 0):,} frauds, days {_days(test)})", L["dataset"])][: MAX_ITEMS["datasets"]],
            "metric_groups": [{"name": clip("Classification", L["metric_group_name"]), "metric_data": metrics}],
        }],
        "additional_information": {
            "ethical_considerations": clip(ethical, L["ethical"]),
            "caveats_and_recommendations": clip("Educational use, synthetic data. Re-evaluate whenever the monitor reports drift or the fraud "
                                                "rate changes; never promote without the gate and a human approval.", L["caveats"]),
            "custom_details": {
                clip(k, 255): clip(v, L["custom"]) for k, v in {
                    "top_features_by_mean_abs_shap": top_features, "cutoff": cutoff, "test_window": json.dumps(test_window),
                    "mlflow_run_id": mlflow_run_id or "", "pipeline_execution_arn": execution_arn or "", "data": "synthetic (PaySim); educational"}.items()},
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
