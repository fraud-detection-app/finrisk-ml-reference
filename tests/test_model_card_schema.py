"""The Model Card is checked against AWS's OWN schema before CreateModelPackage ever sees it.

A live pipeline run got through BuildDataset, Tune, Evaluate and the Gate, and then Register failed with "The ModelCardContent JSON isn't
valid": one `type` key too many in training_metrics. The official schema is the CloudFormation resource schema of AWS::SageMaker::ModelCard
(fetched with `aws cloudformation describe-type --type RESOURCE --type-name AWS::SageMaker::ModelCard --query Schema`, vendored in
tests/fixtures/model_card_cfn_schema.json). CloudFormation spells the properties PascalCase; the card content is snake_case. The
converter below only renames properties and `required` lists, keeps every `$ref`, and validates with Draft 7 over the `Content` definition.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest
from conftest import needs_app
from test_jobs import built, packages  # noqa: F401 - the module-scoped fixtures that run the real build-dataset / train jobs

jsonschema = pytest.importorskip("jsonschema")

from finrisk_jobs import card  # noqa: E402

SCHEMA_FILE = Path(__file__).parent / "fixtures" / "model_card_cfn_schema.json"


def snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def convert(node):
    """PascalCase property names and required lists -> snake_case; everything else (including $ref targets) untouched."""
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key == "properties":
                out[key] = {snake(p): convert(sub) for p, sub in value.items()}
            elif key == "required":
                out[key] = [snake(p) for p in value]
            elif key == "$ref":
                out[key] = value
            else:
                out[key] = convert(value)
        return out
    if isinstance(node, list):
        return [convert(x) for x in node]
    return node


def validator():
    definitions = convert(json.loads(SCHEMA_FILE.read_text())["definitions"])
    schema = {"$schema": "http://json-schema.org/draft-07/schema#", "definitions": definitions, **definitions["Content"]}
    return jsonschema.Draft7Validator(schema)


def problems(content) -> list[str]:
    return [f"{'/'.join(str(p) for p in e.path) or '(root)'}: {e.message[:160]}" for e in sorted(validator().iter_errors(content), key=lambda e: list(e.path))]


# ---------------------------------------------------------------- the validator itself works


def test_the_vendored_schema_is_the_official_one_and_holds_no_account_id():
    text = SCHEMA_FILE.read_text()
    schema = json.loads(text)
    assert "ModelCard" in json.dumps(schema)[:2000] or "Content" in schema["definitions"]
    assert {"Content", "TrainingMetric", "SimpleMetric", "TrainingDetails"} <= set(schema["definitions"])
    assert not re.search(r"(?<!\d)\d{12}(?!\d)", text)


def good_args(**over):
    candidate = {"pr_auc": 0.9541, "roc_auc": 0.99, "precision": 0.5, "recall": 0.8, "threshold": 0.4, "validation_pr_auc": 0.9572}
    base = dict(model_type="optimized", training_job="finrisk-tune-20261020-003-abcd1234", image="123.dkr.ecr.ap-southeast-2.amazonaws.com/finrisk-train@sha256:" + "a" * 64,
                model_artifact="s3://b/models/job/output/model.tar.gz", hyperparameters={"max_depth": 5, "learning_rate": 0.0912, "model_type": "optimized"},
                cutoff="2026-09-19T23:00:00Z", dataset_uri="s3://b/datasets/e",
                splits={"train": {"rows": 4_000_000, "frauds": 3500, "prevalence": 0.0009, "days": [1, 13]}, "validation": {"rows": 800_000, "days": [14, 15]},
                        "test": {"rows": 800_000, "frauds": 600, "days": [16, 17]}},
                candidate=candidate, champion=None, gate={"floor": 0.7, "tolerance": 0.02, "champion_floor": 0.0},
                shap_top=[{"feature": "origin_new_balance", "mean_abs_shap": 2.1}, {"feature": "amount", "mean_abs_shap": 1.2}],
                bias_highlights=["transaction_type=TRANSFER: recall 90.1 % is 3.2 points lower than the reference CASH_OUT (disparate impact 0.80)"],
                test_window={"from": "2026-09-16T00:00:00Z", "to": "2026-09-17T23:00:00Z", "rows": 800_000, "frauds": 600},
                mlflow_run_id="abc123", execution_arn="arn:aws:sagemaker:ap-southeast-2:123456789012:pipeline/finrisk-train/execution/xyz")
    return {**base, **over}


def test_a_good_card_has_no_violations():
    assert problems(card.build_model_card(**good_args())) == []
    assert problems(card.build_model_card(**good_args(champion={"version": 3, "pr_auc": 0.93}))) == []


def test_the_validator_catches_the_bug_the_live_run_hit_a_type_key_in_training_metrics():
    bad = card.build_model_card(**good_args())
    bad["training_details"]["training_job_details"]["training_metrics"][0]["type"] = "number"      # what the card used to contain
    found = problems(bad)
    assert len(found) == 1 and "training_metrics/0" in found[0] and "type" in found[0]


@pytest.mark.parametrize("mutate, where", [
    (lambda c: c["intended_uses"].__setitem__("risk_rating", "Moderate"), "risk_rating"),
    (lambda c: c["model_overview"].__setitem__("surprise", 1), "surprise"),
    (lambda c: c["training_details"]["objective_function"]["function"].__setitem__("condition", "x" * 64), "condition"),
    (lambda c: c["evaluation_details"][0]["metric_groups"][0]["metric_data"][0].__setitem__("value", None), "metric_data"),
    (lambda c: c["evaluation_details"][0]["metric_groups"][0].__setitem__("name", ""), "metric_groups"),   # the schema's name patterns are unanchored: only an EMPTY name fails
])
def test_the_validator_is_strict_enough_to_notice_other_mistakes(mutate, where):
    bad = card.build_model_card(**good_args())
    mutate(bad)
    found = problems(bad)
    assert found and any(where in line for line in found)


# ---------------------------------------------------------------- the real card, from the real evaluate job


@needs_app
def test_the_card_the_evaluate_job_writes_is_valid(tmp_path, built, packages):
    """Both copies: model_card.json and the string the Register step sends to CreateModelPackage (register_info.json)."""
    from test_jobs import clients, evaluate_args, make_inputs

    from finrisk_jobs import evaluate

    for with_champion in (False, True):
        out = tmp_path / ("champ" if with_champion else "first") / "evaluation"
        clients(champion_tar=packages["baseline"]["tar"] if with_champion else None, version=3)
        evaluate.run(evaluate_args(make_inputs(built, packages, out.parent), out))
        written = json.loads((out / "model_card.json").read_text())
        registered = json.loads(json.loads((out / "register_info.json").read_text())["model_card_content"])
        assert problems(written) == [] and problems(registered) == [] and written == registered
        assert "type" not in written["training_details"]["training_job_details"]["training_metrics"][0]


# ---------------------------------------------------------------- limits: clip, never fail


def test_every_free_text_field_is_clipped_to_its_limit_with_an_ellipsis():
    long = "x" * 5000
    args = good_args(training_job=long, image=long, model_artifact=long, dataset_uri=long, bias_highlights=[long, long],
                     hyperparameters={"k" * 400: long, "short": long}, cutoff=long, execution_arn=long, mlflow_run_id=long,
                     shap_top=[{"feature": long, "mean_abs_shap": 1.0}])
    content = card.build_model_card(**args)
    assert problems(content) == []
    assert content["training_details"]["training_job_details"]["training_arn"].endswith("…")
    assert len(content["training_details"]["training_job_details"]["training_arn"]) == 1024
    assert len(content["training_details"]["training_observations"]) <= 1024
    assert len(content["additional_information"]["ethical_considerations"]) <= 2048
    assert all(len(v) <= 1024 for v in content["additional_information"]["custom_details"].values())
    hp = content["training_details"]["training_job_details"]["hyper_parameters"]
    assert all(1 <= len(h["name"]) <= 255 and 1 <= len(h["value"]) <= 255 for h in hp)
    assert len(card.card_content(content)) < card.MAX_CARD_CHARS


def test_clip_leaves_short_text_alone_and_marks_cut_text():
    assert card.clip("short", 63) == "short" and card.clip("a" * 63, 63) == "a" * 63
    cut = card.clip("a" * 100, 63)
    assert len(cut) == 63 and cut.endswith("…")


def test_missing_numbers_make_no_metric_instead_of_a_null_value():
    candidate = {"pr_auc": 0.9, "roc_auc": None, "precision": None, "recall": None, "threshold": None, "validation_pr_auc": None}
    content = card.build_model_card(**good_args(candidate=candidate))
    assert problems(content) == []
    assert content["training_details"]["training_job_details"]["training_metrics"] == []
    names = [m["name"] for m in content["evaluation_details"][0]["metric_groups"][0]["metric_data"]]
    assert names == ["PR-AUC (test)"]


def test_training_metrics_have_only_name_and_value_evaluation_metrics_also_a_type():
    content = card.build_model_card(**good_args())
    (training,) = content["training_details"]["training_job_details"]["training_metrics"]
    assert set(training) == {"name", "value"} and training["value"] == 0.9572
    for metric in content["evaluation_details"][0]["metric_groups"][0]["metric_data"]:
        assert set(metric) == {"name", "type", "value"} and metric["type"] == "number"


def test_too_many_items_are_cut_to_the_limits():
    assert card.MAX_ITEMS == {"model_artifact": 15, "container_image": 15, "training_datasets": 15, "datasets": 10}
