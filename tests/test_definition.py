"""The two pipeline definitions: a golden test of the DAG, consistency of every reference, and every step's Arguments checked
against botocore's input shapes (required members present, no unknown members). Dev-only SDK parity test at the end."""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import botocore.session
import pytest
import yaml
from botocore.validate import validate_parameters
from fakes import BUCKET, IMAGE, ROLE, check_shape

import definition as d

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "train.yaml").read_text())
SESSION = botocore.session.get_session()
ARN = f"arn:aws:sagemaker:ap-southeast-2:123456789012:mlflow-app/app-x"


@pytest.fixture(scope="module")
def settings():
    return d.Settings(BUCKET, ROLE, "ap-southeast-2", IMAGE, CFG, ARN, "c" * 40)


@pytest.fixture(scope="module")
def train(settings):
    return d.train_definition(settings)


@pytest.fixture(scope="module")
def monitor(settings):
    return d.monitor_definition(settings)


def by_name(definition):
    return {s["Name"]: s for s in d.walk_steps(definition)}


# ---------------------------------------------------------------- golden: the DAG


def test_train_pipeline_steps_types_and_dependencies(train):
    steps = by_name(train)
    assert [s["Name"] for s in train["Steps"]] == ["BuildDataset", "Tune", "Evaluate", "Gate"]
    assert {n: s["Type"] for n, s in steps.items()} == {"BuildDataset": "Processing", "Tune": "Tuning", "Evaluate": "Processing", "Gate": "Condition",
                                                       "Register": "RegisterModel", "GateFailed": "Fail"}
    assert d.dependencies(steps["BuildDataset"]) == set()
    assert d.dependencies(steps["Tune"]) == {"BuildDataset"}
    assert d.dependencies(steps["Evaluate"]) == {"BuildDataset", "Tune"}
    assert "Evaluate" in d.dependencies(steps["Gate"])
    assert steps["Register"]["_branch"] == "if" and steps["GateFailed"]["_branch"] == "else"


def test_parameters_are_exactly_the_spec_list(train):
    assert [(p["Name"], p["Type"], p["DefaultValue"]) for p in train["Parameters"]] == [
        ("Cutoff", "String", "auto"), ("TriggerReason", "String", "manual"), ("ModelType", "String", "optimized"), ("MaxTuningJobs", "Integer", 8),
        ("ParallelTuningJobs", "Integer", 4), ("GateFloor", "Float", 0.70), ("GateTolerance", "Float", 0.02), ("TrainImage", "String", IMAGE)]
    assert train["Version"] == "2020-12-01"


def test_the_gate_is_a_condition_on_the_floor_and_the_champion_floor(train):
    gate = by_name(train)["Gate"]["Arguments"]
    left = {"Std:JsonGet": {"PropertyFile": {"Get": "Steps.Evaluate.PropertyFiles.EvaluationReport"}, "Path": "candidate.pr_auc"}}
    assert gate["Conditions"] == [
        {"Type": "GreaterThanOrEqualTo", "LeftValue": left, "RightValue": {"Get": "Parameters.GateFloor"}},
        {"Type": "GreaterThanOrEqualTo", "LeftValue": left,
         "RightValue": {"Std:JsonGet": {"PropertyFile": {"Get": "Steps.Evaluate.PropertyFiles.EvaluationReport"}, "Path": "gate.champion_floor"}}}]
    assert [s["Name"] for s in gate["IfSteps"]] == ["Register"] and [s["Name"] for s in gate["ElseSteps"]] == ["GateFailed"]


def test_property_files(train):
    steps = by_name(train)
    assert steps["BuildDataset"]["PropertyFiles"] == [{"PropertyFileName": "DatasetReport", "OutputName": "dataset", "FilePath": "dataset.json"}]
    assert steps["Evaluate"]["PropertyFiles"] == [{"PropertyFileName": "EvaluationReport", "OutputName": "evaluation", "FilePath": "evaluation.json"},
                                                  {"PropertyFileName": "RegisterInfo", "OutputName": "evaluation", "FilePath": "register_info.json"}]


def test_tuning_step_matches_the_spec(train):
    args = by_name(train)["Tune"]["Arguments"]
    config = args["HyperParameterTuningJobConfig"]
    assert config["Strategy"] == "Bayesian" and config["HyperParameterTuningJobObjective"] == {"Type": "Maximize", "MetricName": "validation:pr_auc"}
    assert config["ResourceLimits"] == {"MaxNumberOfTrainingJobs": {"Get": "Parameters.MaxTuningJobs"}, "MaxParallelTrainingJobs": {"Get": "Parameters.ParallelTuningJobs"}}
    ints = {r["Name"]: r for r in config["ParameterRanges"]["IntegerParameterRanges"]}
    floats = {r["Name"]: r for r in config["ParameterRanges"]["ContinuousParameterRanges"]}
    assert set(ints) == {"max_depth", "n_estimators", "min_child_weight"} and set(floats) == {"learning_rate", "subsample", "colsample_bytree"}
    assert (ints["max_depth"]["MinValue"], ints["max_depth"]["MaxValue"]) == ("3", "8") and floats["learning_rate"]["ScalingType"] == "Logarithmic"
    assert (floats["learning_rate"]["MinValue"], floats["learning_rate"]["MaxValue"]) == ("0.02", "0.3")
    definition = args["TrainingJobDefinition"]
    assert definition["StaticHyperParameters"] == {"model_type": {"Get": "Parameters.ModelType"}}
    assert definition["AlgorithmSpecification"]["TrainingImage"] == {"Get": "Parameters.TrainImage"}
    assert definition["StoppingCondition"] == {"MaxRuntimeInSeconds": 1800} and definition["ResourceConfig"]["InstanceType"] == "ml.m5.xlarge"
    assert [c["ChannelName"] for c in definition["InputDataConfig"]] == ["train", "validation"]
    names = {m["Name"]: m["Regex"] for m in definition["AlgorithmSpecification"]["MetricDefinitions"]}
    assert names["validation:pr_auc"] == r"validation:pr_auc=([0-9\.]+)" and names["train:pr_auc"] == r"train:pr_auc=([0-9\.]+)"
    assert definition["Environment"]["MLFLOW_TRACKING_ARN"] == ARN and definition["Environment"]["ML_BUCKET"] == BUCKET
    assert "ProcessingJobName" not in json.dumps(args) and "HyperParameterTuningJobName" not in args


def test_evaluate_scores_the_top_trial_the_pipeline_picks(train):
    step = by_name(train)["Evaluate"]
    arguments = step["Arguments"]["AppSpecification"]["ContainerArguments"]
    assert arguments[:3] == ["evaluate", "--candidate-job", {"Get": "Steps.Tune.TrainingJobSummaries[0].TrainingJobName"}]
    model_input = next(i for i in step["Arguments"]["ProcessingInputs"] if i["InputName"] == "model")
    assert model_input["S3Input"]["S3Uri"]["Std:Join"]["Values"] == [f"s3://{BUCKET}/models/", {"Get": "Steps.Tune.TrainingJobSummaries[0].TrainingJobName"},
                                                                     "/output/model.tar.gz"]
    assert {i["InputName"] for i in step["Arguments"]["ProcessingInputs"]} == {"test", "profile", "report", "model"}
    assert step["Arguments"]["AppSpecification"]["ContainerEntrypoint"] == ["python", "-m", "finrisk_jobs"]
    assert step["Arguments"]["AppSpecification"]["ImageUri"] == {"Get": "Parameters.TrainImage"}


def test_register_step_matches_the_spec(train):
    args = by_name(train)["Register"]["Arguments"]
    assert args["ModelPackageGroupName"] == "finrisk-fraud-model" and args["ModelApprovalStatus"] == "PendingManualApproval"
    assert args["InferenceSpecification"]["SupportedRealtimeInferenceInstanceTypes"] == ["ml.t2.medium", "ml.m5.large"]
    assert args["InferenceSpecification"]["SupportedContentTypes"] == ["application/json"]
    assert set(args["ModelMetrics"]) == {"ModelQuality", "Bias", "Explainability"}
    assert args["ModelMetrics"]["ModelQuality"]["Statistics"]["ContentType"] == "application/json"
    assert args["ModelCard"]["ModelCardStatus"] == "Draft" and "Std:JsonGet" in args["ModelCard"]["ModelCardContent"]
    assert set(args["CustomerMetadataProperties"]) >= {"train_image_digest", "cutoff", "dataset_uri", "baseline_uri", "pipeline_execution_arn", "ml_git_sha",
                                                       "trigger_reason", "mlflow_run_id", "champion_version", "pr_auc", "champion_pr_auc"}
    assert args["CustomerMetadataProperties"]["ml_git_sha"] == "c" * 40


def test_the_fail_step_message_comes_from_the_evaluate_step(train):
    step = by_name(train)["GateFailed"]
    assert step["Arguments"]["ErrorMessage"]["Std:JsonGet"]["Path"] == "fail_message"


def test_monitor_pipeline(monitor):
    assert [p["Name"] for p in monitor["Parameters"]] == ["Env", "WindowHours", "Reference", "TrainImage"]
    assert [(p["Type"], p["DefaultValue"]) for p in monitor["Parameters"][:3]] == [("String", "prod"), ("Integer", 24), ("String", "auto")]
    (step,) = monitor["Steps"]
    assert step["Name"] == "Monitor" and step["Type"] == "Processing"
    args = step["Arguments"]["AppSpecification"]["ContainerArguments"]
    assert args[:4] == ["monitor", "--env", {"Get": "Parameters.Env"}, "--window-hours"] and {"Get": "Parameters.Reference"} in args


# ---------------------------------------------------------------- consistency: every reference points at something real


def all_values(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from all_values(v)
    elif isinstance(node, list):
        for v in node:
            yield from all_values(v)


@pytest.mark.parametrize("which", ["train", "monitor"])
def test_every_reference_resolves(which, train, monitor):
    definition = train if which == "train" else monitor
    steps = by_name(definition)
    params = {p["Name"] for p in definition["Parameters"]}
    order = [s["Name"] for s in d.walk_steps(definition)]
    for step in steps.values():
        outputs = {o["OutputName"] for o in step["Arguments"].get("ProcessingOutputConfig", {}).get("Outputs", [])} if "Arguments" in step else set()
        for path in d.references(step.get("Arguments", {})):
            if path.startswith("Parameters."):
                assert path.split(".")[1] in params, path
            elif path.startswith("Execution."):
                assert path in ("Execution.PipelineExecutionId", "Execution.PipelineExecutionArn"), path
            else:
                assert path.startswith("Steps."), path
                target = path.split(".")[1]
                assert target in steps and order.index(target) < order.index(step["Name"]), f"{step['Name']} reads {path}"
                m = re.search(r"Outputs\['(\w+)'\]", path)
                if m:
                    assert m.group(1) in {o["OutputName"] for o in steps[target]["Arguments"]["ProcessingOutputConfig"]["Outputs"]}
        for node in all_values(step.get("Arguments", {})):
            if "Std:JsonGet" in node:                      # the property file exists on that step and its output is declared
                target, file_name = node["Std:JsonGet"]["PropertyFile"]["Get"].split(".")[1], node["Std:JsonGet"]["PropertyFile"]["Get"].split(".")[3]
                declared = {f["PropertyFileName"]: f for f in steps[target]["PropertyFiles"]}
                assert file_name in declared
                assert declared[file_name]["OutputName"] in {o["OutputName"] for o in steps[target]["Arguments"]["ProcessingOutputConfig"]["Outputs"]}


def test_json_gets_read_paths_that_the_evaluate_job_really_writes(train):
    """The Path of every JsonGet must exist in what finrisk_jobs/evaluate.py writes (names are the contract between the two)."""
    written = {"evaluation.json": {"candidate.pr_auc", "gate.champion_floor", "champion.version"},
               "register_info.json": {"fail_message", "model_card_content", "training_job", "train_image_digest", "cutoff", "dataset_uri", "baseline_uri",
                                      "pipeline_execution_arn", "ml_git_sha", "trigger_reason", "mlflow_run_id", "champion_version", "pr_auc", "champion_pr_auc"}}
    files = {f["PropertyFileName"]: f["FilePath"] for s in d.walk_steps(train) for f in s.get("PropertyFiles", [])}
    for step in d.walk_steps(train):
        for node in all_values(step.get("Arguments", {})):
            if "Std:JsonGet" in node and node["Std:JsonGet"]["PropertyFile"]["Get"].split(".")[1] == "Evaluate":
                file = files[node["Std:JsonGet"]["PropertyFile"]["Get"].split(".")[3]]
                assert node["Std:JsonGet"]["Path"] in written[file], node


def test_the_definition_is_json_and_small(train, monitor):
    for definition in (train, monitor):
        assert len(d.to_json(definition)) < 100_000 and json.loads(d.to_json(definition)) == definition


def test_no_account_id_in_the_definition_except_the_ones_it_is_given(settings):
    text = d.to_json(d.train_definition(d.Settings("bucket-x", "arn:aws:iam::999999999999:role/r", "ap-southeast-2", "repo@sha256:" + "a" * 64, CFG)))
    assert text.count("999999999999") == 3                     # role in: BuildDataset, Evaluate, Tune


# ---------------------------------------------------------------- botocore: every step's Arguments is a valid AWS request


def concrete(value, params):
    """Replace the references the service resolves at run time with a value of the type the API field needs."""
    if isinstance(value, dict):
        if set(value) == {"Get"}:
            path = value["Get"]
            if path.startswith("Parameters."):
                return {"Integer": 3, "Float": 0.5, "String": "text"}[params[path.split(".")[1]]]
            return "s3://bucket/resolved-at-run-time"
        if set(value) == {"Std:Join"} or set(value) == {"Std:JsonGet"}:
            return "resolved-at-run-time"
        return {k: concrete(v, params) for k, v in value.items()}
    if isinstance(value, list):
        return [concrete(v, params) for v in value]
    return value


def validate(service, operation, request):
    model = SESSION.get_service_model(service).operation_model(operation)
    validate_parameters(request, model.input_shape)


REQUESTS = {"Processing": ("CreateProcessingJob", "ProcessingJobName"), "Tuning": ("CreateHyperParameterTuningJob", "HyperParameterTuningJobName"),
            "RegisterModel": ("CreateModelPackage", None)}


@pytest.mark.parametrize("which", ["train", "monitor"])
def test_every_step_is_a_valid_aws_request(which, train, monitor):
    definition = train if which == "train" else monitor
    params = {p["Name"]: p["Type"] for p in definition["Parameters"]}
    checked = []
    for step in d.walk_steps(definition):
        if step["Type"] not in REQUESTS:
            continue
        operation, name_member = REQUESTS[step["Type"]]
        request = concrete(copy.deepcopy(step["Arguments"]), params)
        if name_member:
            request[name_member] = "named-by-the-pipeline"          # required by the API, supplied by SageMaker Pipelines
        validate("sagemaker", operation, request)
        checked.append((step["Name"], operation))
    expected = [("BuildDataset", "CreateProcessingJob"), ("Tune", "CreateHyperParameterTuningJob"), ("Evaluate", "CreateProcessingJob"),
                ("Register", "CreateModelPackage")] if which == "train" else [("Monitor", "CreateProcessingJob")]
    assert checked == expected


def test_a_misspelled_member_would_be_caught_by_this_check(train):
    step = copy.deepcopy(by_name(train)["BuildDataset"])
    step["Arguments"]["ProcessingResource"] = step["Arguments"].pop("ProcessingResources")
    request = concrete(step["Arguments"], {p["Name"]: p["Type"] for p in train["Parameters"]}) | {"ProcessingJobName": "x"}
    with pytest.raises(Exception, match="ProcessingResource"):
        validate("sagemaker", "CreateProcessingJob", request)


def test_create_and_update_pipeline_and_start_execution_requests_are_valid(train):
    text = d.to_json(train)
    check_shape("sagemaker", "CreatePipeline", {"PipelineName": "finrisk-train", "PipelineDefinition": text, "RoleArn": ROLE, "ClientRequestToken": "t" * 32})
    check_shape("sagemaker", "UpdatePipeline", {"PipelineName": "finrisk-train", "PipelineDefinition": text, "RoleArn": ROLE})
    check_shape("sagemaker", "StartPipelineExecution", {"PipelineName": "finrisk-train", "PipelineParameters": [{"Name": "TriggerReason", "Value": "drift"}],
                                                       "ClientRequestToken": "t" * 32})


def test_the_step_arguments_use_no_member_that_botocore_does_not_know_even_nested(train):
    """validate_parameters already rejects unknown members at every depth; this pins that for the deepest one we have."""
    args = copy.deepcopy(by_name(train)["Tune"]["Arguments"])
    args["TrainingJobDefinition"]["Environment"]["X"] = 1
    request = concrete(args, {p["Name"]: p["Type"] for p in train["Parameters"]}) | {"HyperParameterTuningJobName": "x"}
    with pytest.raises(Exception):
        validate("sagemaker", "CreateHyperParameterTuningJob", request)


# ---------------------------------------------------------------- the SDK parity test (dev-only; skipped without sagemaker 3.x)


def test_sdk_v3_builds_the_same_dag(train, tmp_path):
    python = os.environ.get("FINRISK_SDK_PYTHON", sys.executable)
    (tmp_path / "def.json").write_text(d.to_json(train))
    env = {**os.environ, "AWS_ACCESS_KEY_ID": "x", "AWS_SECRET_ACCESS_KEY": "x", "AWS_DEFAULT_REGION": "ap-southeast-2", "AWS_CONFIG_FILE": os.devnull,
           "AWS_SHARED_CREDENTIALS_FILE": os.devnull}
    done = subprocess.run([python, str(ROOT / "tests" / "sdk_parity_script.py"), str(tmp_path / "def.json")], capture_output=True, text=True, env=env)
    if done.returncode == 77:
        pytest.skip("the SageMaker Python SDK v3 (sagemaker, sagemaker-mlops) is not installed here: " + done.stderr.strip()[-120:])
    assert done.returncode == 0, done.stderr[-1500:]
    sdk = json.loads(done.stdout[done.stdout.index("{"):])
    assert sdk["Version"] == train["Version"] == "2020-12-01"
    assert sdk["Parameters"] == train["Parameters"]
    summary = lambda steps: [(s["Name"], s["Type"], s.get("DependsOn")) for s in steps]            # noqa: E731
    assert summary(sdk["Steps"]) == summary(train["Steps"])
    ours_gate, sdk_gate = train["Steps"][3]["Arguments"], sdk["Steps"][3]["Arguments"]
    assert sdk_gate["Conditions"] == ours_gate["Conditions"]                                       # the SDK writes the same Condition
    assert summary(sdk_gate["IfSteps"]) == summary(ours_gate["IfSteps"]) and summary(sdk_gate["ElseSteps"]) == summary(ours_gate["ElseSteps"])
    assert sdk_gate["ElseSteps"][0]["Arguments"] == ours_gate["ElseSteps"][0]["Arguments"]         # Fail: the same ErrorMessage JsonGet
