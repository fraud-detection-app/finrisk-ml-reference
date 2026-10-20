"""The two SageMaker Pipelines, as plain JSON built here (decision D3: no SageMaker SDK at run time).

A pipeline definition is a document: {"Version": "2020-12-01", "Parameters": [...], "Steps": [...]}. Each step's `Arguments`
are the SAME request you would send to the underlying API (CreateProcessingJob, CreateHyperParameterTuningJob,
CreateModelPackage), minus the job name (the pipeline names the job). "A pipeline is a list of AWS requests with arrows."
The arrows are property references that the service resolves while it runs:

    {"Get": "Parameters.GateFloor"}                                  a pipeline parameter
    {"Get": "Steps.Tune.TrainingJobSummaries[0].TrainingJobName"}    a property of a finished step
    {"Std:Join": {"On": "", "Values": [...]}}                        glue strings (and numbers) together
    {"Std:JsonGet": {"PropertyFile": {"Get": "Steps.Evaluate.PropertyFiles.EvaluationReport"}, "Path": "candidate.pr_auc"}}
                                                                    read one value out of a JSON file a step wrote

finrisk-train      BuildDataset -> Tune -> Evaluate -> Gate (Condition) -> Register | GateFailed (Fail)
finrisk-monitor    Monitor

The step Arguments are checked against botocore's input shapes in tests/test_definition.py (required members present, no
unknown members): a typo fails in CI, not in a 20-minute pipeline run.
"""

from __future__ import annotations

import json
from typing import Any

TRAIN_PIPELINE = "finrisk-train"
MONITOR_PIPELINE = "finrisk-monitor"
VERSION = "2020-12-01"
ENTRYPOINT = ["python", "-m", "finrisk_jobs"]
IN = "/opt/ml/processing/input"
OUT = "/opt/ml/processing/output"
TRIGGER_REASONS = ("manual", "push", "schedule", "drift", "backfill")


# ---------------------------------------------------------------------------- the reference language

def ref(path: str) -> dict:
    return {"Get": path}


def param(name: str) -> dict:
    return ref(f"Parameters.{name}")


def join(values: list[Any], on: str = "") -> dict:
    return {"Std:Join": {"On": on, "Values": values}}


def json_get(step: str, property_file: str, path: str) -> dict:
    return {"Std:JsonGet": {"PropertyFile": ref(f"Steps.{step}.PropertyFiles.{property_file}"), "Path": path}}


def output_uri(step: str, output: str) -> dict:
    return ref(f"Steps.{step}.ProcessingOutputConfig.Outputs['{output}'].S3Output.S3Uri")


def parameter(name: str, kind: str, default: Any) -> dict:
    return {"Name": name, "Type": kind, "DefaultValue": default}


class Settings:
    """What the definition needs to know about THIS account and THIS commit (resolved at upsert time)."""

    def __init__(self, ml_bucket: str, role_arn: str, region: str, image: str, train_cfg: dict, mlflow_arn: str = "",
                 git_sha: str = "unknown"):
        self.ml_bucket, self.role_arn, self.region, self.image = ml_bucket, role_arn, region, image
        self.cfg, self.mlflow_arn, self.git_sha = train_cfg, mlflow_arn, git_sha

    @property
    def group(self) -> str:
        return self.cfg.get("registry", {}).get("group", "finrisk-fraud-model")

    def environment(self, **extra: Any) -> dict[str, Any]:
        env = {"AWS_REGION": self.region, "ML_BUCKET": self.ml_bucket, "ML_GIT_SHA": self.git_sha}
        if self.mlflow_arn:
            env["MLFLOW_TRACKING_ARN"] = self.mlflow_arn
        env.update(extra)
        return env

    def bucket_uri(self, key: str) -> str:
        return f"s3://{self.ml_bucket}/{key}"


def processing_arguments(s: Settings, arguments: list[Any], *, inputs: list[dict] | None = None, outputs: list[dict] | None = None,
                         environment: dict | None = None) -> dict[str, Any]:
    """CreateProcessingJob minus ProcessingJobName: the image is the TrainImage parameter, the code is `python -m finrisk_jobs`."""
    inst = s.cfg["instance"]
    args: dict[str, Any] = {
        "ProcessingResources": {"ClusterConfig": {"InstanceCount": int(inst.get("count", 1)), "InstanceType": inst["processing_type"],
                                                  "VolumeSizeInGB": int(inst.get("volume_gb", 50))}},
        "AppSpecification": {"ImageUri": param("TrainImage"), "ContainerEntrypoint": list(ENTRYPOINT), "ContainerArguments": arguments},
        "RoleArn": s.role_arn,
        "StoppingCondition": {"MaxRuntimeInSeconds": int(inst.get("processing_max_runtime_seconds", 3600))},
        "Environment": environment or s.environment(),
    }
    if inputs:
        args["ProcessingInputs"] = inputs
    if outputs:
        args["ProcessingOutputConfig"] = {"Outputs": outputs}
    return args


def s3_input(name: str, uri: Any, local: str) -> dict:
    return {"InputName": name, "S3Input": {"S3Uri": uri, "LocalPath": local, "S3DataType": "S3Prefix", "S3InputMode": "File",
                                           "S3DataDistributionType": "FullyReplicated"}}


def s3_output(name: str, uri: Any, local: str) -> dict:
    return {"OutputName": name, "S3Output": {"S3Uri": uri, "LocalPath": local, "S3UploadMode": "EndOfJob"}}


# ---------------------------------------------------------------------------- finrisk-train

def train_parameters(s: Settings) -> list[dict]:
    tuning, gate = s.cfg["tuning"], s.cfg["gate"]
    return [
        parameter("Cutoff", "String", "auto"),
        parameter("TriggerReason", "String", "manual"),
        parameter("ModelType", "String", s.cfg.get("hyperparameters", {}).get("model_type", "optimized")),
        parameter("MaxTuningJobs", "Integer", int(tuning["max_jobs"])),
        parameter("ParallelTuningJobs", "Integer", int(tuning["parallel_jobs"])),
        parameter("GateFloor", "Float", float(gate["floor"])),
        parameter("GateTolerance", "Float", float(gate["tolerance"])),
        parameter("TrainImage", "String", s.image),
    ]


def build_dataset_step(s: Settings) -> dict:
    dataset_uri = join([s.bucket_uri("datasets/"), ref("Execution.PipelineExecutionId")])
    return {
        "Name": "BuildDataset", "Type": "Processing",
        "Arguments": processing_arguments(
            s, ["build-dataset", "--cutoff", param("Cutoff")],
            outputs=[s3_output("dataset", dataset_uri, f"{OUT}/dataset")],
            environment=s.environment(PIPELINE_EXECUTION_ID=ref("Execution.PipelineExecutionId"))),
        "PropertyFiles": [{"PropertyFileName": "DatasetReport", "OutputName": "dataset", "FilePath": "dataset.json"}],
    }


def _range(name: str, spec: dict) -> dict:
    return {"Name": name, "MinValue": str(spec["min"]), "MaxValue": str(spec["max"]), "ScalingType": spec.get("scaling", "Auto")}


def metric_definitions() -> list[dict]:
    from finrisk_jobs.contract import METRIC_NAMES

    import re

    return [{"Name": name, "Regex": rf"{re.escape(name)}=([0-9\.]+)"} for name in METRIC_NAMES]


def tune_step(s: Settings) -> dict:
    tuning, inst = s.cfg["tuning"], s.cfg["instance"]
    ranges = tuning["ranges"]
    integers = [_range(n, r) for n, r in ranges.items() if r["type"] == "int"]
    continuous = [_range(n, r) for n, r in ranges.items() if r["type"] == "float"]
    dataset = output_uri("BuildDataset", "dataset")

    def channel(name: str) -> dict:
        return {"ChannelName": name, "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix", "S3Uri": join([dataset, f"/{name}/"]), "S3DataDistributionType": "FullyReplicated"}}}

    return {
        "Name": "Tune", "Type": "Tuning", "DependsOn": ["BuildDataset"],
        "Arguments": {
            "HyperParameterTuningJobConfig": {
                "Strategy": tuning.get("strategy", "Bayesian"),
                "HyperParameterTuningJobObjective": {"Type": "Maximize", "MetricName": tuning["objective"]},
                "ResourceLimits": {"MaxNumberOfTrainingJobs": param("MaxTuningJobs"), "MaxParallelTrainingJobs": param("ParallelTuningJobs")},
                "ParameterRanges": {"IntegerParameterRanges": integers, "ContinuousParameterRanges": continuous},
                "TrainingJobEarlyStoppingType": "Off",
                "RandomSeed": 42,
            },
            "TrainingJobDefinition": {
                "StaticHyperParameters": {"model_type": param("ModelType")},
                "AlgorithmSpecification": {"TrainingImage": param("TrainImage"), "TrainingInputMode": "File",
                                           "MetricDefinitions": metric_definitions()},
                "RoleArn": s.role_arn,
                "InputDataConfig": [channel("train"), channel("validation")],
                "OutputDataConfig": {"S3OutputPath": s.bucket_uri("models/")},
                "ResourceConfig": {"InstanceType": inst["training_type"], "InstanceCount": int(inst.get("count", 1)),
                                   "VolumeSizeInGB": int(inst.get("volume_gb", 50))},
                "StoppingCondition": {"MaxRuntimeInSeconds": int(tuning.get("max_runtime_seconds", 1800))},
                "Environment": s.environment(),
            },
        },
    }


def top_trial(prop: str = "TrainingJobName") -> dict:
    """The best trial of the tuning job: TrainingJobSummaries[0] is the one with the best objective."""
    return ref(f"Steps.Tune.TrainingJobSummaries[0].{prop}")


def top_model_uri(s: Settings) -> dict:
    return join([s.bucket_uri("models/"), top_trial(), "/output/model.tar.gz"])


def evaluate_step(s: Settings) -> dict:
    dataset = output_uri("BuildDataset", "dataset")
    gate = s.cfg["gate"]
    evaluation_uri = join([s.bucket_uri("metrics/"), ref("Execution.PipelineExecutionId")])
    arguments = [
        "evaluate", "--candidate-job", top_trial(), "--tuning-job-arn", ref("Steps.Tune.HyperParameterTuningJobArn"),
        "--gate-floor", join([param("GateFloor")]), "--gate-tolerance", join([param("GateTolerance")]),
        "--champion-env", gate.get("champion_env", "prod"), "--train-image", param("TrainImage"),
        "--trigger-reason", param("TriggerReason"), "--execution-arn", ref("Execution.PipelineExecutionArn"),
        "--git-sha", s.git_sha, "--ml-bucket", s.ml_bucket,
    ]
    return {
        "Name": "Evaluate", "Type": "Processing", "DependsOn": ["Tune"],
        "Arguments": processing_arguments(
            s, arguments,
            inputs=[s3_input("test", join([dataset, "/test/"]), f"{IN}/test"),
                    s3_input("profile", join([dataset, "/profile/"]), f"{IN}/profile"),
                    s3_input("report", join([dataset, "/dataset.json"]), f"{IN}/report"),
                    s3_input("model", top_model_uri(s), f"{IN}/model")],
            outputs=[s3_output("evaluation", evaluation_uri, f"{OUT}/evaluation")],
            environment=s.environment(PIPELINE_EXECUTION_ID=ref("Execution.PipelineExecutionId"))),
        "PropertyFiles": [{"PropertyFileName": "EvaluationReport", "OutputName": "evaluation", "FilePath": "evaluation.json"},
                          {"PropertyFileName": "RegisterInfo", "OutputName": "evaluation", "FilePath": "register_info.json"}],
    }


def register_step(s: Settings) -> dict:
    evaluation = output_uri("Evaluate", "evaluation")

    def report(name: str) -> dict:
        return {"ContentType": "application/json", "S3Uri": join([evaluation, f"/{name}"])}

    def info(key: str) -> dict:
        return json_get("Evaluate", "RegisterInfo", key)

    return {
        "Name": "Register", "Type": "RegisterModel",
        "Arguments": {
            "ModelPackageGroupName": s.group,
            "ModelPackageDescription": join(["Fraud model from ", top_trial(), ": PR-AUC ", info("pr_auc"), " on the test days"]),
            "InferenceSpecification": {
                "Containers": [{"Image": param("TrainImage"), "ModelDataUrl": top_model_uri(s)}],
                "SupportedContentTypes": ["application/json"], "SupportedResponseMIMETypes": ["application/json"],
                "SupportedRealtimeInferenceInstanceTypes": list(s.cfg["registry"]["realtime_instance_types"]),
            },
            "ModelMetrics": {"ModelQuality": {"Statistics": report("evaluation.json")}, "Bias": {"Report": report("bias.json")},
                             "Explainability": {"Report": report("explainability.json")}},
            "ModelApprovalStatus": "PendingManualApproval",
            "ModelCard": {"ModelCardContent": info("model_card_content"), "ModelCardStatus": "Draft"},
            "CustomerMetadataProperties": {
                "training_job": info("training_job"), "train_image_digest": info("train_image_digest"), "cutoff": info("cutoff"),
                "dataset_uri": info("dataset_uri"), "baseline_uri": info("baseline_uri"),
                "pipeline_execution_arn": ref("Execution.PipelineExecutionArn"), "ml_git_sha": s.git_sha,
                "trigger_reason": param("TriggerReason"), "mlflow_run_id": info("mlflow_run_id"),
                "champion_version": info("champion_version"), "pr_auc": info("pr_auc"), "champion_pr_auc": info("champion_pr_auc"),
            },
        },
    }


def gate_failed_step() -> dict:
    """The Fail step's message is built by the Evaluate job (it knows if there is a champion; a JsonGet on a null champion
    would fail the failure message itself): 'GATE FAILED: candidate pr_auc X; needs >= floor 0.70 and >= champion floor Y (...)'. It is written whether or
    not the gate passed (the key is fail_message), and only the Fail step ever shows it."""
    return {"Name": "GateFailed", "Type": "Fail",
            "Arguments": {"ErrorMessage": json_get("Evaluate", "RegisterInfo", "fail_message")}}


def gate_step(s: Settings) -> dict:
    cand = json_get("Evaluate", "EvaluationReport", "candidate.pr_auc")
    return {
        "Name": "Gate", "Type": "Condition", "DependsOn": ["Evaluate"],
        "Arguments": {
            "Conditions": [
                {"Type": "GreaterThanOrEqualTo", "LeftValue": cand, "RightValue": param("GateFloor")},
                {"Type": "GreaterThanOrEqualTo", "LeftValue": cand, "RightValue": json_get("Evaluate", "EvaluationReport", "gate.champion_floor")},
            ],
            "IfSteps": [register_step(s)], "ElseSteps": [gate_failed_step()],
        },
    }


def train_definition(s: Settings) -> dict:
    return {"Version": VERSION, "Metadata": {"repository": "finrisk-ml", "git_sha": s.git_sha},
            "Parameters": train_parameters(s),
            "Steps": [build_dataset_step(s), tune_step(s), evaluate_step(s), gate_step(s)]}


# ---------------------------------------------------------------------------- finrisk-monitor

def monitor_definition(s: Settings) -> dict:
    parameters = [parameter("Env", "String", "prod"), parameter("WindowHours", "Integer", 24), parameter("Reference", "String", "auto"),
                  parameter("TrainImage", "String", s.image)]
    step = {
        "Name": "Monitor", "Type": "Processing",
        "Arguments": processing_arguments(
            s, ["monitor", "--env", param("Env"), "--window-hours", join([param("WindowHours")]), "--reference", param("Reference"),
                "--ml-bucket", s.ml_bucket],
            environment=s.environment(PIPELINE_EXECUTION_ID=ref("Execution.PipelineExecutionId"))),
    }
    return {"Version": VERSION, "Metadata": {"repository": "finrisk-ml", "git_sha": s.git_sha}, "Parameters": parameters, "Steps": [step]}


def to_json(definition: dict) -> str:
    """The string CreatePipeline takes (compact; the service limit is 1 MB)."""
    return json.dumps(definition, separators=(",", ":"))


# ---------------------------------------------------------------------------- reading a definition back (tests, docs)

def walk_steps(definition: dict) -> list[dict]:
    """Every step, with the steps inside Condition branches too (each gets `_branch` = 'if' / 'else' / None)."""
    found: list[dict] = []

    def visit(steps: list[dict], branch: str | None) -> None:
        for step in steps:
            found.append({**step, "_branch": branch})
            if step["Type"] == "Condition":
                visit(step["Arguments"]["IfSteps"], "if")
                visit(step["Arguments"]["ElseSteps"], "else")

    visit(definition["Steps"], None)
    return found


def references(value: Any) -> list[str]:
    """Every {"Get": ...} path inside a value."""
    out: list[str] = []
    if isinstance(value, dict):
        if set(value) == {"Get"}:
            out.append(value["Get"])
        for v in value.values():
            out += references(v)
    elif isinstance(value, list):
        for v in value:
            out += references(v)
    return out


def dependencies(step: dict) -> set[str]:
    """Step names this step waits for: explicit DependsOn plus every Steps.<Name>. reference."""
    found = set(step.get("DependsOn", []))
    for path in references(step.get("Arguments", {})):
        if path.startswith("Steps."):
            found.add(path.split(".")[1])
    return found
