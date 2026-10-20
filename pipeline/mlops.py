"""The MLOps command-line tool: data -> pipeline (tune, evaluate, gate, register) -> approve -> endpoints -> monitor.

    backfill   load the FULL PaySim data once (SageMaker Processing job): raw -> S3, Feature Store history, labels
    upsert     write the two pipelines (finrisk-train, finrisk-monitor) to SageMaker from pipeline/definition.py
    run        start a pipeline and follow it step by step; for finrisk-train the end shows the gate and the new version
    status     one pipeline execution: its steps and where it is
    list/show  look at the registry
    approve    a person says yes (or no) to a version; the Model Card status follows
    promote    deploy a version to an environment's endpoint (serverless dev/sit; real-time + blue/green pre-prod/prod),
               smoke-test it, move the pointer, wait for the app to report it
    replay     the simulated customers: days 18-22 through the app (and the features into the Feature Store)
    feedback   the simulated analysts: the true labels of those days, written late
    monitor    run the finrisk-monitor pipeline and show the drift table and the alarm states
    mlflow-url a presigned link to the MLflow App UI
    teardown   delete an environment's endpoint, configs and models

    python pipeline/mlops.py upsert --image <repo>@sha256:...
    python pipeline/mlops.py run --trigger-reason manual
    python pipeline/mlops.py approve --version 1 --decision approve
    python pipeline/mlops.py promote --env dev --version 1
    python pipeline/mlops.py replay --from-day 18 --to-day 22 && python pipeline/mlops.py feedback --from-day 18 --to-day 22
    python pipeline/mlops.py monitor --env prod

What you will see: every AWS call this tool makes is printed first -- the operation and its main parameters, account ids
masked as <account-id>. Nothing happens behind your back. The LAST line printed on success is the value the next step needs
(the version for run/promote, the status for approve). Add --quiet to print nothing else.

Exit codes: 0 ok · 1 something failed (message on stderr) · 2 the model version cannot be used here (not found, Rejected,
or not Approved for pre-prod / prod) · 3 the gate said no.

Needs only boto3 and pyyaml (replay and feedback also pandas and pyarrow: requirements-sim.txt). The heavy work runs
INSIDE SageMaker, in the image built from train/Dockerfile (finrisk_jobs/).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
for _folder in (REPO_ROOT, REPO_ROOT / "pipeline"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import deploy  # noqa: E402
import definition  # noqa: E402
from errors import EXIT_FAILED, EXIT_GATE, EXIT_OK, EXIT_UNUSABLE, MlopsError, error_code, mask  # noqa: E402
from deploy import find_load_error, parse_time, wait_for_model  # noqa: E402,F401 - kept importable from here
from finrisk_jobs import gate as gatelib  # noqa: E402

DEFAULT_REGION = "ap-southeast-2"
DEFAULT_GROUP = "finrisk-fraud-model"
DEFAULT_CONFIG = REPO_ROOT / "config" / "train.yaml"
ENVIRONMENTS = ("dev", "sit", "pre-prod", "prod")
NEEDS_APPROVAL = ("pre-prod", "prod")           # dev and sit may run a version nobody approved yet
DIGEST_IMAGE = re.compile(r"^\S+@sha256:[0-9a-f]{64}$")
PIPELINES = (definition.TRAIN_PIPELINE, definition.MONITOR_PIPELINE)
EXECUTION_DONE = {"Succeeded", "Failed", "Stopped"}
JOB_DONE = {"Completed", "Failed", "Stopped"}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp(fmt: str) -> str:
    return utc_now().strftime(fmt)


def idempotency_token() -> str:
    """ClientRequestToken: 32-64 characters; a retry of the SAME call would reuse it, a new call gets a new one."""
    import uuid

    return uuid.uuid4().hex


# ---------------------------------------------------------------------------- configuration


def load_train_config(path: Path) -> dict[str, Any]:
    """config/train.yaml as a dict (see the file for every key)."""
    import yaml

    if not Path(path).is_file():
        raise MlopsError(f"config file not found: {path}")
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for key in ("tuning", "instance", "gate", "registry", "endpoints"):
        if key not in raw:
            raise MlopsError(f"{path} has no '{key}:' section")
    return raw


@dataclass
class Config:
    ml_bucket: str = ""
    group: str = DEFAULT_GROUP
    role_arn: str = ""
    region: str = DEFAULT_REGION

    def need(self, *names: str) -> None:
        env_names = {"ml_bucket": "ML_BUCKET", "group": "MODEL_PACKAGE_GROUP", "role_arn": "SAGEMAKER_ROLE_ARN"}
        missing = [n for n in names if not getattr(self, n)]
        if missing:
            how = ", ".join(f"{env_names[n]} (--{n.replace('_', '-')})" for n in missing)
            raise MlopsError(f"not set: {how}.")


def load_config(args: argparse.Namespace, env: dict[str, str]) -> Config:
    return Config(
        ml_bucket=args.ml_bucket or env.get("ML_BUCKET", ""),
        group=args.group or env.get("MODEL_PACKAGE_GROUP", "") or DEFAULT_GROUP,
        role_arn=args.role_arn or env.get("SAGEMAKER_ROLE_ARN", ""),
        region=args.region or env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION") or DEFAULT_REGION,
    )


# ---------------------------------------------------------------------------- showing every AWS call


class Traced:
    """Wraps a boto3 client: prints each call (operation + main parameters) and then makes it.

    ``quiet_after_first`` names polling calls (and bulk calls): shown once, then silent, so a 25-minute wait does not
    print 50 identical blocks."""

    def __init__(self, client: Any, service: str, out: Callable[[str], None], quiet_after_first: tuple[str, ...] = ()):
        self._client, self._service, self._out = client, service, out
        self._quiet, self._seen = set(quiet_after_first), set()

    def __getattr__(self, operation: str) -> Callable[..., Any]:
        call = getattr(self._client, operation)

        def traced(**params: Any) -> Any:
            if operation not in self._quiet or operation not in self._seen:
                self._out(mask(f"  AWS  {self._service}.{operation}"))
                for key, value in params.items():
                    if isinstance(value, (bytes, bytearray)):
                        text = f"<{len(value):,} bytes>"                      # never print a Parquet file into a log
                    elif hasattr(value, "read"):
                        text = "<a file>"
                    else:
                        text = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"), default=str)
                    text = text if len(text) <= 240 else text[:237] + "..."
                    self._out(mask(f"         {key:<26} {text}"))
            self._seen.add(operation)
            return call(**params)

        return traced


CLIENT_NAMES = {"sagemaker": "sagemaker", "s3": "s3", "sts": "sts", "ssm": "ssm", "runtime": "sagemaker-runtime",
                "fs": "sagemaker-featurestore-runtime", "cw": "cloudwatch"}


def make_clients(region: str) -> dict[str, Any]:
    import boto3

    return {key: boto3.client(service, region_name=region) for key, service in CLIENT_NAMES.items()}


def http_get(url: str, timeout: float) -> tuple[int, str]:
    """GET a URL; return (status, body). An error status (503) is an answer, not an exception."""
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"not an http(s) address: {url!r}")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:      # noqa: S310 - scheme checked above
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def http_post(url: str, body: dict, timeout: float) -> tuple[int, dict]:
    """POST JSON; return (status, parsed JSON). An error status is an answer, not an exception."""
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"not an http(s) address: {url!r}")
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:    # noqa: S310
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, {"detail": raw[:300]}


@dataclass
class Context:
    cfg: Config
    tcfg: dict
    sm: Any
    s3: Any
    sts: Any
    ssm: Any
    out: Callable[[str], None]
    sleep: Callable[[float], None]
    clock: Callable[[], float]
    env: dict[str, str]
    runtime: Any = None
    fs: Any = None
    cw: Any = None
    http: Callable[[str, float], tuple[int, str]] = http_get
    post: Callable[[str, dict, float], tuple[int, dict]] = http_post
    now: Callable[[], datetime] = utc_now
    _account: str | None = None

    def account(self) -> str:
        if self._account is None:
            self._account = self.sts.get_caller_identity()["Account"]
        return self._account

    def package_arn(self, version: int) -> str:
        """arn:aws:sagemaker:<region>:<account>:model-package/<group>/<version> -- the account comes from STS."""
        return f"arn:aws:sagemaker:{self.cfg.region}:{self.account()}:model-package/{self.cfg.group}/{version}"

    def shared(self, attr: str) -> str:
        """A name that contains the account id (the ML bucket, the SageMaker role).

        Order: flag / environment variable -> SSM /finrisk/shared/<name> (written by finrisk-infra) -> the naming
        convention from the course. The source is printed, so you always know which won."""
        value = getattr(self.cfg, attr)
        if value:
            return value
        ssm_name = {"ml_bucket": "ml_bucket", "role_arn": "sagemaker_role_arn"}[attr]
        try:
            value = self.ssm.get_parameter(Name=f"/finrisk/shared/{ssm_name}")["Parameter"]["Value"]
            self.out(f"  {attr} taken from SSM /finrisk/shared/{ssm_name}")
        except Exception as exc:  # noqa: BLE001
            if error_code(exc) not in ("ParameterNotFound", "AccessDeniedException", "AccessDenied"):
                raise
            account = self.account()
            value = {
                "ml_bucket": f"finrisk-ml-{account}-{self.cfg.region}-an",
                "role_arn": f"arn:aws:iam::{account}:role/finrisk-sagemaker-exec",
            }[attr]
            self.out(f"  {attr} built from the naming convention (SSM said {error_code(exc)})")
        setattr(self.cfg, attr, value)
        return value

    def shared_optional(self, name: str) -> str:
        """/finrisk/shared/<name> or '' (a value that may not exist yet, like the MLflow app ARN)."""
        try:
            return self.ssm.get_parameter(Name=f"/finrisk/shared/{name}")["Parameter"]["Value"]
        except Exception as exc:  # noqa: BLE001
            if error_code(exc) in ("ParameterNotFound", "AccessDeniedException", "AccessDenied"):
                return ""
            raise


def write_summary(path: str | None, lines: list[str]) -> None:
    """Append markdown to the GitHub job summary (any file works). Account ids are masked."""
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(mask("\n".join(lines)) + "\n")


def run_url(env: dict[str, str]) -> str:
    server, repo, run_id = env.get("GITHUB_SERVER_URL"), env.get("GITHUB_REPOSITORY"), env.get("GITHUB_RUN_ID")
    return f"{server}/{repo}/actions/runs/{run_id}" if server and repo and run_id else "local"


def bucket_key(uri: str) -> tuple[str, str]:
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return bucket, key


# ---------------------------------------------------------------------------- backfill


def split_days(days: list[int], parts: int) -> list[list[int]]:
    """Contiguous, near-equal day ranges: [1..17] over 3 instances -> 6 + 6 + 5 days."""
    parts = max(1, min(parts, len(days)))
    size, extra = divmod(len(days), parts)
    out, start = [], 0
    for i in range(parts):
        end = start + size + (1 if i < extra else 0)
        out.append(days[start:end])
        start = end
    return out


def days_text(days: list[int]) -> str:
    return f"{days[0]}-{days[-1]}" if len(days) > 1 else str(days[0])


def build_backfill_request(ctx: Context, image: str, name: str, days: list[int], production: bool, index: int = 1, count: int = 1) -> dict[str, Any]:
    inst = ctx.tcfg["instance"]
    return {
        "ProcessingJobName": name,
        "ProcessingResources": {"ClusterConfig": {"InstanceCount": 1, "InstanceType": inst["processing_type"],
                                                  "VolumeSizeInGB": int(inst.get("volume_gb", 50)) + 30}},
        "AppSpecification": {"ImageUri": image, "ContainerEntrypoint": list(definition.ENTRYPOINT),
                             "ContainerArguments": ["backfill", "--days", days_text(days), "--production", "yes" if production else "no",
                                                    "--instance-index", str(index), "--instance-count", str(count), "--ml-bucket", ctx.cfg.ml_bucket]},
        "RoleArn": ctx.cfg.role_arn,
        "StoppingCondition": {"MaxRuntimeInSeconds": 4 * 3600},
        "Environment": {"AWS_REGION": ctx.cfg.region, "ML_BUCKET": ctx.cfg.ml_bucket},
        "Tags": [{"Key": "Project", "Value": "finrisk"}, {"Key": "ManagedBy", "Value": "mlops.py"}],
    }


def cmd_backfill(ctx: Context, a: argparse.Namespace) -> str:
    import yaml

    ctx.shared("ml_bucket")
    ctx.shared("role_arn")
    image = a.image or ctx.env.get("TRAIN_IMAGE", "")
    if not DIGEST_IMAGE.match(image or ""):
        raise MlopsError("--image must be an image DIGEST (<repo>@sha256:<64 hex>); the backfill runs the training image.")
    data = yaml.safe_load((REPO_ROOT / "config" / "data.yaml").read_text())
    from finrisk_jobs.contract import parse_days

    excluded = set(data["calendar"].get("exclude_days", []))
    days = [d for d in parse_days(a.days or data["history_days"]) if d not in excluded]
    chunks = split_days(days, a.instance_count)
    stamp = utc_stamp("%Y%m%d-%H%M%S")
    names = []
    for k, chunk in enumerate(chunks):
        name = f"finrisk-backfill-{stamp}" + (f"-{k + 1}" if len(chunks) > 1 else "")
        ctx.out(f"backfill job {name}: history days {days_text(chunk)}" + ("  (+ the production set)" if k == 0 else ""))
        ctx.sm.create_processing_job(**build_backfill_request(ctx, image, name, chunk, production=(k == 0), index=k + 1, count=len(chunks)))
        names.append(name)
    if a.no_wait:
        ctx.out("not waiting (--no-wait)." + (" --then-train needs the jobs to finish first: start the training yourself afterwards (train.yml)." if a.then_train else ""))
        return names[-1]
    pending, last = list(names), {}
    started = ctx.clock()
    while pending:
        for name in list(pending):
            d = ctx.sm.describe_processing_job(ProcessingJobName=name)
            status = d["ProcessingJobStatus"]
            if last.get(name) != status:
                ctx.out(f"  [{ctx.clock() - started:5.0f}s] {name}: {status}")
                last[name] = status
            if status in JOB_DONE:
                pending.remove(name)
                if status != "Completed":
                    reason = mask(d.get("FailureReason") or d.get("ExitMessage") or "no reason given")
                    hint = ""
                    if "STOPPED: the rest would take" in reason:
                        hint = "\n  The job measured the Feature Store write speed and stopped on purpose. Run again with the --instance-count it suggests."
                    raise MlopsError(f"backfill job {name} {status}: {reason}{hint}\n  logs: CloudWatch /aws/sagemaker/ProcessingJobs, stream {name}/...")
        if pending:
            if ctx.clock() - started > a.max_wait:
                raise MlopsError(f"still running after {a.max_wait:.0f} s; the job(s) {pending} keep running in AWS.")
            ctx.sleep(a.poll)
    report = f"s3://{ctx.cfg.ml_bucket}/{data['dataset']['raw_prefix']}/backfill_report.json"
    ctx.out(f"backfill finished. Report: {mask(report)}")
    ctx.out("  (each job only finishes after Athena can see everything it wrote: the offline store is complete and queryable)")
    lines = ["### Backfill finished", ""] + [f"- job `{n}`" for n in names] + [
        "", f"History days {days_text(days)} are in the Feature Store and visible in Athena; labels and the production set (days 18-30) are in S3. "
        "The report is `backfill_report.json` next to the raw data."]
    if not a.then_train:
        write_summary(a.summary, lines + ["", "Next: run **train** (or start `finrisk-train` with `python pipeline/mlops.py run`)."])
        return names[-1]
    ctx.out("--then-train: writing the pipelines, then starting finrisk-train")
    cmd_upsert(ctx, argparse.Namespace(image=image, pipeline="both", summary=None))
    arn = start_training(ctx, "backfill")
    write_summary(a.summary, lines + ["", "### Training started", "", f"- execution `{arn}` (TriggerReason=backfill)",
                                      f"- follow it: `python pipeline/mlops.py status {arn}` or SageMaker Studio -> Pipelines -> finrisk-train"])
    return arn


# ---------------------------------------------------------------------------- upsert / run / status


def settings_for(ctx: Context, image: str) -> definition.Settings:
    return definition.Settings(ml_bucket=ctx.shared("ml_bucket"), role_arn=ctx.shared("role_arn"), region=ctx.cfg.region, image=image,
                               train_cfg=ctx.tcfg, mlflow_arn=ctx.shared_optional("mlflow_app_arn"),
                               git_sha=ctx.env.get("ML_GIT_SHA") or ctx.env.get("GITHUB_SHA") or "unknown")


def pipeline_exists(ctx: Context, name: str) -> bool:
    try:
        ctx.sm.describe_pipeline(PipelineName=name)
        return True
    except Exception as exc:  # noqa: BLE001
        if error_code(exc) in ("ResourceNotFound", "ValidationException", "ResourceNotFoundException"):
            return False
        raise


def cmd_upsert(ctx: Context, a: argparse.Namespace) -> str:
    image = a.image or ctx.env.get("TRAIN_IMAGE", "")
    if not DIGEST_IMAGE.match(image or ""):
        raise MlopsError("--image must be an image DIGEST (<repo>@sha256:<64 hex>), not a tag: a tag can move, a digest cannot, "
                         "and the model must be trained with exactly the image you can name later.")
    settings = settings_for(ctx, image)
    built = {definition.TRAIN_PIPELINE: definition.train_definition(settings), definition.MONITOR_PIPELINE: definition.monitor_definition(settings)}
    wanted = PIPELINES if a.pipeline == "both" else (a.pipeline,)
    done = []
    for name in wanted:
        text = definition.to_json(built[name])
        if len(text) > 1_000_000:
            raise MlopsError(f"the {name} definition is {len(text)} bytes; CreatePipeline takes at most 1 MB")
        ctx.out(f"{name}: {len(built[name]['Steps'])} step(s), {len(text):,} bytes of definition")
        if pipeline_exists(ctx, name):
            ctx.sm.update_pipeline(PipelineName=name, PipelineDefinition=text, RoleArn=settings.role_arn)
        else:
            ctx.sm.create_pipeline(PipelineName=name, PipelineDefinition=text, RoleArn=settings.role_arn,
                                   ClientRequestToken=idempotency_token(),
                                   Tags=[{"Key": "Project", "Value": "finrisk"}, {"Key": "ManagedBy", "Value": "mlops.py"}])
        done.append(name)
    write_summary(a.summary, ["### Pipelines written", ""] + [f"- `{n}`" for n in done] + ["", f"Training image: `{image.partition('@')[2][:19]}...`"])
    return ", ".join(done)


def execution_id(arn: str) -> str:
    return arn.rsplit("/", 1)[-1]


def list_steps(ctx: Context, arn: str) -> list[dict]:
    steps, token = [], None
    while True:
        params: dict[str, Any] = {"PipelineExecutionArn": arn, "SortOrder": "Ascending"}
        if token:
            params["NextToken"] = token
        page = ctx.sm.list_pipeline_execution_steps(**params)
        steps += page.get("PipelineExecutionSteps", [])
        token = page.get("NextToken")
        if not token:
            return steps


def follow_execution(ctx: Context, arn: str, poll: float, max_wait: float) -> tuple[dict, list[dict]]:
    """Poll until the execution ends, printing each step's status whenever it changes."""
    started, shown = ctx.clock(), {}
    ctx.out(f"  following {execution_id(arn)}; checking every {poll:.0f} s and printing only changes")
    while True:
        d = ctx.sm.describe_pipeline_execution(PipelineExecutionArn=arn)
        steps = list_steps(ctx, arn)
        for step in steps:
            state = step["StepStatus"]
            if shown.get(step["StepName"]) != state:
                shown[step["StepName"]] = state
                extra = f"  {mask(step.get('FailureReason', ''))[:200]}" if state == "Failed" else ""
                ctx.out(f"  [{ctx.clock() - started:5.0f}s] {step['StepName']:<14} {state}{extra}")
        if d["PipelineExecutionStatus"] in EXECUTION_DONE:
            ctx.out(f"  execution {d['PipelineExecutionStatus']}")
            return d, steps
        if ctx.clock() - started > max_wait:
            raise MlopsError(f"still {d['PipelineExecutionStatus']} after {max_wait:.0f} s. The execution keeps running in AWS: "
                             f"python pipeline/mlops.py status {execution_id(arn)}")
        ctx.sleep(poll)


def parse_params(pairs: list[str]) -> dict[str, str]:
    out = {}
    for pair in pairs or []:
        name, sep, value = pair.partition("=")
        if not sep or not name:
            raise MlopsError(f"--param must look like NAME=VALUE, got {pair!r}")
        out[name] = value
    return out


def read_evaluation(ctx: Context, execution: str) -> dict | None:
    try:
        key = f"metrics/{execution}/evaluation.json"
        return json.loads(ctx.s3.get_object(Bucket=ctx.shared("ml_bucket"), Key=key)["Body"].read())
    except Exception:  # noqa: BLE001
        return None


def print_gate(ctx: Context, evaluation: dict, label: str) -> list[str]:
    checks = gatelib.checks_from_evaluation(evaluation)
    lines = gatelib.gate_table(label, float(evaluation["candidate"]["pr_auc"]), checks)
    for line in lines:
        ctx.out(line)
    return lines


def start_training(ctx: Context, reason: str) -> str:
    """StartPipelineExecution(finrisk-train) without following it; prints the ARN and where to watch it. Returns the ARN."""
    try:
        arn = ctx.sm.start_pipeline_execution(PipelineName=definition.TRAIN_PIPELINE,
                                              PipelineParameters=[{"Name": "TriggerReason", "Value": reason}],
                                              ClientRequestToken=idempotency_token())["PipelineExecutionArn"]
    except Exception as exc:  # noqa: BLE001
        if error_code(exc) in ("ResourceNotFound", "ResourceNotFoundException"):
            raise MlopsError("the pipeline finrisk-train does not exist yet: run train.yml (or `python pipeline/mlops.py upsert --image ...`) first.") from exc
        raise
    ctx.out(f"started {definition.TRAIN_PIPELINE}: {arn}")
    ctx.out(f"  follow it:  python pipeline/mlops.py status {arn}")
    ctx.out(f"  or look at it in SageMaker Studio -> Pipelines -> {definition.TRAIN_PIPELINE} -> Executions")
    return arn


def data_config() -> dict[str, Any]:
    import yaml

    return yaml.safe_load((REPO_ROOT / "config" / "data.yaml").read_text())


def backfill_report_exists(ctx: Context) -> bool:
    """Has the backfill finished? Its LAST act is writing s3://<ml>/<raw_prefix>/backfill_report.json (after Athena sees the data).

    S3 answers 403 instead of 404 for a missing key when the caller may not list the bucket: both mean 'not there for us'."""
    bucket = ctx.shared("ml_bucket")
    key = f"{data_config()['dataset']['raw_prefix']}/backfill_report.json"
    try:
        ctx.s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as exc:  # noqa: BLE001
        if error_code(exc) in ("404", "NoSuchKey", "NotFound", "403", "AccessDenied", "Forbidden"):
            return False
        raise


NO_DATA_NOTICE = ("no data yet: run backfill.yml first (it loads the dataset and, with then_train, starts the first training run). "
                  "The pipelines are upserted and ready.")


def cmd_run(ctx: Context, a: argparse.Namespace) -> str | None:
    name = a.pipeline
    if name == definition.TRAIN_PIPELINE and not backfill_report_exists(ctx):
        message = f"{NO_DATA_NOTICE} (s3://<ml bucket>/{data_config()['dataset']['raw_prefix']}/backfill_report.json does not exist)"
        reason = a.trigger_reason or "manual"
        if reason == "manual":
            raise MlopsError(message)                       # a person asked for a run: tell them it cannot happen (exit 1)
        ctx.out(f"NOTICE: {message}")
        ctx.out(f"  nothing was started (trigger: {reason}); this is not a failure.")
        write_summary(a.summary, ["### No training run started", "", f"{NO_DATA_NOTICE}", "", f"Trigger: `{reason}`. The job is green on purpose."])
        return None
    params = parse_params(a.param)
    if name == definition.TRAIN_PIPELINE:
        if a.trigger_reason:
            params.setdefault("TriggerReason", a.trigger_reason)
        if a.model_type:
            params.setdefault("ModelType", a.model_type)
        if a.cutoff:
            params.setdefault("Cutoff", a.cutoff)
    if a.image:
        if not DIGEST_IMAGE.match(a.image):
            raise MlopsError("--image must be an image DIGEST (<repo>@sha256:<64 hex>)")
        params["TrainImage"] = a.image
    ctx.out(f"starting {name} with " + (", ".join(f"{k}={mask(v)}" for k, v in params.items()) or "the default parameters"))
    arn = ctx.sm.start_pipeline_execution(PipelineName=name, PipelineParameters=[{"Name": k, "Value": v} for k, v in params.items()],
                                          ClientRequestToken=idempotency_token())["PipelineExecutionArn"]
    execution = execution_id(arn)
    ctx.out(f"execution {execution}")
    if a.no_wait:
        ctx.out("not waiting (--no-wait).")
        return execution
    return finish_execution(ctx, name, arn, a)


def finish_execution(ctx: Context, name: str, arn: str, a: argparse.Namespace) -> str | None:
    execution = execution_id(arn)
    d, steps = follow_execution(ctx, arn, a.poll, a.max_wait)
    by_name = {s["StepName"]: s for s in steps}
    if name != definition.TRAIN_PIPELINE:
        if d["PipelineExecutionStatus"] != "Succeeded":
            failed = next((s for s in steps if s["StepStatus"] == "Failed"), {})
            raise MlopsError(f"{name} {d['PipelineExecutionStatus']}: {mask(failed.get('FailureReason') or d.get('FailureReason', 'no reason given'))}")
        return execution
    evaluation = read_evaluation(ctx, execution)
    gate_lines: list[str] = []
    if evaluation:
        gate_lines = print_gate(ctx, evaluation, by_name.get("Tune", {}).get("StepName", "tuning") + " best trial")
    if "GateFailed" in by_name:
        reason = mask(by_name["GateFailed"].get("FailureReason") or "GATE FAILED")
        write_summary(a.summary, ["### Gate: FAILED", "", "```", *gate_lines, "```", "", f"**{reason}**", "",
                                  "Nothing was registered; the models in use are unchanged."])
        raise MlopsError(reason, EXIT_GATE)
    if d["PipelineExecutionStatus"] != "Succeeded":
        failed = next((s for s in steps if s["StepStatus"] == "Failed"), {})
        raise MlopsError(f"{name} {d['PipelineExecutionStatus']} at step {failed.get('StepName', '?')}: "
                         f"{mask(failed.get('FailureReason') or d.get('FailureReason', 'no reason given'))}")
    register = by_name.get("Register")
    arn_model = (register or {}).get("Metadata", {}).get("RegisterModel", {}).get("Arn", "")
    if not arn_model:
        raise MlopsError("the pipeline succeeded but no model version was registered (no Register step in the execution).")
    version = int(arn_model.rsplit("/", 1)[1])
    ctx.out(f"registered model version {version} (PendingManualApproval)")
    cand = evaluation["candidate"] if evaluation else {}
    write_summary(a.summary, ["### Gate: PASSED", "", "```", *gate_lines, "```", "", f"### Registered: model version {version} (PendingManualApproval)", "",
                              f"- candidate PR-AUC {cand.get('pr_auc', float('nan')):.4f} on the test days",
                              f"- execution `{execution}`", "", f"Next: run **approve-model** with version {version}, then **promote-model**."])
    return str(version)


def cmd_status(ctx: Context, a: argparse.Namespace) -> str | None:
    arn = a.execution
    if not arn.startswith("arn:"):
        arn = f"arn:aws:sagemaker:{ctx.cfg.region}:{ctx.account()}:pipeline/{a.pipeline}/execution/{a.execution}"
    d = ctx.sm.describe_pipeline_execution(PipelineExecutionArn=arn)
    ctx.out(f"execution {execution_id(arn)}: {d['PipelineExecutionStatus']}")
    for step in list_steps(ctx, arn):
        extra = f"  {mask(step.get('FailureReason', ''))[:160]}" if step.get("FailureReason") else ""
        ctx.out(f"  {step['StepName']:<14} {step['StepStatus']}{extra}")
    return None


# ---------------------------------------------------------------------------- approve / list / show


def describe_version(ctx: Context, version: int) -> dict[str, Any]:
    try:
        return ctx.sm.describe_model_package(ModelPackageName=ctx.package_arn(version))
    except MlopsError:
        raise
    except Exception as exc:  # noqa: BLE001
        if error_code(exc) in ("ValidationException", "ResourceNotFound", "ResourceNotFoundException"):
            raise MlopsError(f"model version {version} not found in group {ctx.cfg.group}. See the versions: python pipeline/mlops.py list", EXIT_UNUSABLE) from exc
        raise MlopsError(f"could not read model version {version}: {mask(str(exc))}") from exc


def cmd_approve(ctx: Context, a: argparse.Namespace) -> str:
    detail = describe_version(ctx, a.version)            # a clear "not found" before we try to change anything
    new_status = "Approved" if a.decision == "approve" else "Rejected"
    params: dict[str, Any] = {"ModelPackageArn": ctx.package_arn(a.version), "ModelApprovalStatus": new_status}
    if a.note:
        params["ApprovalDescription"] = a.note[:1024]
    if detail.get("ModelCard"):                           # the Model Card follows the decision (versions made before cards have none)
        params["ModelCard"] = {"ModelCardStatus": "Approved" if a.decision == "approve" else "Archived"}
    ctx.sm.update_model_package(**params)
    after = describe_version(ctx, a.version)
    status = after["ModelApprovalStatus"]
    card = (after.get("ModelCard") or {}).get("ModelCardStatus")
    ctx.out(f"model version {a.version} is now {status}" + (f" (Model Card: {card})" if card else ""))
    write_summary(a.summary, [f"### Model version {a.version}: {a.decision} -> {status}"] + ([f"Model Card status: {card}"] if card else [])
                  + ([f"Note: {a.note}"] if a.note else []))
    return status


def when(value: Any) -> str:
    """A time for the tables, always in UTC (the columns say so). boto3 hands back datetimes in the LOCAL time zone,
    and strftime would print that local clock; astimezone(utc) converts first. A datetime with no zone is taken as UTC."""
    if isinstance(value, datetime):
        value = (value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value).astimezone(timezone.utc)
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value)[:16]


def cmd_list(ctx: Context, a: argparse.Namespace) -> str | None:
    ctx.cfg.need("group")
    page = ctx.sm.list_model_packages(ModelPackageGroupName=ctx.cfg.group, SortBy="CreationTime", SortOrder="Descending", MaxResults=a.limit)
    rows = []
    for summary in page.get("ModelPackageSummaryList", []):
        detail = describe_version(ctx, int(summary["ModelPackageVersion"]))
        meta = detail.get("CustomerMetadataProperties", {})
        rows.append((summary["ModelPackageVersion"], summary["ModelApprovalStatus"], meta.get("pr_auc", "-"),
                     when(summary["CreationTime"]), meta.get("training_job", "-")))
    if not rows:
        ctx.out(f"no versions in {ctx.cfg.group} yet. Run the pipeline: python pipeline/mlops.py run")
        return None
    ctx.out(f"{'version':>7}  {'status':<22} {'PR-AUC':>8}  {'created (UTC)':<16}  training job")
    for version, status, pr_auc, created, job in rows:
        shown = f"{float(pr_auc):.4f}" if pr_auc not in ("-", None, "") else "-"
        ctx.out(f"{version:>7}  {status:<22} {shown:>8}  {created:<16}  {job}")
    return None


def cmd_show(ctx: Context, a: argparse.Namespace) -> str | None:
    d = describe_version(ctx, a.version)
    meta = d.get("CustomerMetadataProperties", {})
    container = d["InferenceSpecification"]["Containers"][0]
    ctx.out(f"model version {a.version} of {d['ModelPackageGroupName']}")
    ctx.out(f"  status        {d['ModelApprovalStatus']}" + (f"  ({d['ApprovalDescription']})" if d.get("ApprovalDescription") else ""))
    if d.get("ModelCard"):
        ctx.out(f"  model card    {d['ModelCard'].get('ModelCardStatus')}")
    ctx.out(f"  created       {when(d.get('CreationTime'))} UTC")
    ctx.out(f"  model file    {container['ModelDataUrl']}")
    ctx.out(f"  image         {container['Image']}")
    for key, value in meta.items():
        ctx.out(f"  {key:<22} {value}")
    return None


# ---------------------------------------------------------------------------- promote


def cmd_promote(ctx: Context, a: argparse.Namespace) -> str:
    detail = describe_version(ctx, a.version)
    status = detail["ModelApprovalStatus"]
    ctx.out(f"model version {a.version}: {status}")
    if status == "Rejected":
        raise MlopsError(f"model version {a.version} was Rejected: it must not run in any environment. Pick another version.", EXIT_UNUSABLE)
    if a.env in NEEDS_APPROVAL and status != "Approved":
        raise MlopsError(
            f"model version {a.version} is {status}, not Approved: {a.env} only runs approved models. "
            f"Run approve-model first (python pipeline/mlops.py approve --version {a.version} --decision approve).", EXIT_UNUSABLE)

    app_url = ""
    if not a.no_wait:
        # Look the app up BEFORE changing anything: promoting what we cannot verify is how a bad model hides.
        try:
            app_url = ctx.ssm.get_parameter(Name=f"/finrisk/{a.env}/app_url")["Parameter"]["Value"].strip()
        except Exception as exc:  # noqa: BLE001
            if error_code(exc) != "ParameterNotFound":
                raise
            raise MlopsError(f"/finrisk/{a.env}/app_url does not exist: the app is not deployed to {a.env} yet, so there is nothing "
                             "to check the model against. Deploy finrisk-app first (or use --no-wait to skip that check).") from exc

    ecfg = ctx.tcfg["endpoints"]
    result = deploy.deploy(ctx, a.env, a.version, detail, ctx.package_arn(a.version), status, app_url, run_url(ctx.env), no_wait=a.no_wait,
                           ready_timeout=a.timeout if a.timeout else float(ecfg.get("ready_wait_minutes", 10)) * 60, poll=a.poll)
    write_summary(a.summary, [f"- **{a.env}**: model version {a.version} ({status}) on endpoint `{deploy.endpoint_name(a.env)}` "
                              f"({result['kind']}, {'created' if result['created'] else 'updated'})"
                              + ("" if a.no_wait else f", confirmed by `/ready` after {result['ready_seconds']:.0f} s")])
    return str(a.version)


# ---------------------------------------------------------------------------- replay / feedback


def sim_app_url(ctx: Context, env_name: str) -> str:
    try:
        return ctx.ssm.get_parameter(Name=f"/finrisk/{env_name}/app_url")["Parameter"]["Value"].strip()
    except Exception as exc:  # noqa: BLE001
        if error_code(exc) == "ParameterNotFound":
            raise MlopsError(f"/finrisk/{env_name}/app_url does not exist: the app is not deployed to {env_name}.") from exc
        raise


def cmd_replay(ctx: Context, a: argparse.Namespace) -> str:
    import replay

    bucket = ctx.shared("ml_bucket")
    summary = replay.replay(ctx, a.app_url or sim_app_url(ctx, a.env), a.from_day, a.to_day, a.batch, bucket, a.explain_every)
    write_summary(a.summary, ["### Replay", "", "| day | rows | batches | seconds | flagged | true frauds |", "| --- | --- | --- | --- | --- | --- |"]
                  + [f"| {d['day']} | {d['rows']:,} | {d['batches']} | {d['seconds']} | {d['flagged']:,} | {d['frauds']:,} |" for d in summary["days"]]
                  + ["", f"Model version(s) that answered: {', '.join(summary['model_versions'])}. Features recorded in the Feature Store."])
    return f"{summary['rows']}"


def cmd_feedback(ctx: Context, a: argparse.Namespace) -> str:
    import feedback

    bucket = ctx.shared("ml_bucket")
    run_id = a.run_id or ctx.env.get("GITHUB_RUN_ID") or utc_stamp("%Y%m%d%H%M%S")
    summary = feedback.feedback(ctx, a.from_day, a.to_day, bucket, run_id, delay_minutes=a.label_delay_minutes)
    write_summary(a.summary, ["### Feedback", "", f"{summary['rows']:,} labels ({summary['frauds']:,} frauds) with label_time `{summary['label_time']}`."])
    return summary["key"]


# ---------------------------------------------------------------------------- monitor


def cmd_monitor(ctx: Context, a: argparse.Namespace) -> str | None:
    params = {"Env": a.env, "WindowHours": str(a.window_hours), "Reference": a.reference}
    ctx.out(f"starting {definition.MONITOR_PIPELINE} for {a.env} (window {a.window_hours} h)")
    arn = ctx.sm.start_pipeline_execution(PipelineName=definition.MONITOR_PIPELINE,
                                          PipelineParameters=[{"Name": k, "Value": v} for k, v in params.items()],
                                          ClientRequestToken=idempotency_token())["PipelineExecutionArn"]
    run_id = execution_id(arn)
    ctx.out(f"execution {run_id}")
    if a.no_wait:
        return run_id
    follow_args = argparse.Namespace(poll=a.poll, max_wait=a.max_wait, summary=a.summary)
    finish_execution(ctx, definition.MONITOR_PIPELINE, arn, follow_args)
    metrics: dict[str, Any] = {}
    try:
        key = f"monitoring/{a.env}/{run_id}/metrics.json"
        metrics = json.loads(ctx.s3.get_object(Bucket=ctx.shared("ml_bucket"), Key=key)["Body"].read())
    except Exception as exc:  # noqa: BLE001
        ctx.out(f"  (metrics.json is not readable yet: {error_code(exc) or type(exc).__name__})")
    lines = monitor_table(metrics)
    for line in lines:
        ctx.out(line)
    alarms = ctx.cw.describe_alarms(AlarmNames=model_alarm_names(a.env)).get("MetricAlarms", [])
    alarm_lines = [f"  {alarm['AlarmName']:<36} {alarm['StateValue']}" for alarm in sorted(alarms, key=lambda x: x["AlarmName"])] or \
        [f"  (no alarms named finrisk-{a.env}-model-*: finrisk-infra creates them in prod only)"]
    ctx.out("alarms:")
    for line in alarm_lines:
        ctx.out(line)
    write_summary(a.summary, ["### Monitor", "", "```", *lines, "```", "", "Alarms", "", "```", *alarm_lines, "```"])
    return run_id


def model_alarm_names(env: str) -> list[str]:
    """finrisk-<env>-model-drift / -prior-shift / -quality, from config/monitor.yaml. Exact names, never a prefix: IAM evaluates a
    prefix search on alarm:* and the least-privilege role is refused (found in the live rehearsal)."""
    import yaml

    cfg = yaml.safe_load((REPO_ROOT / "config" / "monitor.yaml").read_text())["alarms"]
    return [cfg["name_template"].format(env=env, kind=kind) for kind in cfg["kinds"]]


def monitor_table(summary: dict) -> list[str]:
    if not summary:
        return []
    psi = summary.get("psi") or {}
    lines = [f"  {'feature':<28} {'PSI':>8}  band"]
    for feature, v in sorted(psi.items(), key=lambda kv: -kv[1]["psi"]):
        lines.append(f"  {feature:<28} {v['psi']:>8.4f}  {v['band']}")
    metrics = summary.get("metrics") or {}
    if not metrics:
        lines.append("  no captured traffic in the window")
    for name, value in metrics.items():
        lines.append(f"  {name:<22} {value:.6g}")
    crossed = summary.get("thresholds_crossed") or []
    lines.append("  thresholds crossed: " + ("none" if not crossed else ""))
    lines += [f"    - {c}" for c in crossed]
    return lines


# ---------------------------------------------------------------------------- mlflow-url / teardown


def cmd_mlflow_url(ctx: Context, a: argparse.Namespace) -> str:
    arn = a.arn or ctx.shared_optional("mlflow_app_arn")
    if not arn:
        raise MlopsError("no MLflow app ARN: /finrisk/shared/mlflow_app_arn does not exist (finrisk-infra shared/ creates the app); pass --arn.")
    return ctx.sm.create_presigned_mlflow_app_url(Arn=arn)["AuthorizedUrl"]


def cmd_teardown(ctx: Context, a: argparse.Namespace) -> str:
    if a.confirm != a.env:
        raise MlopsError(f"teardown deletes the endpoint, configs and models of {a.env}. Type the environment name to confirm: --confirm {a.env}")
    result = deploy.teardown(ctx, a.env)
    write_summary(a.summary, [f"- **{a.env}** torn down: endpoint {'deleted' if result['endpoint_deleted'] else 'was absent'}, "
                              f"{result['configs']} config(s), {result['models']} model(s); pointer reset to null."])
    return a.env


# ---------------------------------------------------------------------------- command line


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ml-bucket", help="the ML bucket (env ML_BUCKET; else SSM /finrisk/shared/ml_bucket)")
    common.add_argument("--group", help=f"the model package group (env MODEL_PACKAGE_GROUP, default {DEFAULT_GROUP})")
    common.add_argument("--role-arn", help="the SageMaker execution role (env SAGEMAKER_ROLE_ARN; else SSM)")
    common.add_argument("--region", help=f"AWS Region (env AWS_REGION, default {DEFAULT_REGION})")
    common.add_argument("--config", default=str(DEFAULT_CONFIG), help="the training config (default config/train.yaml)")
    common.add_argument("--quiet", action="store_true", help="print only the final value (and errors)")
    common.add_argument("--summary", help="append a markdown note here (the GitHub job summary)")

    follow = argparse.ArgumentParser(add_help=False)
    follow.add_argument("--no-wait", action="store_true")
    follow.add_argument("--poll", type=float, default=30.0, help="seconds between status checks")
    follow.add_argument("--max-wait", type=float, default=6000.0, help="stop waiting (not the job) after this many seconds")

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("backfill", parents=[common, follow], help="load the full PaySim data (raw, Feature Store history, labels, production set)")
    p.add_argument("--image", help="the training image digest (env TRAIN_IMAGE)")
    p.add_argument("--days", help="history days, e.g. 1-17 (default config/data.yaml)")
    p.add_argument("--then-train", action="store_true", help="when the backfill succeeded: upsert the pipelines and start finrisk-train (TriggerReason=backfill); not followed")
    p.add_argument("--instance-count", type=int, default=1, help="split the history days over N parallel jobs (use it when the job says it is too slow)")

    p = sub.add_parser("upsert", parents=[common], help="write the pipelines finrisk-train and finrisk-monitor to SageMaker")
    p.add_argument("--image", help="the training image digest (env TRAIN_IMAGE): becomes the default of the TrainImage parameter")
    p.add_argument("--pipeline", choices=["both", *PIPELINES], default="both")

    p = sub.add_parser("run", parents=[common, follow], help="start a pipeline and follow it (default finrisk-train)")
    p.add_argument("--pipeline", choices=list(PIPELINES), default=definition.TRAIN_PIPELINE)
    p.add_argument("--image", help="override the TrainImage parameter (a digest)")
    p.add_argument("--trigger-reason", default="manual", help="manual | push | schedule | drift (recorded with the model version)")
    p.add_argument("--model-type", choices=["optimized", "baseline"], help="override ModelType (baseline = a worse model: watch the gate say no)")
    p.add_argument("--cutoff", help="override Cutoff (ISO time, or auto)")
    p.add_argument("--param", action="append", help="NAME=VALUE, any other pipeline parameter (repeatable)")

    p = sub.add_parser("status", parents=[common], help="the steps of one pipeline execution")
    p.add_argument("execution", help="an execution ARN or id")
    p.add_argument("--pipeline", default=definition.TRAIN_PIPELINE)

    p = sub.add_parser("approve", parents=[common], help="approve (or reject) a model version; the Model Card status follows")
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--decision", choices=["approve", "reject"], default="approve")
    p.add_argument("--note", help="why (stored with the decision)")

    p = sub.add_parser("promote", parents=[common], help="deploy a version to an environment's endpoint; wait for /ready to confirm")
    p.add_argument("--env", choices=ENVIRONMENTS, required=True)
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--poll", type=float, default=15.0, help="seconds between /ready checks")
    p.add_argument("--timeout", type=float, default=0.0, help="give up waiting for /ready after this many seconds (default: config, 10 min)")
    p.add_argument("--no-wait", action="store_true", help="do not wait for the app's /ready (the endpoint itself is always waited for)")

    p = sub.add_parser("list", parents=[common], help="the versions in the registry")
    p.add_argument("--limit", type=int, default=20)

    p = sub.add_parser("show", parents=[common], help="everything about one version")
    p.add_argument("--version", type=int, required=True)

    p = sub.add_parser("replay", parents=[common], help="send production days through the app (the simulated customers)")
    p.add_argument("--from-day", type=int, default=18)
    p.add_argument("--to-day", type=int, default=22)
    p.add_argument("--env", choices=ENVIRONMENTS, default="prod")
    p.add_argument("--batch", type=int, default=500)
    p.add_argument("--explain-every", type=int, default=100, help="ask for reasons on every Nth batch (slower)")
    p.add_argument("--app-url", help="override /finrisk/<env>/app_url")

    p = sub.add_parser("feedback", parents=[common], help="write the true labels of the replayed days (the simulated analysts)")
    p.add_argument("--from-day", type=int, default=18)
    p.add_argument("--to-day", type=int, default=22)
    p.add_argument("--label-delay-minutes", type=float, default=0.0)
    p.add_argument("--run-id", help="names the label batch (default GITHUB_RUN_ID)")

    p = sub.add_parser("monitor", parents=[common, follow], help="run finrisk-monitor and show the drift table and the alarm states")
    p.add_argument("--env", choices=ENVIRONMENTS, default="prod")
    p.add_argument("--window-hours", type=int, default=24)
    p.add_argument("--reference", default="auto")

    p = sub.add_parser("mlflow-url", parents=[common], help="a presigned URL to the MLflow App UI")
    p.add_argument("--arn")

    p = sub.add_parser("teardown", parents=[common], help="delete an environment's endpoint, configs and models")
    p.add_argument("--env", choices=ENVIRONMENTS, required=True)
    p.add_argument("--confirm", default="", help="type the environment name again")
    return parser


COMMANDS = {
    "backfill": cmd_backfill, "upsert": cmd_upsert, "run": cmd_run, "status": cmd_status, "approve": cmd_approve, "promote": cmd_promote,
    "list": cmd_list, "show": cmd_show, "replay": cmd_replay, "feedback": cmd_feedback, "monitor": cmd_monitor,
    "mlflow-url": cmd_mlflow_url, "teardown": cmd_teardown,
}


def main(argv: list[str] | None = None, clients: dict[str, Any] | None = None, env: dict[str, str] | None = None,
         out: Callable[[str], None] | None = None, sleep: Callable[[float], None] = time.sleep,
         clock: Callable[[], float] = time.monotonic, http: Callable[[str, float], tuple[int, str]] = http_get,
         post: Callable[[str, dict, float], tuple[int, dict]] = http_post, now: Callable[[], datetime] = utc_now) -> int:
    """Run one sub-command; return the exit code. ``clients`` lets a test pass fakes instead of boto3."""
    args = build_parser().parse_args(argv)
    env = dict(os.environ if env is None else env)
    echo = out or (lambda line: print(line, flush=True))
    say = (lambda line: None) if args.quiet else (lambda line: echo(mask(line)))   # ids hidden in everything but the result
    try:
        cfg = load_config(args, env)
        tcfg = load_train_config(Path(args.config))
        raw = clients or make_clients(cfg.region)
        quiet = {
            "sagemaker": ("describe_pipeline_execution", "list_pipeline_execution_steps", "describe_processing_job", "describe_endpoint"),
            "fs": ("batch_write_record",),
        }
        wrapped = {key: Traced(client, CLIENT_NAMES.get(key, key), say, quiet.get(key, ())) for key, client in raw.items()}
        ctx = Context(
            cfg=cfg, tcfg=tcfg, sm=wrapped["sagemaker"], s3=wrapped["s3"], sts=wrapped["sts"], ssm=wrapped["ssm"],
            runtime=wrapped.get("runtime"), fs=wrapped.get("fs"), cw=wrapped.get("cw"),
            out=say, sleep=sleep, clock=clock, env=env, http=http, post=post, now=now)
        result = COMMANDS[args.command](ctx, args)
    except MlopsError as exc:
        print(f"error: {mask(str(exc))}", file=sys.stderr)
        return exc.code
    except Exception as exc:  # noqa: BLE001 - an AWS error should be one readable line, not a traceback
        print(f"error: {type(exc).__name__}: {mask(str(exc))}", file=sys.stderr)
        if "botocore[crt]" in str(exc):
            print("hint: you signed in with `aws login`. Either `pip install \"botocore[crt]\"`, or hand boto3 plain "
                  "credentials: eval \"$(aws configure export-credentials --format env)\"", file=sys.stderr)
        return EXIT_FAILED
    if result is not None:
        echo(result)                  # the value the next step needs -- always the last line, never masked
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
