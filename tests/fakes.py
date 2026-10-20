"""A fake AWS, small enough to read, strict enough to catch a misspelled request.

Every request the code sends is checked against botocore's own copy of the AWS API definition (the same check boto3 runs
before it sends anything), so a wrong parameter name or type fails HERE, not in a live run. Nothing touches the network.

    FakeS3           objects in a dict (upload/download/put/get/list/head/delete)
    FakeSTS, FakeSSM
    FakeSageMaker    registry, pipelines, processing jobs, models/endpoint configs/endpoints (with scripted status changes)
    FakeRuntime      invoke_endpoint (sagemaker-runtime): answers like the endpoint contract
    FakeFeatureStore batch_write_record (sagemaker-featurestore-runtime), optional throttling / partial failures
    FakeCloudWatch   put_metric_data, describe_alarms
    FakeAthena       runs the REAL SQL on DuckDB tables (finrisk_ml.transactions / finrisk_ml.labels); UNLOAD writes parquet to FakeS3
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import re
import tempfile
from pathlib import Path

import botocore.session
from botocore.exceptions import ClientError
from botocore.validate import validate_parameters

import mlops

ACCOUNT = "123456789012"
REGION = "ap-southeast-2"
BUCKET = f"finrisk-ml-{ACCOUNT}-{REGION}-an"
GROUP = "finrisk-fraud-model"
ROLE = f"arn:aws:iam::{ACCOUNT}:role/finrisk-sagemaker-exec"
IMAGE = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/finrisk-train@sha256:{'ab' * 32}"
APP_URL = "https://abc123.lambda-url.ap-southeast-2.on.aws"
ENV = {"ML_BUCKET": BUCKET, "MODEL_PACKAGE_GROUP": GROUP, "SAGEMAKER_ROLE_ARN": ROLE, "AWS_REGION": REGION}
NOW = dt.datetime(2026, 10, 16, 3, 0, 0, tzinfo=dt.timezone.utc)

_SESSION = botocore.session.get_session()
CHECKED: set[tuple[str, str]] = set()          # every (service, operation) whose request shape a test validated


def check_shape(service: str, operation: str, params: dict) -> None:
    """Raise if AWS would reject the request's SHAPE (names, types, required fields)."""
    model = _SESSION.get_service_model(service).operation_model(operation)
    validate_parameters(params, model.input_shape)
    CHECKED.add((service, operation))


def client_error(code: str, operation: str = "Op", message: str | None = None) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": message or code}}, operation)


class FakeSTS:
    def get_caller_identity(self):
        return {"Account": ACCOUNT}


class FakeS3:
    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}
        self.uploads: list[tuple[str, str, str]] = []
        self.puts: list[tuple[str, str]] = []

    # -- writing
    def upload_file(self, Filename, Bucket, Key):                       # noqa: N803 - boto3's own names
        self.uploads.append((Filename, Bucket, Key))
        self.objects[(Bucket, Key)] = Path(Filename).read_bytes()

    def put_object(self, **kw):
        check_shape("s3", "PutObject", kw)
        body = kw["Body"]
        self.puts.append((kw["Bucket"], kw["Key"]))
        self.objects[(kw["Bucket"], kw["Key"])] = body if isinstance(body, bytes) else body.encode()
        return {}

    def delete_objects(self, **kw):
        check_shape("s3", "DeleteObjects", kw)
        for item in kw["Delete"]["Objects"]:
            self.objects.pop((kw["Bucket"], item["Key"]), None)
        return {}

    # -- reading
    def download_file(self, Bucket, Key, Filename):                     # noqa: N803
        if (Bucket, Key) not in self.objects:
            raise client_error("404", "HeadObject")
        Path(Filename).parent.mkdir(parents=True, exist_ok=True)
        Path(Filename).write_bytes(self.objects[(Bucket, Key)])

    def get_object(self, **kw):
        check_shape("s3", "GetObject", kw)
        if (kw["Bucket"], kw["Key"]) not in self.objects:
            raise client_error("NoSuchKey", "GetObject")
        return {"Body": io.BytesIO(self.objects[(kw["Bucket"], kw["Key"])])}

    def head_object(self, **kw):
        check_shape("s3", "HeadObject", kw)
        if (kw["Bucket"], kw["Key"]) not in self.objects:
            raise client_error("404", "HeadObject")
        return {}

    def list_objects_v2(self, **kw):
        check_shape("s3", "ListObjectsV2", kw)
        bucket, prefix, delimiter = kw["Bucket"], kw.get("Prefix", ""), kw.get("Delimiter")
        keys = sorted(k for (b, k) in self.objects if b == bucket and k.startswith(prefix))
        if delimiter:
            folders = sorted({k[len(prefix):].split(delimiter)[0] for k in keys if delimiter in k[len(prefix):]})
            return {"CommonPrefixes": [{"Prefix": f"{prefix}{f}{delimiter}"} for f in folders], "IsTruncated": False}
        return {"Contents": [{"Key": k, "Size": len(self.objects[(bucket, k)])} for k in keys], "IsTruncated": False}

    # -- helpers for tests
    def put(self, uri_or_bucket, key=None, body=b""):
        if key is None:
            bucket, _, key = uri_or_bucket.removeprefix("s3://").partition("/")
        else:
            bucket = uri_or_bucket
        self.objects[(bucket, key)] = body if isinstance(body, bytes) else body.encode()

    def text(self, key, bucket=BUCKET):
        return self.objects[(bucket, key)].decode()

    def json(self, key, bucket=BUCKET):
        return json.loads(self.objects[(bucket, key)])

    def keys(self, prefix="", bucket=BUCKET):
        return sorted(k for (b, k) in self.objects if b == bucket and k.startswith(prefix))


class FakeSSM:
    def __init__(self):
        self.params: dict[str, str] = {}
        self.puts: list[dict] = []
        self.denied: set[str] = set()                                   # names whose read is AccessDenied

    def get_parameter(self, Name):                                      # noqa: N803
        check_shape("ssm", "GetParameter", {"Name": Name})
        if Name in self.denied:
            raise client_error("AccessDeniedException", "GetParameter")
        if Name not in self.params:
            raise client_error("ParameterNotFound", "GetParameter")
        return {"Parameter": {"Name": Name, "Value": self.params[Name]}}

    def put_parameter(self, **kw):
        check_shape("ssm", "PutParameter", kw)
        self.puts.append(kw)
        self.params[kw["Name"]] = kw["Value"]
        return {"Version": len(self.puts)}


class FakeSageMaker:
    """Registry, pipelines, processing jobs, endpoints. State changes are scripted so a test can watch a wait loop."""

    def __init__(self, s3: FakeS3, pr_auc=0.8631):
        self.s3, self.pr_auc = s3, pr_auc
        self.calls: list[tuple[str, dict]] = []
        # registry
        self.packages: dict[int, dict] = {}
        # pipelines
        self.pipelines: dict[str, dict] = {}
        self.executions: dict[str, dict] = {}
        self.exec_script: list[dict] = []                               # snapshots served one per describe_pipeline_execution
        self._exec_polls = 0
        # processing jobs
        self.jobs: dict[str, dict] = {}
        self.job_script: dict[str, list[tuple[str, str | None]]] = {}
        self._job_polls: dict[str, int] = {}
        # endpoints
        self.models: dict[str, dict] = {}
        self.configs: dict[str, dict] = {}
        self.endpoints: dict[str, dict] = {}
        self.endpoint_script: list[tuple[str, str | None, str | None]] = []   # (status, config override, failure) after create/update
        self._created = 0

    def _log(self, name, kw):
        self.calls.append((name, kw))

    def called(self, name):
        return [kw for n, kw in self.calls if n == name]

    # ------------------------------------------------------------ registry
    def add_package(self, version=None, status="PendingManualApproval", pr_auc=None, **meta):
        version = version or len(self.packages) + 1
        arn = f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:model-package/{GROUP}/{version}"
        job = f"finrisk-train-{version:03d}-aaaaaaaa"
        metadata = {"training_job": job, "pr_auc": f"{(pr_auc if pr_auc is not None else self.pr_auc):.6f}", "train_image_digest": "sha256:" + "ab" * 32,
                    "dataset_uri": f"s3://{BUCKET}/datasets/exec{version}", "baseline_uri": f"s3://{BUCKET}/datasets/exec{version}/profile/baseline.parquet",
                    "cutoff": "2026-09-19T23:00:00Z", "ml_git_sha": "c" * 40, "trigger_reason": "manual", **meta}
        package = {
            "ModelPackageArn": arn, "ModelPackageGroupName": GROUP, "ModelPackageVersion": version,
            "InferenceSpecification": {"Containers": [{"Image": IMAGE, "ModelDataUrl": f"s3://{BUCKET}/models/{job}/output/model.tar.gz"}],
                                       "SupportedContentTypes": ["application/json"], "SupportedResponseMIMETypes": ["application/json"]},
            "ModelApprovalStatus": status, "CustomerMetadataProperties": metadata,
            "CreationTime": dt.datetime(2026, 10, 14, 3, version), "ModelCard": {"ModelCardContent": "{}", "ModelCardStatus": "Draft"},
        }
        self.packages[version] = package
        return version

    def _find(self, arn):
        version = int(arn.rsplit("/", 1)[1])
        if version not in self.packages:
            raise client_error("ValidationException", "DescribeModelPackage")
        return self.packages[version]

    def describe_model_package(self, ModelPackageName):                 # noqa: N803
        check_shape("sagemaker", "DescribeModelPackage", {"ModelPackageName": ModelPackageName})
        return dict(self._find(ModelPackageName))

    def update_model_package(self, **kw):
        check_shape("sagemaker", "UpdateModelPackage", kw)
        self._log("update_model_package", kw)
        package = self._find(kw["ModelPackageArn"])
        package["ModelApprovalStatus"] = kw["ModelApprovalStatus"]
        if "ApprovalDescription" in kw:
            package["ApprovalDescription"] = kw["ApprovalDescription"]
        if "ModelCard" in kw:
            package["ModelCard"] = {**package.get("ModelCard", {}), **kw["ModelCard"]}
        return {"ModelPackageArn": kw["ModelPackageArn"]}

    def list_model_packages(self, **kw):
        check_shape("sagemaker", "ListModelPackages", kw)
        newest_first = sorted(self.packages.values(), key=lambda p: -p["ModelPackageVersion"])
        return {"ModelPackageSummaryList": [{k: p[k] for k in ("ModelPackageArn", "ModelPackageVersion", "ModelApprovalStatus", "CreationTime")}
                                            for p in newest_first]}

    # ------------------------------------------------------------ pipelines
    def describe_pipeline(self, **kw):
        check_shape("sagemaker", "DescribePipeline", kw)
        if kw["PipelineName"] not in self.pipelines:
            raise client_error("ResourceNotFound", "DescribePipeline", "Pipeline does not exist")
        return {"PipelineName": kw["PipelineName"], **self.pipelines[kw["PipelineName"]]}

    def create_pipeline(self, **kw):
        check_shape("sagemaker", "CreatePipeline", kw)
        self._log("create_pipeline", kw)
        json.loads(kw["PipelineDefinition"])
        self.pipelines[kw["PipelineName"]] = {"PipelineDefinition": kw["PipelineDefinition"], "RoleArn": kw["RoleArn"]}
        return {"PipelineArn": f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:pipeline/{kw['PipelineName']}"}

    def update_pipeline(self, **kw):
        check_shape("sagemaker", "UpdatePipeline", kw)
        self._log("update_pipeline", kw)
        json.loads(kw["PipelineDefinition"])
        self.pipelines[kw["PipelineName"]].update({"PipelineDefinition": kw["PipelineDefinition"], "RoleArn": kw["RoleArn"]})
        return {"PipelineArn": f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:pipeline/{kw['PipelineName']}"}

    def start_pipeline_execution(self, **kw):
        check_shape("sagemaker", "StartPipelineExecution", kw)
        self._log("start_pipeline_execution", kw)
        arn = f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:pipeline/{kw['PipelineName']}/execution/abc{len(self.executions) + 1:05d}"
        self.executions[arn] = kw
        self._exec_polls = 0
        if not self.exec_script:
            self.script_execution("registered")
        return {"PipelineExecutionArn": arn}

    def script_execution(self, outcome="registered", version=None, monitor=False):
        """Snapshots, one per poll: running -> ... -> the final state. outcome: registered | gate_failed | step_failed | monitor_ok."""
        version = version or len(self.packages) + 1
        pending = lambda n: {"StepName": n, "StepStatus": "Executing"}              # noqa: E731
        done = lambda n, **x: {"StepName": n, "StepStatus": "Succeeded", **x}       # noqa: E731
        if monitor or outcome == "monitor_ok":
            self.exec_script = [{"status": "Executing", "steps": [pending("Monitor")]}, {"status": "Succeeded", "steps": [done("Monitor")]}]
            return
        base = [done("BuildDataset"), done("Tune")]
        if outcome == "registered":
            arn = f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:model-package/{GROUP}/{version}"
            final = {"status": "Succeeded", "steps": base + [done("Evaluate"), done("Gate"), done("Register", Metadata={"RegisterModel": {"Arn": arn}})]}
        elif outcome == "gate_failed":
            final = {"status": "Failed", "steps": base + [done("Evaluate"), done("Gate"),
                     {"StepName": "GateFailed", "StepStatus": "Failed", "FailureReason": "GATE FAILED: candidate pr_auc 0.6000 < floor 0.7"}]}
        else:
            final = {"status": "Failed", "steps": [done("BuildDataset"), {"StepName": "Tune", "StepStatus": "Failed", "FailureReason": "ClientError: trial 3 failed"}]}
        self.exec_script = [{"status": "Executing", "steps": [pending("BuildDataset")]},
                            {"status": "Executing", "steps": [done("BuildDataset"), pending("Tune")]}, final]

    def _snapshot(self):
        return self.exec_script[min(self._exec_polls, len(self.exec_script) - 1)]

    def describe_pipeline_execution(self, **kw):
        check_shape("sagemaker", "DescribePipelineExecution", kw)
        snap = self._snapshot()
        self._exec_polls += 1
        return {"PipelineExecutionArn": kw["PipelineExecutionArn"], "PipelineExecutionStatus": snap["status"]}

    def list_pipeline_execution_steps(self, **kw):
        check_shape("sagemaker", "ListPipelineExecutionSteps", kw)
        return {"PipelineExecutionSteps": self._snapshot()["steps"]}

    # ------------------------------------------------------------ processing
    def create_processing_job(self, **kw):
        check_shape("sagemaker", "CreateProcessingJob", kw)
        self._log("create_processing_job", kw)
        self.jobs[kw["ProcessingJobName"]] = kw
        self.job_script.setdefault(kw["ProcessingJobName"], [("InProgress", None), ("Completed", None)])
        return {"ProcessingJobArn": f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:processing-job/{kw['ProcessingJobName']}"}

    def describe_processing_job(self, **kw):
        check_shape("sagemaker", "DescribeProcessingJob", kw)
        name = kw["ProcessingJobName"]
        script = self.job_script[name]
        i = self._job_polls.get(name, 0)
        self._job_polls[name] = i + 1
        status, reason = script[min(i, len(script) - 1)]
        out = {"ProcessingJobName": name, "ProcessingJobStatus": status}
        if reason:
            out["FailureReason"] = reason
        return out

    def describe_hyper_parameter_tuning_job(self, **kw):
        check_shape("sagemaker", "DescribeHyperParameterTuningJob", kw)
        return {"HyperParameterTuningJobName": kw["HyperParameterTuningJobName"], "HyperParameterTuningJobStatus": "Completed",
                "TrainingJobStatusCounters": {"Completed": 8}, "BestTrainingJob": {"FinalHyperParameterTuningJobObjectiveMetric": {"Value": 0.9}}}

    def create_presigned_mlflow_app_url(self, **kw):
        check_shape("sagemaker", "CreatePresignedMlflowAppUrl", kw)
        self._log("create_presigned_mlflow_app_url", kw)
        return {"AuthorizedUrl": "https://mlflow.example/presigned?token=abc"}

    # ------------------------------------------------------------ models, configs, endpoints
    def create_model(self, **kw):
        check_shape("sagemaker", "CreateModel", kw)
        self._log("create_model", kw)
        self._created += 1
        self.models[kw["ModelName"]] = {**kw, "_n": self._created}
        return {"ModelArn": f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:model/{kw['ModelName']}"}

    def create_endpoint_config(self, **kw):
        check_shape("sagemaker", "CreateEndpointConfig", kw)
        self._log("create_endpoint_config", kw)
        self._created += 1
        self.configs[kw["EndpointConfigName"]] = {**kw, "_n": self._created}
        return {"EndpointConfigArn": f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:endpoint-config/{kw['EndpointConfigName']}"}

    def _start_script(self, name, config, previous):
        default = [("Creating", config, None), ("InService", config, None)] if previous is None else \
                  [("Updating", previous, None), ("Updating", previous, None), ("InService", config, None)]
        self.endpoints[name]["_script"] = list(self.endpoint_script or default)
        self.endpoint_script = []
        self.endpoints[name]["_i"] = 0

    def create_endpoint(self, **kw):
        check_shape("sagemaker", "CreateEndpoint", kw)
        self._log("create_endpoint", kw)
        self.endpoints[kw["EndpointName"]] = {"EndpointName": kw["EndpointName"], "EndpointConfigName": kw["EndpointConfigName"]}
        self._start_script(kw["EndpointName"], kw["EndpointConfigName"], None)
        return {"EndpointArn": f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:endpoint/{kw['EndpointName']}"}

    def update_endpoint(self, **kw):
        check_shape("sagemaker", "UpdateEndpoint", kw)
        self._log("update_endpoint", kw)
        endpoint = self.endpoints[kw["EndpointName"]]
        previous = endpoint["EndpointConfigName"]
        endpoint["EndpointConfigName"] = previous
        self._start_script(kw["EndpointName"], kw["EndpointConfigName"], previous)
        return {"EndpointArn": f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:endpoint/{kw['EndpointName']}"}

    def describe_endpoint(self, **kw):
        check_shape("sagemaker", "DescribeEndpoint", kw)
        name = kw["EndpointName"]
        if name not in self.endpoints:
            raise client_error("ValidationException", "DescribeEndpoint", f"Could not find endpoint \"arn:aws:sagemaker:{REGION}:{ACCOUNT}:endpoint/{name}\".")
        ep = self.endpoints[name]
        script = ep.get("_script")
        if script:
            status, config, failure = script[min(ep["_i"], len(script) - 1)]
            ep["_i"] += 1
            ep["EndpointStatus"], ep["EndpointConfigName"] = status, config
            if failure:
                ep["FailureReason"] = failure
        ep.setdefault("EndpointStatus", "InService")
        return {k: v for k, v in ep.items() if not k.startswith("_")}

    def delete_endpoint(self, **kw):
        check_shape("sagemaker", "DeleteEndpoint", kw)
        self._log("delete_endpoint", kw)
        self.endpoints.pop(kw["EndpointName"], None)
        return {}

    def delete_endpoint_config(self, **kw):
        check_shape("sagemaker", "DeleteEndpointConfig", kw)
        self._log("delete_endpoint_config", kw)
        self.configs.pop(kw["EndpointConfigName"], None)
        return {}

    def delete_model(self, **kw):
        check_shape("sagemaker", "DeleteModel", kw)
        self._log("delete_model", kw)
        self.models.pop(kw["ModelName"], None)
        return {}

    def list_endpoint_configs(self, **kw):
        check_shape("sagemaker", "ListEndpointConfigs", kw)
        rows = [c for c in self.configs.values() if kw.get("NameContains", "") in c["EndpointConfigName"]]
        rows.sort(key=lambda c: -c["_n"])
        return {"EndpointConfigs": [{"EndpointConfigName": c["EndpointConfigName"], "CreationTime": NOW} for c in rows]}

    def list_models(self, **kw):
        check_shape("sagemaker", "ListModels", kw)
        rows = [m for m in self.models.values() if kw.get("NameContains", "") in m["ModelName"]]
        rows.sort(key=lambda m: -m["_n"])
        return {"Models": [{"ModelName": m["ModelName"], "CreationTime": NOW} for m in rows]}


class FakeRuntime:
    """sagemaker-runtime: answers like the endpoint contract. The version it reports is the one in the config name (-v<N>-)."""

    def __init__(self, sm: FakeSageMaker):
        self.sm = sm
        self.requests: list[dict] = []
        self.fail_times = 0
        self.force_version = None
        self.status = 200

    def invoke_endpoint(self, **kw):
        check_shape("sagemaker-runtime", "InvokeEndpoint", kw)
        self.requests.append(kw)
        if self.fail_times:
            self.fail_times -= 1
            raise client_error("ModelError", "InvokeEndpoint", "timed out while the serverless endpoint was starting")
        body = json.loads(kw["Body"])
        config = self.sm.endpoints[kw["EndpointName"]]["EndpointConfigName"]
        version = self.force_version or re.search(r"-v(\d+)-", config).group(1)
        predictions = [{"row_id": str(i["row_id"]), "score": 0.9 if float(i["amount_to_origin_balance"]) >= 1 else 0.02,
                        "flagged": float(i["amount_to_origin_balance"]) >= 1} for i in body["instances"]]
        payload = {"model_version": str(version), "threshold": 0.4812, "predictions": predictions}
        return {"Body": io.BytesIO(json.dumps(payload).encode()), "ContentType": "application/json", "ResponseMetadata": {"HTTPStatusCode": self.status}}


class FakeFeatureStore:
    def __init__(self):
        self.records: dict[str, dict[str, str]] = {}
        self.calls = 0
        self.throttle_first = 0
        self.fail_entries_once = 0           # report this many entries of the first call as failed
        self._failed = False
        self.lock = __import__("threading").Lock()

    def batch_write_record(self, **kw):
        check_shape("sagemaker-featurestore-runtime", "BatchWriteRecord", kw)
        with self.lock:
            self.calls += 1
            if self.throttle_first > 0:
                self.throttle_first -= 1
                raise client_error("ThrottlingException", "BatchWriteRecord")
            errors = []
            if self.fail_entries_once and not self._failed:
                self._failed = True
                errors = [{"Entry": e, "ErrorCode": "InternalFailure", "ErrorMessage": "try again"} for e in kw["Entries"][: self.fail_entries_once]]
            skip = {id(x["Entry"]) for x in errors}
            for entry in kw["Entries"]:
                if id(entry) in skip:
                    continue
                values = {f["FeatureName"]: f["ValueAsString"] for f in entry["Record"]}
                self.records[values["row_id"]] = values
        return {"Errors": errors, "UnprocessedEntries": []}


class FakeCloudWatch:
    def __init__(self):
        self.puts: list[dict] = []
        self.alarms: dict[str, str] = {}

    def put_metric_data(self, **kw):
        check_shape("cloudwatch", "PutMetricData", kw)
        self.puts.append(kw)
        return {}

    def describe_alarms(self, **kw):
        check_shape("cloudwatch", "DescribeAlarms", kw)
        names = kw.get("AlarmNames")
        prefix = kw.get("AlarmNamePrefix", "")
        if prefix:    # IAM evaluates a prefix search on alarm:* -- the least-privilege role finrisk-ml-train is refused (live rehearsal)
            raise client_error("AccessDenied", "DescribeAlarms", "not authorized to perform: cloudwatch:DescribeAlarms on the resource alarm:*")
        rows = [{"AlarmName": n, "StateValue": s} for n, s in self.alarms.items() if (n in names if names else n.startswith(prefix))]
        return {"MetricAlarms": rows}


class FakeAthena:
    """start_query_execution runs the SQL on DuckDB (the Trino function from_iso8601_timestamp is defined as a macro).

    SELECTs write their result as CSV to s3://<bucket>/athena/<id>.csv like the real service; UNLOAD writes one parquet file under the
    destination prefix. Tables: register("transactions", df) / register("labels", df) -> finrisk_ml.<name>."""

    def __init__(self, s3: FakeS3, bucket: str = BUCKET):
        import duckdb

        self.s3, self.bucket = s3, bucket
        self.con = duckdb.connect()
        self.con.execute("CREATE SCHEMA finrisk_ml")
        self.con.execute("CREATE MACRO from_iso8601_timestamp(s) AS CAST(s AS TIMESTAMPTZ)")
        self.queries: list[str] = []
        self.fail_with: str | None = None
        self._n = 0
        self._state: dict = {}

    def register(self, name, frame):
        self.con.register(f"_{name}_df", frame)
        self.con.execute(f"CREATE OR REPLACE TABLE finrisk_ml.{name} AS SELECT * FROM _{name}_df")

    def start_query_execution(self, **kw):
        check_shape("athena", "StartQueryExecution", kw)
        self._n += 1
        qid = f"q{self._n:04d}"
        sql = kw["QueryString"].strip()
        self.queries.append(sql)
        state, reason, location = "SUCCEEDED", None, f"s3://{self.bucket}/athena/{qid}.csv"
        if self.fail_with:
            state, reason = "FAILED", self.fail_with
        else:
            unload = re.match(r"UNLOAD \((.*)\)\s+TO '([^']+)'\s+WITH \((.*)\)\s*$", sql, re.S)
            if unload:
                dest = unload.group(2).removeprefix("s3://")
                bucket, _, prefix = dest.partition("/")
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "out.parquet"
                    self.con.execute(f"COPY ({unload.group(1)}) TO '{path}' (FORMAT PARQUET)")
                    self.s3.put(bucket, f"{prefix}{qid}_00000", path.read_bytes())
            else:
                cursor = self.con.execute(sql)
                header = [d[0] for d in cursor.description]
                buffer = io.StringIO()
                writer = csv.writer(buffer, quoting=csv.QUOTE_ALL)
                writer.writerow(header)
                for row in cursor.fetchall():
                    writer.writerow(["" if v is None else v for v in row])
                self.s3.put(self.bucket, f"athena/{qid}.csv", buffer.getvalue())
        self._state[qid] = (state, reason, location)
        return {"QueryExecutionId": qid}

    def get_query_execution(self, **kw):
        check_shape("athena", "GetQueryExecution", kw)
        state, reason, location = self._state[kw["QueryExecutionId"]]
        return {"QueryExecution": {"QueryExecutionId": kw["QueryExecutionId"], "Status": {"State": state, **({"StateChangeReason": reason} if reason else {})},
                                   "Statistics": {"DataScannedInBytes": 1_000_000, "TotalExecutionTimeInMillis": 1500},
                                   "ResultConfiguration": {"OutputLocation": location}}}


class LaggingAthena(FakeAthena):
    """An Athena whose view of the Feature Store's offline store GROWS: before each query, `finrisk_ml.transactions` is rebuilt from the
    records written so far, of which only the first `schedule[i]` (a fraction) are 'visible' at the i-th query (the last value repeats).
    This is the delay between a BatchWriteRecord and the record being queryable."""

    def __init__(self, s3: FakeS3, fs: FakeFeatureStore, schedule=(1.0,)):
        super().__init__(s3)
        self.fs, self.schedule, self.polls = fs, list(schedule), 0

    def start_query_execution(self, **kw):
        import pandas as pd

        fraction = self.schedule[min(self.polls, len(self.schedule) - 1)]
        self.polls += 1
        records = list(self.fs.records.values())[: int(len(self.fs.records) * fraction)]
        self.register("transactions", pd.DataFrame({"row_id": [r["row_id"] for r in records], "event_time": [r["event_time"] for r in records],
                                                    "is_deleted": [False] * len(records)}) if records else
                      pd.DataFrame({"row_id": pd.Series(dtype=str), "event_time": pd.Series(dtype=str), "is_deleted": pd.Series(dtype=bool)}))
        return super().start_query_execution(**kw)


class FakeApp:
    """The running app's /ready, scripted: a list of (status, body) answers, the last one repeats."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.urls: list[str] = []

    def __call__(self, url, timeout):
        self.urls.append(url)
        answer = self.answers[min(len(self.urls) - 1, len(self.answers) - 1)]
        if isinstance(answer, Exception):
            raise answer
        status, body = answer
        return status, body if isinstance(body, str) else json.dumps(body)


class FakeScoreApp:
    """POST /api/score: checks the contract (exactly row_id + the 14 features) and answers like the app."""

    def __init__(self):
        self.requests: list[dict] = []
        self.statuses: list[int] = []        # scripted statuses before the first 200 (e.g. [503, 503])

    def __call__(self, url, body, timeout):
        from finrisk_jobs.contract import FEATURE_COLUMNS

        self.requests.append({"url": url, **body})
        if self.statuses:
            status = self.statuses.pop(0)
            return status, {"error": "model_unavailable", "detail": "model endpoint is starting, retry"}
        keys = {"row_id", *FEATURE_COLUMNS}
        for tx in body["transactions"]:
            assert set(tx) == keys, f"bad keys: {set(tx) ^ keys}"
            assert isinstance(tx["row_id"], str)
        assert 1 <= len(body["transactions"]) <= 500
        results = [{"row_id": t["row_id"], "score": 0.8 if t["amount_to_origin_balance"] >= 1 else 0.01, "flagged": t["amount_to_origin_balance"] >= 1}
                   for t in body["transactions"]]
        return 200, {"model_version": "1", "threshold": 0.5, "scored_by": "endpoint", "results": results}


def ready(version, source="endpoint", **extra):
    """What /ready says (the part that matters here): which model version is loaded, and from where."""
    return (200, {"ready": True, "model": {"version": version, "source": source, "loaded_at": "2026-10-16T03:00:20+00:00"}, **extra})


class World:
    """One fake AWS plus a way to run the CLI against it and read what it printed."""

    def __init__(self, capsys, backfilled=True, **sm_options):
        self.capsys = capsys
        self.s3 = FakeS3()
        self.ssm = FakeSSM()
        self.sm = FakeSageMaker(self.s3, **sm_options)
        self.runtime = FakeRuntime(self.sm)
        self.fs = FakeFeatureStore()
        self.cw = FakeCloudWatch()
        self.athena = FakeAthena(self.s3)
        self.clients = {"sagemaker": self.sm, "s3": self.s3, "sts": FakeSTS(), "ssm": self.ssm, "runtime": self.runtime, "fs": self.fs, "cw": self.cw}
        self.clock_now = 0.0
        self.app = FakeApp([ready(1)])
        self.score_app = FakeScoreApp()
        if backfilled:
            self.s3.put(BUCKET, "raw/paysim/v1/backfill_report.json", "{}")        # the backfill has run (it writes this file LAST)

    def run(self, *argv: str, env: dict | None = None, app: FakeApp | None = None, post=None) -> tuple[int, str, str]:
        def clock():                       # time only moves when the pipeline sleeps
            return self.clock_now

        def sleep(seconds):
            self.clock_now += seconds

        code = mlops.main(list(argv), clients=self.clients, env={**ENV, **(env or {})}, sleep=sleep, clock=clock,
                          http=app or self.app, post=post or self.score_app, now=lambda: NOW + dt.timedelta(seconds=self.clock_now))
        captured = self.capsys.readouterr()
        return code, captured.out, captured.err

    # helpers that set up the state a later command needs
    def set_pointer(self, env_name, version, **extra):
        self.ssm.params[f"/finrisk/{env_name}/model"] = json.dumps(
            {"version": version, "serving": "endpoint", "endpoint": f"finrisk-{env_name}-fraud", "endpoint_config": "x",
             "model_package_arn": f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:model-package/{GROUP}/{version}", "status": "Approved",
             "promoted_at": "2026-10-01T00:00:00+00:00", "run": "local", **extra})

    def set_app_url(self, env_name):
        self.ssm.params[f"/finrisk/{env_name}/app_url"] = APP_URL

    def add_alarms(self, env_name):
        for name in (f"finrisk-{env_name}-endpoint-5xx", f"finrisk-{env_name}-endpoint-latency"):
            self.cw.alarms[name] = "OK"

    def approve(self, version, decision="approve"):
        code, out, err = self.run("approve", "--version", str(version), "--decision", decision)
        assert code == 0, err


def last_line(text: str) -> str:
    return text.strip().splitlines()[-1]
