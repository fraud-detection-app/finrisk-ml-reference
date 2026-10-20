"""Helpers shared by the jobs: AWS clients, S3, Athena, config files, the finrisk-app import, MLflow-safe logging.

Every job prints what it does (`say`) and every AWS call that matters (an Athena query, a PutMetricData): a job log
is read by a person in CloudWatch, so it has to tell the story.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from . import contract

PACKAGE_DIR = Path(__file__).resolve().parent
CODE_ROOT = PACKAGE_DIR.parent              # the repo root, or /opt/program inside the image (same relative layout)
CONFIG_DIR = Path(os.environ.get("FINRISK_CONFIG_DIR", CODE_ROOT / "config"))
PIPELINE_DIR = CODE_ROOT / "pipeline"
IMAGE_APP_SRC = Path("/app/src")            # where the app image keeps the financial_risk package
ACCOUNT_ID = re.compile(r"(?<!\d)\d{12}(?!\d)")


def mask(text: Any) -> str:
    return ACCOUNT_ID.sub("<account-id>", str(text))


def say(message: str = "") -> None:
    print(mask(message), flush=True)


class JobError(RuntimeError):
    """Something the job cannot go on from: printed as one line, exit code 1 (or `code`)."""

    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------- AWS clients (tests register fakes)

_CLIENTS: dict[str, Any] = {}


def use_clients(**clients: Any) -> None:
    """Tests: hand the jobs fake clients, e.g. use_clients(s3=fake_s3, athena=fake_athena)."""
    _CLIENTS.update({name.replace("_", "-"): client for name, client in clients.items()})


def reset_clients() -> None:
    _CLIENTS.clear()


def region() -> str:
    return os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or contract.DEFAULT_REGION


def aws(service: str) -> Any:
    service = service.replace("_", "-")
    if service not in _CLIENTS:
        import boto3

        _CLIENTS[service] = boto3.client(service, region_name=region())
    return _CLIENTS[service]


def error_code(exc: Exception) -> str:
    return (getattr(exc, "response", None) or {}).get("Error", {}).get("Code", "")


# ---------------------------------------------------------------------------- config

def load_yaml(name: str) -> dict:
    import yaml

    path = CONFIG_DIR / name
    if not path.is_file():
        raise JobError(f"config file not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def data_config() -> dict:
    return load_yaml("data.yaml")


def train_config() -> dict:
    return load_yaml("train.yaml")


def monitor_config() -> dict:
    return load_yaml("monitor.yaml")


def ml_bucket(explicit: str | None = None) -> str:
    bucket = explicit or os.environ.get("ML_BUCKET", "")
    if not bucket:
        raise JobError("the ML bucket is not set: pass --ml-bucket or set ML_BUCKET")
    return bucket


# ---------------------------------------------------------------------------- the app's package

def add_app_package_to_path(env: dict[str, str] | None = None) -> str:
    """Make ``import financial_risk`` work and say where it came from.

    Order: FINRISK_APP_SRC (your laptop, the tests), then whatever PYTHONPATH already gives us (the image has
    PYTHONPATH=/app/src), then /app/src itself in case the platform dropped the image's ENV."""
    env = dict(os.environ if env is None else env)
    explicit = env.get("FINRISK_APP_SRC")
    if explicit:
        src = Path(explicit).resolve()
        if not (src / "financial_risk").is_dir():
            raise JobError(f"FINRISK_APP_SRC={explicit} has no financial_risk/ package inside")
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        return str(src)
    import importlib.util

    if importlib.util.find_spec("financial_risk") is None and (IMAGE_APP_SRC / "financial_risk").is_dir():
        sys.path.insert(0, str(IMAGE_APP_SRC))
    spec = importlib.util.find_spec("financial_risk")
    if spec is None or not spec.submodule_search_locations:
        raise JobError("cannot import financial_risk: set FINRISK_APP_SRC to the app's src folder, or run inside the "
                       "image (it is built FROM the app image)")
    return str(Path(next(iter(spec.submodule_search_locations))).parent)


def check_feature_contract() -> None:
    """The app's FEATURE_COLUMNS must equal ours, in the same order, or the model and the endpoint disagree."""
    add_app_package_to_path()
    from financial_risk.models.training import FEATURE_COLUMNS as app_columns

    if list(app_columns) != contract.FEATURE_COLUMNS:
        raise JobError("the app's FEATURE_COLUMNS differ from finrisk_jobs/contract.py (order matters):\n"
                       f"  app : {list(app_columns)}\n  here: {contract.FEATURE_COLUMNS}\n"
                       "bump base_image.txt to an app image with the same features, or update the contract in a pull request.")


# ---------------------------------------------------------------------------- S3

def parse_s3(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise JobError(f"not an s3:// address: {mask(uri)!r}")
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    if not bucket:
        raise JobError(f"no bucket in {mask(uri)!r}")
    return bucket, key


def s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def s3_list(prefix_uri: str, s3: Any = None) -> list[dict]:
    """Every object under a prefix: [{'Key', 'Size', 'LastModified'?}], following pagination."""
    s3 = s3 or aws("s3")
    bucket, prefix = parse_s3(prefix_uri)
    found: list[dict] = []
    token = None
    while True:
        params: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            params["ContinuationToken"] = token
        page = s3.list_objects_v2(**params)
        found += page.get("Contents", [])
        token = page.get("NextContinuationToken") if page.get("IsTruncated") else None
        if not token:
            return found


def s3_exists(uri: str, s3: Any = None) -> bool:
    s3 = s3 or aws("s3")
    bucket, key = parse_s3(uri)
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as exc:  # noqa: BLE001
        if error_code(exc) in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def s3_get_bytes(uri: str, s3: Any = None) -> bytes:
    s3 = s3 or aws("s3")
    bucket, key = parse_s3(uri)
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def s3_get_json(uri: str, s3: Any = None) -> Any:
    return json.loads(s3_get_bytes(uri, s3))


def s3_put_bytes(uri: str, body: bytes, s3: Any = None, content_type: str | None = None) -> None:
    s3 = s3 or aws("s3")
    bucket, key = parse_s3(uri)
    params: dict[str, Any] = {"Bucket": bucket, "Key": key, "Body": body}
    if content_type:
        params["ContentType"] = content_type
    s3.put_object(**params)


def s3_put_json(uri: str, payload: Any, s3: Any = None) -> None:
    s3_put_bytes(uri, json.dumps(payload, indent=2, default=str).encode("utf-8"), s3, "application/json")


def s3_download(uri: str, path: Path, s3: Any = None) -> Path:
    s3 = s3 or aws("s3")
    bucket, key = parse_s3(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(Bucket=bucket, Key=key, Filename=str(path))
    return path


def s3_upload(path: Path, uri: str, s3: Any = None) -> None:
    s3 = s3 or aws("s3")
    bucket, key = parse_s3(uri)
    s3.upload_file(Filename=str(path), Bucket=bucket, Key=key)


def s3_download_prefix(prefix_uri: str, folder: Path, s3: Any = None, suffix: str = "") -> list[Path]:
    """Download every object under a prefix into a folder (keeping the part of the key after the prefix)."""
    s3 = s3 or aws("s3")
    bucket, prefix = parse_s3(prefix_uri)
    prefix = prefix if prefix.endswith("/") or not prefix else prefix + "/"
    files = []
    for obj in s3_list(prefix_uri if prefix_uri.endswith("/") else prefix_uri + "/", s3):
        key = obj["Key"]
        if key.endswith("/") or not key.endswith(suffix):
            continue
        files.append(s3_download(s3_uri(bucket, key), folder / key[len(prefix):], s3))
    return files


def s3_upload_dir(folder: Path, prefix_uri: str, s3: Any = None) -> list[str]:
    s3 = s3 or aws("s3")
    prefix_uri = prefix_uri.rstrip("/")
    uploaded = []
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        uri = f"{prefix_uri}/{path.relative_to(folder).as_posix()}"
        s3_upload(path, uri, s3)
        uploaded.append(uri)
    return uploaded


def s3_delete_prefix(prefix_uri: str, s3: Any = None) -> int:
    s3 = s3 or aws("s3")
    bucket, _ = parse_s3(prefix_uri)
    keys = [o["Key"] for o in s3_list(prefix_uri, s3)]
    for start in range(0, len(keys), 1000):
        s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in keys[start:start + 1000]]})
    return len(keys)


def extract_model_tar(tar_path: Path, folder: Path) -> Path:
    """Unpack a SageMaker model.tar.gz (flat layout) into a folder, refusing paths that escape it."""
    folder.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as archive:
        for member in archive.getmembers():
            target = (folder / member.name).resolve()
            if folder.resolve() not in target.parents and target != folder.resolve():
                raise JobError(f"unsafe path in {tar_path.name}: {member.name}")
        archive.extractall(folder, filter="data") if sys.version_info >= (3, 12) else archive.extractall(folder)
    return folder


def make_model_tar(model_dir: Path, tar_path: Path) -> Path:
    with tarfile.open(tar_path, "w:gz") as archive:
        for path in sorted(model_dir.iterdir()):
            archive.add(path, arcname=path.name)
    return tar_path


def workdir(explicit: str | None = None, prefix: str = "finrisk-") -> Path:
    if explicit:
        path = Path(explicit)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return Path(tempfile.mkdtemp(prefix=prefix))


def clean(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------- Athena

class AthenaError(JobError):
    pass


def run_athena(sql: str, database: str | None = None, workgroup: str = contract.ATHENA_WORKGROUP, athena: Any = None,
               poll: float = 2.0, timeout: float = 1800.0, sleep: Callable[[float], None] = time.sleep,
               clock: Callable[[], float] = time.monotonic, show: bool = True) -> dict:
    """Run one query and wait for it. Returns the QueryExecution dict (id, statistics, result location)."""
    athena = athena or aws("athena")
    if show:
        say(f"  AWS  athena.start_query_execution  workgroup={workgroup}")
        for line in sql.strip().splitlines():
            say(f"         {line}")
    params: dict[str, Any] = {"QueryString": sql, "WorkGroup": workgroup}
    if database:
        params["QueryExecutionContext"] = {"Database": database}
    query_id = athena.start_query_execution(**params)["QueryExecutionId"]
    started = clock()
    while True:
        execution = athena.get_query_execution(QueryExecutionId=query_id)["QueryExecution"]
        state = execution["Status"]["State"]
        if state == "SUCCEEDED":
            stats = execution.get("Statistics", {})
            scanned = stats.get("DataScannedInBytes")
            say(f"  athena query {query_id}: SUCCEEDED in {stats.get('TotalExecutionTimeInMillis', 0) / 1000:.1f} s"
                + (f", scanned {scanned / 1e6:,.0f} MB" if scanned is not None else ""))
            return execution
        if state in ("FAILED", "CANCELLED"):
            raise AthenaError(f"Athena query {query_id} {state}: {execution['Status'].get('StateChangeReason', 'no reason given')}")
        if clock() - started > timeout:
            raise AthenaError(f"Athena query {query_id} still {state} after {timeout:.0f} s")
        sleep(poll)


def athena_rows(execution: dict, s3: Any = None) -> list[dict]:
    """The result of a finished SELECT, read from the CSV Athena wrote (faster than paging GetQueryResults)."""
    location = execution["ResultConfiguration"]["OutputLocation"]
    text = s3_get_bytes(location, s3).decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


# ---------------------------------------------------------------------------- MLflow, failure-proof

class Tracker:
    """One MLflow run. EVERY method swallows its own errors: MLflow failing never fails a job (the registry is the
    system of record). If MLflow is not configured (no MLFLOW_TRACKING_ARN / MLFLOW_TRACKING_URI) the tracker is off
    and says so once.

    ``module`` lets a test pass a fake in place of the real ``mlflow``."""

    def __init__(self, experiment: str, run_name: str, tags: dict[str, str] | None = None, module: Any = None,
                 tracking_uri: str | None = None):
        self.experiment, self.run_name, self.tags = experiment, run_name, dict(tags or {})
        self.run_id: str | None = None
        self.enabled = False
        self._mlflow = module
        self._uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_ARN") or os.environ.get("MLFLOW_TRACKING_URI")

    def _warn(self, what: str, exc: Exception) -> None:
        say(f"  [mlflow] {what} failed and is skipped: {type(exc).__name__}: {str(exc)[:160]}")

    def start(self) -> "Tracker":
        if not self._uri:
            say("  [mlflow] no MLFLOW_TRACKING_ARN: runs are not logged (the job carries on)")
            return self
        try:
            mlflow = self._mlflow
            if mlflow is None:
                import mlflow  # type: ignore

                self._mlflow = mlflow
            mlflow.set_tracking_uri(self._uri)
            mlflow.set_experiment(self.experiment)
            run = mlflow.start_run(run_name=self.run_name, tags=self.tags or None)
            self.run_id = run.info.run_id
            self.enabled = True
            say(f"  [mlflow] experiment {self.experiment}, run {self.run_id}")
        except Exception as exc:  # noqa: BLE001
            self._warn("start", exc)
        return self

    @staticmethod
    def clean_name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_\-. /]", "_", name)

    def params(self, values: dict[str, Any]) -> None:
        if self.enabled:
            try:
                self._mlflow.log_params({self.clean_name(k): str(v)[:250] for k, v in values.items()})
            except Exception as exc:  # noqa: BLE001
                self._warn("log_params", exc)

    def metrics(self, values: dict[str, Any], step: int | None = None) -> None:
        if self.enabled:
            for name, value in values.items():
                try:
                    if value is not None:
                        self._mlflow.log_metric(self.clean_name(name), float(value), step=step)
                except Exception as exc:  # noqa: BLE001
                    self._warn(f"log_metric {name}", exc)

    def set_tags(self, values: dict[str, Any]) -> None:
        if self.enabled:
            try:
                self._mlflow.set_tags({self.clean_name(k): str(v)[:5000] for k, v in values.items()})
            except Exception as exc:  # noqa: BLE001
                self._warn("set_tags", exc)

    def artifact(self, path: Path | str, folder: str | None = None) -> None:
        if self.enabled and Path(path).exists():
            try:
                self._mlflow.log_artifact(str(path), artifact_path=folder)
            except Exception as exc:  # noqa: BLE001
                self._warn(f"log_artifact {Path(path).name}", exc)

    def end(self, status: str = "FINISHED") -> None:
        if self.enabled:
            try:
                self._mlflow.end_run(status=status)
            except Exception as exc:  # noqa: BLE001
                self._warn("end_run", exc)


def json_default(value: Any) -> Any:
    """numpy / pandas scalars -> plain Python, for json.dumps(default=...)."""
    for attr in ("item",):
        if hasattr(value, attr):
            return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    return path
