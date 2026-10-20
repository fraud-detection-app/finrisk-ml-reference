"""Deploying a model version to an environment's SageMaker endpoint, and taking it away again.

    dev, sit          SERVERLESS endpoint (3072 MB = a new account's limit, scales to zero, no Data Capture: serverless does not support it).
                      Replaced in place (UpdateEndpoint).
    pre-prod, prod    REAL-TIME endpoint (ml.t2.medium x1) with Data Capture (100 %, input and output, JSON) and a BLUE/GREEN
                      update: a new fleet comes up next to the old one, two CloudWatch alarms watch it
                      (finrisk-<env>-endpoint-5xx and -latency) and roll back by themselves if either goes off, and the old
                      fleet is kept 120 s after the switch for a quick rollback. HOW the traffic moves depends on the instance
                      count of the variant (AWS: "Cannot create/update endpoint with single instance of variant AllTraffic, when
                      using CANARY traffic shifting type"; a canary needs a second instance to hold the first slice):
                        1 instance  ALL_AT_ONCE: all traffic moves to the new fleet, then it is WATCHED for `bake_seconds` (120)
                                    by the alarms before the old fleet goes. This is what we run (one ml.t2.medium).
                        2+ instances CANARY: 50 % of the capacity first, `canary_wait_seconds` (120) watched, then the rest.
                      The very first deploy is a create: there is nothing to roll back to.

`deploy()` does, in order (each AWS call is printed first, by the Traced clients in mlops.py):

    CreateModel -> CreateEndpointConfig -> CreateEndpoint | UpdateEndpoint -> wait InService (a rollback is detected and
    reported) -> smoke test (3 fixed rows, HTTP 200 and the version we just deployed) -> write the pointer
    /finrisk/<env>/model -> wait for the app's /ready to report the version -> delete configs and models older than the two
    newest (the rollback target stays).

The order matters: the pointer is written only AFTER the endpoint answered a real request, so the app is never pointed at
an endpoint that cannot answer. The context `ctx` is mlops.Context (clients, printing, clock); this module imports nothing from it.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from errors import EXIT_FAILED, EXIT_UNUSABLE, MlopsError, error_code, mask

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_FILE = REPO_ROOT / "config" / "smoke.json"
VARIANT = "AllTraffic"
NEEDS_APPROVAL = ("pre-prod", "prod")
GUARDED = ("pre-prod", "prod")                  # real-time + data capture + blue/green
TERMINAL_BAD = {"Failed", "OutOfService"}


# ---------------------------------------------------------------------------- names and requests

def endpoint_name(env: str) -> str:
    return f"finrisk-{env}-fraud"


def alarm_names(env: str) -> list[str]:
    return [f"finrisk-{env}-endpoint-5xx", f"finrisk-{env}-endpoint-latency"]


def resource_name(env: str, version: int, stamp: str) -> str:
    """finrisk-<env>-fraud-v<N>-<YYYYMMDDHHMMSS>: the model and its endpoint config share one name."""
    return f"{endpoint_name(env)}-v{version}-{stamp}"


def model_request(env: str, version: int, name: str, container: dict, role_arn: str, region: str, package_arn: str) -> dict:
    """CreateModel: the package's image and model data, the version in the environment (the endpoint echoes it back)."""
    return {
        "ModelName": name,
        "ExecutionRoleArn": role_arn,
        "PrimaryContainer": {
            "Image": container["Image"],
            "ModelDataUrl": container["ModelDataUrl"],
            "Mode": "SingleModel",
            "Environment": {"MODEL_VERSION": str(version), "AWS_REGION": region, "FINRISK_ENV": env},
        },
        "Tags": [{"Key": "Project", "Value": "finrisk"}, {"Key": "Env", "Value": env}, {"Key": "ModelVersion", "Value": str(version)}],
    }


def endpoint_config_request(env: str, name: str, ml_bucket: str, ecfg: dict) -> dict:
    variant: dict[str, Any] = {"VariantName": VARIANT, "ModelName": name}
    request: dict[str, Any] = {"EndpointConfigName": name, "ProductionVariants": [variant],
                               "Tags": [{"Key": "Project", "Value": "finrisk"}, {"Key": "Env", "Value": env}]}
    if env in GUARDED:
        rt = ecfg["realtime"]
        variant.update({"InstanceType": rt["model_instance_type"], "InitialInstanceCount": int(rt["instance_count"]), "InitialVariantWeight": 1.0})
        request["DataCaptureConfig"] = {
            "EnableCapture": True, "InitialSamplingPercentage": int(rt.get("capture_sampling_percent", 100)),
            "DestinationS3Uri": f"s3://{ml_bucket}/capture/{env}/",
            "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
            "CaptureContentTypeHeader": {"JsonContentTypes": ["application/json"]},
        }
    else:
        sv = ecfg["serverless"]
        variant["ServerlessConfig"] = {"MemorySizeInMB": int(sv["memory_mb"]), "MaxConcurrency": int(sv["max_concurrency"])}
    return request


def routing_strategy(ecfg: dict) -> tuple[dict, str]:
    """(TrafficRoutingConfiguration, the sentence that says which and why), chosen by the variant's instance count.

    botocore: Type and WaitIntervalInSeconds are required; CanarySize belongs to CANARY only (so ALL_AT_ONCE carries none).
    For ALL_AT_ONCE the wait is the BAKE: SageMaker shifts all traffic, then watches the rollback alarms this long before it
    terminates the old fleet."""
    bg, count = ecfg["blue_green"], int(ecfg["realtime"]["instance_count"])
    if count <= 1:
        bake = int(bg["bake_seconds"])
        return ({"Type": "ALL_AT_ONCE", "WaitIntervalInSeconds": bake},
                f"blue/green ALL_AT_ONCE (1 instance: CANARY needs 2+), bake {bake} s")
    wait, percent = int(bg["canary_wait_seconds"]), int(bg["canary_percent"])
    return ({"Type": "CANARY", "WaitIntervalInSeconds": wait, "CanarySize": {"Type": "CAPACITY_PERCENT", "Value": percent}},
            f"blue/green CANARY ({count} instances): {percent} % for {wait} s, then the rest")


def deployment_config(env: str, ecfg: dict) -> dict | None:
    """Blue/green with automatic rollback, for pre-prod and prod only (see routing_strategy for how traffic moves)."""
    if env not in GUARDED:
        return None
    bg = ecfg["blue_green"]
    return {
        "BlueGreenUpdatePolicy": {
            "TrafficRoutingConfiguration": routing_strategy(ecfg)[0],
            "TerminationWaitInSeconds": int(bg["termination_wait_seconds"]),
            "MaximumExecutionTimeoutInSeconds": int(bg["max_execution_seconds"]),
        },
        "AutoRollbackConfiguration": {"Alarms": [{"AlarmName": n} for n in alarm_names(env)]},
    }


def load_smoke(path: Path = SMOKE_FILE) -> dict:
    smoke = json.loads(path.read_text(encoding="utf-8"))
    if len(smoke.get("instances", [])) != 3:
        raise MlopsError(f"{path.name} must hold exactly 3 instances (1 fraud, 2 legit)")
    return smoke


# ---------------------------------------------------------------------------- endpoint state

def describe_endpoint(ctx, env: str) -> dict | None:
    """The endpoint, or None when it does not exist."""
    try:
        return ctx.sm.describe_endpoint(EndpointName=endpoint_name(env))
    except Exception as exc:  # noqa: BLE001
        if error_code(exc) == "ValidationException" and ("Could not find" in str(exc) or "does not exist" in str(exc)):
            return None
        raise


def endpoint_role(ctx, env: str) -> str:
    """The model execution role: SSM /finrisk/<env>/endpoint_role_arn, else the naming convention (the account is public to STS)."""
    name = f"/finrisk/{env}/endpoint_role_arn"
    try:
        value = ctx.ssm.get_parameter(Name=name)["Parameter"]["Value"].strip()
        ctx.out(f"  endpoint role taken from SSM {name}")
        return value
    except Exception as exc:  # noqa: BLE001
        if error_code(exc) not in ("ParameterNotFound", "AccessDeniedException", "AccessDenied"):
            raise
        ctx.out(f"  endpoint role built from the naming convention (SSM said {error_code(exc)})")
        return f"arn:aws:iam::{ctx.account()}:role/finrisk-{env}-endpoint"


def check_alarms(ctx, env: str) -> None:
    """The auto-rollback alarms must exist before an update names them: say so plainly instead of an AWS validation error."""
    found = {a["AlarmName"] for a in ctx.cw.describe_alarms(AlarmNames=alarm_names(env)).get("MetricAlarms", [])}
    missing = [n for n in alarm_names(env) if n not in found]
    if missing:
        raise MlopsError(f"the rollback alarm(s) {missing} do not exist. They are made by finrisk-infra (envs/{env}); apply it first. "
                         "A blue/green update without its guardrails is exactly what this pipeline refuses to do.")


def wait_in_service(ctx, env: str, new_config: str, old_config: str | None, minutes: float, poll: float = 30.0) -> dict:
    """Poll until the endpoint is InService ON THE NEW CONFIG. Detects a rollback (InService again, but on the old config)."""
    started, last, seen_busy = ctx.clock(), None, old_config is None
    ctx.out(f"  waiting for {endpoint_name(env)} to be InService on {new_config} (budget {minutes:.0f} min; status every {poll:.0f} s)")
    while True:
        d = ctx.sm.describe_endpoint(EndpointName=endpoint_name(env))
        status, config = d["EndpointStatus"], d.get("EndpointConfigName")
        if status != last:
            ctx.out(f"  [{ctx.clock() - started:5.0f}s] {status}")
            last = status
        if status in TERMINAL_BAD:
            raise MlopsError(f"{endpoint_name(env)} is {status}: {mask(d.get('FailureReason', 'no reason given'))}. "
                             "Logs: CloudWatch /aws/sagemaker/Endpoints/" + endpoint_name(env))
        if status != "InService":
            seen_busy = True
        elif config == new_config:
            return d
        elif seen_busy and old_config and config == old_config:
            raise MlopsError(f"{endpoint_name(env)} ROLLED BACK to {old_config}: the new version did not pass the guardrail "
                             f"({', '.join(alarm_names(env))}) or failed to start. {mask(d.get('FailureReason', ''))} "
                             "The old version is still serving; nothing was changed in the pointer.")
        if ctx.clock() - started > minutes * 60:
            raise MlopsError(f"{endpoint_name(env)} is still {status} after {minutes:.0f} minutes. It keeps going in AWS; look at it with "
                             f"`aws sagemaker describe-endpoint --endpoint-name {endpoint_name(env)}`.")
        ctx.sleep(poll)


# ---------------------------------------------------------------------------- the smoke test

def smoke_test(ctx, env: str, version: int, attempts: int = 4, wait: float = 30.0) -> dict:
    """Send the 3 fixed rows to the endpoint. Require HTTP 200 and the version we deployed. Serverless endpoints start cold
    (up to about a minute), so a timeout is retried a few times."""
    smoke = load_smoke()
    body = json.dumps({"instances": smoke["instances"], "explain": False, "top_k": 3})
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = ctx.runtime.invoke_endpoint(EndpointName=endpoint_name(env), ContentType="application/json",
                                                    Accept="application/json", Body=body)
        except Exception as exc:  # noqa: BLE001
            last = exc
            ctx.out(f"  smoke test attempt {attempt}/{attempts}: {type(exc).__name__}: {mask(str(exc))[:140]}")
            if attempt < attempts:
                ctx.sleep(wait)
            continue
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        payload = json.loads(response["Body"].read())
        if status != 200:
            raise MlopsError(f"smoke test: the endpoint answered HTTP {status}: {mask(str(payload)[:200])}")
        if str(payload.get("model_version")) != str(version):
            raise MlopsError(f"smoke test: the endpoint reports model_version {payload.get('model_version')!r}, we deployed {version}")
        predictions = payload.get("predictions", [])
        if len(predictions) != 3:
            raise MlopsError(f"smoke test: expected 3 predictions, got {len(predictions)}")
        for pred in predictions:
            ctx.out(f"    row {pred['row_id']:>6}  score {pred['score']:.4f}  flagged {pred['flagged']}")
        ctx.out(f"  smoke test passed: HTTP 200, model_version {payload['model_version']}, threshold {payload.get('threshold')}")
        return payload
    raise MlopsError(f"smoke test: {endpoint_name(env)} did not answer after {attempts} tries ({type(last).__name__ if last else '?'}: "
                     f"{mask(str(last))[:160]}). The pointer was NOT changed.")


# ---------------------------------------------------------------------------- the pointer and the app

def pointer_document(version: int, env: str, config_name: str, package_arn: str, status: str, promoted_at: datetime, run: str) -> dict:
    """Spec 5.4. The app rejects the old shape (no `serving`): this is the contract with finrisk-app."""
    return {"version": version, "serving": "endpoint", "endpoint": endpoint_name(env), "endpoint_config": config_name,
            "model_package_arn": package_arn, "status": status, "promoted_at": promoted_at.isoformat(timespec="seconds"), "run": run}


def find_load_error(ready: dict[str, Any]) -> str:
    """The app's own explanation of why it could not load an artifact, wherever /ready puts it."""
    model = ready.get("model") if isinstance(ready.get("model"), dict) else {}
    if "load_error" in model:
        return str(model["load_error"]) if model["load_error"] else ""
    for holder in (ready, model):
        for key in ("load_error", "error", "last_error"):
            if holder.get(key):
                return str(holder[key])
    return ""


def parse_time(value: Any) -> datetime | None:
    from datetime import timezone

    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), timezone.utc)
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def wait_for_model(ctx, env_name: str, app_url: str, version: int, promoted_at: datetime, timeout: float, poll: float) -> float:
    """Poll <app_url>/ready until model.version == version (and say which source serves it). Returns the seconds it took.

    A load_error reported AFTER the pointer was written is a verdict: stop with its text. One reported before is not."""
    url = app_url.rstrip("/") + "/ready"
    started = ctx.clock()
    last_error = ""
    ctx.out(f"  waiting for {mask(url)} to report model.version {version} (timeout {timeout:.0f} s, polling every {poll:.0f} s)")
    while True:
        elapsed = ctx.clock() - started
        data: dict[str, Any] | None = None
        status = None
        try:
            status, body = ctx.http(url, 10)
            data = json.loads(body)
            if not isinstance(data, dict):
                raise ValueError("not a JSON object")
        except Exception as exc:  # noqa: BLE001 - a cold Lambda or a starting task may not answer yet
            ctx.out(f"  [{elapsed:5.0f}s] GET /ready -> no usable answer ({type(exc).__name__}: {mask(str(exc))[:80]})")
        if data is not None:
            model = data.get("model") if isinstance(data.get("model"), dict) else {}
            seen = model.get("version")
            ctx.out(f"  [{elapsed:5.0f}s] GET /ready -> {status}  model.version={seen}  (want {version})"
                    + (f"  source={model['source']}" if model.get("source") else ""))
            if seen is not None and str(seen) == str(version):
                return elapsed
            error = find_load_error(data)
            if error:
                last_error = error
                checked = parse_time(data.get("pointers_checked_at"))
                if checked is None or checked >= promoted_at:
                    raise MlopsError(f"{env_name} could not load model version {version}; the app says: {mask(error)}. "
                                     f"It keeps serving version {seen}. The pointer was written; promote the previous version to put it back.")
        if ctx.clock() - started > timeout:
            more = f" Last load_error from the app: {mask(last_error)}" if last_error else ""
            raise MlopsError(f"{env_name} did not report model version {version} within {timeout:.0f} s (last seen: "
                             f"{(data or {}).get('model', {}).get('version') if data else 'no answer'}).{more}")
        ctx.sleep(poll)


# ---------------------------------------------------------------------------- cleanup

def list_env_resources(ctx, env: str) -> tuple[list[dict], list[dict]]:
    """(endpoint configs, models) of this environment, newest first."""
    prefix = endpoint_name(env) + "-"
    configs = ctx.sm.list_endpoint_configs(NameContains=prefix, SortBy="CreationTime", SortOrder="Descending", MaxResults=100)
    models = ctx.sm.list_models(NameContains=prefix, SortBy="CreationTime", SortOrder="Descending", MaxResults=100)
    return ([c for c in configs.get("EndpointConfigs", []) if c["EndpointConfigName"].startswith(prefix)],
            [m for m in models.get("Models", []) if m["ModelName"].startswith(prefix)])


def delete_quietly(ctx, call, **params) -> None:
    try:
        call(**params)
    except Exception as exc:  # noqa: BLE001
        if error_code(exc) not in ("ValidationException", "ResourceNotFound", "ResourceNotFoundException"):
            raise


def cleanup_old(ctx, env: str, keep: int, current_config: str) -> list[str]:
    """Keep the newest ``keep`` configs/models (the rollback target) and whatever the endpoint serves right now."""
    configs, models = list_env_resources(ctx, env)
    names = [c["EndpointConfigName"] for c in configs]
    protected = set(names[:keep]) | {current_config}
    removed = []
    for name in names:
        if name not in protected:
            delete_quietly(ctx, ctx.sm.delete_endpoint_config, EndpointConfigName=name)
            removed.append(name)
    for model in models:
        if model["ModelName"] not in protected:
            delete_quietly(ctx, ctx.sm.delete_model, ModelName=model["ModelName"])
    if removed:
        ctx.out(f"  deleted {len(removed)} older config/model pair(s); kept {sorted(protected)} (the rollback target stays)")
    return removed


# ---------------------------------------------------------------------------- deploy

def deploy(ctx, env: str, version: int, detail: dict, package_arn: str, status: str, app_url: str, run: str, *, no_wait: bool,
           ready_timeout: float, poll: float) -> dict:
    ecfg = ctx.tcfg["endpoints"]
    bucket = ctx.shared("ml_bucket")
    stamp = ctx.now().strftime("%Y%m%d%H%M%S")
    name = resource_name(env, version, stamp)
    container = detail["InferenceSpecification"]["Containers"][0]
    role = endpoint_role(ctx, env)
    existing = describe_endpoint(ctx, env)
    if existing and existing["EndpointStatus"] not in ("InService", "Failed"):
        raise MlopsError(f"{endpoint_name(env)} is {existing['EndpointStatus']}: wait until it is InService (or Failed), then promote again.")
    guarded_update = existing is not None and env in GUARDED
    if guarded_update:
        check_alarms(ctx, env)
    old_config = existing.get("EndpointConfigName") if existing else None

    ctx.sm.create_model(**model_request(env, version, name, container, role, ctx.cfg.region, package_arn))
    ctx.sm.create_endpoint_config(**endpoint_config_request(env, name, bucket, ecfg))
    kind = "serverless" if env not in GUARDED else "real-time + Data Capture"
    if existing is None:
        ctx.out(f"  {endpoint_name(env)} does not exist: CREATE ({kind}; the first deploy has nothing to roll back to)")
        ctx.sm.create_endpoint(EndpointName=endpoint_name(env), EndpointConfigName=name,
                               Tags=[{"Key": "Project", "Value": "finrisk"}, {"Key": "Env", "Value": env}])
    else:
        ctx.out(f"  {endpoint_name(env)} exists ({existing['EndpointStatus']}, config {old_config}): UPDATE"
                + (f" with {routing_strategy(ecfg)[1]}, auto-rollback on {' and '.join(alarm_names(env))}" if guarded_update else " in place"))
        params: dict[str, Any] = {"EndpointName": endpoint_name(env), "EndpointConfigName": name}
        dc = deployment_config(env, ecfg)
        if dc:
            params["DeploymentConfig"] = dc
        ctx.sm.update_endpoint(**params)
    wait_in_service(ctx, env, name, old_config if existing and existing["EndpointStatus"] == "InService" else None,
                    float(ecfg.get("wait_minutes", 25)), poll=30.0)
    smoke_test(ctx, env, version)

    promoted_at = ctx.now()
    pointer = pointer_document(version, env, name, package_arn, status, promoted_at, run)
    ctx.ssm.put_parameter(Name=f"/finrisk/{env}/model", Value=json.dumps(pointer), Type="String", Overwrite=True)
    ctx.out(f"  /finrisk/{env}/model now points at version {version} (endpoint {endpoint_name(env)})")
    took = 0.0
    if no_wait:
        ctx.out("not waiting for the app (--no-wait): it picks the pointer up within its POINTER_TTL_SECONDS.")
    else:
        took = wait_for_model(ctx, env, app_url, version, promoted_at, ready_timeout, poll)
        ctx.out(f"  {env} serves model version {version} from the endpoint (confirmed by /ready after {took:.0f} s)")
    cleanup_old(ctx, env, int(ecfg.get("keep_configs", 2)), name)
    return {"config": name, "pointer": pointer, "ready_seconds": took, "created": existing is None, "kind": kind}


# ---------------------------------------------------------------------------- teardown

def teardown(ctx, env: str, wait_minutes: float = 10.0, poll: float = 15.0) -> dict:
    """Delete the endpoint, then its configs and models; put the pointer back to {"version": null} so the app serves its baked
    model instead of calling an endpoint that is gone."""
    existing = describe_endpoint(ctx, env)
    if existing:
        ctx.sm.delete_endpoint(EndpointName=endpoint_name(env))
        started = ctx.clock()
        while describe_endpoint(ctx, env) is not None:
            if ctx.clock() - started > wait_minutes * 60:
                raise MlopsError(f"{endpoint_name(env)} is still being deleted after {wait_minutes:.0f} minutes; run teardown again later.")
            ctx.sleep(poll)
        ctx.out(f"  {endpoint_name(env)} deleted")
    else:
        ctx.out(f"  {endpoint_name(env)} does not exist (nothing to delete)")
    configs, models = list_env_resources(ctx, env)
    for c in configs:
        delete_quietly(ctx, ctx.sm.delete_endpoint_config, EndpointConfigName=c["EndpointConfigName"])
    for m in models:
        delete_quietly(ctx, ctx.sm.delete_model, ModelName=m["ModelName"])
    ctx.ssm.put_parameter(Name=f"/finrisk/{env}/model", Value=json.dumps({"version": None}), Type="String", Overwrite=True)
    ctx.out(f"  /finrisk/{env}/model reset to {{\"version\": null}}: the app serves its baked model again")
    return {"endpoint_deleted": existing is not None, "configs": len(configs), "models": len(models)}
