"""monitor: the only step of the pipeline `finrisk-monitor`. A Processing job. Replaces SageMaker Model Monitor.

SageMaker Model Monitor closed to new customers on 30 Jul 2026. What AWS tells customers to do now is what this job does:
the endpoint's DATA CAPTURE (still available) writes every request and response to S3; this job reads them, compares them
with the training data and with the labels that arrive late, and reports to CloudWatch (alarms) and MLflow (history).

    1. capture    the JSONL files of the last WindowHours hours -> one row per scored transaction
                  (features + score + flagged + model_version)
    2. data drift PSI of every feature against the reference (the `baseline.parquet` of the model being served: a sample of
                  its training data), with the app's population_stability_index (quantile bins); an Evidently
                  DataDriftPreset report as HTML. The score gets a PSI too (prediction drift), and so does the flagged rate.
    3. quality    the feedback labels (finrisk_ml.labels, source=feedback) joined by row_id -> live PR-AUC, recall, precision,
                  and the live fraud rate against the training prevalence (prior shift)
    4. report     s3://<ml>/monitoring/<env>/<run>/{metrics.json, report.html}; one MLflow run (finrisk-monitoring);
                  CloudWatch namespace FinRisk/Model, dimension Env: MaxFeaturePSI, DriftedFeatureShare, ScorePSI,
                  FlaggedRate, LiveFraudRate, FraudRateRatio, LivePrAuc, LiveRecall, LabelledRows
    5. print      the table and which thresholds were crossed. Exit 0 either way: drift is information, not a job failure.

The alarms (Terraform, finrisk-infra) watch these metrics; the alarm `finrisk-prod-model-drift` or `...-prior-shift`
starts the training pipeline (EventBridge). The alarms are ON THE LARGEST PSI, not on "the share of drifting features":
in the simulated go-live one feature of fourteen moves (hour_of_day, PSI ~0.5) and a share would never fire.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import contract
from .common import (JobError, PIPELINE_DIR, Tracker, add_app_package_to_path, athena_rows, aws, ml_bucket, monitor_config, mask,
                     run_athena, s3_download, s3_get_json, s3_list, s3_get_bytes, s3_put_bytes, say, workdir, write_json)

BAND_SIGNIFICANT = "significant shift - investigate/refit"
BAND_MODERATE = "moderate shift - monitor"
BAND_STABLE = "stable"
VARIANT = "AllTraffic"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="finrisk_jobs monitor", description=__doc__.split("\n\n")[0])
    p.add_argument("--env", default="prod")
    p.add_argument("--window-hours", type=int, help="captured requests of the last N hours (default config/monitor.yaml)")
    p.add_argument("--reference", default="auto", help="s3 URI of a baseline.parquet, or auto = the serving version's baseline")
    p.add_argument("--ml-bucket")
    p.add_argument("--run-id", default=os.environ.get("PIPELINE_EXECUTION_ID"))
    p.add_argument("--no-evidently", action="store_true")
    p.add_argument("--workdir")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------- PSI

def psi_band(value: float, significant: float = 0.25, moderate: float = 0.10) -> str:
    if value >= significant:
        return BAND_SIGNIFICANT
    if value >= moderate:
        return BAND_MODERATE
    return BAND_STABLE


def local_psi_numeric(expected, actual, bins: int = 10) -> float:
    """Identical to financial_risk.eda.profiling.population_stability_index (kept so the monitor still runs when the app
    changes its signature): baseline quantile bins, shares floored at 1e-4."""
    expected = np.asarray(pd.Series(expected).dropna(), dtype=float)
    actual = np.asarray(pd.Series(actual).dropna(), dtype=float)
    if expected.size == 0 or actual.size == 0:
        raise ValueError("PSI needs non-empty expected and actual samples")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    exp_share = np.clip(np.histogram(expected, bins=edges)[0] / expected.size, 1e-4, None)
    act_share = np.clip(np.histogram(actual, bins=edges)[0] / actual.size, 1e-4, None)
    return float(np.sum((act_share - exp_share) * np.log(act_share / exp_share)))


def psi_numeric(expected, actual, bins: int = 10) -> float:
    try:
        add_app_package_to_path()
        from financial_risk.eda.profiling import population_stability_index

        return float(population_stability_index(expected, actual, bins=bins))
    except (ImportError, JobError):
        return local_psi_numeric(expected, actual, bins)


def psi_categorical(expected, actual) -> float:
    """The same formula over category shares (a category absent on one side counts as the 1e-4 floor)."""
    e = pd.Series(expected).dropna().astype(str).value_counts(normalize=True)
    a = pd.Series(actual).dropna().astype(str).value_counts(normalize=True)
    if e.empty or a.empty:
        raise ValueError("PSI needs non-empty expected and actual samples")
    keys = sorted(set(e.index) | set(a.index))
    exp_share = np.clip(np.array([e.get(k, 0.0) for k in keys]), 1e-4, None)
    act_share = np.clip(np.array([a.get(k, 0.0) for k in keys]), 1e-4, None)
    return float(np.sum((act_share - exp_share) * np.log(act_share / exp_share)))


def feature_psi(reference: pd.DataFrame, live: pd.DataFrame, bins: int, significant: float, moderate: float) -> dict[str, dict]:
    result = {}
    for column in contract.FEATURE_COLUMNS:
        if column in contract.CATEGORICAL_FEATURES:
            value = psi_categorical(reference[column], live[column])
        else:
            value = psi_numeric(reference[column], live[column], bins)
        result[column] = {"psi": value, "band": psi_band(value, significant, moderate)}
    return result


# ---------------------------------------------------------------------------- capture files

def capture_prefixes(bucket: str, env: str, now: datetime, window_hours: int) -> list[str]:
    """One S3 prefix per hour of the window: capture/<env>/<endpoint>/<variant>/YYYY/MM/DD/HH/"""
    base = f"capture/{env}/{contract.endpoint_name(env)}/{VARIANT}"
    hours = []
    start = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=window_hours)
    for i in range(window_hours + 1):
        h = start + timedelta(hours=i)
        hours.append(f"s3://{bucket}/{base}/{h:%Y/%m/%d/%H}/")
    return hours


def _decode(part: dict) -> Any:
    data = part.get("data", "")
    if str(part.get("encoding", "")).upper() == "BASE64":
        data = base64.b64decode(data).decode("utf-8")
    return json.loads(data)


def parse_capture_text(text: str) -> tuple[list[dict], int]:
    """JSONL text -> (one dict per scored instance, number of unusable lines). Features come from the request, score and
    flagged from the response (same position), the model version and threshold from the response."""
    rows: list[dict] = []
    skipped = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            data = record["captureData"]
            request, response = _decode(data["endpointInput"]), _decode(data["endpointOutput"])
            instances, predictions = request["instances"], response["predictions"]
            if len(instances) != len(predictions):
                raise ValueError("instances and predictions differ in length")
        except (ValueError, KeyError, TypeError):
            skipped += 1
            continue
        when = (record.get("eventMetadata") or {}).get("inferenceTime")
        for inst, pred in zip(instances, predictions):
            row = {"row_id": str(inst.get("row_id", pred.get("row_id"))), **{c: inst.get(c) for c in contract.FEATURE_COLUMNS},
                   "score": pred.get("score"), "flagged": bool(pred.get("flagged")),
                   "model_version": str(response.get("model_version")), "threshold": response.get("threshold"), "inference_time": when}
            rows.append(row)
    return rows, skipped


def read_capture(bucket: str, env: str, now: datetime, window_hours: int, s3: Any) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    files = skipped = 0
    for prefix in capture_prefixes(bucket, env, now, window_hours):
        for obj in s3_list(prefix, s3):
            if not obj["Key"].endswith((".jsonl", ".json")):
                continue
            files += 1
            part, bad = parse_capture_text(s3_get_bytes(f"s3://{bucket}/{obj['Key']}", s3).decode("utf-8"))
            rows += part
            skipped += bad
    frame = pd.DataFrame(rows)
    return frame, {"files": files, "rows": len(frame), "unusable_lines": skipped}


# ---------------------------------------------------------------------------- what is being served, the reference

def serving_info(env: str, ssm: Any, sm: Any, sts: Any, group: str, region: str) -> dict[str, Any] | None:
    try:
        value = ssm.get_parameter(Name=f"/finrisk/{env}/model")["Parameter"]["Value"]
    except Exception as exc:  # noqa: BLE001
        if (getattr(exc, "response", {}) or {}).get("Error", {}).get("Code") == "ParameterNotFound":
            return None
        raise
    pointer = json.loads(value)
    if not isinstance(pointer, dict) or pointer.get("version") is None:
        return None
    arn = pointer.get("model_package_arn") or f"arn:aws:sagemaker:{region}:{sts.get_caller_identity()['Account']}:model-package/{group}/{pointer['version']}"
    detail = sm.describe_model_package(ModelPackageName=arn)
    meta = detail.get("CustomerMetadataProperties") or {}
    return {"version": int(pointer["version"]), "endpoint": pointer.get("endpoint"), "arn": arn,
            "model_url": detail["InferenceSpecification"]["Containers"][0]["ModelDataUrl"],
            "baseline_uri": meta.get("baseline_uri"), "dataset_uri": meta.get("dataset_uri")}


def training_prevalence(serving: dict | None, reference_uri: str, baseline: pd.DataFrame, s3: Any) -> tuple[float | None, str]:
    """The fraud rate of the training split (dataset.json), else the baseline sample's own rate (said in the report)."""
    candidates = []
    if serving and serving.get("dataset_uri"):
        candidates.append(serving["dataset_uri"].rstrip("/") + "/dataset.json")
    if reference_uri.startswith("s3://") and "/profile/" in reference_uri:
        candidates.append(reference_uri.split("/profile/")[0] + "/dataset.json")
    for uri in candidates:
        try:
            return float(s3_get_json(uri, s3)["splits"]["train"]["prevalence"]), uri
        except Exception:  # noqa: BLE001
            continue
    if "is_fraud" in baseline.columns and len(baseline):
        return float(baseline["is_fraud"].mean()), "baseline sample (dataset.json not found)"
    return None, "unknown"


def score_reference(serving: dict | None, baseline: pd.DataFrame, work: Path, s3: Any) -> np.ndarray | None:
    """The serving model scores the reference sample: the score distribution live scores are compared with."""
    if not serving:
        return None
    try:
        add_app_package_to_path()
        from financial_risk.models.package import load_model_package, unpack_model_package
        from financial_risk.models.training import predict_proba_positive

        tar = s3_download(serving["model_url"], work / "serving.tar.gz", s3)
        unpack_model_package(tar, work / "serving")
        artifact = load_model_package(work / "serving")
        return np.asarray(predict_proba_positive(artifact["model"], baseline[contract.FEATURE_COLUMNS]), dtype=float)
    except Exception as exc:  # noqa: BLE001
        say(f"  (ScorePSI skipped: cannot score the reference with the serving model: {type(exc).__name__}: {mask(str(exc))[:120]})")
        return None


# ---------------------------------------------------------------------------- quality on delayed ground truth

def labels_for(live: pd.DataFrame, athena: Any, s3: Any) -> pd.DataFrame:
    ids = pd.to_numeric(live["row_id"], errors="coerce").dropna()
    if ids.empty:
        return pd.DataFrame(columns=["row_id", "is_fraud"])
    if str(PIPELINE_DIR) not in sys.path:
        sys.path.insert(0, str(PIPELINE_DIR))
    import queries

    execution = run_athena(queries.labels_for_rows(int(ids.min()), int(ids.max())), queries.DATABASE, athena=athena)
    rows = athena_rows(execution, s3)
    frame = pd.DataFrame(rows, columns=["row_id", "is_fraud", "label_time"]) if rows else pd.DataFrame(columns=["row_id", "is_fraud", "label_time"])
    frame["is_fraud"] = pd.to_numeric(frame["is_fraud"]).astype(int) if len(frame) else frame["is_fraud"]
    return frame


def quality_metrics(live: pd.DataFrame, labels: pd.DataFrame, threshold: float | None, prevalence: float | None) -> dict[str, float]:
    """LabelledRows is always there; the rest only when it can be measured (both classes for PR-AUC)."""
    from sklearn.metrics import average_precision_score, precision_score, recall_score

    joined = live.drop_duplicates("row_id", keep="last").merge(labels[["row_id", "is_fraud"]], on="row_id", how="inner")
    out: dict[str, float] = {"LabelledRows": float(len(joined))}
    if joined.empty:
        return out
    y = joined["is_fraud"].astype(int).to_numpy()
    out["LiveFraudRate"] = float(y.mean())
    if prevalence and prevalence > 0:
        out["FraudRateRatio"] = float(y.mean() / prevalence)
    if len(set(y)) > 1:
        scores = joined["score"].astype(float).to_numpy()
        out["LivePrAuc"] = float(average_precision_score(y, scores))
        pred = (scores >= (threshold if threshold is not None else 0.5)).astype(int)
        out["LiveRecall"] = float(recall_score(y, pred, zero_division=0))
        out["LivePrecision"] = float(precision_score(y, pred, zero_division=0))
    return out


# ---------------------------------------------------------------------------- reports

def evidently_html(reference: pd.DataFrame, live: pd.DataFrame, max_rows: int) -> tuple[str | None, str]:
    """Evidently's DataDriftPreset as HTML. (None, reason) when it is not installed or fails: the monitor writes its own."""
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset
    except ImportError as exc:
        return None, f"evidently not installed ({exc})"
    try:
        import tempfile

        definition = DataDefinition(numerical_columns=list(contract.NUMERIC_FEATURES), categorical_columns=list(contract.CATEGORICAL_FEATURES))
        ref = Dataset.from_pandas(reference[contract.FEATURE_COLUMNS].sample(min(max_rows, len(reference)), random_state=42), data_definition=definition)
        cur = Dataset.from_pandas(live[contract.FEATURE_COLUMNS].astype({c: float for c in contract.NUMERIC_FEATURES}).sample(min(max_rows, len(live)), random_state=42),
                                  data_definition=definition)
        snapshot = Report([DataDriftPreset()]).run(cur, ref)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.html"
            snapshot.save_html(str(path))
            return path.read_text(encoding="utf-8"), "evidently DataDriftPreset"
    except Exception as exc:  # noqa: BLE001
        return None, f"evidently failed ({type(exc).__name__}: {str(exc)[:100]})"


def own_html(title: str, table: dict[str, dict], metrics: dict[str, float], notes: list[str]) -> str:
    rows = "".join(f"<tr><td>{html.escape(f)}</td><td>{v['psi']:.4f}</td><td>{html.escape(v['band'])}</td></tr>" for f, v in table.items())
    mrows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v:.6g}</td></tr>" for k, v in metrics.items())
    notes_html = "".join(f"<li>{html.escape(n)}</li>" for n in notes)
    return (f"<!doctype html>\n<html lang='en'><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>"
            "body{font-family:system-ui;margin:2rem}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 10px;text-align:left}</style></head>"
            f"<body><h1>{html.escape(title)}</h1><ul>{notes_html}</ul><h2>Feature drift (PSI)</h2>"
            f"<table><tr><th>feature</th><th>PSI</th><th>band</th></tr>{rows}</table><h2>Metrics</h2><table>{mrows}</table></body></html>")


def crossed(metrics: dict[str, float], thresholds: dict[str, float], env: str) -> list[str]:
    out = []
    worst = max(metrics.get("MaxFeaturePSI", 0.0), metrics.get("ScorePSI", 0.0))
    if worst >= thresholds["psi_significant"]:
        out.append(f"data/prediction drift: max PSI {worst:.3f} >= {thresholds['psi_significant']}  -> alarm finrisk-{env}-model-drift")
    if metrics.get("FraudRateRatio", 0.0) >= thresholds["fraud_rate_ratio"]:
        out.append(f"prior shift: fraud rate x{metrics['FraudRateRatio']:.1f} >= x{thresholds['fraud_rate_ratio']}  -> alarm finrisk-{env}-model-prior-shift")
    if metrics.get("LabelledRows", 0) >= thresholds["min_labelled_rows"] and metrics.get("LivePrAuc", 1.0) < thresholds["live_pr_auc_floor"]:
        out.append(f"quality: live PR-AUC {metrics['LivePrAuc']:.3f} < {thresholds['live_pr_auc_floor']}  -> alarm finrisk-{env}-model-quality")
    return out


def put_metrics(cw: Any, namespace: str, dimension: str, env: str, metrics: dict[str, float], now: datetime) -> None:
    names = [m for m in contract.MONITOR_METRICS if m in metrics]
    data = [{"MetricName": m, "Dimensions": [{"Name": dimension, "Value": env}], "Timestamp": now, "Value": float(metrics[m]), "Unit": "None"}
            for m in names]
    say(f"  AWS  cloudwatch.put_metric_data  Namespace={namespace}  Dimensions={dimension}={env}  {len(data)} metrics: {', '.join(names)}")
    for start in range(0, len(data), 1000):
        cw.put_metric_data(Namespace=namespace, MetricData=data[start:start + 1000])


# ---------------------------------------------------------------------------- the job

def run(argv: list[str] | None = None, *, now: Callable[[], datetime] | None = None) -> int:
    a = parse_args(argv)
    cfg = monitor_config()
    th = cfg["thresholds"]
    bucket = ml_bucket(a.ml_bucket)
    clock = (now or (lambda: datetime.now(timezone.utc)))()
    window = a.window_hours or int(cfg["window_hours"])
    s3, ssm, sm, sts, cw, athena = aws("s3"), aws("ssm"), aws("sagemaker"), aws("sts"), aws("cloudwatch"), aws("athena")
    run_id = a.run_id or clock.strftime("%Y%m%d-%H%M%S")
    work = workdir(a.workdir, "finrisk-monitor-")
    region = os.environ.get("AWS_REGION", contract.DEFAULT_REGION)
    say(f"monitor: env {a.env}, window {window} h, run {run_id}")

    serving = serving_info(a.env, ssm, sm, sts, contract.MODEL_GROUP, region)
    if serving:
        say(f"  serving model version {serving['version']} (endpoint {serving.get('endpoint')})")
    else:
        say(f"  /finrisk/{a.env}/model has no version: nothing has been promoted here yet")
    reference_uri = a.reference if a.reference != "auto" else (serving or {}).get("baseline_uri")
    if not reference_uri:
        raise JobError("no reference sample: the serving version has no baseline_uri and no --reference was given")
    baseline = pd.read_parquet(s3_download(reference_uri, work / "baseline.parquet", s3))
    say(f"  reference: {mask(reference_uri)} ({len(baseline):,} rows)")

    live, capture = read_capture(bucket, a.env, clock, window, s3)
    say(f"  captured: {capture['files']} file(s), {capture['rows']:,} scored transactions"
        + (f", {capture['unusable_lines']} unusable line(s)" if capture["unusable_lines"] else ""))
    summary: dict[str, Any] = {"env": a.env, "run": run_id, "created_at": clock.isoformat(timespec="seconds"), "window_hours": window,
                               "serving": {k: serving[k] for k in ("version", "endpoint", "arn", "baseline_uri")} if serving else None,
                               "reference": {"uri": reference_uri, "rows": len(baseline)}, "capture": capture, "thresholds": th}
    if live.empty:
        say("  no captured requests in the window: nothing to compare. (Is the endpoint pre-prod/prod, with Data Capture on, and has traffic arrived?)")
        summary["metrics"] = {}
        s3_put_bytes(f"s3://{bucket}/monitoring/{a.env}/{run_id}/metrics.json", json.dumps(summary, indent=2, default=str).encode(), s3, "application/json")
        return 0

    versions = sorted(live["model_version"].dropna().unique())
    threshold = pd.to_numeric(live["threshold"], errors="coerce").dropna()
    threshold_value = float(threshold.iloc[-1]) if len(threshold) else None
    if len(versions) > 1:
        say(f"  note: the window holds more than one model version {versions} (a rollout?)")
    live = live.copy()
    for column in contract.NUMERIC_FEATURES:
        live[column] = pd.to_numeric(live[column], errors="coerce")

    # data drift
    table = feature_psi(baseline, live, int(cfg["psi_bins"]), th["psi_significant"], th["psi_moderate"])
    metrics: dict[str, float] = {
        "MaxFeaturePSI": max(v["psi"] for v in table.values()),
        "DriftedFeatureShare": sum(v["psi"] >= th["psi_significant"] for v in table.values()) / len(table),
        "FlaggedRate": float(live["flagged"].mean()),
    }
    # prediction drift
    reference_scores = score_reference(serving, baseline, work, s3)
    if reference_scores is not None:
        metrics["ScorePSI"] = psi_numeric(reference_scores, pd.to_numeric(live["score"]).to_numpy(), int(cfg["psi_bins"]))
        summary["reference_flagged_rate"] = float((reference_scores >= (threshold_value if threshold_value is not None else 0.5)).mean())
    # quality on delayed ground truth
    prevalence, prevalence_source = training_prevalence(serving, reference_uri, baseline, s3)
    labels = labels_for(live, athena, s3)
    metrics.update(quality_metrics(live, labels, threshold_value, prevalence))
    summary.update({"psi": table, "metrics": metrics, "training_prevalence": prevalence, "training_prevalence_source": prevalence_source,
                    "model_versions_seen": versions, "threshold": threshold_value})

    # the report
    notes = [f"env {a.env}, window {window} h, run {run_id}", f"live rows {len(live):,}, reference rows {len(baseline):,}"]
    report_html, how = (None, "disabled") if a.no_evidently else evidently_html(baseline, live, int(cfg["evidently_max_rows"]))
    summary["report"] = {"kind": how if report_html else "own table (" + how + ")"}
    say(f"  report: {summary['report']['kind']}")
    hit = crossed(metrics, th, a.env)
    summary["thresholds_crossed"] = hit
    page = report_html or own_html(f"finrisk monitor {a.env} {run_id}", table, metrics, notes)
    prefix = f"s3://{bucket}/monitoring/{a.env}/{run_id}"
    s3_put_bytes(f"{prefix}/report.html", page.encode("utf-8"), s3, "text/html")
    s3_put_bytes(f"{prefix}/metrics.json", json.dumps(summary, indent=2, default=str).encode("utf-8"), s3, "application/json")
    say(f"  written: {mask(prefix)}/metrics.json and report.html")

    put_metrics(cw, cfg["cloudwatch"]["namespace"], cfg["cloudwatch"]["dimension"], a.env, metrics, clock)

    tracker = Tracker(contract.MLFLOW_EXPERIMENT_MONITOR, f"monitor-{a.env}-{run_id}", {"job_type": "monitor", "env": a.env,
                      "serving_version": str(serving["version"]) if serving else "none", "models_seen": ",".join(versions)}).start()
    tracker.params({"env": a.env, "window_hours": window, "reference": reference_uri, "live_rows": len(live)})
    tracker.metrics({**metrics, **{f"psi_{f}": v["psi"] for f, v in table.items()}})
    local = work / "out"
    tracker.artifact(write_json(local / "metrics.json", summary))
    (local / "report.html").write_text(page, encoding="utf-8")
    tracker.artifact(local / "report.html")
    tracker.end("FINISHED")

    # the table
    say("")
    say(f"  {'feature':<28} {'PSI':>8}  band")
    for feature, v in sorted(table.items(), key=lambda kv: -kv[1]["psi"]):
        say(f"  {feature:<28} {v['psi']:>8.4f}  {v['band']}")
    if "ScorePSI" in metrics:
        say(f"  {'(score)':<28} {metrics['ScorePSI']:>8.4f}  {psi_band(metrics['ScorePSI'], th['psi_significant'], th['psi_moderate'])}")
    say("")
    for name in contract.MONITOR_METRICS:
        if name in metrics:
            say(f"  {name:<20} {metrics[name]:.6g}")
    say("  thresholds crossed: " + ("none" if not hit else ""))
    for line in hit:
        say(f"    - {line}")
    return 0
