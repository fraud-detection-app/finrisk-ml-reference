"""The monitor job: capture parsing, PSI and its bands, prediction drift, the delayed-label join, CloudWatch, the report."""

from __future__ import annotations

import base64
import io
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from conftest import needs_app
from fakes import ACCOUNT, BUCKET, FakeAthena, FakeCloudWatch, FakeS3, FakeSSM, FakeSTS, FakeSageMaker, check_shape
from helpers import FakeMlflow, demo_gold, history_tables, trained_package

from finrisk_jobs import common, contract, monitor

NOW = datetime(2026, 10, 16, 3, 30, tzinfo=timezone.utc)


# ---------------------------------------------------------------- PSI (pure)


def test_psi_is_zero_for_the_same_data_and_large_for_a_shifted_one():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 20000)
    assert monitor.local_psi_numeric(a, rng.normal(0, 1, 20000)) < 0.02
    assert monitor.local_psi_numeric(a, rng.normal(2, 1, 20000)) > 0.5


def test_psi_bands():
    assert monitor.psi_band(0.05) == "stable" and monitor.psi_band(0.10) == "moderate shift - monitor" and monitor.psi_band(0.25) == "significant shift - investigate/refit"


def test_categorical_psi_sees_a_mix_change_and_a_new_category():
    ref = ["A"] * 80 + ["B"] * 20
    assert monitor.psi_categorical(ref, ref) == pytest.approx(0, abs=1e-9)
    assert monitor.psi_categorical(ref, ["A"] * 40 + ["B"] * 60) > 0.25
    assert monitor.psi_categorical(ref, ["A"] * 50 + ["Z"] * 50) > 1.0                 # Z was never seen: floored share, big PSI


@needs_app
def test_the_local_psi_is_the_apps_psi():
    from financial_risk.eda.profiling import population_stability_index

    rng = np.random.default_rng(1)
    a, b = rng.lognormal(8, 1, 5000), rng.lognormal(8.3, 1.1, 3000)
    assert monitor.local_psi_numeric(a, b) == pytest.approx(population_stability_index(a, b))
    assert monitor.psi_numeric(a, b) == pytest.approx(population_stability_index(a, b))


# ---------------------------------------------------------------- capture files


def capture_line(instances, predictions, version="1", encoding="JSON", threshold=0.5, when="2026-10-16T02:10:00Z"):
    request, response = json.dumps({"instances": instances, "explain": False, "top_k": 3}), json.dumps({"model_version": version, "threshold": threshold, "predictions": predictions})
    if encoding == "BASE64":
        request, response = base64.b64encode(request.encode()).decode(), base64.b64encode(response.encode()).decode()
    return json.dumps({"captureData": {"endpointInput": {"observedContentType": "application/json", "mode": "INPUT", "data": request, "encoding": encoding},
                                       "endpointOutput": {"observedContentType": "application/json", "mode": "OUTPUT", "data": response, "encoding": encoding}},
                       "eventMetadata": {"eventId": "e", "inferenceTime": when}, "eventVersion": "0"})


def instance(row_id, **over):
    base = {"row_id": row_id, "transaction_type": "PAYMENT", **{c: 1.0 for c in contract.NUMERIC_FEATURES}}
    return {**base, **over}


def test_capture_lines_become_one_row_per_scored_transaction():
    text = "\n".join([capture_line([instance("1"), instance("2", amount=5.0)], [{"row_id": "1", "score": 0.1, "flagged": False}, {"row_id": "2", "score": 0.9, "flagged": True}]),
                      "", "not json at all", capture_line([instance("3")], [{"row_id": "3", "score": 0.2, "flagged": False}], encoding="BASE64", version="2")])
    rows, skipped = monitor.parse_capture_text(text)
    assert skipped == 1 and [r["row_id"] for r in rows] == ["1", "2", "3"]
    assert rows[1]["amount"] == 5.0 and rows[1]["flagged"] is True and rows[1]["score"] == 0.9 and rows[1]["model_version"] == "1" and rows[2]["model_version"] == "2"
    assert rows[0]["threshold"] == 0.5 and rows[0]["inference_time"] == "2026-10-16T02:10:00Z"


def test_a_capture_line_with_mismatched_lengths_is_skipped_not_misaligned():
    rows, skipped = monitor.parse_capture_text(capture_line([instance("1"), instance("2")], [{"row_id": "1", "score": 0.1, "flagged": False}]))
    assert rows == [] and skipped == 1


def test_hour_prefixes_cover_the_window():
    prefixes = monitor.capture_prefixes("b", "prod", NOW, 3)
    assert prefixes[0] == "s3://b/capture/prod/finrisk-prod-fraud/AllTraffic/2026/10/16/00/" and prefixes[-1].endswith("/2026/10/16/03/") and len(prefixes) == 4


# ---------------------------------------------------------------- the whole job, on real small data


@pytest.fixture(scope="module")
def world_data(tmp_path_factory):
    """A trained baseline model, its reference sample, and 'live' traffic: days 18-22 with the hours squeezed and many more frauds."""
    from financial_risk.models.package import load_model_package
    from financial_risk.models.training import predict_proba_positive

    tmp = tmp_path_factory.mktemp("monitor")
    gold = demo_gold(30000)
    package = trained_package(tmp, gold, "baseline")
    tar = common.make_model_tar(package, tmp / "model.tar.gz")
    artifact = load_model_package(package)
    reference = gold[gold["day"] <= 13][["row_id", "event_time", "step", *contract.FEATURE_COLUMNS, "is_fraud"]]
    live = gold[gold["day"].between(18, 22)].copy()
    rng = np.random.default_rng(3)
    live["hour_of_day"] = rng.integers(0, 6, len(live)).astype(float)                      # drift: the traffic moved to the small hours
    live["is_fraud"] = (rng.random(len(live)) < 0.10).astype(int)                          # prior shift: ~10 % fraud, 10x the demo's ~1 %
    live["score"] = predict_proba_positive(artifact["model"], live[contract.FEATURE_COLUMNS])
    return {"tar": tar, "reference": reference, "live": live, "threshold": float(artifact["threshold"]), "prevalence": float(reference["is_fraud"].mean())}


def put_capture(s3, live, threshold, version="1", per_file=300, hour="02"):
    for i in range(0, len(live), per_file):
        part = live.iloc[i:i + per_file]
        instances = [{"row_id": str(r.row_id), "transaction_type": r.transaction_type, **{c: float(getattr(r, c)) for c in contract.NUMERIC_FEATURES}} for r in part.itertuples()]
        preds = [{"row_id": str(r.row_id), "score": float(r.score), "flagged": bool(r.score >= threshold)} for r in part.itertuples()]
        s3.put(BUCKET, f"capture/prod/finrisk-prod-fraud/AllTraffic/2026/10/16/{hour}/{i:06d}.jsonl", capture_line(instances, preds, version, threshold=threshold))


def setup_aws(world_data, label_fraction=1.0, with_labels=True, version=1):
    s3, ssm, cw = FakeS3(), FakeSSM(), FakeCloudWatch()
    sm = FakeSageMaker(s3)
    athena = FakeAthena(s3)
    live = world_data["live"]
    sm.add_package(version=version, baseline_uri=f"s3://{BUCKET}/datasets/e1/profile/baseline.parquet", dataset_uri=f"s3://{BUCKET}/datasets/e1")
    sm.packages[version]["InferenceSpecification"]["Containers"][0]["ModelDataUrl"] = f"s3://{BUCKET}/models/serving/output/model.tar.gz"
    s3.put(BUCKET, "models/serving/output/model.tar.gz", world_data["tar"].read_bytes())
    buffer = io.BytesIO()
    world_data["reference"].to_parquet(buffer)
    s3.put(BUCKET, "datasets/e1/profile/baseline.parquet", buffer.getvalue())
    s3.put(BUCKET, "datasets/e1/dataset.json", json.dumps({"splits": {"train": {"prevalence": world_data["prevalence"]}}}))
    ssm.params["/finrisk/prod/model"] = json.dumps({"version": version, "serving": "endpoint", "endpoint": "finrisk-prod-fraud"})
    labelled = live.sample(frac=label_fraction, random_state=1) if with_labels else live.iloc[0:0]
    athena.register("labels", pd.DataFrame({"row_id": labelled["row_id"].astype(str), "is_fraud": labelled["is_fraud"].astype(int), "event_time": labelled["event_time"],
                                            "label_time": "2026-09-24T23:00:00Z", "source": "feedback", "batch_id": "b1"}))
    put_capture(s3, live, world_data["threshold"], str(version))
    common.use_clients(s3=s3, ssm=ssm, sagemaker=sm, sts=FakeSTS(), cloudwatch=cw, athena=athena)
    return s3, cw, athena


def run_monitor(*extra):
    return monitor.run(["--env", "prod", "--ml-bucket", BUCKET, "--run-id", "run-1", "--window-hours", "24", *extra], now=lambda: NOW)


@needs_app
def test_end_to_end_metrics_drift_and_alarms(world_data, capsys):
    s3, cw, _ = setup_aws(world_data)
    assert run_monitor("--no-evidently") == 0
    summary = s3.json("monitoring/prod/run-1/metrics.json")
    m = summary["metrics"]
    assert set(m) == {"MaxFeaturePSI", "DriftedFeatureShare", "FlaggedRate", "ScorePSI", "LabelledRows", "LiveFraudRate", "FraudRateRatio", "LivePrAuc", "LiveRecall", "LivePrecision"}
    assert m["LabelledRows"] == len(world_data["live"]) and 0.08 < m["LiveFraudRate"] < 0.12
    assert m["MaxFeaturePSI"] == pytest.approx(summary["psi"]["hour_of_day"]["psi"]) and summary["psi"]["hour_of_day"]["psi"] > 0.25
    assert summary["psi"]["hour_of_day"]["band"].startswith("significant")
    assert summary["psi"]["amount"]["psi"] < 0.1 and summary["psi"]["transaction_type"]["psi"] < 0.1             # the money looks the same
    assert m["DriftedFeatureShare"] == pytest.approx(1 / 14)
    assert m["FraudRateRatio"] == pytest.approx(m["LiveFraudRate"] / world_data["prevalence"]) and m["FraudRateRatio"] > 5
    assert 0 < m["ScorePSI"] and 0 <= m["LivePrAuc"] <= 1
    assert summary["serving"]["version"] == 1 and summary["model_versions_seen"] == ["1"] and summary["training_prevalence_source"].endswith("/datasets/e1/dataset.json")
    crossed = " ".join(summary["thresholds_crossed"])
    assert "finrisk-prod-model-drift" in crossed and "finrisk-prod-model-prior-shift" in crossed
    assert "<html" in s3.text("monitoring/prod/run-1/report.html").lower() and summary["report"]["kind"].startswith("own table")


@needs_app
def test_cloudwatch_gets_the_exact_metric_names_in_the_right_namespace(world_data):
    s3, cw, _ = setup_aws(world_data)
    run_monitor("--no-evidently")
    (put,) = cw.puts
    check_shape("cloudwatch", "PutMetricData", put)
    assert put["Namespace"] == "FinRisk/Model"
    names = [d["MetricName"] for d in put["MetricData"]]
    assert names == [n for n in contract.MONITOR_METRICS if n in names] and set(names) >= {"MaxFeaturePSI", "ScorePSI", "FlaggedRate", "LiveFraudRate", "FraudRateRatio", "LivePrAuc", "LiveRecall", "LabelledRows", "DriftedFeatureShare"}
    for datum in put["MetricData"]:
        assert datum["Dimensions"] == [{"Name": "Env", "Value": "prod"}] and datum["Timestamp"] == NOW and isinstance(datum["Value"], float)


@needs_app
def test_the_table_and_the_crossed_thresholds_are_printed(world_data, capsys):
    setup_aws(world_data)
    run_monitor("--no-evidently")
    out = capsys.readouterr().out
    assert out.index("hour_of_day") < out.index("transaction_type") and "significant shift" in out and "thresholds crossed:" in out
    assert "AWS  cloudwatch.put_metric_data" in out and "alarm finrisk-prod-model-drift" in out and ACCOUNT not in out


@needs_app
def test_without_labels_only_the_drift_metrics_and_zero_labelled_rows(world_data):
    s3, cw, _ = setup_aws(world_data, with_labels=False)
    run_monitor("--no-evidently")
    m = s3.json("monitoring/prod/run-1/metrics.json")["metrics"]
    assert m["LabelledRows"] == 0.0 and not {"LivePrAuc", "LiveRecall", "FraudRateRatio", "LiveFraudRate"} & set(m)
    assert {"MaxFeaturePSI", "ScorePSI", "FlaggedRate"} <= set(m)


@needs_app
def test_few_labels_do_not_make_the_quality_alarm_fire_in_the_report(world_data):
    s3, cw, _ = setup_aws(world_data, label_fraction=0.002)
    run_monitor("--no-evidently")
    summary = s3.json("monitoring/prod/run-1/metrics.json")
    assert summary["metrics"]["LabelledRows"] < 500 and not any("quality" in c for c in summary["thresholds_crossed"])


@needs_app
def test_a_rollout_window_with_two_versions_is_reported(world_data, capsys):
    s3, cw, _ = setup_aws(world_data)
    put_capture(s3, world_data["live"].head(200), world_data["threshold"], version="2", hour="03")
    run_monitor("--no-evidently")
    assert s3.json("monitoring/prod/run-1/metrics.json")["model_versions_seen"] == ["1", "2"] and "more than one model version" in capsys.readouterr().out


@needs_app
def test_no_capture_in_the_window_is_a_clean_exit_that_says_why(world_data, capsys):
    s3, cw, _ = setup_aws(world_data)
    for key in [k for (b, k) in list(s3.objects) if k.startswith("capture/")]:
        del s3.objects[(BUCKET, key)]
    assert run_monitor("--no-evidently") == 0
    assert "no captured requests" in capsys.readouterr().out and not cw.puts
    assert s3.json("monitoring/prod/run-1/metrics.json")["metrics"] == {}


@needs_app
def test_nothing_promoted_yet_needs_an_explicit_reference(world_data):
    s3, cw, _ = setup_aws(world_data)
    common.aws("ssm").params["/finrisk/prod/model"] = json.dumps({"version": None})
    with pytest.raises(common.JobError, match="no reference sample"):
        run_monitor("--no-evidently")
    assert run_monitor("--no-evidently", "--reference", f"s3://{BUCKET}/datasets/e1/profile/baseline.parquet") == 0
    summary = s3.json("monitoring/prod/run-1/metrics.json")
    assert "ScorePSI" not in summary["metrics"] and summary["serving"] is None and summary["training_prevalence_source"].endswith("/datasets/e1/dataset.json")


@needs_app
def test_mlflow_gets_one_run_with_the_metrics_and_the_report(world_data, monkeypatch):
    setup_aws(world_data)
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setenv("MLFLOW_TRACKING_ARN", "arn:aws:sagemaker:ap-southeast-2:123456789012:mlflow-app/app-x")
    run_monitor("--no-evidently")
    assert fake.experiment == "finrisk-monitoring" and fake.tags["serving_version"] == "1" and fake.metrics["MaxFeaturePSI"] > 0.25
    assert "psi_hour_of_day" in fake.metrics and {a[0].rsplit("/", 1)[1] for a in fake.artifacts} == {"metrics.json", "report.html"}


@needs_app
def test_a_broken_mlflow_does_not_fail_the_monitor(world_data, monkeypatch):
    s3, cw, _ = setup_aws(world_data)
    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow(fail="start_run"))
    monkeypatch.setenv("MLFLOW_TRACKING_ARN", "arn:x")
    assert run_monitor("--no-evidently") == 0 and cw.puts


@needs_app
def test_evidently_report_when_installed_else_our_own(world_data):
    s3, cw, _ = setup_aws(world_data)
    run_monitor()
    kind = s3.json("monitoring/prod/run-1/metrics.json")["report"]["kind"]
    try:
        import evidently  # noqa: F401

        assert kind == "evidently DataDriftPreset" and len(s3.text("monitoring/prod/run-1/report.html")) > 100_000
    except ImportError:
        assert kind.startswith("own table (evidently not installed")


def test_own_html_escapes_and_lists_every_feature():
    page = monitor.own_html("t <x>", {"a<b": {"psi": 0.3, "band": "significant shift - investigate/refit"}}, {"MaxFeaturePSI": 0.3}, ["note & more"])
    assert "a&lt;b" in page and "t &lt;x&gt;" in page and "0.3000" in page and "note &amp; more" in page


def test_crossed_follows_the_three_alarm_rules():
    th = {"psi_significant": 0.25, "fraud_rate_ratio": 5, "live_pr_auc_floor": 0.7, "min_labelled_rows": 500}
    assert monitor.crossed({"MaxFeaturePSI": 0.1, "ScorePSI": 0.3}, th, "prod")[0].startswith("data/prediction drift")
    assert monitor.crossed({"FraudRateRatio": 12.0}, th, "prod")[0].startswith("prior shift")
    assert monitor.crossed({"LabelledRows": 600, "LivePrAuc": 0.5}, th, "prod")[0].startswith("quality")
    assert monitor.crossed({"LabelledRows": 100, "LivePrAuc": 0.5}, th, "prod") == []
    assert monitor.crossed({"MaxFeaturePSI": 0.05, "ScorePSI": 0.05, "FraudRateRatio": 2.0, "LabelledRows": 900, "LivePrAuc": 0.9}, th, "prod") == []
