"""mlops.py: upsert, run (follow, gate, register), status, backfill, monitor, mlflow-url; approve, list, show."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone

import pytest
from fakes import ACCOUNT, BUCKET, GROUP, IMAGE, last_line

import definition
import mlops


def evaluation(pr_auc=0.91, champion=None, floor=0.7):
    from finrisk_jobs.gate import champion_floor

    return {"candidate": {"pr_auc": pr_auc, "training_job": "finrisk-tune-x-001-abcd1234"},
            "champion": champion, "gate": {"floor": floor, "tolerance": 0.02, "champion_floor": champion_floor(champion["pr_auc"] if champion else None, 0.02)},
            "test_window": {"rows": 10}}


# ---------------------------------------------------------------- upsert


def test_upsert_creates_both_pipelines_then_updates_them(world, tmp_path):
    summary = tmp_path / "s.md"
    code, out, err = world.run("upsert", "--image", IMAGE, "--summary", str(summary))
    assert code == 0, err and last_line(out) == "finrisk-train, finrisk-monitor"
    assert [c["PipelineName"] for c in world.sm.called("create_pipeline")] == ["finrisk-train", "finrisk-monitor"]
    assert not world.sm.called("update_pipeline")
    created = world.sm.called("create_pipeline")[0]
    assert created["RoleArn"] == f"arn:aws:iam::{ACCOUNT}:role/finrisk-sagemaker-exec" and len(created["ClientRequestToken"]) >= 32
    defn = json.loads(created["PipelineDefinition"])
    assert defn["Parameters"][-1]["DefaultValue"] == IMAGE and [s["Name"] for s in defn["Steps"]] == ["BuildDataset", "Tune", "Evaluate", "Gate"]
    assert "pipeline_exists" not in out and ACCOUNT not in out
    code, out, err = world.run("upsert", "--image", IMAGE)
    assert code == 0 and [c["PipelineName"] for c in world.sm.called("update_pipeline")] == ["finrisk-train", "finrisk-monitor"]
    assert "Pipelines written" in summary.read_text()


def test_upsert_takes_the_mlflow_arn_and_commit_from_the_environment(world):
    world.ssm.params["/finrisk/shared/mlflow_app_arn"] = "arn:aws:sagemaker:ap-southeast-2:123456789012:mlflow-app/app-9"
    world.run("upsert", "--image", IMAGE, "--pipeline", "finrisk-train", env={"GITHUB_SHA": "e" * 40})
    defn = json.loads(world.sm.called("create_pipeline")[0]["PipelineDefinition"])
    tune = next(s for s in defn["Steps"] if s["Name"] == "Tune")
    assert tune["Arguments"]["TrainingJobDefinition"]["Environment"]["MLFLOW_TRACKING_ARN"].endswith("mlflow-app/app-9")
    assert defn["Metadata"]["git_sha"] == "e" * 40 and len(world.sm.called("create_pipeline")) == 1


def test_upsert_refuses_a_tag_and_the_placeholder_free_check_is_for_digests_only(world):
    code, out, err = world.run("upsert", "--image", "repo/finrisk-train:latest")
    assert code == 1 and "DIGEST" in err and not world.sm.called("create_pipeline")


# ---------------------------------------------------------------- run


def registered(world, pr_auc=0.91):
    world.sm.add_package(version=1, pr_auc=pr_auc)
    world.sm.script_execution("registered", version=1)


def test_run_follows_the_steps_prints_the_gate_and_ends_with_the_version(world, tmp_path):
    registered(world)
    world.s3.put(BUCKET, "metrics/abc00001/evaluation.json", json.dumps(evaluation(0.91)))
    summary = tmp_path / "s.md"
    code, out, err = world.run("run", "--trigger-reason", "push", "--model-type", "baseline", "--summary", str(summary))
    assert code == 0, err and last_line(out) == "1"
    (start,) = world.sm.called("start_pipeline_execution")
    assert start["PipelineName"] == "finrisk-train" and len(start["ClientRequestToken"]) >= 32
    assert {p["Name"]: p["Value"] for p in start["PipelineParameters"]} == {"TriggerReason": "push", "ModelType": "baseline"}
    for line in ("BuildDataset", "Tune", "Evaluate", "Gate", "Register"):
        assert line in out
    assert "Executing" in out and "Succeeded" in out and "floor" in out and "PASS" in out and "registered model version 1" in out
    text = summary.read_text()
    assert "Gate: PASSED" in text and "model version 1 (PendingManualApproval)" in text and "0.9100" in text


def test_run_with_a_cutoff_and_an_extra_parameter(world):
    registered(world)
    world.run("run", "--cutoff", "2026-09-19T23:00:00Z", "--param", "MaxTuningJobs=2", "--image", IMAGE)
    params = {p["Name"]: p["Value"] for p in world.sm.called("start_pipeline_execution")[0]["PipelineParameters"]}
    assert params == {"TriggerReason": "manual", "Cutoff": "2026-09-19T23:00:00Z", "MaxTuningJobs": "2", "TrainImage": IMAGE}
    code, out, err = world.run("run", "--param", "oops")
    assert code == 1 and "NAME=VALUE" in err


def test_a_gate_that_says_no_is_exit_3_with_the_table_and_nothing_registered(world, tmp_path):
    world.sm.script_execution("gate_failed")
    world.s3.put(BUCKET, "metrics/abc00001/evaluation.json", json.dumps(evaluation(0.60, champion={"version": 3, "pr_auc": 0.88})))
    summary = tmp_path / "s.md"
    code, out, err = world.run("run", "--summary", str(summary))
    assert code == 3
    assert "GATE FAILED" in err and "0.6000" in err
    assert "FAIL" in out and "champion v3 (0.8800)" in out
    text = summary.read_text()
    assert "Gate: FAILED" in text and "Nothing was registered" in text and not world.sm.packages


def test_a_failed_step_is_exit_1_with_the_reason(world):
    world.sm.script_execution("step_failed")
    code, out, err = world.run("run")
    assert code == 1 and "Tune" in err and "trial 3 failed" in err


def test_no_wait_prints_only_the_execution_id(world):
    code, out, err = world.run("run", "--no-wait")
    assert code == 0 and last_line(out) == "abc00001" and world.sm._exec_polls == 0


def test_following_gives_up_after_max_wait_but_the_execution_goes_on(world):
    world.sm.exec_script = [{"status": "Executing", "steps": [{"StepName": "Tune", "StepStatus": "Executing"}]}]
    code, out, err = world.run("run", "--max-wait", "100", "--poll", "30")
    assert code == 1 and "still Executing after 100 s" in err and "status abc00001" in err


def test_status_shows_each_step(world):
    registered(world)
    world.run("run", "--no-wait")
    arn = next(iter(world.sm.executions))
    code, out, err = world.run("status", "abc00001")
    assert code == 0, err and "BuildDataset" in out and "Executing" in out
    code, out, err = world.run("status", arn)
    assert code == 0


def test_the_monitor_pipeline_can_be_run_with_run_too(world):
    world.sm.script_execution(monitor=True)
    code, out, err = world.run("run", "--pipeline", "finrisk-monitor", "--param", "Env=prod")
    assert code == 0, err and last_line(out) == "abc00001"
    assert world.sm.called("start_pipeline_execution")[0]["PipelineName"] == "finrisk-monitor"


# ---------------------------------------------------------------- backfill (the command that starts the processing job)


def test_backfill_starts_one_processing_job_with_the_image_and_follows_it(world, tmp_path):
    world.sm.job_script = {}
    code, out, err = world.run("backfill", "--image", IMAGE, "--poll", "1")
    assert code == 0, err
    (job,) = world.sm.called("create_processing_job")
    assert job["ProcessingJobName"].startswith("finrisk-backfill-") and job["AppSpecification"]["ImageUri"] == IMAGE
    args = job["AppSpecification"]["ContainerArguments"]
    assert args[:2] == ["backfill", "--days"] and args[2] == "1-17" and args[args.index("--production") + 1] == "yes"
    assert job["ProcessingResources"]["ClusterConfig"]["InstanceType"] == "ml.m5.xlarge" and job["RoleArn"].endswith("finrisk-sagemaker-exec")
    assert "InProgress" in out and "Completed" in out and "backfill_report.json" in out


def test_backfill_over_three_instances_splits_the_days_and_writes_the_production_set_once(world):
    code, out, err = world.run("backfill", "--image", IMAGE, "--instance-count", "3", "--poll", "1")
    assert code == 0, err
    jobs = world.sm.called("create_processing_job")
    days = [j["AppSpecification"]["ContainerArguments"][2] for j in jobs]
    assert days == ["1-6", "7-12", "13-17"]
    assert [j["AppSpecification"]["ContainerArguments"][j["AppSpecification"]["ContainerArguments"].index("--production") + 1] for j in jobs] == ["yes", "no", "no"]
    names = [j["ProcessingJobName"] for j in jobs]
    assert names[0].endswith("-1") and names[2].endswith("-3")
    def option(job, name):
        arguments = job["AppSpecification"]["ContainerArguments"]
        return arguments[arguments.index(name) + 1]

    assert [option(j, "--instance-index") for j in jobs] == ["1", "2", "3"] and {option(j, "--instance-count") for j in jobs} == {"3"}


def test_backfill_days_and_a_failed_job_with_the_too_slow_hint(world):
    world.sm.job_script = {}
    original = world.sm.create_processing_job

    def create(**kw):
        result = original(**kw)
        world.sm.job_script[kw["ProcessingJobName"]] = [("InProgress", None), ("Failed", "AlgorithmError: STOPPED: the rest would take 95 minutes")]
        return result

    world.sm.create_processing_job = create
    code, out, err = world.run("backfill", "--image", IMAGE, "--days", "1-5", "--poll", "1")
    assert code == 1 and "Failed" in err and "--instance-count" in err
    assert world.sm.called("create_processing_job")[0]["AppSpecification"]["ContainerArguments"][2] == "1-5"


def test_backfill_needs_an_image_digest(world):
    code, out, err = world.run("backfill")
    assert code == 1 and "DIGEST" in err and not world.sm.called("create_processing_job")


def test_split_days_helper():
    assert mlops.split_days(list(range(1, 18)), 1) == [list(range(1, 18))]
    assert [len(x) for x in mlops.split_days(list(range(1, 18)), 4)] == [5, 4, 4, 4]
    assert len(mlops.split_days([1, 2], 9)) == 2 and mlops.days_text([3]) == "3" and mlops.days_text([3, 4, 5]) == "3-5"


# ---------------------------------------------------------------- monitor


def test_monitor_starts_the_pipeline_prints_the_table_and_the_alarms(world, tmp_path):
    world.sm.script_execution(monitor=True)
    world.s3.put(BUCKET, "monitoring/prod/abc00001/metrics.json", json.dumps({
        "psi": {"hour_of_day": {"psi": 0.49, "band": "significant shift - investigate/refit"}, "amount": {"psi": 0.003, "band": "stable"}},
        "metrics": {"MaxFeaturePSI": 0.49, "FraudRateRatio": 12.1, "LabelledRows": 130056.0}, "thresholds_crossed": ["data drift: ... -> alarm finrisk-prod-model-drift"]}))
    world.cw.alarms.update({"finrisk-prod-model-drift": "ALARM", "finrisk-prod-model-prior-shift": "OK", "finrisk-prod-model-quality": "OK"})
    summary = tmp_path / "s.md"
    code, out, err = world.run("monitor", "--env", "prod", "--window-hours", "12", "--summary", str(summary))
    assert code == 0, err
    params = {p["Name"]: p["Value"] for p in world.sm.called("start_pipeline_execution")[0]["PipelineParameters"]}
    assert params == {"Env": "prod", "WindowHours": "12", "Reference": "auto"}
    assert out.index("hour_of_day") < out.index("amount") and "FraudRateRatio" in out and "12.1" in out and "finrisk-prod-model-drift" in out and "ALARM" in out
    assert "Monitor" in summary.read_text() and last_line(out) == "abc00001"


def test_monitor_says_so_when_there_are_no_model_alarms(world):
    world.sm.script_execution(monitor=True)
    code, out, err = world.run("monitor", "--env", "pre-prod")
    assert code == 0 and "no alarms named finrisk-pre-prod-model-" in out


# ---------------------------------------------------------------- mlflow-url


def test_mlflow_url(world):
    world.ssm.params["/finrisk/shared/mlflow_app_arn"] = "arn:aws:sagemaker:ap-southeast-2:123456789012:mlflow-app/app-9"
    code, out, err = world.run("mlflow-url")
    assert code == 0 and last_line(out).startswith("https://") and world.sm.called("create_presigned_mlflow_app_url")[0]["Arn"].endswith("app-9")
    world.ssm.params.clear()
    code, out, err = world.run("mlflow-url")
    assert code == 1 and "mlflow_app_arn" in err


# ---------------------------------------------------------------- approve / list / show


def test_approve_moves_the_status_and_the_model_card(world):
    v = world.sm.add_package()
    code, out, err = world.run("approve", "--version", str(v), "--decision", "approve", "--note", "reviewed the metrics")
    assert code == 0, err and last_line(out) == "Approved" and "Model Card: Approved" in out
    (update,) = world.sm.called("update_model_package")
    assert update["ModelApprovalStatus"] == "Approved" and update["ApprovalDescription"] == "reviewed the metrics" and update["ModelCard"] == {"ModelCardStatus": "Approved"}
    code, out, err = world.run("approve", "--version", str(v), "--decision", "reject", "--note", "recall too low")
    assert code == 0 and last_line(out) == "Rejected" and world.sm.called("update_model_package")[-1]["ModelCard"] == {"ModelCardStatus": "Archived"}


def test_approve_leaves_out_the_card_for_a_version_without_one(world):
    v = world.sm.add_package()
    del world.sm.packages[v]["ModelCard"]
    code, out, err = world.run("approve", "--version", str(v))
    assert code == 0 and last_line(out) == "Approved" and "ModelCard" not in world.sm.called("update_model_package")[0]


def test_approving_a_version_that_does_not_exist_exits_2(world):
    code, out, err = world.run("approve", "--version", "9", "--decision", "approve")
    assert code == 2 and "not found" in err and not world.sm.called("update_model_package")


def test_list_shows_versions_newest_first_and_with_nothing_says_what_to_do(world):
    code, out, err = world.run("list")
    assert code == 0 and "no versions" in out and "python pipeline/mlops.py run" in out
    world.sm.add_package(pr_auc=0.8)
    world.sm.add_package(pr_auc=0.9)
    code, out, err = world.run("list")
    rows = [line for line in out.splitlines() if line.strip()[:1].isdigit() and "PendingManualApproval" in line]
    assert [r.split()[0] for r in rows] == ["2", "1"] and "0.9000" in rows[0]


def test_show_prints_the_provenance_with_ids_masked(world):
    v = world.sm.add_package()
    code, out, err = world.run("show", "--version", str(v))
    assert code == 0, err
    for key in ("train_image_digest", "dataset_uri", "baseline_uri", "ml_git_sha", "cutoff", "trigger_reason", "model card    Draft"):
        assert key in out
    assert ACCOUNT not in out


def test_quiet_prints_only_the_value(world):
    v = world.sm.add_package()
    code, out, err = world.run("approve", "--version", str(v), "--quiet")
    assert code == 0 and out.strip() == "Approved"


def test_a_missing_config_file_is_one_readable_error(world, tmp_path):
    code, out, err = world.run("list", "--config", str(tmp_path / "nope.yaml"))
    assert code == 1 and "config file not found" in err


def test_the_config_must_have_its_sections(world, tmp_path):
    (tmp_path / "c.yaml").write_text("tuning: {}\n")
    code, out, err = world.run("list", "--config", str(tmp_path / "c.yaml"))
    assert code == 1 and "no 'instance:' section" in err


def test_names_come_from_ssm_then_the_naming_convention(world):
    code, out, err = world.run("upsert", "--image", IMAGE, "--ml-bucket", "", env={"ML_BUCKET": "", "SAGEMAKER_ROLE_ARN": ""})
    assert "ml_bucket built from the naming convention" in out
    world2 = type(world)(world.capsys)
    world2.ssm.params["/finrisk/shared/ml_bucket"] = "bucket-from-ssm"
    world2.ssm.params["/finrisk/shared/sagemaker_role_arn"] = "arn:aws:iam::123456789012:role/from-ssm"
    code, out, err = world2.run("upsert", "--image", IMAGE, env={"ML_BUCKET": "", "SAGEMAKER_ROLE_ARN": ""})
    assert "taken from SSM /finrisk/shared/ml_bucket" in out and world2.sm.called("create_pipeline")[0]["RoleArn"].endswith("role/from-ssm")


# ---------------------------------------------------------------- times are real UTC


@pytest.fixture
def melbourne(monkeypatch):
    """The machine's clock is in Melbourne (UTC+10/+11), as on the instructor's laptop. boto3 returns datetimes in LOCAL time."""
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset needs a Unix system")
    monkeypatch.setenv("TZ", "Australia/Melbourne")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_when_prints_utc_not_the_local_clock(melbourne):
    created = datetime(2026, 10, 17, 11, 41, tzinfo=timezone(timedelta(hours=11)))
    assert created.strftime("%H:%M") == "11:41"
    assert mlops.when(created) == "2026-10-17 00:41"
    assert mlops.when(datetime(2026, 10, 17, 0, 41, tzinfo=timezone.utc)) == "2026-10-17 00:41"
    assert mlops.when(datetime(2026, 10, 17, 0, 41)) == "2026-10-17 00:41"
    assert mlops.when("2026-10-17T00:41:00+00:00") == "2026-10-17T00:41"
    assert os.environ["TZ"] == "Australia/Melbourne"


def test_list_and_show_print_the_utc_time_of_a_package_created_in_another_zone(world, melbourne):
    v = world.sm.add_package()
    world.sm.packages[v]["CreationTime"] = datetime(2026, 10, 17, 11, 41, tzinfo=timezone(timedelta(hours=11)))
    code, out, err = world.run("list")
    assert code == 0, err and "created (UTC)" in out and "2026-10-17 00:41" in out and "11:41" not in out
    code, out, err = world.run("show", "--version", str(v))
    assert "created       2026-10-17 00:41 UTC" in out


# ---------------------------------------------------------------- the order guard: no data yet


@pytest.mark.parametrize("reason", ["push", "schedule", "drift"])
def test_run_without_a_backfill_report_is_a_green_notice_for_automatic_triggers(world, tmp_path, reason):
    del world.s3.objects[(BUCKET, "raw/paysim/v1/backfill_report.json")]
    summary = tmp_path / "s.md"
    code, out, err = world.run("run", "--trigger-reason", reason, "--summary", str(summary))
    assert code == 0 and err == ""
    assert "no data yet: run backfill.yml first" in out and "pipelines are upserted and ready" in out and f"trigger: {reason}" in out
    assert "no data yet: run backfill.yml first" in summary.read_text() and f"`{reason}`" in summary.read_text()
    assert not world.sm.called("start_pipeline_execution")                # nothing was started


def test_run_without_a_backfill_report_fails_a_manual_run_with_the_same_words(world, tmp_path):
    del world.s3.objects[(BUCKET, "raw/paysim/v1/backfill_report.json")]
    code, out, err = world.run("run")
    assert code == 1 and "no data yet: run backfill.yml first" in err and "backfill_report.json does not exist" in err
    assert not world.sm.called("start_pipeline_execution")


def test_s3_403_for_a_missing_key_counts_as_not_there(world):
    del world.s3.objects[(BUCKET, "raw/paysim/v1/backfill_report.json")]
    original = world.s3.head_object

    def forbidden(**kw):
        from fakes import client_error

        raise client_error("403", "HeadObject")

    world.s3.head_object = forbidden
    code, out, err = world.run("run", "--trigger-reason", "push")
    assert code == 0 and "no data yet" in out


def test_with_the_report_present_run_goes_ahead_and_the_guard_checks_the_right_key(world):
    registered(world)
    code, out, err = world.run("run")
    assert code == 0, err and world.sm.called("start_pipeline_execution")


def test_the_guard_does_not_apply_to_the_monitor_pipeline(world):
    del world.s3.objects[(BUCKET, "raw/paysim/v1/backfill_report.json")]
    world.sm.script_execution(monitor=True)
    code, out, err = world.run("run", "--pipeline", "finrisk-monitor")
    assert code == 0, err


# ---------------------------------------------------------------- backfill --then-train


def test_then_train_starts_finrisk_train_after_the_backfill_and_does_not_follow_it(world, tmp_path):
    world.sm.job_script = {}
    summary = tmp_path / "s.md"
    code, out, err = world.run("backfill", "--image", IMAGE, "--then-train", "--poll", "1", "--summary", str(summary))
    assert code == 0, err
    assert [c["PipelineName"] for c in world.sm.called("create_pipeline")] == ["finrisk-train", "finrisk-monitor"]       # upsert first: definitions match the image
    (start,) = world.sm.called("start_pipeline_execution")
    assert start["PipelineName"] == "finrisk-train" and start["PipelineParameters"] == [{"Name": "TriggerReason", "Value": "backfill"}]
    assert world.sm._exec_polls == 0                                                  # started, never followed
    arn = last_line(out)
    assert arn.startswith("arn:aws:sagemaker:") and "<account-id>" not in arn
    assert f"python pipeline/mlops.py status {arn}" in out.replace("<account-id>", ACCOUNT) or "mlops.py status" in out
    assert "SageMaker Studio" in out and "Training started" in summary.read_text() and "TriggerReason=backfill" in summary.read_text()
    assert out.index("backfill finished") < out.index("--then-train") < out.index("started finrisk-train")


def test_without_then_train_nothing_is_started(world):
    code, out, err = world.run("backfill", "--image", IMAGE, "--poll", "1")
    assert code == 0 and not world.sm.called("start_pipeline_execution") and not world.sm.called("create_pipeline")


def test_a_failed_backfill_never_starts_the_training(world):
    original = world.sm.create_processing_job

    def create(**kw):
        result = original(**kw)
        world.sm.job_script[kw["ProcessingJobName"]] = [("InProgress", None), ("Failed", "AlgorithmError: Athena still sees only 10 of 20")]
        return result

    world.sm.create_processing_job = create
    code, out, err = world.run("backfill", "--image", IMAGE, "--then-train", "--poll", "1")
    assert code == 1 and "Failed" in err and not world.sm.called("start_pipeline_execution") and not world.sm.called("create_pipeline")


def test_then_train_with_several_instances_starts_the_training_once(world):
    code, out, err = world.run("backfill", "--image", IMAGE, "--instance-count", "2", "--then-train", "--poll", "1")
    assert code == 0, err and len(world.sm.called("start_pipeline_execution")) == 1 and len(world.sm.called("create_processing_job")) == 2


def test_then_train_with_no_wait_explains_why_it_cannot(world):
    code, out, err = world.run("backfill", "--image", IMAGE, "--then-train", "--no-wait")
    assert code == 0 and "needs the jobs to finish first" in out and not world.sm.called("start_pipeline_execution")


# ---------------------------------------------------------------- the live rehearsal's two small findings


def test_the_monitor_asks_for_the_alarms_by_exact_name_not_by_prefix(world):
    """Least privilege: DescribeAlarms by prefix is evaluated on alarm:* and finrisk-ml-train is refused. The fake refuses it too."""
    world.sm.script_execution(monitor=True)
    world.cw.alarms.update({"finrisk-prod-model-drift": "ALARM", "finrisk-prod-model-prior-shift": "OK", "finrisk-prod-model-quality": "OK"})
    code, out, err = world.run("monitor", "--env", "prod")
    assert code == 0, err and "finrisk-prod-model-drift" in out and "ALARM" in out
    assert "AlarmNames" in out and "AlarmNamePrefix" not in out
    assert mlops.model_alarm_names("prod") == ["finrisk-prod-model-drift", "finrisk-prod-model-prior-shift", "finrisk-prod-model-quality"]
    assert mlops.model_alarm_names("pre-prod")[0] == "finrisk-pre-prod-model-drift"


def test_a_prefix_search_really_is_refused_by_the_fake(world):
    from botocore.exceptions import ClientError

    with pytest.raises(ClientError, match="AccessDenied"):
        world.cw.describe_alarms(AlarmNamePrefix="finrisk-prod-model-")


def test_binary_bodies_are_printed_as_a_byte_count_not_as_bytes(world):
    printed = []
    traced = mlops.Traced(world.s3, "s3", printed.append)
    traced.put_object(Bucket=BUCKET, Key="labels/x.parquet", Body=b"PAR1\x00\x01binary" * 1000)
    traced.put_object(Bucket=BUCKET, Key="a.json", Body="text body")
    text = "\n".join(printed)
    assert "<12,000 bytes>" in text and "PAR1" not in text and "text body" in text
