"""promote = deploy to an endpoint: approval rules, serverless vs real-time, create vs update, blue/green, rollback detection,
smoke test, the pointer, the /ready wait, clean-up of old configs, teardown."""

from __future__ import annotations

import json
import re

import pytest
from fakes import ACCOUNT, APP_URL, BUCKET, FakeApp, client_error, last_line, ready

import deploy


def add(world, **kw):
    return world.sm.add_package(**kw)


def promote(world, env, version, *extra, app=None):
    world.set_app_url(env)
    if env in ("pre-prod", "prod"):
        world.add_alarms(env)
    return world.run("promote", "--env", env, "--version", str(version), *extra, app=app or FakeApp([ready(version)]))


# ---------------------------------------------------------------- who may run what


@pytest.mark.parametrize("env", ["pre-prod", "prod"])
def test_pre_prod_and_prod_refuse_a_pending_version(world, env):
    v = add(world)
    code, out, err = promote(world, env, v)
    assert code == 2 and "PendingManualApproval" in err and "not Approved" in err
    assert not world.ssm.puts and not world.sm.called("create_model")                      # nothing was created, nothing written


@pytest.mark.parametrize("env", ["dev", "sit"])
def test_dev_and_sit_accept_a_pending_version(world, env):
    v = add(world)
    code, out, err = promote(world, env, v)
    assert code == 0, err
    assert json.loads(world.ssm.params[f"/finrisk/{env}/model"])["status"] == "PendingManualApproval"


@pytest.mark.parametrize("env", ["dev", "sit", "pre-prod", "prod"])
def test_a_rejected_version_is_never_accepted(world, env):
    v = add(world, status="Rejected")
    code, out, err = promote(world, env, v)
    assert code == 2 and "Rejected" in err and not world.ssm.puts and not world.sm.called("create_model")


@pytest.mark.parametrize("env", ["pre-prod", "prod"])
def test_pre_prod_and_prod_accept_an_approved_version(world, env):
    v = add(world, status="Approved")
    code, out, err = promote(world, env, v)
    assert code == 0, err
    assert world.ssm.puts[0]["Name"] == f"/finrisk/{env}/model"


def test_an_unknown_version_exits_2(world):
    world.set_app_url("dev")
    code, out, err = world.run("promote", "--env", "dev", "--version", "7")
    assert code == 2 and "not found" in err


def test_an_unknown_environment_is_refused_by_the_parser(world):
    with pytest.raises(SystemExit):
        world.run("promote", "--env", "staging", "--version", "1")


def test_the_app_url_is_looked_up_before_anything_is_created(world):
    v = add(world)
    code, out, err = world.run("promote", "--env", "dev", "--version", str(v))             # no /finrisk/dev/app_url
    assert code == 1 and "/finrisk/dev/app_url does not exist" in err and "--no-wait" in err
    assert not world.sm.called("create_model") and not world.ssm.puts


# ---------------------------------------------------------------- the first deploy


def test_dev_first_deploy_creates_a_serverless_endpoint(world):
    v = add(world)
    code, out, err = promote(world, "dev", v)
    assert code == 0, err
    (model,), (config,), (endpoint,) = (world.sm.called(n) for n in ("create_model", "create_endpoint_config", "create_endpoint"))
    assert re.fullmatch(r"finrisk-dev-fraud-v1-\d{14}", model["ModelName"]) and config["EndpointConfigName"] == model["ModelName"]
    assert endpoint["EndpointName"] == "finrisk-dev-fraud" and endpoint["EndpointConfigName"] == config["EndpointConfigName"]
    assert config["ProductionVariants"] == [{"VariantName": "AllTraffic", "ModelName": model["ModelName"], "ServerlessConfig": {"MemorySizeInMB": 3072, "MaxConcurrency": 5}}]
    assert "DataCaptureConfig" not in config                                                # serverless does not support capture
    assert not world.sm.called("update_endpoint")
    container = model["PrimaryContainer"]
    assert container["Environment"]["MODEL_VERSION"] == "1" and container["ModelDataUrl"].endswith("/output/model.tar.gz") and "@sha256:" in container["Image"]
    assert model["ExecutionRoleArn"] == f"arn:aws:iam::{ACCOUNT}:role/finrisk-dev-endpoint"      # the naming convention when SSM has no value


def test_the_endpoint_role_comes_from_ssm_when_infra_wrote_it(world):
    v = add(world)
    world.ssm.params["/finrisk/dev/endpoint_role_arn"] = "arn:aws:iam::123456789012:role/finrisk-dev-endpoint-x"
    code, out, err = promote(world, "dev", v)
    assert code == 0 and world.sm.called("create_model")[0]["ExecutionRoleArn"].endswith("finrisk-dev-endpoint-x") and "taken from SSM" in out


@pytest.mark.parametrize("env", ["pre-prod", "prod"])
def test_pre_prod_and_prod_are_real_time_with_data_capture(world, env):
    v = add(world, status="Approved")
    code, out, err = promote(world, env, v)
    assert code == 0, err
    config = world.sm.called("create_endpoint_config")[0]
    variant = config["ProductionVariants"][0]
    assert variant["InstanceType"] == "ml.t2.medium" and variant["InitialInstanceCount"] == 1 and "ServerlessConfig" not in variant
    assert config["DataCaptureConfig"] == {
        "EnableCapture": True, "InitialSamplingPercentage": 100, "DestinationS3Uri": f"s3://{BUCKET}/capture/{env}/",
        "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}], "CaptureContentTypeHeader": {"JsonContentTypes": ["application/json"]}}
    assert world.sm.called("create_endpoint") and "DeploymentConfig" not in world.sm.called("create_endpoint")[0]   # the first deploy: nothing to roll back to


def test_calls_are_printed_before_they_are_made_and_account_ids_are_masked(world):
    v = add(world)
    code, out, err = promote(world, "dev", v)
    assert ACCOUNT not in out and ACCOUNT not in err and "<account-id>" in out
    order = [out.index(f"AWS  sagemaker.{op}") for op in ("describe_model_package", "create_model", "create_endpoint_config", "create_endpoint")]
    assert order == sorted(order) and out.index("AWS  sagemaker-runtime.invoke_endpoint") > order[-1] and out.index("AWS  ssm.put_parameter") > out.index("invoke_endpoint")


def test_the_pointer_json_is_exactly_the_contract(world):
    v = add(world, status="Approved")
    env = {"GITHUB_SERVER_URL": "https://github.com", "GITHUB_REPOSITORY": "fraud-detection-app/finrisk-ml", "GITHUB_RUN_ID": "99"}
    world.set_app_url("prod"); world.add_alarms("prod")
    code, out, err = world.run("promote", "--env", "prod", "--version", str(v), env=env, app=FakeApp([ready(v)]))
    assert code == 0, err
    put = world.ssm.puts[-1]
    from fakes import check_shape

    check_shape("ssm", "PutParameter", put)
    assert put["Overwrite"] is True and put["Name"] == "/finrisk/prod/model" and put["Type"] == "String"
    pointer = json.loads(put["Value"])
    assert list(pointer) == ["version", "serving", "endpoint", "endpoint_config", "model_package_arn", "status", "promoted_at", "run"]
    config = world.sm.called("create_endpoint_config")[0]["EndpointConfigName"]
    assert pointer == {"version": 1, "serving": "endpoint", "endpoint": "finrisk-prod-fraud", "endpoint_config": config,
                       "model_package_arn": f"arn:aws:sagemaker:ap-southeast-2:{ACCOUNT}:model-package/finrisk-fraud-model/1", "status": "Approved",
                       "promoted_at": pointer["promoted_at"], "run": "https://github.com/fraud-detection-app/finrisk-ml/actions/runs/99"}
    assert isinstance(pointer["version"], int) and pointer["promoted_at"].startswith("2026-10-16T03:")


def test_the_pointer_is_written_only_after_the_smoke_test_passed(world):
    v = add(world)
    world.runtime.force_version = "99"                       # the endpoint answers, but with the wrong version
    code, out, err = promote(world, "dev", v)
    assert code == 1 and "reports model_version '99'" in err
    assert not world.ssm.puts                                # the app was never pointed at it


def test_smoke_test_sends_the_three_fixed_rows_and_requires_http_200(world):
    v = add(world)
    world.runtime.status = 500
    code, out, err = promote(world, "dev", v)
    assert code == 1 and "HTTP 500" in err and not world.ssm.puts
    world2 = type(world)(world.capsys)
    v = add(world2)
    code, out, err = promote(world2, "dev", v)
    (request,) = world2.runtime.requests
    body = json.loads(request["Body"])
    assert request["ContentType"] == "application/json" and len(body["instances"]) == 3 and body["explain"] is False
    assert "row   4667" in out and "flagged True" in out


def test_a_cold_serverless_endpoint_is_retried_by_the_smoke_test(world):
    v = add(world)
    world.runtime.fail_times = 2
    code, out, err = promote(world, "dev", v)
    assert code == 0, err and "smoke test attempt 1/4" in out and "smoke test passed" in out
    assert len(world.runtime.requests) == 3


def test_a_smoke_test_that_never_answers_leaves_the_pointer_alone(world):
    v = add(world)
    world.runtime.fail_times = 99
    code, out, err = promote(world, "dev", v)
    assert code == 1 and "did not answer after 4 tries" in err and "pointer was NOT changed" in err and not world.ssm.puts


# ---------------------------------------------------------------- update, blue/green, rollback


def deployed(world, env, version=1):
    """Run a first deploy so the endpoint exists; returns the config name."""
    add(world, version=version, status="Approved")
    code, out, err = promote(world, env, version)
    assert code == 0, err
    return world.sm.endpoints[f"finrisk-{env}-fraud"]["EndpointConfigName"]


@pytest.mark.parametrize("env", ["pre-prod", "prod"])
def test_a_second_deploy_to_pre_prod_or_prod_is_a_blue_green_update_with_rollback_alarms(world, env):
    old = deployed(world, env)
    add(world, version=2, status="Approved")
    code, out, err = promote(world, env, 2, app=FakeApp([ready(2)]))
    assert code == 0, err
    (update,) = world.sm.called("update_endpoint")
    assert update["EndpointName"] == f"finrisk-{env}-fraud" and update["EndpointConfigName"] != old
    assert update["DeploymentConfig"] == {
        "BlueGreenUpdatePolicy": {"TrafficRoutingConfiguration": {"Type": "ALL_AT_ONCE", "WaitIntervalInSeconds": 120},      # one instance: no canary
                                  "TerminationWaitInSeconds": 120, "MaximumExecutionTimeoutInSeconds": 1800},
        "AutoRollbackConfiguration": {"Alarms": [{"AlarmName": f"finrisk-{env}-endpoint-5xx"}, {"AlarmName": f"finrisk-{env}-endpoint-latency"}]}}
    assert len(world.sm.called("create_endpoint")) == 1                                    # only the first deploy created it
    assert ("UPDATE with blue/green ALL_AT_ONCE (1 instance: CANARY needs 2+), bake 120 s, auto-rollback on "
            f"finrisk-{env}-endpoint-5xx and finrisk-{env}-endpoint-latency") in out


@pytest.mark.parametrize("env", ["dev", "sit"])
def test_dev_and_sit_are_replaced_in_place_without_blue_green(world, env):
    deployed(world, env)
    add(world, version=2)
    code, out, err = promote(world, env, 2, app=FakeApp([ready(2)]))
    assert code == 0, err
    (update,) = world.sm.called("update_endpoint")
    assert "DeploymentConfig" not in update and "blue/green" not in out


def test_the_rollback_alarms_must_exist_before_a_guarded_update(world):
    deployed(world, "prod")
    add(world, version=2, status="Approved")
    world.cw.alarms.clear()
    code, out, err = world.run("promote", "--env", "prod", "--version", "2", app=FakeApp([ready(2)]))
    assert code == 1 and "finrisk-prod-endpoint-5xx" in err and "finrisk-infra" in err
    assert len(world.sm.called("create_model")) == 1 and len(world.sm.called("update_endpoint")) == 0


def test_an_automatic_rollback_is_reported_and_the_pointer_stays(world):
    old = deployed(world, "prod")
    pointer_before = world.ssm.params["/finrisk/prod/model"]
    add(world, version=2, status="Approved")
    world.sm.endpoint_script = [("Updating", old, None), ("Updating", old, None), ("InService", old, None)]      # back on the old config = rolled back
    code, out, err = promote(world, "prod", 2)
    assert code == 1 and "ROLLED BACK" in err and old in err and "finrisk-prod-endpoint-5xx" in err
    assert world.ssm.params["/finrisk/prod/model"] == pointer_before
    assert len(world.runtime.requests) == 1                          # no smoke test of the new version: it never served


def test_a_failed_endpoint_stops_with_the_failure_reason(world):
    add(world)
    world.sm.endpoint_script = [("Creating", "x", None), ("Failed", "x", "The primary container for production variant AllTraffic did not pass the ping health check")]
    code, out, err = promote(world, "dev", 1)
    assert code == 1 and "is Failed" in err and "ping health check" in err and not world.ssm.puts


def test_an_endpoint_that_is_busy_cannot_be_updated(world):
    deployed(world, "dev")
    world.sm.endpoints["finrisk-dev-fraud"]["EndpointStatus"] = "Updating"
    world.sm.endpoints["finrisk-dev-fraud"].pop("_script", None)
    add(world, version=2)
    code, out, err = promote(world, "dev", 2)
    assert code == 1 and "wait until it is InService" in err


def test_the_wait_times_out_with_a_pointer_to_the_endpoint(world):
    add(world)
    world.sm.endpoint_script = [("Creating", "x", None)]                # never finishes
    code, out, err = promote(world, "dev", 1)
    assert code == 1 and "still Creating after 25 minutes" in err and "describe-endpoint" in err


def test_status_changes_are_printed_once_each(world):
    add(world)
    code, out, err = promote(world, "dev", 1)
    assert out.count("] Creating") == 1 and out.count("] InService") == 1


# ---------------------------------------------------------------- the app's /ready


def test_promote_waits_for_ready_to_report_the_new_version_from_the_endpoint(world):
    v = add(world)
    app = FakeApp([ready(0, source="baked"), ready(0, source="baked"), ready(v)])
    code, out, err = promote(world, "dev", v, app=app)
    assert code == 0, err and len(app.urls) == 3 and app.urls[0] == f"{APP_URL}/ready"
    assert "source=baked" in out and "source=endpoint" in out and "serves model version 1 from the endpoint" in out


def test_a_load_error_after_the_pointer_was_written_is_a_verdict(world):
    v = add(world)
    bad = (503, {"ready": False, "model": {"version": 0, "source": "baked", "load_error": "endpoint finrisk-dev-fraud is not InService"},
                 "pointers_checked_at": "2026-10-16T03:30:00+00:00"})
    code, out, err = promote(world, "dev", v, app=FakeApp([bad]))
    assert code == 1 and "could not load model version 1" in err and "not InService" in err


def test_a_load_error_from_before_the_pointer_is_not(world):
    v = add(world)
    old = (503, {"ready": False, "model": {"version": 0, "load_error": "old problem"}, "pointers_checked_at": "2026-10-16T02:00:00+00:00"})
    code, out, err = promote(world, "dev", v, app=FakeApp([old, ready(v)]))
    assert code == 0, err


def test_no_wait_skips_the_app_but_not_the_endpoint(world):
    v = add(world)
    code, out, err = world.run("promote", "--env", "dev", "--version", str(v), "--no-wait")            # no app_url needed
    assert code == 0, err and "not waiting for the app" in out and world.sm.called("create_endpoint") and world.ssm.puts


def test_the_ready_wait_times_out_with_the_last_answer(world):
    v = add(world)
    code, out, err = promote(world, "dev", v, "--timeout", "30", app=FakeApp([ready(0, source="baked")]))
    assert code == 1 and "did not report model version 1 within 30 s" in err


def test_the_last_line_is_the_version_and_the_summary_names_the_endpoint(world, tmp_path):
    v = add(world, status="Approved")
    summary = tmp_path / "s.md"
    code, out, err = promote(world, "prod", v, "--summary", str(summary))
    assert code == 0 and last_line(out) == "1"
    text = summary.read_text()
    assert "**prod**: model version 1" in text and "finrisk-prod-fraud" in text and "real-time + Data Capture" in text and "created" in text and ACCOUNT not in text


# ---------------------------------------------------------------- clean-up and rollback


def test_older_configs_and_models_are_deleted_but_the_two_newest_stay(world):
    for v in (1, 2, 3, 4):
        add(world, version=v)
        code, out, err = promote(world, "dev", v, app=FakeApp([ready(v)]))
        assert code == 0, err
    names = sorted(world.sm.configs)
    assert len(names) == 2 and [re.search(r"-v(\d)-", n).group(1) for n in names] == ["3", "4"]
    assert sorted(world.sm.models) == names                                       # a model for every kept config, none left over
    assert world.sm.endpoints["finrisk-dev-fraud"]["EndpointConfigName"] in names


def test_the_config_the_endpoint_serves_is_never_deleted_even_if_older(world):
    for v in (1, 2, 3):
        add(world, version=v)
        promote(world, "dev", v, app=FakeApp([ready(v)]))
    oldest = sorted(world.sm.configs)[0]
    removed = deploy.cleanup_old(_Ctx(world), "dev", 1, oldest)               # keep only the newest 1, but the serving one is protected
    assert oldest in world.sm.configs and oldest not in removed and len(world.sm.configs) == 2


class _Ctx:
    """The two attributes cleanup_old needs from mlops.Context."""

    def __init__(self, world):
        self.sm, self.out = world.sm, lambda line: None


def test_rolling_back_is_the_same_command_with_an_older_version(world):
    for v in (1, 2):
        add(world, version=v, status="Approved")
        promote(world, "prod", v, app=FakeApp([ready(v)]))
    code, out, err = promote(world, "prod", 1, app=FakeApp([ready(1)]))
    assert code == 0, err
    pointer = json.loads(world.ssm.params["/finrisk/prod/model"])
    assert pointer["version"] == 1 and "-v1-" in pointer["endpoint_config"]
    assert len(world.sm.called("update_endpoint")) == 2 and "DeploymentConfig" in world.sm.called("update_endpoint")[-1]


# ---------------------------------------------------------------- teardown


def test_teardown_deletes_endpoint_configs_and_models_and_resets_the_pointer(world):
    for v in (1, 2):
        add(world, version=v, status="Approved")
        promote(world, "prod", v, app=FakeApp([ready(v)]))
    code, out, err = world.run("teardown", "--env", "prod", "--confirm", "prod")
    assert code == 0, err and last_line(out) == "prod"
    assert not world.sm.endpoints and not world.sm.configs and not world.sm.models
    assert json.loads(world.ssm.params["/finrisk/prod/model"]) == {"version": None}
    assert world.sm.called("delete_endpoint") == [{"EndpointName": "finrisk-prod-fraud"}]


def test_teardown_needs_the_environment_typed_again(world):
    deployed(world, "dev")
    code, out, err = world.run("teardown", "--env", "dev", "--confirm", "prod")
    assert code == 1 and "--confirm dev" in err and world.sm.endpoints
    code, out, err = world.run("teardown", "--env", "dev")
    assert code == 1 and world.sm.endpoints


def test_teardown_of_an_environment_with_nothing_is_fine(world):
    code, out, err = world.run("teardown", "--env", "sit", "--confirm", "sit")
    assert code == 0, err and "does not exist" in out
    assert json.loads(world.ssm.params["/finrisk/sit/model"]) == {"version": None}


def test_teardown_only_touches_its_own_environment(world):
    deployed(world, "dev")
    add(world, version=2)
    promote(world, "sit", 2, app=FakeApp([ready(2)]))
    world.run("teardown", "--env", "dev", "--confirm", "dev")
    assert "finrisk-sit-fraud" in world.sm.endpoints and all("-sit-" in n for n in world.sm.configs)


# ---------------------------------------------------------------- how traffic moves: by the variant's instance count


import copy

import yaml
from pathlib import Path

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "train.yaml").read_text())["endpoints"]


def with_instances(count):
    cfg = copy.deepcopy(CFG)
    cfg["realtime"]["instance_count"] = count
    return cfg


def test_one_instance_means_all_at_once_with_a_bake_and_no_canary_size():
    config = deploy.deployment_config("prod", with_instances(1))
    routing = config["BlueGreenUpdatePolicy"]["TrafficRoutingConfiguration"]
    assert routing == {"Type": "ALL_AT_ONCE", "WaitIntervalInSeconds": 120} and "CanarySize" not in routing
    assert config["BlueGreenUpdatePolicy"]["TerminationWaitInSeconds"] == 120 and config["BlueGreenUpdatePolicy"]["MaximumExecutionTimeoutInSeconds"] == 1800
    assert config["AutoRollbackConfiguration"]["Alarms"] == [{"AlarmName": "finrisk-prod-endpoint-5xx"}, {"AlarmName": "finrisk-prod-endpoint-latency"}]


def test_two_or_more_instances_keep_the_canary():
    for count in (2, 3):
        routing = deploy.deployment_config("pre-prod", with_instances(count))["BlueGreenUpdatePolicy"]["TrafficRoutingConfiguration"]
        assert routing == {"Type": "CANARY", "WaitIntervalInSeconds": 120, "CanarySize": {"Type": "CAPACITY_PERCENT", "Value": 50}}


def test_the_bake_comes_from_the_config_and_the_sentence_says_which_strategy_and_why():
    cfg = with_instances(1)
    cfg["blue_green"]["bake_seconds"] = 300
    routing, why = deploy.routing_strategy(cfg)
    assert routing["WaitIntervalInSeconds"] == 300 and why == "blue/green ALL_AT_ONCE (1 instance: CANARY needs 2+), bake 300 s"
    assert deploy.routing_strategy(with_instances(2))[1] == "blue/green CANARY (2 instances): 50 % for 120 s, then the rest"
    assert CFG["blue_green"]["bake_seconds"] == 120


def test_dev_and_sit_have_no_deployment_config_whatever_the_count():
    assert deploy.deployment_config("dev", with_instances(2)) is None and deploy.deployment_config("sit", with_instances(1)) is None


@pytest.mark.parametrize("count", [1, 2])
def test_both_deployment_configs_are_valid_update_endpoint_requests(count):
    from fakes import check_shape

    for env in ("pre-prod", "prod"):
        check_shape("sagemaker", "UpdateEndpoint", {"EndpointName": f"finrisk-{env}-fraud", "EndpointConfigName": "finrisk-x-v1-20261016030000",
                                                    "DeploymentConfig": deploy.deployment_config(env, with_instances(count))})


def test_a_two_instance_deploy_uses_the_canary_end_to_end(world, monkeypatch):
    import mlops

    original = mlops.load_train_config

    def two(path):
        cfg = original(path)
        cfg["endpoints"]["realtime"]["instance_count"] = 2
        return cfg

    monkeypatch.setattr(mlops, "load_train_config", two)
    for v in (1, 2):
        add(world, version=v, status="Approved")
        code, out, err = promote(world, "prod", v, app=FakeApp([ready(v)]))
        assert code == 0, err
    routing = world.sm.called("update_endpoint")[0]["DeploymentConfig"]["BlueGreenUpdatePolicy"]["TrafficRoutingConfiguration"]
    assert routing["Type"] == "CANARY" and "blue/green CANARY (2 instances)" in out
    assert world.sm.called("create_endpoint_config")[0]["ProductionVariants"][0]["InitialInstanceCount"] == 2
