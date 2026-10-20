"""The contracts between the repos: feature columns, calendar, smoke rows."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import needs_app

from finrisk_jobs import contract

ROOT = Path(__file__).resolve().parents[1]


@needs_app
def test_feature_columns_equal_the_apps_in_the_same_order():
    from financial_risk.models.training import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

    assert contract.FEATURE_COLUMNS == FEATURE_COLUMNS
    assert contract.CATEGORICAL_FEATURES == CATEGORICAL_FEATURES and contract.NUMERIC_FEATURES == NUMERIC_FEATURES
    assert "hour_of_day" in contract.FEATURE_COLUMNS and "step" not in contract.FEATURE_COLUMNS


@needs_app
def test_the_serving_contract_accepts_exactly_row_id_and_our_features():
    from financial_risk.serving.contract import INSTANCE_KEYS

    assert list(INSTANCE_KEYS) == ["row_id", *contract.FEATURE_COLUMNS]


@needs_app
def test_check_feature_contract_passes_and_catches_a_difference(monkeypatch):
    from finrisk_jobs import common

    common.check_feature_contract()
    monkeypatch.setattr(contract, "FEATURE_COLUMNS", contract.FEATURE_COLUMNS[:-1])
    with pytest.raises(common.JobError, match="FEATURE_COLUMNS differ"):
        common.check_feature_contract()


def test_the_feature_group_columns_hold_every_model_input_plus_metadata():
    assert set(contract.FEATURE_COLUMNS) <= set(contract.FEATURE_GROUP_COLUMNS)
    assert contract.FEATURE_GROUP_COLUMNS[:5] == ["row_id", "event_time", "step", "transaction_type", "origin_customer"]
    assert set(contract.INTEGRAL_FEATURES) <= set(contract.FEATURE_GROUP_COLUMNS)


def test_the_calendar_maps_steps_onto_september_2026():
    assert contract.event_time_of_step(1) == "2026-09-01T00:00:00Z"
    assert contract.event_time_of_step(24) == "2026-09-01T23:00:00Z"
    assert contract.event_time_of_step(25) == "2026-09-02T00:00:00Z"
    assert contract.event_time_of_step(743) == "2026-10-01T22:00:00Z" and contract.day_of_step(743) == 31
    assert [contract.day_of_step(s) for s in (1, 24, 25, 408, 409, 720, 721)] == [1, 1, 2, 17, 18, 30, 31]
    assert [contract.hour_of_step(s) for s in (1, 24, 25)] == [0, 23, 0]


def test_label_time_is_the_event_time_plus_the_analyst_delay():
    assert contract.label_time_of("2026-09-17T23:00:00Z") == "2026-09-19T23:00:00Z"


def test_parse_days():
    assert contract.parse_days("1-17") == list(range(1, 18))
    assert contract.parse_days("1,3,5-6") == [1, 3, 5, 6]
    assert contract.parse_days(4) == [4]
    with pytest.raises(ValueError, match="first number"):
        contract.parse_days("9-3")
    with pytest.raises(ValueError):
        contract.parse_days("")


def test_data_config_matches_the_contract():
    import yaml

    cfg = yaml.safe_load((ROOT / "config" / "data.yaml").read_text())
    assert cfg["dataset"]["sha256"] == "16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b"
    assert cfg["dataset"]["url"].endswith("/paysim.csv")
    assert cfg["calendar"]["start"] == "2026-09-01T00:00:00Z" and cfg["calendar"]["exclude_days"] == [31]
    assert contract.parse_days(cfg["history_days"]) == list(range(1, 18))
    assert contract.parse_days(cfg["production_days"]) == list(range(18, 31))
    assert cfg["label_delay_days"] == contract.LABEL_DELAY_DAYS
    assert cfg["backfill"]["batch_size"] == 25 and cfg["backfill"]["measure_records"] == 100000 and cfg["backfill"]["max_minutes"] == 60


def test_monitor_config_has_the_spec_thresholds():
    import yaml

    cfg = yaml.safe_load((ROOT / "config" / "monitor.yaml").read_text())
    t = cfg["thresholds"]
    assert (t["psi_significant"], t["fraud_rate_ratio"], t["live_pr_auc_floor"], t["min_labelled_rows"]) == (0.25, 5, 0.70, 500)
    assert cfg["cloudwatch"] == {"namespace": "FinRisk/Model", "dimension": "Env"}
    assert contract.CLOUDWATCH_NAMESPACE == "FinRisk/Model"


def test_train_config_has_the_spec_ranges_and_gate():
    import yaml

    cfg = yaml.safe_load((ROOT / "config" / "train.yaml").read_text())
    r = cfg["tuning"]["ranges"]
    assert (r["max_depth"]["min"], r["max_depth"]["max"]) == (3, 8) and r["learning_rate"]["scaling"] == "Logarithmic"
    assert (r["n_estimators"]["min"], r["n_estimators"]["max"]) == (100, 500)
    assert (r["subsample"]["min"], r["colsample_bytree"]["min"]) == (0.6, 0.6) and (r["min_child_weight"]["min"], r["min_child_weight"]["max"]) == (1, 10)
    assert cfg["tuning"]["objective"] == contract.HPO_OBJECTIVE == "validation:pr_auc"
    assert (cfg["tuning"]["max_jobs"], cfg["tuning"]["parallel_jobs"], cfg["tuning"]["max_runtime_seconds"]) == (8, 4, 1800)
    assert (cfg["gate"]["floor"], cfg["gate"]["tolerance"], cfg["gate"]["champion_env"]) == (0.70, 0.02, "prod")
    assert (cfg["splits"]["test_days"], cfg["splits"]["validation_days"], cfg["splits"]["baseline_rows"]) == (2, 2, 200000)
    e = cfg["endpoints"]
    assert e["serverless"] == {"memory_mb": 3072, "max_concurrency": 5} and e["realtime"]["model_instance_type"] == "ml.t2.medium"
    assert (e["blue_green"]["canary_percent"], e["blue_green"]["canary_wait_seconds"], e["blue_green"]["termination_wait_seconds"]) == (50, 120, 120)
    assert e["blue_green"]["bake_seconds"] == 120 and e["realtime"]["instance_count"] == 1
    assert e["keep_configs"] == 2


# ---------------------------------------------------------------- the smoke rows


def smoke():
    return json.loads((ROOT / "config" / "smoke.json").read_text())


def test_smoke_has_three_instances_with_exactly_the_contract_keys():
    doc = smoke()
    assert len(doc["instances"]) == 3
    for instance in doc["instances"]:
        assert set(instance) == {"row_id", *contract.FEATURE_COLUMNS}
        assert isinstance(instance["row_id"], str)


def test_smoke_rows_obey_the_rules_they_say_they_were_chosen_by():
    fraud, legit1, legit2 = smoke()["instances"]
    assert fraud["transaction_type"] == "TRANSFER" and fraud["amount"] == fraud["origin_old_balance"] and fraud["origin_new_balance"] == 0
    assert fraud["destination_old_balance"] == 0 and fraud["tx_index_for_origin"] == 1 and 100_000 <= fraud["amount"] <= 200_000
    assert legit1["transaction_type"] == "PAYMENT" and 5_000 <= legit1["amount"] <= 10_000 and legit1["origin_new_balance"] > 0
    assert legit2["transaction_type"] == "CASH_OUT" and 20_000 <= legit2["amount"] <= 60_000 and legit2["origin_new_balance"] > 0
    for row in (legit1, legit2):
        assert 8 <= row["hour_of_day"] <= 18
    labels = smoke()["labels"]
    assert [labels[i["row_id"]] for i in smoke()["instances"]] == [1, 0, 0]


def test_smoke_features_are_internally_consistent():
    import math

    for row in smoke()["instances"]:
        assert row["origin_balance_delta"] == pytest.approx(row["origin_old_balance"] - row["origin_new_balance"], abs=0.01)
        assert row["destination_balance_delta"] == pytest.approx(row["destination_new_balance"] - row["destination_old_balance"], abs=0.01)
        assert row["customer_amount_so_far"] == pytest.approx(row["amount"]) and row["customer_tx_count_so_far"] == row["tx_index_for_origin"] == 1
        assert row["log_amount"] == pytest.approx(math.log10(1 + row["amount"]), rel=1e-4)       # DuckDB's log() is base 10 (the app's SQL)
        assert row["amount_to_origin_balance"] == pytest.approx(row["amount"] / row["origin_old_balance"], rel=1e-4)
