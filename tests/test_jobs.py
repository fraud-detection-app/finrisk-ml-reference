"""build-dataset, train and evaluate on small real data: the Athena query (DuckDB), the splits, the trained package, the evaluation files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import needs_app
from fakes import BUCKET, FakeAthena, FakeS3, FakeSSM, FakeSTS, FakeSageMaker, IMAGE
from helpers import FakeMlflow, demo_gold, history_tables, trained_package

import definition
from finrisk_jobs import build_dataset, common, contract, evaluate, train

pytestmark = needs_app


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """The real build-dataset job on a fake Athena (DuckDB) loaded with 17 days of history."""
    tmp = tmp_path_factory.mktemp("dataset")
    s3 = FakeS3()
    athena = FakeAthena(s3)
    tx, labels = history_tables(demo_gold(30000))
    athena.register("transactions", tx)
    athena.register("labels", labels)
    out = tmp / "out"
    build_dataset.run(["--ml-bucket", BUCKET, "--output", str(out), "--execution-id", "exec-001", "--workdir", str(tmp / "w")], athena=athena, s3=s3,
                      sleep=lambda s: None)
    return {"out": out, "s3": s3, "athena": athena, "tmp": tmp}


# ---------------------------------------------------------------- build-dataset


def test_splits_are_by_day_test_last_validation_before_train_the_rest(built):
    report = json.loads((built["out"] / "dataset.json").read_text())
    s = report["splits"]
    assert s["test"]["days"] == [16, 17] and s["validation"]["days"] == [14, 15] and s["train"]["days"] == [1, 13]
    train_df, valid_df, test_df = (pd.read_parquet(built["out"] / n / "part-0.parquet") for n in ("train", "validation", "test"))
    days = lambda df: sorted(set(df["step"].map(contract.day_of_step)))            # noqa: E731
    assert days(train_df) == list(range(1, 14)) and days(valid_df) == [14, 15] and days(test_df) == [16, 17]
    assert (s["train"]["rows"], s["validation"]["rows"], s["test"]["rows"]) == (len(train_df), len(valid_df), len(test_df))
    assert s["test"]["frauds"] == int(test_df["is_fraud"].sum()) and 0 < s["train"]["prevalence"] < 0.1


def test_dataset_json_records_the_cutoff_the_queries_and_the_columns(built):
    report = json.loads((built["out"] / "dataset.json").read_text())
    assert report["cutoff_input"] == "auto" and report["cutoff"] == "2026-09-19T23:00:00Z"          # last history label_time = day 17 23:00 + 2 days
    assert report["queries"]["cutoff"] and report["queries"]["point_in_time"] and report["execution_id"] == "exec-001"
    assert report["dataset_uri"] == f"s3://{BUCKET}/datasets/exec-001"
    assert report["baseline_uri"].endswith("/profile/baseline.parquet") and report["feature_columns"] == contract.FEATURE_COLUMNS
    assert "is_fraud" in report["columns"] and "step" in report["columns"] and "event_time" in report["columns"]


def test_baseline_is_a_stratified_sample_of_train(built):
    base = pd.read_parquet(built["out"] / "profile" / "baseline.parquet")
    train_df = pd.read_parquet(built["out"] / "train" / "part-0.parquet")
    report = json.loads((built["out"] / "dataset.json").read_text())
    assert len(base) == min(200_000, len(train_df)) == report["baseline"]["rows"]
    assert set(base["row_id"]) <= set(train_df["row_id"])
    assert abs(base["is_fraud"].mean() - train_df["is_fraud"].mean()) < 0.002              # proportional, not oversampled


def test_the_sql_that_ran_is_the_one_in_queries_and_unload_files_are_cleaned_up(built):
    import queries

    assert any(q.startswith("UNLOAD (WITH f AS") for q in built["athena"].queries)
    assert queries.point_in_time("2026-09-19T23:00:00Z") in "\n".join(built["athena"].queries)
    assert not [k for (b, k) in built["s3"].objects if k.startswith("athena/unload/exec-001")]


def test_an_explicit_earlier_cutoff_leaves_out_later_days_and_late_labels(tmp_path):
    s3 = FakeS3()
    athena = FakeAthena(s3)
    tx, labels = history_tables(demo_gold(30000))
    athena.register("transactions", tx)
    athena.register("labels", labels)
    build_dataset.run(["--cutoff", "2026-09-11T00:00:00Z", "--ml-bucket", BUCKET, "--output", str(tmp_path / "o"), "--execution-id", "e2",
                       "--workdir", str(tmp_path / "w")], athena=athena, s3=s3, sleep=lambda s: None)
    report = json.loads((tmp_path / "o" / "dataset.json").read_text())
    assert report["queries"]["cutoff"] is None and report["cutoff"] == "2026-09-11T00:00:00Z"
    # labels of day d are known 2 days later, so only events up to 2026-09-09T00:00 can be both before the cutoff and labelled by it
    assert report["days_present"][-1] == 9 and report["splits"]["test"]["days"] == [8, 9]


def test_too_few_labelled_days_say_what_is_needed(tmp_path):
    s3 = FakeS3()
    athena = FakeAthena(s3)
    tx, labels = history_tables(demo_gold(30000))
    athena.register("transactions", tx)
    athena.register("labels", labels)
    with pytest.raises(common.JobError, match="needs at least 5"):
        build_dataset.run(["--cutoff", "2026-09-05T00:00:00Z", "--ml-bucket", BUCKET, "--output", str(tmp_path / "o"), "--execution-id", "e3"],
                          athena=athena, s3=s3, sleep=lambda s: None)


def test_auto_cutoff_without_any_label_says_to_run_the_backfill(tmp_path):
    s3 = FakeS3()
    athena = FakeAthena(s3)
    tx, labels = history_tables(demo_gold(30000))
    athena.register("transactions", tx)
    athena.register("labels", labels.iloc[0:0])
    with pytest.raises(common.JobError, match="run the backfill"):
        build_dataset.run(["--ml-bucket", BUCKET, "--output", str(tmp_path / "o"), "--execution-id", "e4"], athena=athena, s3=s3, sleep=lambda s: None)


def test_split_days_arithmetic():
    assert build_dataset.split_days(list(range(1, 18)), 2, 2) == {"test": [16, 17], "validation": [14, 15], "train": list(range(1, 14))}
    assert build_dataset.split_days([1, 2, 3, 5, 6, 7], 2, 2)["train"] == [1, 2]


# ---------------------------------------------------------------- train


def read_split(built, name):
    return pd.read_parquet(built["out"] / name / "part-0.parquet")


@pytest.fixture(scope="module")
def packages(built, tmp_path_factory):
    """Two real model packages trained by the job: a small XGBoost (the candidate) and the logistic-regression baseline."""
    base = tmp_path_factory.mktemp("models")
    result = {}
    for kind, hyper in (("optimized", {"max_depth": "3", "n_estimators": "30", "learning_rate": "0.2"}), ("baseline", {})):
        model_dir = base / kind / "model"
        env = {"SM_CHANNEL_TRAIN": str(built["out"] / "train"), "SM_CHANNEL_VALIDATION": str(built["out"] / "validation"), "SM_MODEL_DIR": str(model_dir),
               "SM_OUTPUT_DATA_DIR": str(base / kind / "output"),
               "HYPERPARAMETERS": json.dumps({"model_type": json.dumps(kind), **{k: json.dumps(v) for k, v in hyper.items()}}),
               "TRAINING_JOB_NAME": f"finrisk-tune-20261016-0{1 if kind == 'optimized' else 2}1-abcd1234"}
        metrics = train.main(env)
        tar = common.make_model_tar(model_dir, base / kind / "model.tar.gz")
        result[kind] = {"dir": model_dir, "tar": tar, "metrics": metrics, "env": env}
    return result


def test_training_writes_the_flat_package_without_the_feature_table(packages):
    names = sorted(p.name for p in packages["optimized"]["dir"].iterdir())
    assert names == ["latest_model.joblib", "metrics.json", "model_v2.pkl"]
    assert sorted(p.name for p in packages["baseline"]["dir"].iterdir()) == ["latest_model.joblib", "metrics.json", "model_v1.pkl"]
    assert "features.parquet" not in names


def test_the_package_loads_with_the_apps_own_loader_and_uses_the_contract_features(packages):
    from financial_risk.models.package import load_model_package

    artifact = load_model_package(packages["optimized"]["dir"])
    assert artifact["feature_columns"] == contract.FEATURE_COLUMNS and 0 < artifact["threshold"] < 1
    assert artifact["model"].named_steps["model"].get_params()["max_depth"] == 3          # the tuned value reached the XGBoost model
    assert artifact["model"].named_steps["model"].get_params()["n_estimators"] == 30
    assert json.loads((packages["optimized"]["dir"] / "metrics.json").read_text())["evaluated_on"] == "validation"


def test_training_prints_the_metric_lines_the_tuning_job_reads(packages, capsys):
    import re

    train.main(packages["optimized"]["env"])
    printed = capsys.readouterr().out
    for definition_ in definition.metric_definitions():
        assert re.search(definition_["Regex"], printed), f"{definition_['Name']} not printed in the shape its regex expects"
    assert re.search(r"^validation:pr_auc=\d\.\d{6}$", printed, re.M) and re.search(r"^train:pr_auc=\d\.\d{6}$", printed, re.M)
    assert "e-" not in "".join(l for l in printed.splitlines() if "=" in l and ":" in l.split("=")[0])


def test_hyperparameters_arrive_json_encoded_and_are_decoded():
    assert train.read_hyperparameters({"HYPERPARAMETERS": json.dumps({"model_type": '"baseline"', "max_depth": "5", "learning_rate": "0.1"})}) == {
        "model_type": "baseline", "max_depth": 5, "learning_rate": 0.1}
    with pytest.raises(common.JobError, match="model_type must be"):
        train.read_hyperparameters({"HYPERPARAMETERS": json.dumps({"model_type": "forest"})})


def test_a_trial_name_gives_the_tuning_job():
    assert train.tuning_job_of("finrisk-tune-20261016-003-1a2b3c4d") == "finrisk-tune-20261016"
    assert train.tuning_job_of("finrisk-train-20261016") is None


def test_training_logs_to_mlflow_and_a_broken_mlflow_never_fails_the_job(packages, monkeypatch, capsys):
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    env = {**packages["baseline"]["env"], "MLFLOW_TRACKING_ARN": "arn:aws:sagemaker:ap-southeast-2:123456789012:mlflow-app/app-x"}
    train.main(env)
    assert fake.experiment == "finrisk-fraud" and fake.tracking_uri.startswith("arn:aws:sagemaker") and fake.ended == "FINISHED"
    assert fake.tags["model_type"] == "baseline" and fake.tags["tuning_job"] == "finrisk-tune-20261016" and "validation_pr_auc" in fake.metrics
    assert "train_pr_auc" in fake.metrics and "validation:pr_auc" not in fake.metrics          # ':' is not a safe MLflow metric character
    for broken in ("start_run", "log_params", "log_metric", "log_artifact", "end_run"):
        monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow(fail=broken))
        metrics = train.main(env)
        assert metrics["pr_auc"] > 0
    assert "[mlflow]" in capsys.readouterr().out


def test_without_an_mlflow_arn_the_job_says_so_and_carries_on(packages, capsys):
    train.main(packages["baseline"]["env"])
    assert "no MLFLOW_TRACKING_ARN" in capsys.readouterr().out


def test_missing_channel_and_one_class_are_clear_errors(built, tmp_path):
    env = {"SM_CHANNEL_TRAIN": str(tmp_path / "nope"), "SM_CHANNEL_VALIDATION": str(built["out"] / "validation"), "SM_MODEL_DIR": str(tmp_path / "m"),
           "SM_OUTPUT_DATA_DIR": str(tmp_path / "o"), "HYPERPARAMETERS": "{}"}
    with pytest.raises(common.JobError, match="channel folder is missing"):
        train.main(env)
    one = tmp_path / "one"
    one.mkdir()
    read_split(built, "train").assign(is_fraud=0).to_parquet(one / "p.parquet")
    with pytest.raises(common.JobError, match="one class only"):
        train.main({**env, "SM_CHANNEL_TRAIN": str(one)})


# ---------------------------------------------------------------- evaluate


def make_inputs(built, packages, tmp_path, kind="optimized"):
    base = tmp_path / "in"
    (base / "test").mkdir(parents=True)
    (base / "profile").mkdir()
    (base / "report").mkdir()
    (base / "model").mkdir()
    read_split(built, "test").to_parquet(base / "test" / "part-0.parquet")
    pd.read_parquet(built["out"] / "profile" / "baseline.parquet").to_parquet(base / "profile" / "baseline.parquet")
    (base / "report" / "dataset.json").write_text((built["out"] / "dataset.json").read_text())
    (base / "model" / "model.tar.gz").write_bytes(packages[kind]["tar"].read_bytes())
    return base


def evaluate_args(base, out, **extra):
    args = ["--candidate-job", "finrisk-tune-20261016-001-abcd1234", "--tuning-job-arn", "arn:aws:sagemaker:ap-southeast-2:123456789012:hyper-parameter-tuning-job/finrisk-tune-20261016",
            "--train-image", IMAGE, "--trigger-reason", "drift", "--execution-arn", "arn:aws:sagemaker:ap-southeast-2:123456789012:pipeline/finrisk-train/execution/abc",
            "--git-sha", "d" * 40, "--ml-bucket", BUCKET, "--test-dir", str(base / "test"), "--profile-dir", str(base / "profile"),
            "--report-dir", str(base / "report"), "--model-dir", str(base / "model"), "--output", str(out), "--workdir", str(out.parent / "work"),
            "--shap-rows", "500"]
    for k, v in extra.items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    return args


def clients(champion_tar=None, version=1):
    s3, ssm, sm = FakeS3(), FakeSSM(), FakeSageMaker(FakeS3())
    if champion_tar is not None:
        v = sm.add_package(version=version, pr_auc=0.5)
        s3.put(BUCKET, f"models/champion/output/model.tar.gz", champion_tar.read_bytes())
        sm.packages[v]["InferenceSpecification"]["Containers"][0]["ModelDataUrl"] = f"s3://{BUCKET}/models/champion/output/model.tar.gz"
        ssm.params["/finrisk/prod/model"] = json.dumps({"version": v, "serving": "endpoint",
                                                        "model_package_arn": f"arn:aws:sagemaker:ap-southeast-2:123456789012:model-package/finrisk-fraud-model/{v}"})
    common.use_clients(s3=s3, ssm=ssm, sagemaker=sm, sts=FakeSTS())
    return s3, ssm, sm


def test_evaluate_without_a_champion_writes_the_five_files(built, packages, tmp_path):
    clients()
    base, out = make_inputs(built, packages, tmp_path), tmp_path / "evaluation"
    assert evaluate.run(evaluate_args(base, out)) == 0
    assert sorted(p.name for p in out.iterdir()) == ["bias.json", "evaluation.json", "explainability.json", "model_card.json", "register_info.json"]
    ev = json.loads((out / "evaluation.json").read_text())
    assert set(ev) == {"candidate", "champion", "gate", "test_window"} and ev["champion"] is None
    assert ev["gate"] == {"floor": 0.7, "tolerance": 0.02, "champion_floor": 0.0}
    assert set(ev["candidate"]) >= {"pr_auc", "roc_auc", "threshold", "precision", "recall", "training_job"}
    assert ev["candidate"]["training_job"] == "finrisk-tune-20261016-001-abcd1234"
    window = ev["test_window"]
    assert window["rows"] == len(read_split(built, "test")) and window["days"] == [16, 17] and window["from"] < window["to"]


def test_evaluate_scores_the_candidate_and_the_champion_on_the_same_test_rows(built, packages, tmp_path):
    clients(champion_tar=packages["baseline"]["tar"], version=3)
    base, out = make_inputs(built, packages, tmp_path), tmp_path / "evaluation"
    evaluate.run(evaluate_args(base, out))
    ev = json.loads((out / "evaluation.json").read_text())
    assert ev["champion"]["version"] == 3 and ev["champion"]["model_type"] == "baseline"
    test = read_split(built, "test")
    from sklearn.metrics import average_precision_score
    from financial_risk.models.package import load_model_package
    from financial_risk.models.training import predict_proba_positive

    champ = load_model_package(packages["baseline"]["dir"])
    expected = average_precision_score(test["is_fraud"], predict_proba_positive(champ["model"], test[contract.FEATURE_COLUMNS]))
    assert ev["champion"]["pr_auc"] == pytest.approx(expected)
    from finrisk_jobs.gate import champion_floor

    assert ev["gate"]["champion_floor"] == champion_floor(ev["champion"]["pr_auc"], 0.02) <= ev["champion"]["pr_auc"] - 0.02 + 1e-6


def test_explainability_names_original_features_and_sorts_them(built, packages, tmp_path):
    clients()
    base, out = make_inputs(built, packages, tmp_path), tmp_path / "evaluation"
    evaluate.run(evaluate_args(base, out))
    ex = json.loads((out / "explainability.json").read_text())
    values = [t["mean_abs_shap"] for t in ex["top"]]
    assert len(ex["top"]) <= 10 and values == sorted(values, reverse=True) and values[0] > 0
    assert {t["feature"] for t in ex["top"]} <= set(contract.FEATURE_COLUMNS) and not any("__" in t["feature"] for t in ex["top"])
    assert set(ex["all"]) <= set(contract.FEATURE_COLUMNS) and "SHAP" in ex["method"]


def test_bias_json_has_both_segments_and_the_clarify_numbers(built, packages, tmp_path):
    clients()
    base, out = make_inputs(built, packages, tmp_path), tmp_path / "evaluation"
    evaluate.run(evaluate_args(base, out))
    bias = json.loads((out / "bias.json").read_text())
    assert set(bias["segments"]) == {"transaction_type", "amount_band"} and "no protected attribute" in bias["note"]
    types = bias["segments"]["transaction_type"]
    assert types["reference"] == max(types["groups"], key=lambda g: types["groups"][g]["rows"])
    other = next(g for g in types["groups"] if g != types["reference"])
    assert set(types["groups"][other]) >= {"rows", "prevalence", "positive_rate", "tpr", "fpr", "precision", "ci", "dpl", "dppl", "di", "delta_tpr", "delta_fpr"}
    assert len(bias["segments"]["amount_band"]["groups"]) == 4 and sum(g["rows"] for g in types["groups"].values()) == bias["rows"]


def test_register_info_holds_registry_sized_strings_and_a_card_that_is_json(built, packages, tmp_path):
    clients(champion_tar=packages["baseline"]["tar"], version=3)
    base, out = make_inputs(built, packages, tmp_path), tmp_path / "evaluation"
    evaluate.run(evaluate_args(base, out))
    info = json.loads((out / "register_info.json").read_text())
    expected_keys = {"fail_message", "model_card_content", "training_job", "train_image_digest", "cutoff", "dataset_uri", "baseline_uri", "pipeline_execution_arn",
                     "ml_git_sha", "trigger_reason", "mlflow_run_id", "champion_version", "pr_auc", "champion_pr_auc"}
    assert set(info) == expected_keys
    assert all(isinstance(v, str) and len(v) <= 256 for k, v in info.items() if k != "model_card_content")
    assert info["train_image_digest"] == "sha256:" + "ab" * 32 and info["trigger_reason"] == "drift" and info["champion_version"] == "3"
    card = json.loads(info["model_card_content"])
    assert card == json.loads((out / "model_card.json").read_text())
    assert set(card) == {"model_overview", "intended_uses", "business_details", "training_details", "evaluation_details", "additional_information"}
    metrics = {m["name"]: m["value"] for m in card["evaluation_details"][0]["metric_groups"][0]["metric_data"]}
    assert metrics["PR-AUC (test)"] == pytest.approx(json.loads((out / "evaluation.json").read_text())["candidate"]["pr_auc"], abs=1e-6)
    assert any("Champion v3" in name for name in metrics) and len(info["model_card_content"]) < 100_000
    assert "educational" in json.dumps(card).lower() and card["intended_uses"]["risk_rating"] in ("High", "Medium", "Low", "Unknown")


def test_fail_message_states_the_rule_without_claiming_a_verdict():
    gate = {"floor": 0.7, "tolerance": 0.02, "champion_floor": 0.0}
    assert evaluate.fail_message(0.9541, gate, None) == "GATE FAILED: candidate pr_auc 0.9541; needs >= floor 0.70 (no champion yet)"
    gate = {"floor": 0.7, "tolerance": 0.02, "champion_floor": 0.86}
    assert evaluate.fail_message(0.6123, gate, {"version": 3, "pr_auc": 0.88}) == \
        "GATE FAILED: candidate pr_auc 0.6123; needs >= floor 0.70 and >= champion floor 0.8600 (champion v3 0.8800 - 0.02)"


def test_evaluate_logs_one_mlflow_run_and_links_it_in_the_register_info(built, packages, tmp_path, monkeypatch):
    clients()
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setenv("MLFLOW_TRACKING_ARN", "arn:aws:sagemaker:ap-southeast-2:123456789012:mlflow-app/app-x")
    base, out = make_inputs(built, packages, tmp_path), tmp_path / "evaluation"
    evaluate.run(evaluate_args(base, out))
    info = json.loads((out / "register_info.json").read_text())
    assert info["mlflow_run_id"] == "run-0001" and fake.experiment == "finrisk-fraud" and fake.tags["job_type"] == "evaluate"
    assert fake.tags["trigger_reason"] == "drift" and "candidate_pr_auc" in fake.metrics and fake.metrics["gate_passed"] in (0.0, 1.0)
    assert {a[0].rsplit("/", 1)[1] for a in fake.artifacts} >= {"evaluation.json", "bias.json", "explainability.json", "model_card.json"}


def test_a_candidate_that_fails_the_gate_is_still_a_normal_exit(built, packages, tmp_path):
    clients(champion_tar=packages["optimized"]["tar"], version=2)
    base, out = make_inputs(built, packages, tmp_path, kind="baseline"), tmp_path / "evaluation"
    assert evaluate.run(evaluate_args(base, out, gate_floor=0.99)) == 0              # the Condition step decides, not this job
    ev = json.loads((out / "evaluation.json").read_text())
    assert ev["gate"]["floor"] == 0.99 and ev["candidate"]["pr_auc"] < 0.99
    assert "GATE FAILED" in json.loads((out / "register_info.json").read_text())["fail_message"]


def test_an_unreadable_champion_pointer_stops_the_job_instead_of_guessing(built, packages, tmp_path):
    s3, ssm, sm = clients()
    ssm.params["/finrisk/prod/model"] = "not json"
    base, out = make_inputs(built, packages, tmp_path), tmp_path / "evaluation"
    with pytest.raises(common.JobError, match="does not hold JSON"):
        evaluate.run(evaluate_args(base, out))
    ssm.params["/finrisk/prod/model"] = json.dumps({"version": None})
    assert evaluate.run(evaluate_args(base, out)) == 0


def test_original_feature_names_come_back_from_transformed_columns():
    assert evaluate.original_feature("numeric__amount") == "amount"
    assert evaluate.original_feature("categorical__transaction_type_CASH_OUT") == "transaction_type"


def test_the_tracker_works_with_the_real_mlflow_library(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.chdir(tmp_path)                                         # MLflow writes artifacts under ./mlruns by default
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    tracker = common.Tracker("finrisk-fraud", "real-run", {"job_type": "train"}, tracking_uri=uri).start()
    assert tracker.enabled and tracker.run_id
    tracker.params({"max_depth": 5, "model:type": "x"})
    tracker.metrics({"validation:pr_auc": 0.9, "missing": None})
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}")
    tracker.artifact(artifact, "evaluation")
    tracker.set_tags({"gate": "PASS"})
    tracker.end("FINISHED")
    run = mlflow.MlflowClient(tracking_uri=uri).get_run(tracker.run_id)
    assert run.data.params["max_depth"] == "5" and run.data.metrics == {"validation_pr_auc": 0.9}
    assert run.data.tags["gate"] == "PASS" and run.data.tags["job_type"] == "train" and run.info.status == "FINISHED"


# ---------------------------------------------------------------- build-dataset never trains on a half-filled offline store


class FillingStore(FakeAthena):
    """The labels table is complete at once (plain S3); the offline store's rows become visible over time. Each completeness check
    sees the first `schedule[i]` fraction of the feature rows (the last value repeats); every other query sees all of them."""

    def __init__(self, s3, tx, labels, schedule):
        super().__init__(s3)
        self.tx, self.schedule, self.checks = tx, list(schedule), 0
        self.register("labels", labels)
        self.register("transactions", tx)

    def start_query_execution(self, **kw):
        sql = kw["QueryString"]
        if "LEFT JOIN" in sql:
            fraction = self.schedule[min(self.checks, len(self.schedule) - 1)]
            self.checks += 1
            keep = self.tx.iloc[: int(len(self.tx) * fraction)]
        else:
            keep = self.tx
        self.register("transactions", keep)
        return super().start_query_execution(**kw)


class Clock:
    def __init__(self):
        self.t, self.sleeps = 0.0, []

    def sleep(self, seconds):
        self.t += seconds
        self.sleeps.append(seconds)

    def __call__(self):
        self.t += 0.001
        return self.t


def run_dataset(tmp_path, athena, clock, name="o"):
    s3 = athena.s3
    return build_dataset.run(["--ml-bucket", BUCKET, "--output", str(tmp_path / name), "--execution-id", "e9", "--workdir", str(tmp_path / "w" / name)],
                             athena=athena, s3=s3, sleep=clock.sleep, clock=clock)


def tables():
    return history_tables(demo_gold(30000))


def test_build_dataset_waits_while_the_store_fills_and_then_goes_ahead(tmp_path, capsys):
    tx, labels = tables()
    athena, clock = FillingStore(FakeS3(), tx, labels, schedule=(0.5, 0.9, 1.0)), Clock()
    assert run_dataset(tmp_path, athena, clock) == 0
    out = capsys.readouterr().out
    assert out.count("the offline store is still filling") == 2 and "checking again in 60 s" in out and "offline store complete" in out
    assert [s for s in clock.sleeps if s == 60.0] == [60.0, 60.0] and athena.checks == 3
    report = json.loads((tmp_path / "o" / "dataset.json").read_text())
    assert report["completeness"]["missing"] == 0 and report["completeness"]["labelled"] == len(labels) and report["completeness"]["waited_seconds"] >= 120
    assert any(q.startswith("UNLOAD") for q in athena.queries) and "LEFT JOIN" in athena.queries[1 if "max(label_time)" in athena.queries[0] else 0]


def test_the_check_happens_before_the_dataset_query(tmp_path):
    tx, labels = tables()
    athena = FillingStore(FakeS3(), tx, labels, schedule=(1.0,))
    run_dataset(tmp_path, athena, Clock())
    order = [("LEFT JOIN" in q, q.startswith("UNLOAD")) for q in athena.queries]
    assert order.index((True, False)) < order.index((False, True))


def test_a_store_that_stays_partial_fails_with_exit_1_and_the_reason_after_the_budget(tmp_path, capsys):
    tx, labels = tables()
    athena, clock = FillingStore(FakeS3(), tx, labels, schedule=(0.95,)), Clock()          # 5 % of the labelled rows never get features
    with pytest.raises(common.JobError) as info:
        run_dataset(tmp_path, athena, clock)
    message = str(info.value)
    assert info.value.code == 1 and "the offline store is still filling" in message and "have no features yet" in message and "retry in a few minutes" in message
    assert "~15 minutes" in message and f"of {len(labels):,}" in message
    assert 29 <= athena.checks <= 31 and clock.t >= 29 * 60                               # polled every 60 s for ~30 minutes
    assert not any(q.startswith("UNLOAD") for q in athena.queries)                       # it never trained (never even unloaded) on the partial store
    assert not (tmp_path / "o" / "dataset.json").exists()


def test_a_tiny_gap_within_the_tolerance_is_accepted_and_reported(tmp_path, capsys):
    tx, labels = tables()
    athena = FillingStore(FakeS3(), tx, labels, schedule=(1 - 0.0004,))                    # 0.04 % missing, 0.1 % allowed
    assert run_dataset(tmp_path, athena, Clock()) == 0
    report = json.loads((tmp_path / "o" / "dataset.json").read_text())["completeness"]
    assert 0 < report["missing"] <= report["allowed"] and "within the 0.1% allowed" in capsys.readouterr().out


def test_the_tolerance_is_configurable_zero_means_none_at_all():
    tx, labels = tables()
    athena = FillingStore(FakeS3(), tx, labels, schedule=(1 - 0.0004,))
    clock = Clock()
    with pytest.raises(common.JobError, match="still filling"):
        build_dataset.wait_for_complete_store("2026-09-19T23:00:00Z", {"max_missing_share": 0, "wait_minutes": 3, "poll_seconds": 60}, athena, athena.s3, clock.sleep, clock)
    assert athena.checks <= 4
    ok = build_dataset.wait_for_complete_store("2026-09-19T23:00:00Z", {"max_missing_share": 0.001}, athena, athena.s3, clock.sleep, clock)
    assert ok["missing"] > 0 and ok["allowed"] >= ok["missing"]


def test_the_defaults_in_train_yaml_are_thirty_minutes_every_minute_and_a_tenth_of_a_percent():
    from finrisk_jobs.common import load_yaml

    assert load_yaml("train.yaml")["completeness"] == {"max_missing_share": 0.001, "wait_minutes": 30, "poll_seconds": 60}


def test_no_labels_at_all_say_to_run_the_backfill(tmp_path):
    tx, labels = tables()
    athena = FillingStore(FakeS3(), tx, labels.iloc[0:0], schedule=(1.0,))
    with pytest.raises(common.JobError, match="(run the backfill|Run the backfill)"):
        run_dataset(tmp_path, athena, Clock())


def test_the_wait_also_covers_the_features_a_replay_wrote_minutes_before_the_drift_retrain(tmp_path):
    """After a replay the labels (S3) are there at once; the replay's features (offline store) lag. The same check protects that run."""
    gold = demo_gold(30000)
    tx, labels = history_tables(gold, days=range(1, 23))
    athena, clock = FillingStore(FakeS3(), tx, labels, schedule=(0.85, 1.0)), Clock()
    assert run_dataset(tmp_path, athena, clock) == 0 and athena.checks == 2
