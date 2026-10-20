"""replay (the simulated customers) and feedback (the simulated analysts), and the loop they close with the next training set."""

from __future__ import annotations

import io
import json

import pandas as pd
import pytest
from conftest import needs_app
from fakes import ACCOUNT, APP_URL, BUCKET, FakeAthena, FakeScoreApp, last_line
from helpers import demo_gold, history_tables

from finrisk_jobs import build_dataset, contract

pytestmark = needs_app


@pytest.fixture(scope="module")
def gold():
    return demo_gold(30000)


def put_replay(world, gold, days=range(18, 23)):
    for day in days:
        part = gold[gold["day"] == day][[*contract.FEATURE_GROUP_COLUMNS, "is_fraud"]]
        buffer = io.BytesIO()
        part.to_parquet(buffer)
        world.s3.put(BUCKET, f"replay/v1/day={day}/part.parquet", buffer.getvalue())
    world.set_app_url("prod")


def rows_of(gold, days):
    return int(gold["day"].isin(list(days)).sum())


# ---------------------------------------------------------------- replay


def test_replay_posts_the_contract_batches_in_event_order_and_prints_a_line_per_day(world, gold, tmp_path):
    put_replay(world, gold, range(18, 20))
    summary = tmp_path / "s.md"
    code, out, err = world.run("replay", "--from-day", "18", "--to-day", "19", "--batch", "500", "--summary", str(summary))
    assert code == 0, err
    requests = world.score_app.requests
    assert all(r["url"] == f"{APP_URL}/api/score" for r in requests) and all(1 <= len(r["transactions"]) <= 500 for r in requests)
    sent = [t["row_id"] for r in requests for t in r["transactions"]]
    assert len(sent) == rows_of(gold, [18, 19]) and len(set(sent)) == len(sent)
    steps = [int(gold.loc[gold["row_id"] == i, "step"].iloc[0]) for i in sent[:200]]
    assert steps == sorted(steps)                                                                 # event-time order
    assert "day 18:" in out and "day 19:" in out and "batches" in out and "flagged" in out and ACCOUNT not in out
    assert out.index("day 18:") < out.index("day 19:") and "replayed" in out and last_line(out) == str(len(sent))
    text = summary.read_text()
    assert "| 18 |" in text and "Model version(s) that answered: 1" in text


def test_explain_is_asked_for_on_a_small_share_of_the_batches(world, gold):
    put_replay(world, gold, [18])
    world.run("replay", "--from-day", "18", "--to-day", "18", "--batch", "100", "--explain-every", "4")
    flags = [r["explain"] for r in world.score_app.requests]
    assert flags[3] is True and flags.count(True) == len(flags) // 4 and flags[0] is False


def test_the_features_of_every_replayed_row_are_recorded_in_the_feature_store(world, gold):
    put_replay(world, gold, [18, 19])
    world.run("replay", "--from-day", "18", "--to-day", "19")
    assert len(world.fs.records) == rows_of(gold, [18, 19])
    record = next(iter(world.fs.records.values()))
    assert list(record) == contract.FEATURE_GROUP_COLUMNS and record["event_time"].startswith("2026-09-1")
    assert world.fs.calls >= 2


def test_a_503_means_the_endpoint_is_starting_so_replay_waits_and_retries(world, gold):
    put_replay(world, gold, [18])
    world.score_app.statuses = [503, 503]
    code, out, err = world.run("replay", "--from-day", "18", "--to-day", "18")
    assert code == 0, err and out.count("503") == 2 and world.clock_now >= 30


def test_a_422_stops_the_replay_with_the_apps_own_words(world, gold):
    put_replay(world, gold, [18])

    def post(url, body, timeout):
        return 422, {"error": "invalid_transactions", "detail": "instance 0: missing ['amount']"}

    code, out, err = world.run("replay", "--from-day", "18", "--to-day", "18", post=post)
    assert code == 1 and "HTTP 422" in err and "missing ['amount']" in err and not world.fs.records


def test_replay_needs_the_backfill_to_have_run_and_a_sane_batch(world, gold):
    world.set_app_url("prod")
    code, out, err = world.run("replay", "--from-day", "18", "--to-day", "18")
    assert code == 1 and "has the backfill run" in err
    put_replay(world, gold, [18])
    code, out, err = world.run("replay", "--from-day", "18", "--to-day", "18", "--batch", "501")
    assert code == 1 and "batch must be 1-500" in err


def test_replay_reads_the_app_url_from_ssm_or_says_it_is_missing(world, gold):
    put_replay(world, gold, [18])
    del world.ssm.params["/finrisk/prod/app_url"]
    code, out, err = world.run("replay", "--from-day", "18", "--to-day", "18")
    assert code == 1 and "app_url does not exist" in err
    code, out, err = world.run("replay", "--from-day", "18", "--to-day", "18", "--app-url", "https://x.example")
    assert code == 0 and world.score_app.requests[0]["url"] == "https://x.example/api/score"


# ---------------------------------------------------------------- feedback


def test_feedback_writes_the_true_labels_late(world, gold, tmp_path):
    put_replay(world, gold)
    summary = tmp_path / "s.md"
    code, out, err = world.run("feedback", "--from-day", "18", "--to-day", "22", "--run-id", "run42", "--summary", str(summary))
    assert code == 0, err and last_line(out) == "labels/source=feedback/batch=run42/part.parquet"
    frame = pd.read_parquet(io.BytesIO(world.s3.objects[(BUCKET, "labels/source=feedback/batch=run42/part.parquet")]))
    assert list(frame.columns) == contract.LABEL_COLUMNS and len(frame) == rows_of(gold, range(18, 23))
    assert (frame["source"] == "feedback").all() and (frame["batch_id"] == "run42").all() and frame["row_id"].map(type).eq(str).all()
    last_event = gold[gold["day"] == 22]["event_time"].max()
    assert set(frame["label_time"]) == {contract.label_time_of(last_event, 2)} and frame["is_fraud"].sum() == int(gold[gold["day"].between(18, 22)]["is_fraud"].sum())
    assert "labels" in summary.read_text() and not world.score_app.requests and not world.sm.calls          # feedback touches neither the app nor SageMaker


def test_feedback_waits_for_the_label_delay(world, gold):
    put_replay(world, gold, [18])
    world.run("feedback", "--from-day", "18", "--to-day", "18", "--label-delay-minutes", "5", "--run-id", "r")
    assert world.clock_now >= 300


def test_feedback_defaults_its_batch_to_the_github_run(world, gold):
    put_replay(world, gold, [18])
    code, out, err = world.run("feedback", "--from-day", "18", "--to-day", "18", env={"GITHUB_RUN_ID": "777"})
    assert last_line(out) == "labels/source=feedback/batch=777/part.parquet"


# ---------------------------------------------------------------- the loop: what the next training run sees


def test_after_replay_and_feedback_the_next_dataset_trains_through_day_18_and_tests_on_21_22(world, gold, tmp_path):
    """History + replayed features + feedback labels -> Cutoff=auto: train days 1-18, validation 19-20, test 21-22 (spec 6.4)."""
    put_replay(world, gold)
    world.run("replay", "--from-day", "18", "--to-day", "22")
    world.run("feedback", "--from-day", "18", "--to-day", "22", "--run-id", "r1")
    tx, labels = history_tables(gold)
    athena = FakeAthena(world.s3)
    replayed = pd.DataFrame(list(world.fs.records.values()))
    for column in contract.INTEGRAL_FEATURES:
        replayed[column] = replayed[column].astype(int)
    for column in contract.NUMERIC_FEATURES:
        if column not in contract.INTEGRAL_FEATURES:
            replayed[column] = replayed[column].astype(float)
    replayed["write_time"] = pd.Timestamp("2026-10-16T03:00:00Z")
    replayed["api_invocation_time"] = replayed["write_time"]
    replayed["is_deleted"] = False
    athena.register("transactions", pd.concat([tx, replayed[tx.columns]], ignore_index=True))
    feedback = pd.read_parquet(io.BytesIO(world.s3.objects[(BUCKET, "labels/source=feedback/batch=r1/part.parquet")]))
    athena.register("labels", pd.concat([labels, feedback], ignore_index=True))
    out = tmp_path / "dataset"
    build_dataset.run(["--ml-bucket", BUCKET, "--output", str(out), "--execution-id", "next", "--workdir", str(tmp_path / "w")], athena=athena, s3=world.s3, sleep=lambda s: None)
    report = json.loads((out / "dataset.json").read_text())
    assert report["splits"]["train"]["days"] == [1, 18] and report["splits"]["validation"]["days"] == [19, 20] and report["splits"]["test"]["days"] == [21, 22]
    assert report["cutoff"] == contract.label_time_of(gold[gold["day"] == 22]["event_time"].max(), 2)
    assert report["splits"]["test"]["rows"] == rows_of(gold, [21, 22])
