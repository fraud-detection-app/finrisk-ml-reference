"""The backfill job on a small demo CSV: sha check, gold, Feature Store write, labels, production set, report, the measurement rule."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import pytest
from conftest import needs_app
from fakes import BUCKET, FakeFeatureStore, FakeS3, LaggingAthena

from finrisk_jobs import backfill, common, contract

pytestmark = needs_app


@pytest.fixture(scope="module")
def demo_csv(tmp_path_factory):
    from financial_risk.data.paysim import generate_demo_data

    path = generate_demo_data(12000, tmp_path_factory.mktemp("raw") / "paysim.csv", seed=11)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def aws():
    s3, fs = FakeS3(), FakeFeatureStore()
    common.use_clients(s3=s3, sagemaker_featurestore_runtime=fs)
    return s3, fs


def run_backfill(demo_csv, *extra, **kw):
    path, sha = demo_csv
    return backfill.run(["--ml-bucket", BUCKET, "--raw-file", str(path), "--sha256", sha, "--threads", "4", "--measure-records", "100", "--no-visibility-wait", *extra], **kw)


def test_end_to_end_counts_match_the_data(demo_csv, aws, capsys):
    s3, fs = aws
    assert run_backfill(demo_csv) == 0
    report = s3.json("raw/paysim/v1/backfill_report.json")
    days = {int(d): v for d, v in report["days"].items()}
    history_rows = sum(days[d]["rows"] for d in range(1, 18))
    assert report["history"]["rows"] == history_rows == len(fs.records)                      # every history row, once
    assert report["history"]["feature_store"]["records"] == history_rows and report["history"]["feature_store"]["write_units"] >= history_rows
    assert report["history"]["days"] == "1-17" and report["history"]["frauds"] == sum(days[d]["frauds"] for d in range(1, 18))
    assert report["calendar"]["excluded_days"] == [31]
    out = capsys.readouterr().out
    assert "day 31 EXCLUDED" in out and "measuring first" in out and "projection" in out and "Feature Store:" in out


def test_feature_store_records_carry_the_contract_columns_and_calendar(demo_csv, aws):
    s3, fs = aws
    run_backfill(demo_csv)
    record = next(iter(fs.records.values()))
    assert list(record) == contract.FEATURE_GROUP_COLUMNS
    assert record["event_time"].endswith("Z") and len(record["event_time"]) == 20
    day = contract.day_of_step(int(record["step"]))
    assert 1 <= day <= 17 and record["event_time"] == contract.event_time_of_step(int(record["step"]))
    assert record["hour_of_day"] == str(contract.hour_of_step(int(record["step"])))
    assert record["transaction_type"] in {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}


def test_history_labels_have_the_analyst_delay_and_the_table_columns(demo_csv, aws):
    s3, fs = aws
    run_backfill(demo_csv)
    keys = s3.keys("labels/source=history/")
    assert len(keys) == 17 and keys[0].endswith("part-day01.parquet")
    frame = pd.read_parquet(io.BytesIO(s3.objects[(BUCKET, keys[0])]))
    assert list(frame.columns) == contract.LABEL_COLUMNS
    assert (frame["source"] == "history").all()
    first = frame.iloc[0]
    assert first["label_time"] == contract.label_time_of(first["event_time"], 2)
    assert sum(len(pd.read_parquet(io.BytesIO(s3.objects[(BUCKET, k)]))) for k in keys) == len(fs.records)


def test_production_set_is_days_18_to_30_with_labels_and_no_day_31(demo_csv, aws):
    s3, fs = aws
    run_backfill(demo_csv)
    keys = s3.keys("replay/v1/")
    assert sorted(k.split("/")[2] for k in keys) == sorted(f"day={d}" for d in range(18, 31))
    assert not any("day=31" in k for k in keys) and not any("day=17" in k for k in keys)
    frame = pd.read_parquet(io.BytesIO(s3.objects[(BUCKET, "replay/v1/day=18/part.parquet")]))
    assert {"row_id", "is_fraud", *contract.FEATURE_COLUMNS} <= set(frame.columns)
    assert frame["row_id"].map(type).eq(str).all() and set(frame["is_fraud"]) <= {0, 1}
    assert (frame["step"].map(contract.day_of_step) == 18).all()
    assert list(frame["step"]) == sorted(frame["step"])                                         # event-time order
    report = s3.json("raw/paysim/v1/backfill_report.json")
    assert report["production"]["days"] == list(range(18, 31))


def test_raw_is_uploaded_once_with_its_sha(demo_csv, aws):
    s3, _ = aws
    run_backfill(demo_csv)
    assert (BUCKET, "raw/paysim/v1/paysim.csv") in s3.objects
    assert s3.text("raw/paysim/v1/paysim.csv.sha256").split()[0] == demo_csv[1]


def test_a_wrong_sha_refuses_before_anything_is_written(demo_csv, aws):
    s3, fs = aws
    path, _ = demo_csv
    with pytest.raises(common.JobError, match="sha256 mismatch"):
        backfill.run(["--ml-bucket", BUCKET, "--raw-file", str(path), "--sha256", "0" * 64])
    assert not fs.records and not s3.objects


def test_download_from_the_course_url_is_verified_and_uploaded(demo_csv, aws, tmp_path):
    s3, fs = aws
    path, sha = demo_csv
    backfill.run(["--ml-bucket", BUCKET, "--sha256", sha, "--dataset-url", "https://example.org/paysim.csv", "--threads", "2", "--skip-feature-store",
                  "--workdir", str(tmp_path / "w")], opener=_Opener(path))
    assert s3.text("raw/paysim/v1/paysim.csv.sha256").split()[0] == sha


class _Opener:
    """urllib.request.urlopen, replaced: hands back the bytes of a local file as a context manager."""

    def __init__(self, path):
        self.path = path

    def __call__(self, url, timeout=0):
        return _Ctx(io.BytesIO(self.path.read_bytes()))


class _Ctx:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self.data

    def __exit__(self, *a):
        return False


def test_a_download_with_the_wrong_content_is_refused(demo_csv, aws, tmp_path):
    s3, fs = aws
    bad = tmp_path / "other.csv"
    bad.write_text("step,type\n1,PAYMENT\n")
    with pytest.raises(common.JobError, match="sha256 mismatch"):
        backfill.run(["--ml-bucket", BUCKET, "--sha256", demo_csv[1], "--dataset-url", "https://example.org/paysim.csv", "--workdir", str(tmp_path / "w")],
                     opener=_Opener(bad))
    assert (BUCKET, "raw/paysim/v1/paysim.csv") not in s3.objects                     # nothing wrong ever reaches the bucket


def test_second_run_takes_the_raw_data_from_s3_not_the_url(demo_csv, aws, tmp_path, capsys):
    s3, fs = aws
    path, sha = demo_csv
    run_backfill(demo_csv, "--skip-feature-store")
    capsys.readouterr()

    def opener(*a, **k):
        raise AssertionError("must not download again")

    backfill.run(["--ml-bucket", BUCKET, "--sha256", sha, "--skip-feature-store", "--workdir", str(tmp_path / "w2")], opener=opener)
    assert "already in s3" in capsys.readouterr().out


def test_days_shrinks_the_history_and_the_report_says_so(demo_csv, aws):
    s3, fs = aws
    run_backfill(demo_csv, "--days", "1-3", "--production", "no")
    report = s3.json("raw/paysim/v1/backfill_report.json")
    assert report["history"]["days"] == "1-3" and len(fs.records) == report["history"]["rows"]
    assert s3.keys("replay/") == [] and report["production"]["days"] == []
    assert len(s3.keys("labels/source=history/")) == 3


def test_row_ids_are_the_csv_positions_and_stable_across_runs(demo_csv, aws):
    s3, fs = aws
    path, _ = demo_csv
    run_backfill(demo_csv, "--days", "1-17", "--production", "no")
    raw = pd.read_csv(path)
    ids = sorted(int(i) for i in fs.records)
    sample = fs.records[str(ids[len(ids) // 2])]
    row = raw.iloc[int(sample["row_id"])]
    assert int(row["step"]) == int(sample["step"]) and float(row["amount"]) == pytest.approx(float(sample["amount"]))
    first = dict(fs.records)
    fs.records.clear()
    run_backfill(demo_csv, "--days", "1-17", "--production", "no")
    assert fs.records == first


def test_measurement_first_stops_when_the_projection_is_over_the_limit(demo_csv, aws):
    s3, fs = aws
    ticks = iter(range(0, 10 ** 7, 40))                    # every clock() call is 40 s later: the write looks very slow
    with pytest.raises(common.JobError) as info:
        run_backfill(demo_csv, "--max-minutes", "1", clock=lambda: next(ticks))
    assert info.value.code == backfill.EXIT_TOO_SLOW == 4
    assert "STOPPED" in str(info.value) and "--instance-count" in str(info.value) and "safe to write again" in str(info.value)
    assert 0 < len(fs.records) < 5000                      # only the measurement batch was written
    assert not s3.keys("labels/") and (BUCKET, "raw/paysim/v1/backfill_report.json") not in s3.objects


def test_the_main_entry_point_turns_the_too_slow_stop_into_exit_code_4(demo_csv, aws, capsys, monkeypatch):
    from finrisk_jobs import __main__ as entry

    path, sha = demo_csv
    monkeypatch.setattr(backfill, "write_history", lambda *a, **k: (_ for _ in ()).throw(common.JobError("STOPPED: too slow", code=4)))
    code = entry.main(["backfill", "--ml-bucket", BUCKET, "--raw-file", str(path), "--sha256", sha])
    assert code == 4 and "STOPPED" in capsys.readouterr().out


def test_local_only_touches_no_aws(demo_csv, capsys, tmp_path):
    path, sha = demo_csv
    assert backfill.run(["--local-only", "--raw-file", str(path), "--sha256", sha, "--workdir", str(tmp_path / "w")]) == 0
    out = capsys.readouterr().out
    assert "SKIPPED" in out and "peak_memory_mb" in out


def test_a_gold_table_without_hour_of_day_is_refused_with_the_reason(demo_csv, tmp_path, monkeypatch):
    import duckdb

    work = tmp_path / "g"
    work.mkdir()
    duckdb.connect().execute(f"COPY (SELECT 1 AS row_id, 1 AS step, 0 AS is_fraud) TO '{work / 'g.parquet'}' (FORMAT PARQUET)")
    con = backfill.open_gold(work / "g.parquet", "2026-09-01T00:00:00Z", 1, "1GB", work / "t")
    with pytest.raises(common.JobError, match="hour_of_day"):
        backfill.check_gold_columns(con)


# ---------------------------------------------------------------- the offline store fills in the background: wait until Athena sees it


def run_with_athena(demo_csv, athena, *extra, sleep=None, clock=None):
    path, sha = demo_csv
    ticks = {"t": 0.0}
    waits = []

    def fake_sleep(seconds):
        ticks["t"] += seconds
        waits.append(seconds)

    def fake_clock():
        ticks["t"] += 0.001                     # every look at the clock costs a millisecond, so a write is never 'instant'
        return ticks["t"]

    common.use_clients(athena=athena)
    code = backfill.run(["--ml-bucket", BUCKET, "--raw-file", str(path), "--sha256", sha, "--threads", "4", "--measure-records", "100", *extra],
                        sleep=fake_sleep, clock=fake_clock)
    return code, waits


def test_the_job_waits_until_athena_sees_every_record_then_writes_the_report(demo_csv, aws, capsys):
    s3, fs = aws
    athena = LaggingAthena(s3, fs, schedule=(0.2, 0.6, 1.0))               # 20 %, 60 %, then everything is visible
    code, waits = run_with_athena(demo_csv, athena, "--days", "1-17")
    assert code == 0 and athena.polls == 3 and waits.count(60.0) == 2        # polled every 60 s
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if "visible" in l and " of " in l]
    assert len(lines) == 3 and f"of {len(fs.records):,}" in lines[0] and lines[-1].split("visible")[1].split(" of ")[0].strip().replace(",", "") == str(len(fs.records))
    report = s3.json("raw/paysim/v1/backfill_report.json")
    assert report["visibility"]["expected"] == len(fs.records) == report["visibility"]["visible"] and report["visibility"]["waited_seconds"] >= 120
    assert out.index("visible") < out.index("report written")                  # the report comes LAST


def test_a_store_that_never_shows_up_fails_naming_the_delay_and_writes_no_report(demo_csv, aws):
    s3, fs = aws
    athena = LaggingAthena(s3, fs, schedule=(0.5,))                           # half of it, forever
    with pytest.raises(common.JobError) as info:
        run_with_athena(demo_csv, athena, "--days", "1-17")
    message = str(info.value)
    assert "still sees only" in message and "~15 minutes" in message and "run the backfill again" in message
    assert (BUCKET, "raw/paysim/v1/backfill_report.json") not in s3.objects    # no report = the pipelines will not start
    assert 25 <= athena.polls <= 32                                           # polled every 60 s for the 30-minute budget
    assert s3.keys("labels/source=history/")                                  # what was safe to write stays


def test_the_wait_counts_only_the_instances_own_days(demo_csv, aws):
    s3, fs = aws
    athena = LaggingAthena(s3, fs, schedule=(1.0,))
    run_with_athena(demo_csv, athena, "--days", "1-5", "--production", "no")
    sql = [q for q in athena.queries if "count(DISTINCT row_id)" in q][0]
    assert "event_time >= '2026-09-01T00:00:00Z'" in sql and "event_time <= '2026-09-05T23:00:00Z'" in sql


def test_parallel_instances_each_write_their_report_and_the_last_one_writes_the_merged_one(demo_csv, aws):
    s3, fs = aws
    one = LaggingAthena(s3, fs, schedule=(1.0,))
    run_with_athena(demo_csv, one, "--days", "1-8", "--instance-index", "1", "--instance-count", "2", "--production", "yes")
    assert (BUCKET, "raw/paysim/v1/backfill_report.1.json") in s3.objects and (BUCKET, "raw/paysim/v1/backfill_report.json") not in s3.objects
    two = LaggingAthena(s3, fs, schedule=(1.0,))
    run_with_athena(demo_csv, two, "--days", "9-17", "--instance-index", "2", "--instance-count", "2", "--production", "no")
    merged = s3.json("raw/paysim/v1/backfill_report.json")
    first, second = s3.json("raw/paysim/v1/backfill_report.1.json"), s3.json("raw/paysim/v1/backfill_report.2.json")
    assert merged["merged_from"] == 2 and merged["history"]["days"] == "1-17" and len(fs.records) == merged["history"]["rows"]
    assert merged["history"]["rows"] == first["history"]["rows"] + second["history"]["rows"]
    assert merged["history"]["feature_store"]["records"] == len(fs.records) and merged["visibility"]["visible_all"] is True
    assert merged["production"]["days"] == list(range(18, 31))                 # the instance that wrote the production set


def test_skipping_the_feature_store_skips_the_wait(demo_csv, aws, capsys):
    s3, fs = aws
    run_backfill(demo_csv, "--skip-feature-store")
    assert "waiting for Athena: skipped" in capsys.readouterr().out and s3.json("raw/paysim/v1/backfill_report.json")["visibility"] == {"skipped": True}
