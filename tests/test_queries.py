"""The point-in-time query, run for real (DuckDB) on tables shaped like the Feature Store's offline table and the labels table."""

from __future__ import annotations

import pandas as pd
import pytest
from fakes import FakeAthena, FakeS3

import queries
from finrisk_jobs import common


def tx(row_id, event_time, write_time, amount=1.0, deleted=False, step=1):
    return {"row_id": row_id, "event_time": event_time, "step": step, "transaction_type": "PAYMENT", "amount": amount, "log_amount": 0.1,
            "origin_old_balance": 1.0, "origin_new_balance": 1.0, "destination_old_balance": 0.0, "destination_new_balance": 0.0,
            "origin_balance_delta": 0.0, "destination_balance_delta": 0.0, "amount_to_origin_balance": 0.5, "hour_of_day": 1, "tx_index_for_origin": 1,
            "customer_tx_count_so_far": 1, "customer_amount_so_far": amount, "write_time": pd.Timestamp(write_time),
            "api_invocation_time": pd.Timestamp(write_time), "is_deleted": deleted}


def label(row_id, is_fraud, label_time, source="history"):
    return {"row_id": row_id, "is_fraud": is_fraud, "event_time": "x", "label_time": label_time, "source": source, "batch_id": "b"}


@pytest.fixture
def athena():
    a = FakeAthena(FakeS3())
    a.register("transactions", pd.DataFrame([
        tx("1", "2026-09-01T00:00:00Z", "2026-10-01T00:00:00Z", amount=10.0),
        tx("1", "2026-09-01T00:00:00Z", "2026-10-02T00:00:00Z", amount=11.0),      # a later write of the same record: this one wins
        tx("2", "2026-09-02T00:00:00Z", "2026-10-01T00:00:00Z"),
        tx("3", "2026-09-03T00:00:00Z", "2026-10-01T00:00:00Z"),                   # its label arrives late
        tx("4", "2026-09-10T00:00:00Z", "2026-10-01T00:00:00Z"),                   # happens after the early cutoff
        tx("5", "2026-09-02T00:00:00Z", "2026-10-01T00:00:00Z", deleted=True),     # deleted
        tx("6", "2026-09-02T00:00:00Z", "2026-10-01T00:00:00Z"),                   # never labelled
    ]))
    a.register("labels", pd.DataFrame([
        label("1", 0, "2026-09-03T00:00:00Z"), label("2", 1, "2026-09-04T00:00:00Z"), label("2", 0, "2026-09-04T01:00:00Z"),
        label("3", 1, "2026-09-12T00:00:00Z"), label("4", 1, "2026-09-12T00:00:00Z"), label("5", 1, "2026-09-04T00:00:00Z"),
    ]))
    return a


def rows(athena, cutoff):
    return athena.con.execute(queries.point_in_time(cutoff)).fetchdf().sort_values("row_id").reset_index(drop=True)


def test_it_keeps_the_latest_write_of_each_record(athena):
    df = rows(athena, "2026-09-05T00:00:00Z")
    assert df.loc[df.row_id == "1", "amount"].iloc[0] == 11.0 and list(df.row_id).count("1") == 1


def test_a_label_that_arrives_after_the_cutoff_is_not_known_yet(athena):
    assert "3" not in set(rows(athena, "2026-09-05T00:00:00Z").row_id)           # its label_time (09-12) is after the cutoff
    late = rows(athena, "2026-09-13T00:00:00Z")
    assert "3" in set(late.row_id) and late.loc[late.row_id == "3", "is_fraud"].iloc[0] == 1


def test_transactions_after_the_cutoff_deleted_and_unlabelled_ones_are_excluded(athena):
    assert set(rows(athena, "2026-09-05T00:00:00Z").row_id) == {"1", "2"}         # 4: after; 5: deleted; 6: no label; 3: label too late
    assert set(rows(athena, "2026-09-13T00:00:00Z").row_id) == {"1", "2", "3", "4"}


def test_two_labels_for_one_row_collapse_to_the_max(athena):
    df = rows(athena, "2026-09-05T00:00:00Z")
    assert df.loc[df.row_id == "2", "is_fraud"].iloc[0] == 1 and list(df.row_id).count("2") == 1


def test_cutoff_is_validated_before_it_reaches_sql():
    for bad in ("2026-09-05", "auto", "2026-09-05T00:00:00Z'; DROP TABLE x; --", ""):
        with pytest.raises(ValueError, match="cutoff must look like"):
            queries.point_in_time(bad)


def test_unload_writes_parquet_files_under_the_destination(athena):
    execution = common.run_athena(queries.unload(queries.point_in_time("2026-09-05T00:00:00Z"), "s3://b/athena/unload/e1"), athena=athena, show=False)
    assert execution["Status"]["State"] == "SUCCEEDED"
    written = [k for (b, k) in athena.s3.objects if b == "b" and k.startswith("athena/unload/e1/")]
    assert len(written) == 1 and athena.s3.objects[("b", written[0])][:4] == b"PAR1"


def test_unload_refuses_odd_destinations():
    with pytest.raises(ValueError):
        queries.unload("SELECT 1", "/tmp/x")
    with pytest.raises(ValueError):
        queries.unload("SELECT 1", "s3://b/x'; --")


def test_latest_label_time_and_rows_per_day(athena):
    assert athena.con.execute(queries.latest_label_time()).fetchone()[0] == "2026-09-12T00:00:00Z"
    days = athena.con.execute(queries.rows_per_day()).fetchall()
    assert ("2026-09-01", 1) in days and ("2026-09-02", 2) in days            # records 2 and 6 on 09-02; 5 is deleted


def test_labels_for_rows_reads_feedback_only(athena):
    athena.register("labels", pd.DataFrame([label("100", 1, "2026-09-25T00:00:00Z", "feedback"), label("101", 0, "2026-09-25T00:00:00Z", "feedback"),
                                            label("7", 1, "2026-09-03T00:00:00Z", "history")]))
    got = athena.con.execute(queries.labels_for_rows(1, 200)).fetchall()
    assert sorted(r[0] for r in got) == ["100", "101"]


def test_athena_failure_is_one_readable_error(athena):
    athena.fail_with = "SYNTAX_ERROR: line 1:1 mismatched input"
    with pytest.raises(common.AthenaError, match="SYNTAX_ERROR"):
        common.run_athena("SELECT 1", athena=athena, show=False)


def test_visible_records_counts_distinct_rows_in_the_event_time_range(athena):
    sql = queries.visible_records("2026-09-01T00:00:00Z", "2026-09-02T23:00:00Z")
    assert athena.con.execute(sql).fetchone()[0] == 3                              # records 1, 2 and 6 (5 is deleted)
    assert athena.con.execute(queries.visible_records("2026-09-03T00:00:00Z", "2026-09-30T00:00:00Z")).fetchone()[0] == 2     # 3 and 4
    with pytest.raises(ValueError, match="cutoff must look like"):
        queries.visible_records("2026-09-01", "2026-09-02T23:00:00Z")


def test_labelled_without_features_counts_labels_that_have_no_feature_row(athena):
    athena.register("labels", pd.DataFrame([label("1", 0, "2026-09-03T00:00:00Z"), label("2", 1, "2026-09-04T00:00:00Z"), label("99", 1, "2026-09-04T00:00:00Z"),
                                            label("98", 0, "2026-09-30T00:00:00Z")]))
    labelled, missing = athena.con.execute(queries.labelled_without_features("2026-09-05T00:00:00Z")).fetchone()
    assert (labelled, missing) == (3, 1)                                           # 99 has no features; 98's label is not known yet
    assert athena.con.execute(queries.labelled_without_features("2026-10-05T00:00:00Z")).fetchone() == (4, 2)
    with pytest.raises(ValueError):
        queries.labelled_without_features("later")
