"""The SQL, written once. Athena (engine v3 = Trino) syntax; workgroup `finrisk`, database `finrisk_ml`.

The point-in-time query is the heart of the training set. The Feature Store's offline table is APPEND-ONLY (every write
is a new row, with `write_time`, `api_invocation_time` and `is_deleted`), and labels arrive LATE (an analyst confirms a
fraud two days after it happened). So "the data as of a moment" needs two rules:

  features  keep the latest write of each record, and only transactions that had happened before the cutoff;
  labels    keep only labels that were KNOWN at the cutoff (label_time <= cutoff).

A transaction whose label arrives after the cutoff is not in the training set at all (not as a 0: that would teach the
model that frauds look legitimate). `tests/test_queries.py` runs this exact text on DuckDB, with small tables, and
checks all three behaviours.

The cutoff is a UTC ISO time `YYYY-MM-DDTHH:MM:SSZ`. It is validated and written into the text (UNLOAD cannot take
query parameters); nothing else is ever concatenated into SQL.
"""

from __future__ import annotations

import re
from typing import Sequence

CUTOFF = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
DATABASE = "finrisk_ml"
FEATURES = f"{DATABASE}.transactions"
LABELS = f"{DATABASE}.labels"

# The model inputs, then what the splits and the monitor need. `step` and `event_time` are metadata, not features.
DEFAULT_COLUMNS = ["row_id", "event_time", "step", "transaction_type", "amount", "log_amount", "origin_old_balance",
                   "origin_new_balance", "destination_old_balance", "destination_new_balance", "origin_balance_delta",
                   "destination_balance_delta", "amount_to_origin_balance", "hour_of_day", "tx_index_for_origin",
                   "customer_tx_count_so_far", "customer_amount_so_far"]


def check_cutoff(cutoff: str) -> str:
    if not CUTOFF.match(cutoff or ""):
        raise ValueError(f"cutoff must look like 2026-09-19T23:00:00Z (UTC), got {cutoff!r}")
    return cutoff


def latest_label_time() -> str:
    """The cutoff for Cutoff=auto: every label that has arrived so far."""
    return f"SELECT max(label_time) AS cutoff FROM {LABELS}"


def point_in_time(cutoff: str, columns: Sequence[str] = DEFAULT_COLUMNS, limit: int | None = None) -> str:
    """The training rows as known at ``cutoff``: features joined to the labels that had arrived by then.

    ``limit`` is for looking (lab 8: the query on 1 000 rows); the pipeline never uses it."""
    check_cutoff(cutoff)
    select = ", ".join(f"f.{c}" for c in columns)
    return f"""WITH f AS (
  SELECT *, row_number() OVER (PARTITION BY row_id ORDER BY write_time DESC, api_invocation_time DESC) AS rn
  FROM {FEATURES}
  WHERE NOT is_deleted AND from_iso8601_timestamp(event_time) < from_iso8601_timestamp('{cutoff}')),
l AS (
  SELECT row_id, max(is_fraud) AS is_fraud FROM {LABELS}
  WHERE from_iso8601_timestamp(label_time) <= from_iso8601_timestamp('{cutoff}') GROUP BY row_id)
SELECT {select}, l.is_fraud
FROM f JOIN l ON f.row_id = l.row_id WHERE f.rn = 1""" + (f"\nLIMIT {int(limit)}" if limit else "")


def unload(select_sql: str, destination_uri: str) -> str:
    """Write a query's result as Parquet files under an (empty) S3 prefix: the fast way to hand millions of rows on."""
    if not destination_uri.startswith("s3://") or "'" in destination_uri:
        raise ValueError(f"bad destination {destination_uri!r}")
    return f"UNLOAD ({select_sql})\nTO '{destination_uri.rstrip('/')}/'\nWITH (format = 'PARQUET', compression = 'SNAPPY')"


def labels_for_rows(first_row_id: int, last_row_id: int) -> str:
    """The delayed ground truth the monitor joins to captured requests: feedback labels, one answer per row."""
    lo, hi = int(first_row_id), int(last_row_id)
    return (f"SELECT row_id, max(is_fraud) AS is_fraud, max(label_time) AS label_time FROM {LABELS}\n"
            f"WHERE source = 'feedback' AND try_cast(row_id AS bigint) BETWEEN {lo} AND {hi}\nGROUP BY row_id")


def rows_per_day() -> str:
    """Lab 8: rows and frauds per calendar day in the feature group (the offline store)."""
    return (f"SELECT substr(event_time, 1, 10) AS day, count(DISTINCT row_id) AS records\nFROM {FEATURES}\n"
            f"WHERE NOT is_deleted GROUP BY 1 ORDER BY 1")


def visible_records(first_event_time: str, last_event_time: str) -> str:
    """How many distinct records of this event-time range Athena can see in the offline store right now.

    The Feature Store writes the offline store to S3 in the background: a record is queryable some minutes (up to ~15) after
    BatchWriteRecord returned. The backfill polls this until it equals what it wrote."""
    check_cutoff(first_event_time), check_cutoff(last_event_time)
    return (f"SELECT count(DISTINCT row_id) AS visible FROM {FEATURES}\n"
            f"WHERE NOT is_deleted AND event_time >= '{first_event_time}' AND event_time <= '{last_event_time}'")


def labelled_without_features(cutoff: str) -> str:
    """Completeness of the offline store as of the cutoff: labels known by then that have NO feature row (yet).

    build-dataset refuses to train while this is not (almost) zero: a point-in-time join would silently drop those rows, and a
    model trained on half the store looks fine and is not."""
    check_cutoff(cutoff)
    return f"""SELECT count(*) AS labelled, count_if(f.row_id IS NULL) AS missing
FROM (SELECT DISTINCT row_id FROM {LABELS}
      WHERE from_iso8601_timestamp(label_time) <= from_iso8601_timestamp('{cutoff}')) l
LEFT JOIN (SELECT DISTINCT row_id FROM {FEATURES} WHERE NOT is_deleted) f ON l.row_id = f.row_id"""
