"""backfill: the one-off (per account) load of the FULL PaySim data. Runs as a SageMaker Processing job.

    1. raw      download the CSV from the course URL, check its sha256 (refuse on a mismatch), upload it ONCE to
                s3://<ml>/raw/paysim/v1/ (next to paysim.csv.sha256). A second run finds it there and skips the download.
    2. gold     DuckDB over ALL days: silver (clean) -> the app's `build_feature_tables` (the same SQL the service uses),
                so the "so far" window features see the whole history. They only look backwards: no leakage.
    3. history  days 1-17 -> the Feature Store `finrisk-transactions` with BatchWriteRecord (offline store only), many
                threads, progress, throughput, write units. MEASUREMENT FIRST: write 100 000 records, time them, project
                the total; above `max_minutes` the job STOPS (exit 4) and tells you how to split by day.
    4. labels   the history labels -> s3://<ml>/labels/source=history/ (label_time = event_time + 2 days: the analyst delay)
    5. replay   days 18-30 (features AND is_fraud) -> s3://<ml>/replay/v1/day=<n>/part.parquet. Only replay.yml reads these.
    6. visible  WAIT until Athena can see what was written. The Feature Store fills its offline store in the background; a record is
                queryable some minutes (up to ~15) after BatchWriteRecord returned. The job polls count(DISTINCT row_id) in
                finrisk_ml.transactions every 60 s (budget 30 min) until it is >= the records written, printing "visible X of Y".
    7. report   s3://<ml>/raw/paysim/v1/backfill_report.json: rows and frauds per day, records written, seconds, write units.
                It is written LAST, after step 6: its existence means "the data is complete and queryable", and `mlops.py run` refuses to
                start the training pipeline without it. (With --instance-count N each instance writes backfill_report.<k>.json; the LAST one
                to finish writes the merged backfill_report.json.)

`row_id` is the row's position in the CSV (0-based): the same row has the same id in every run and on every instance, so a
backfill split by day across instances cannot collide. Day 31 (272 rows, all fraud: an artefact) is excluded and logged.
`history_days` in config/data.yaml is the ONLY way to shrink the history; the report prints what was written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from . import contract
from .common import (JobError, add_app_package_to_path, aws, check_feature_contract, data_config, ml_bucket, s3_exists,
                     s3_download, s3_get_bytes, s3_get_json, s3_put_bytes, s3_put_json, s3_upload, s3_uri, say, workdir)
from .featurestore import WriteStats, write_records

EXIT_TOO_SLOW = 4
RAW_COLUMNS = {"step": "INTEGER", "type": "VARCHAR", "amount": "DOUBLE", "nameOrig": "VARCHAR", "oldbalanceOrg": "DOUBLE",
               "newbalanceOrig": "DOUBLE", "nameDest": "VARCHAR", "oldbalanceDest": "DOUBLE", "newbalanceDest": "DOUBLE",
               "isFraud": "INTEGER", "isFlaggedFraud": "INTEGER"}
TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="finrisk_jobs backfill", description=__doc__.split("\n\n")[0])
    p.add_argument("--ml-bucket", help="the ML bucket (env ML_BUCKET)")
    p.add_argument("--days", help="history days to write to the Feature Store, e.g. 1-17 (default: history_days in config/data.yaml)")
    p.add_argument("--production", choices=["yes", "no"], default="yes",
                   help="also write the production set (days 18-30) to s3://<ml>/replay/ (yes on ONE instance, no on the others)")
    p.add_argument("--instance-index", type=int, default=1, help="which of the parallel jobs this is (1-based)")
    p.add_argument("--instance-count", type=int, default=1, help="how many parallel jobs there are; the last to finish writes the merged report")
    p.add_argument("--no-visibility-wait", action="store_true", help="do not wait for Athena to see the records (tests, laptop)")
    p.add_argument("--raw-file", help="use this local CSV instead of downloading (laptop runs, tests); its sha256 is still checked")
    p.add_argument("--dataset-url", help="override the course URL (env DATASET_URL)")
    p.add_argument("--sha256", help="override the expected sha256 (tests only)")
    p.add_argument("--workdir", help="working folder (default: a new temp folder)")
    p.add_argument("--threads", type=int, help="BatchWriteRecord threads (default from config/data.yaml)")
    p.add_argument("--max-minutes", type=float, help="projected write time above this stops the job (default from config)")
    p.add_argument("--measure-records", type=int, help="records written before the projection (default 100000)")
    p.add_argument("--skip-feature-store", action="store_true", help="build and upload everything except the Feature Store write")
    p.add_argument("--local-only", action="store_true", help="DuckDB part only: no S3, no Feature Store (timing runs on a laptop)")
    p.add_argument("--keep-workdir", action="store_true")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------- 1. raw

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_url(url: str, target: Path, opener: Callable[..., Any] = urllib.request.urlopen) -> str:
    """Stream a URL to a file; return the sha256 of what arrived."""
    if not url.startswith(("http://", "https://")):
        raise JobError(f"not an http(s) address: {url!r}")
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    got = 0
    with opener(url, timeout=120) as response, target.open("wb") as out:        # noqa: S310 - scheme checked above
        for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
            out.write(block)
            digest.update(block)
            got += len(block)
    say(f"  downloaded {got / 1e6:,.1f} MB")
    return digest.hexdigest()


def ensure_raw(bucket: str | None, prefix: str, expected_sha: str, url: str, work: Path, raw_file: str | None = None,
               s3: Any = None, opener: Callable[..., Any] = urllib.request.urlopen) -> tuple[Path, dict]:
    """Return a local path to the verified CSV (and facts about it). Uploads it to S3 the first time only."""
    csv_path = work / "raw" / "paysim.csv"
    info: dict[str, Any] = {"url": url, "sha256": expected_sha}
    if raw_file:
        csv_path = Path(raw_file)
        actual = sha256_file(csv_path)
        info["source"] = "local file"
    elif bucket and s3 is not None and _raw_in_s3(bucket, prefix, expected_sha, s3):
        say(f"  raw data already in s3://{bucket}/{prefix}/ with the expected sha256: not downloading from the course URL")
        s3_download(s3_uri(bucket, f"{prefix}/paysim.csv"), csv_path, s3)
        actual = sha256_file(csv_path)
        info["source"] = "s3"
    else:
        say(f"  downloading {url}")
        actual = download_url(url, csv_path, opener)
        info["source"] = "course url"
    if actual != expected_sha:
        csv_path.unlink(missing_ok=True) if not raw_file else None
        raise JobError(f"sha256 mismatch: the file has {actual} but config/data.yaml expects {expected_sha}. "
                       "Refusing to continue: this is not the dataset the course was built on.")
    say(f"  sha256 ok: {actual[:16]}...  ({csv_path.stat().st_size / 1e6:,.1f} MB)")
    info["bytes"] = csv_path.stat().st_size
    if bucket and s3 is not None and info["source"] != "s3":
        s3_upload(csv_path, s3_uri(bucket, f"{prefix}/paysim.csv"), s3)
        s3_put_bytes(s3_uri(bucket, f"{prefix}/paysim.csv.sha256"), f"{actual}  paysim.csv\n".encode(), s3, "text/plain")
        say(f"  uploaded to s3://{bucket}/{prefix}/paysim.csv (+ .sha256)")
    return csv_path, info


def _raw_in_s3(bucket: str, prefix: str, expected: str, s3: Any) -> bool:
    sha_uri = s3_uri(bucket, f"{prefix}/paysim.csv.sha256")
    if not (s3_exists(sha_uri, s3) and s3_exists(s3_uri(bucket, f"{prefix}/paysim.csv"), s3)):
        return False
    return s3_get_bytes(sha_uri, s3).decode().split()[0].strip() == expected


# ---------------------------------------------------------------------------- 2. silver + gold (DuckDB)

def sql_literal(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def build_gold(csv_path: Path, work: Path, threads: int, memory_limit: str, calendar_start: str) -> tuple[Path, dict]:
    """CSV -> silver -> gold (the app's feature SQL). Returns the gold parquet and timings."""
    import duckdb

    add_app_package_to_path()
    from financial_risk.features.sql_features import build_feature_tables

    silver = work / "silver" / "clean_transactions.parquet"
    silver.parent.mkdir(parents=True, exist_ok=True)
    gold_dir = work / "gold"
    spill = work / "duckdb_tmp"
    spill.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    def timed(name: str, started: float) -> None:
        timings[name] = round(time.monotonic() - started, 1)
        say(f"  {name}: {timings[name]:.1f} s")

    con = duckdb.connect()
    con.execute(f"SET threads = {int(threads)}")
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute(f"SET temp_directory = {sql_literal(spill)}")
    con.execute("SET preserve_insertion_order = true")                       # row_id = the row's position in the file
    columns = ", ".join(f"'{k}': '{v}'" for k, v in RAW_COLUMNS.items())
    types = ", ".join(f"'{t}'" for t in TRANSACTION_TYPES)
    t0 = time.monotonic()
    con.execute(f"""
        COPY (
            SELECT row_id, step, transaction_type, amount, origin_customer, origin_old_balance, origin_new_balance,
                   destination_customer, destination_old_balance, destination_new_balance, is_fraud, is_flagged_fraud,
                   origin_old_balance - origin_new_balance AS origin_balance_delta,
                   destination_new_balance - destination_old_balance AS destination_balance_delta,
                   least(greatest(coalesce(amount / nullif(origin_old_balance, 0), 0), 0), 100) AS amount_to_origin_balance
            FROM (
                SELECT row_number() OVER () - 1 AS row_id, step, upper(trim("type")) AS transaction_type, amount,
                       nameOrig AS origin_customer, oldbalanceOrg AS origin_old_balance, newbalanceOrig AS origin_new_balance,
                       nameDest AS destination_customer, oldbalanceDest AS destination_old_balance,
                       newbalanceDest AS destination_new_balance, isFraud AS is_fraud, isFlaggedFraud AS is_flagged_fraud
                FROM read_csv({sql_literal(csv_path)}, header = true, columns = {{{columns}}})
            )
            WHERE amount >= 0 AND step IS NOT NULL AND amount IS NOT NULL AND is_fraud IS NOT NULL
              AND transaction_type IN ({types})
        ) TO {sql_literal(silver)} (FORMAT PARQUET)
    """)
    con.close()
    timed("silver (csv -> clean parquet)", t0)
    t1 = time.monotonic()
    gold, _customers = build_feature_tables(silver, gold_dir)
    timed("gold (the app's feature SQL)", t1)
    return gold, timings


def open_gold(gold: Path, calendar_start: str, threads: int, memory_limit: str, spill: Path, label_delay_days: int = 2):
    """A DuckDB connection with a view `g` = gold + day + event_time (and row_id as text)."""
    import duckdb

    con = duckdb.connect()
    con.execute(f"SET threads = {int(threads)}")
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute(f"SET temp_directory = {sql_literal(spill)}")
    start = datetime.strptime(calendar_start, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M:%S")
    con.execute(f"""
        CREATE VIEW g AS
        SELECT *, ((step - 1) // 24) + 1 AS day,
               strftime(TIMESTAMP '{start}' + to_hours(CAST(step - 1 AS BIGINT)), '%Y-%m-%dT%H:%M:%SZ') AS event_time,
               strftime(TIMESTAMP '{start}' + to_hours(CAST(step - 1 AS BIGINT)) + INTERVAL {int(label_delay_days)} DAY,
                        '%Y-%m-%dT%H:%M:%SZ') AS label_time,
               CAST(row_id AS VARCHAR) AS row_id_text
        FROM read_parquet({sql_literal(gold)})
    """)
    return con


def check_gold_columns(con) -> None:
    have = {r[0] for r in con.execute("DESCRIBE g").fetchall()}
    need = set(contract.FEATURE_GROUP_COLUMNS) | {contract.TARGET}
    missing = sorted(need - have - {"row_id", "event_time"})
    if missing:
        raise JobError(f"the gold table lacks {missing}: the app image is older than the feature contract "
                       "(finrisk-app must add hour_of_day in sql_features.py). Bump train/base_image.txt.")


def day_stats(con) -> dict[int, dict]:
    rows = con.execute("SELECT day, count(*), sum(is_fraud), min(row_id), max(row_id) FROM g GROUP BY day ORDER BY day").fetchall()
    return {int(d): {"rows": int(n), "frauds": int(f), "first_row_id": int(lo), "last_row_id": int(hi)} for d, n, f, lo, hi in rows}


def null_check(con, days: list[int]) -> None:
    cols = [c for c in contract.FEATURE_GROUP_COLUMNS if c not in ("row_id", "event_time")]
    cond = " OR ".join(f"{c} IS NULL" for c in cols)
    bad = con.execute(f"SELECT count(*) FROM g WHERE day IN ({','.join(map(str, days))}) AND ({cond})").fetchone()[0]
    if bad:
        raise JobError(f"{bad} history rows have a NULL feature: the Feature Store needs a value for every feature. "
                       "Fix the gold SQL in finrisk-app; the backfill will not invent values.")


def feature_rows(con, days: list[int], chunk: int = 50_000) -> Iterator[tuple[str, ...]]:
    """Every history row as a tuple of strings in FEATURE_GROUP_COLUMNS order, oldest first."""
    select = []
    for column in contract.FEATURE_GROUP_COLUMNS:
        if column == "row_id":
            select.append("row_id_text")
        elif column in contract.STRING_FEATURES:
            select.append(column)
        else:
            select.append(f"CAST({column} AS VARCHAR)")
    select = ", ".join(select)
    cursor = con.execute(f"SELECT {select} FROM g WHERE day IN ({','.join(map(str, days))}) ORDER BY step, row_id")
    while True:
        block = cursor.fetchmany(chunk)
        if not block:
            return
        yield from block


# ---------------------------------------------------------------------------- 3. Feature Store (measurement first)

def write_history(con, days: list[int], group: str, client: Any, threads: int, measure_records: int, max_minutes: float,
                  total_rows: int, say_: Callable[[str], None] = say, clock: Callable[[], float] = time.monotonic,
                  sleep: Callable[[float], None] = time.sleep) -> dict:
    """The Feature Store write: measure with `measure_records`, project, stop if too slow, else write the rest."""
    names = contract.FEATURE_GROUP_COLUMNS
    rows = feature_rows(con, days)
    measure = min(math.ceil(measure_records / 25) * 25, total_rows)
    say_(f"  measuring first: {measure:,} records with {threads} threads")
    first = write_records(client, group, names, rows, threads=threads, say=say_, sleep=sleep, clock=clock, limit=measure,
                          label="measure: ")
    rate = first.rate
    remaining = total_rows - first.records
    projected = remaining / rate / 60 if rate > 0 else math.inf
    say_(f"  measured {first.records:,} records in {first.seconds:.1f} s = {rate:,.0f} records/s "
         f"({first.write_units:,} write units, {first.retries} retries)")
    say_(f"  projection: {remaining:,} records left = {projected:.1f} minutes (limit {max_minutes:.0f})")
    measurement = {"records": first.records, "seconds": round(first.seconds, 1), "records_per_second": round(rate, 1),
                   "projected_minutes_for_rest": round(projected, 1)}
    if projected > max_minutes:
        suggested = 4 if math.isinf(projected) else max(2, min(4, math.ceil(projected / (max_minutes * 0.75))))
        raise JobError(
            f"STOPPED: the rest would take {projected:.0f} minutes (> {max_minutes:.0f}). Nothing is lost - the {first.records:,} "
            f"records already written are safe to write again. Split the days across instances:\n"
            f"  python pipeline/mlops.py backfill --instance-count {suggested}\n"
            f"(measured rate {rate:,.0f} records/s with {threads} threads)", code=EXIT_TOO_SLOW)
    rest = write_records(client, group, names, rows, threads=threads, say=say_, sleep=sleep, clock=clock) if remaining > 0 else WriteStats()
    total = WriteStats(records=first.records + rest.records, calls=first.calls + rest.calls, retries=first.retries + rest.retries,
                       write_units=first.write_units + rest.write_units, seconds=first.seconds + rest.seconds)
    if total.records != total_rows:
        raise JobError(f"wrote {total.records:,} records but the history has {total_rows:,}: the Feature Store is incomplete")
    say_(f"  Feature Store: {total.records:,} records, {total.write_units:,} write units, {total.seconds / 60:.1f} min, "
         f"{total.rate:,.0f} records/s")
    return {**total.as_dict(), "threads": threads, "measurement": measurement}


# ---------------------------------------------------------------------------- 4/5. labels and the production set

def write_labels(con, days: list[int], work: Path) -> list[Path]:
    """One parquet per history day under labels/source=history/ (columns = the Glue table)."""
    folder = work / "labels" / "source=history"
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for day in days:
        path = folder / f"part-day{day:02d}.parquet"
        con.execute(f"""
            COPY (
                SELECT row_id_text AS row_id, CAST(is_fraud AS INTEGER) AS is_fraud, event_time, label_time,
                       'history' AS source, 'history' AS batch_id
                FROM g WHERE day = {int(day)} ORDER BY step, row_id
            ) TO {sql_literal(path)} (FORMAT PARQUET)""")
        files.append(path)
    return files


def write_production(con, days: list[int], work: Path) -> list[tuple[int, Path]]:
    """days 18-30: features AND is_fraud, one folder per day, in event-time order."""
    files = []
    cols = ", ".join(("row_id_text AS row_id" if c == "row_id" else c) for c in contract.FEATURE_GROUP_COLUMNS)
    for day in days:
        folder = work / "replay" / f"day={day}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "part.parquet"
        con.execute(f"COPY (SELECT {cols}, CAST(is_fraud AS INTEGER) AS is_fraud FROM g WHERE day = {int(day)} "
                    f"ORDER BY step, row_id) TO {sql_literal(path)} (FORMAT PARQUET)")
        files.append((day, path))
    return files


# ---------------------------------------------------------------------------- 6. wait until Athena sees the records

def wait_for_visibility(first_event_time: str, last_event_time: str, expected: int, bf: dict, athena: Any, s3: Any,
                        sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> dict:
    """Poll Athena until at least `expected` distinct records of the event-time range are visible in finrisk_ml.transactions."""
    from .build_dataset import queries_module
    from .common import athena_rows, run_athena

    q = queries_module()
    poll, budget = float(bf.get("visibility_poll_seconds", 60)), float(bf.get("visibility_minutes", 30))
    started = clock()
    sql = q.visible_records(first_event_time, last_event_time)
    say(f"  waiting until Athena sees {expected:,} records of {first_event_time} .. {last_event_time} "
        f"(poll every {poll:.0f} s, budget {budget:.0f} min; the offline store lags the writes by up to ~15 min)")
    while True:
        execution = run_athena(sql, q.DATABASE, athena=athena, sleep=sleep, clock=clock, show=False)
        visible = int(athena_rows(execution, s3)[0]["visible"])
        waited = clock() - started
        say(f"  [{waited:5.0f}s] visible {visible:,} of {expected:,}")
        if visible >= expected:
            return {"expected": expected, "visible": visible, "waited_seconds": round(waited)}
        if waited + poll > budget * 60:
            raise JobError(f"Athena still sees only {visible:,} of the {expected:,} records written after {waited / 60:.0f} minutes. The Feature Store "
                           "makes written records queryable with a delay (normally up to ~15 minutes); this is longer. The records ARE written "
                           "(idempotent): run the backfill again in a few minutes - it will not write anything new, and it finishes as soon as "
                           "Athena sees them.")
        sleep(poll)


def merge_reports(reports: list[dict]) -> dict:
    """One report from the per-instance reports: the totals of the history, the first instance's data facts (it also wrote the production set)."""
    first = reports[0]
    store = [r["history"]["feature_store"] for r in reports if not r["history"]["feature_store"].get("skipped")]
    lo = min(int(r["history"]["days"].split("-")[0]) for r in reports)
    hi = max(int(r["history"]["days"].split("-")[-1]) for r in reports)
    return {**first, "merged_from": len(reports), "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "history": {**first["history"], "days": f"{lo}-{hi}", "rows": sum(r["history"]["rows"] for r in reports),
                        "frauds": sum(r["history"]["frauds"] for r in reports),
                        "feature_store": {"records": sum(x["records"] for x in store), "write_units": sum(x["write_units"] for x in store),
                                          "seconds": max((x["seconds"] for x in store), default=0), "instances": len(store)} if store else {"skipped": True}},
            "visibility": {"visible_all": all(r.get("visibility", {}).get("visible", 0) >= r.get("visibility", {}).get("expected", 0) for r in reports)},
            "seconds": max(r["seconds"] for r in reports)}


def write_report(s3: Any, bucket: str, prefix: str, report: dict, index: int, count: int) -> list[str]:
    """backfill_report.json (single job) -- or backfill_report.<k>.json, and then the merged one by whoever finishes last."""
    written = []
    if count == 1:
        uri = s3_uri(bucket, f"{prefix}/backfill_report.json")
        s3_put_json(uri, report, s3)
        return [uri]
    own = s3_uri(bucket, f"{prefix}/backfill_report.{index}.json")
    s3_put_json(own, report, s3)
    written.append(own)
    siblings = []
    for k in range(1, count + 1):
        uri = s3_uri(bucket, f"{prefix}/backfill_report.{k}.json")
        if not s3_exists(uri, s3):
            return written                                      # someone else is still running: they will write the merged report
        siblings.append(s3_get_json(uri, s3))
    merged = s3_uri(bucket, f"{prefix}/backfill_report.json")
    s3_put_json(merged, merge_reports(siblings), s3)
    return written + [merged]


# ---------------------------------------------------------------------------- the job

def peak_memory_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024, 1)


def run(argv: list[str] | None = None, *, opener: Callable[..., Any] = urllib.request.urlopen,
        clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> int:
    a = parse_args(argv)
    cfg = data_config()
    ds, bf, cal = cfg["dataset"], cfg["backfill"], cfg["calendar"]
    started = clock()
    local_only = a.local_only
    bucket = None if local_only else ml_bucket(a.ml_bucket)
    s3 = None if local_only else aws("s3")
    history = [d for d in contract.parse_days(a.days or cfg["history_days"]) if d not in cal.get("exclude_days", [])]
    production = [d for d in contract.parse_days(cfg["production_days"]) if d not in cal.get("exclude_days", [])]
    threads = a.threads or int(bf["threads"])
    max_minutes = a.max_minutes or float(bf["max_minutes"])
    measure_records = a.measure_records or int(bf["measure_records"])
    expected = a.sha256 or ds["sha256"]
    url = a.dataset_url or os.environ.get("DATASET_URL") or ds["url"]
    say(f"backfill: history days {history[0]}-{history[-1]} ({len(history)} days), production set "
        f"{'days ' + str(production[0]) + '-' + str(production[-1]) if a.production == 'yes' else 'not on this instance'}, "
        f"day(s) {cal.get('exclude_days')} excluded")
    if a.days:
        say(f"  --days given: this instance writes only days {a.days}")
    work = workdir(a.workdir, "finrisk-backfill-")
    try:
        say("[1/7] raw data")
        csv_path, raw_info = ensure_raw(bucket, ds["raw_prefix"], expected, url, work, a.raw_file, s3, opener)
        say("[2/7] silver and gold (DuckDB, all days)")
        check_feature_contract()
        gold, timings = build_gold(csv_path, work, int(bf["duckdb_threads"]), str(bf["duckdb_memory_limit"]), cal["start"])
        con = open_gold(gold, cal["start"], int(bf["duckdb_threads"]), str(bf["duckdb_memory_limit"]), work / "duckdb_tmp",
                        int(cfg["label_delay_days"]))
        check_gold_columns(con)
        stats = day_stats(con)
        total_rows = sum(v["rows"] for v in stats.values())
        say(f"  gold: {total_rows:,} rows over {len(stats)} days; fraud {sum(v['frauds'] for v in stats.values()):,}")
        for day in cal.get("exclude_days", []):
            if day in stats:
                say(f"  day {day} EXCLUDED: {stats[day]['rows']} rows, {stats[day]['frauds']} frauds (an artefact of the data)")
        missing = [d for d in history if d not in stats]
        if missing:
            raise JobError(f"history day(s) {missing} are not in the data (days present: {sorted(stats)})")
        history_rows = sum(stats[d]["rows"] for d in history)
        null_check(con, history)

        fs_report: dict[str, Any] = {"skipped": True}
        if a.local_only or a.skip_feature_store:
            say("[3/7] Feature Store write SKIPPED (" + ("--local-only" if a.local_only else "--skip-feature-store") + ")")
        else:
            say(f"[3/7] Feature Store {bf['feature_group']}: {history_rows:,} history records")
            fs_report = write_history(con, history, bf["feature_group"], aws("sagemaker-featurestore-runtime"), threads,
                                      measure_records, max_minutes, history_rows, clock=clock, sleep=sleep)

        say("[4/7] history labels")
        label_files = write_labels(con, history, work)
        say(f"  {len(label_files)} label file(s), {sum(stats[d]['frauds'] for d in history):,} frauds in the history")
        production_files: list[tuple[int, Path]] = []
        if a.production == "yes":
            say("[5/7] production set (days 18-30)")
            production_files = write_production(con, [d for d in production if d in stats], work)
            say(f"  {len(production_files)} day files; rows {sum(stats[d]['rows'] for d, _ in production_files):,}, "
                f"frauds {sum(stats[d]['frauds'] for d, _ in production_files):,}")
        else:
            say("[5/7] production set: not on this instance")
        con.close()

        if not local_only:
            for path in label_files:
                s3_upload(path, s3_uri(bucket, f"{bf['labels_prefix']}/source=history/{path.name}"), s3)
            for day, path in production_files:
                s3_upload(path, s3_uri(bucket, f"{bf['replay_prefix']}/day={day}/part.parquet"), s3)
            say(f"  uploaded {len(label_files)} label file(s) and {len(production_files)} replay file(s)")

        visibility: dict[str, Any] = {"skipped": True}
        if not (local_only or a.skip_feature_store or a.no_visibility_wait):
            say("[6/7] wait until Athena sees the records (the offline store fills in the background)")
            first_time = contract.event_time_of_step((min(history) - 1) * 24 + 1)
            last_time = contract.event_time_of_step(max(history) * 24)
            visibility = wait_for_visibility(first_time, last_time, int(fs_report["records"]), bf, aws("athena"), s3, sleep=sleep, clock=clock)
        else:
            say("[6/7] waiting for Athena: skipped")

        report = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": {**raw_info, "rows_in_csv": ds.get("rows"), "rows_in_gold": total_rows},
            "calendar": {"start": cal["start"], "excluded_days": cal.get("exclude_days", [])},
            "days": {str(d): v for d, v in stats.items()},
            "history": {"days": f"{history[0]}-{history[-1]}", "rows": history_rows, "frauds": sum(stats[d]["frauds"] for d in history),
                        "feature_group": bf["feature_group"], "feature_store": fs_report},
            "labels": {"files": [p.name for p in label_files], "label_delay_days": cfg["label_delay_days"]},
            "visibility": visibility, "instance": {"index": a.instance_index, "count": a.instance_count},
            "production": {"days": [d for d, _ in production_files], "prefix": bf["replay_prefix"],
                           "rows": sum(stats[d]["rows"] for d, _ in production_files),
                           "frauds": sum(stats[d]["frauds"] for d, _ in production_files)},
            "duckdb": {**timings, "threads": int(bf["duckdb_threads"]), "memory_limit": bf["duckdb_memory_limit"]},
            "peak_memory_mb": peak_memory_mb(),
            "seconds": round(clock() - started, 1),
        }
        if local_only:
            say(json.dumps({k: report[k] for k in ("history", "production", "duckdb", "peak_memory_mb", "seconds")}, indent=2, default=str))
        else:
            for uri in write_report(s3, bucket, ds["raw_prefix"], report, a.instance_index, a.instance_count):
                say(f"[7/7] report written: {uri}")
        say(f"backfill finished in {report['seconds'] / 60:.1f} minutes")
        return 0
    finally:
        if not a.keep_workdir and not a.workdir:
            import shutil

            shutil.rmtree(work, ignore_errors=True)
