"""build-dataset: step 1 of the pipeline `finrisk-train`. A Processing job.

    1. cutoff     Cutoff=auto means "every label that has arrived": the latest label_time in finrisk_ml.labels.
    2. query      the point-in-time query (pipeline/queries.py) through Athena, UNLOADed as Parquet to S3
    3. split      by DAY of event_time: test = the last `test_days` labelled days, validation = the `validation_days` before,
                  train = everything earlier. Never random: the model is always judged on days after everything it learned from.
    4. baseline   a stratified sample of train (the monitor's reference: "what normal looked like when this model was built")
    5. report     dataset.json: cutoff, rows and frauds per split, the query ids

Output (the Processing output `dataset`, uploaded to s3://<ml>/datasets/<execution id>/):

    train/part-0.parquet  validation/part-0.parquet  test/part-0.parquet  profile/baseline.parquet  dataset.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

from . import contract
from .common import (CODE_ROOT, PIPELINE_DIR, JobError, Tracker, athena_rows, aws, load_yaml, ml_bucket, run_athena, s3_delete_prefix,
                     s3_download_prefix, say, workdir, write_json)

DEFAULT_OUTPUT = "/opt/ml/processing/output/dataset"


def queries_module():
    if str(PIPELINE_DIR) not in sys.path:
        sys.path.insert(0, str(PIPELINE_DIR))
    import queries

    return queries


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="finrisk_jobs build-dataset", description=__doc__.split("\n\n")[0])
    p.add_argument("--cutoff", default="auto", help="UTC ISO time, or auto (every label that has arrived)")
    p.add_argument("--ml-bucket")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="folder SageMaker uploads as the output `dataset`")
    p.add_argument("--execution-id", default=os.environ.get("PIPELINE_EXECUTION_ID"), help="names the unload folder")
    p.add_argument("--workdir")
    p.add_argument("--keep-unload", action="store_true", help="do not delete the Athena unload files afterwards")
    return p.parse_args(argv)


def resolve_cutoff(cutoff: str, athena: Any = None, s3: Any = None, **kw: Any) -> tuple[str, str | None]:
    q = queries_module()
    if cutoff != "auto":
        return q.check_cutoff(cutoff), None
    execution = run_athena(q.latest_label_time(), q.DATABASE, athena=athena, **kw)
    rows = athena_rows(execution, s3)
    value = (rows[0].get("cutoff") if rows else "") or ""
    if not value:
        raise JobError("Cutoff=auto found no labels in finrisk_ml.labels: run the backfill first (backfill.yml).")
    say(f"  Cutoff=auto -> {value} (the latest label_time)")
    return q.check_cutoff(value), execution["QueryExecutionId"]


def wait_for_complete_store(cutoff: str, cfg: dict, athena: Any, s3: Any, sleep: Callable[[float], None] = time.sleep,
                            clock: Callable[[], float] = time.monotonic) -> dict:
    """Never train on a half-filled offline store. The Feature Store writes it in the background (a record is queryable in
    Athena some minutes, up to ~15, after it was written), but the labels table is plain S3 and is complete at once. So count the
    labelled rows (label_time <= cutoff) that have no feature row, and poll until that is 0 -- or at most `max_missing_share` of
    the labelled rows (default 0.1 %) -- for `wait_minutes`. Otherwise fail, saying why."""
    q = queries_module()
    share, budget, poll = float(cfg.get("max_missing_share", 0.001)), float(cfg.get("wait_minutes", 30)), float(cfg.get("poll_seconds", 60))
    started = clock()
    while True:
        execution = run_athena(q.labelled_without_features(cutoff), q.DATABASE, athena=athena, sleep=sleep, clock=clock, show=False)
        row = athena_rows(execution, s3)[0]
        labelled, missing = int(row["labelled"]), int(row["missing"])
        allowed = int(labelled * share)
        waited = clock() - started
        if labelled == 0:
            raise JobError("no labelled rows before the cutoff: nothing to train on. Run the backfill first (backfill.yml).")
        if missing <= allowed:
            say(f"  offline store complete: {labelled - missing:,} of {labelled:,} labelled rows have features"
                + (f" ({missing:,} missing, within the {share:.1%} allowed)" if missing else ""))
            return {"labelled": labelled, "missing": missing, "allowed": allowed, "waited_seconds": round(waited)}
        say(f"  [{waited:5.0f}s] the offline store is still filling: {missing:,} of {labelled:,} labelled rows have no features yet "
            f"(allowed {allowed:,}); checking again in {poll:.0f} s")
        if waited + poll > budget * 60:
            raise JobError(f"the offline store is still filling: {missing:,} labelled rows have no features yet (of {labelled:,}; "
                           f"{allowed:,} allowed) after waiting {waited / 60:.0f} minutes. The Feature Store takes up to ~15 minutes to make "
                           "written records queryable in Athena - retry in a few minutes. If it stays like this, a write failed: look at the "
                           "backfill_report.json and the replay log.")
        sleep(poll)


def split_days(days: list[int], test_days: int, validation_days: int) -> dict[str, list[int]]:
    """The last `test_days` days are test, the `validation_days` before them validation, the rest train."""
    days = sorted(days)
    need = test_days + validation_days + 1
    if len(days) < need:
        raise JobError(f"only {len(days)} labelled day(s) ({days}); a time split needs at least {need} "
                       f"({test_days} test + {validation_days} validation + 1 train). Is the cutoff too early?")
    return {"test": days[-test_days:], "validation": days[-(test_days + validation_days):-test_days], "train": days[:-(test_days + validation_days)]}


def _is_parquet(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(4) == b"PAR1"


def write_splits(unload_dir: Path, out: Path, cfg: dict, cutoff: str) -> dict:
    """DuckDB over the unloaded parquet files -> train/validation/test + the baseline sample + the counts."""
    import duckdb

    files = [f for f in sorted(unload_dir.rglob("*")) if f.is_file() and _is_parquet(f)]
    if not files:
        raise JobError("the Athena query returned no rows: no labelled transactions before the cutoff. Run the backfill first.")
    con = duckdb.connect()
    con.execute(f"SET threads = {int(cfg.get('threads', 4))}")
    listed = ", ".join("'" + str(f).replace("'", "''") + "'" for f in files)
    con.execute(f"CREATE VIEW pit AS SELECT *, ((step - 1) // 24) + 1 AS day FROM read_parquet([{listed}])")
    days = [int(r[0]) for r in con.execute("SELECT DISTINCT day FROM pit ORDER BY day").fetchall()]
    spec = split_days(days, int(cfg["test_days"]), int(cfg["validation_days"]))
    report: dict[str, Any] = {"splits": {}}
    for name, split in spec.items():
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        where = f"day BETWEEN {split[0]} AND {split[-1]}"
        con.execute(f"COPY (SELECT * EXCLUDE (day) FROM pit WHERE {where} ORDER BY step, CAST(row_id AS BIGINT)) "
                    f"TO '{folder / 'part-0.parquet'}' (FORMAT PARQUET)")
        rows, frauds, first, last = con.execute(f"SELECT count(*), sum(CAST(is_fraud AS INTEGER)), min(event_time), max(event_time) FROM pit WHERE {where}").fetchone()
        rows, frauds = int(rows), int(frauds or 0)
        report["splits"][name] = {"days": [split[0], split[-1]], "rows": rows, "frauds": frauds,
                                  "prevalence": frauds / rows if rows else 0.0, "from": first, "to": last}
        say(f"  {name:<10} days {split[0]:>2}-{split[-1]:<2} {rows:>10,} rows {frauds:>7,} frauds ({100 * frauds / max(rows, 1):.3f} %)")
    train = report["splits"]["train"]
    n = min(int(cfg["baseline_rows"]), train["rows"])
    where = f"day BETWEEN {spec['train'][0]} AND {spec['train'][-1]}"
    n_fraud = min(train["frauds"], max(1, round(n * train["prevalence"]))) if train["frauds"] else 0
    n_legit = n - n_fraud
    profile = out / "profile"
    profile.mkdir(parents=True, exist_ok=True)
    con.execute(f"""COPY (
        SELECT * EXCLUDE (day) FROM (
            (SELECT * FROM (SELECT * FROM pit WHERE {where} AND CAST(is_fraud AS INTEGER) = 1) USING SAMPLE reservoir({max(n_fraud, 1)} ROWS) REPEATABLE (42))
            UNION ALL
            (SELECT * FROM (SELECT * FROM pit WHERE {where} AND CAST(is_fraud AS INTEGER) = 0) USING SAMPLE reservoir({max(n_legit, 1)} ROWS) REPEATABLE (42))
        ) ORDER BY step, CAST(row_id AS BIGINT)
    ) TO '{profile / 'baseline.parquet'}' (FORMAT PARQUET)""")
    b_rows, b_frauds = con.execute(f"SELECT count(*), sum(CAST(is_fraud AS INTEGER)) FROM read_parquet('{profile / 'baseline.parquet'}')").fetchone()
    report["baseline"] = {"rows": int(b_rows), "frauds": int(b_frauds or 0), "sampled_from": "train", "method": "stratified by is_fraud, proportional"}
    say(f"  baseline   {b_rows:,} rows sampled from train ({int(b_frauds or 0):,} frauds): the monitor's reference")
    con.close()
    report["days_present"] = days
    report["columns"] = [c for c in _columns(out / "train" / "part-0.parquet")]
    return report


def _columns(path: Path) -> list[str]:
    import duckdb

    return [r[0] for r in duckdb.connect().execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()]


def run(argv: list[str] | None = None, *, athena: Any = None, s3: Any = None, sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic) -> int:
    a = parse_args(argv)
    bucket = ml_bucket(a.ml_bucket)
    athena = athena or aws("athena")
    s3 = s3 or aws("s3")
    q = queries_module()
    train_cfg = load_yaml("train.yaml")
    exec_id = a.execution_id or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    work = workdir(a.workdir, "finrisk-dataset-")
    kw = {"sleep": sleep, "clock": clock}
    say(f"build-dataset: execution {exec_id}, cutoff {a.cutoff}")
    cutoff, cutoff_query = resolve_cutoff(a.cutoff, athena, s3, **kw)
    say("  is the offline store complete? (labelled rows that have no features yet)")
    completeness = wait_for_complete_store(cutoff, train_cfg.get("completeness", {}), athena, s3, sleep=sleep, clock=clock)
    unload_uri = f"s3://{bucket}/athena/unload/{exec_id}"
    say(f"  point-in-time query as of {cutoff}")
    execution = run_athena(q.unload(q.point_in_time(cutoff), unload_uri), q.DATABASE, athena=athena, **kw)
    say(f"  downloading the unloaded files from {unload_uri}/")
    files = s3_download_prefix(unload_uri, work / "unload", s3)
    say(f"  {len(files)} file(s), {sum(f.stat().st_size for f in files) / 1e6:,.1f} MB")
    report = write_splits(work / "unload", out, train_cfg["splits"], cutoff)
    report.update({
        "execution_id": exec_id, "cutoff": cutoff, "cutoff_input": a.cutoff,
        "dataset_uri": f"s3://{bucket}/datasets/{exec_id}", "baseline_uri": f"s3://{bucket}/datasets/{exec_id}/profile/baseline.parquet",
        "queries": {"cutoff": cutoff_query, "point_in_time": execution["QueryExecutionId"], "unload": unload_uri,
                    "data_scanned_bytes": execution.get("Statistics", {}).get("DataScannedInBytes")},
        "feature_columns": contract.FEATURE_COLUMNS, "target": contract.TARGET,
        "completeness": completeness, "mlflow_run_id": None,
    })
    write_json(out / "dataset.json", report)
    if not a.keep_unload:
        deleted = s3_delete_prefix(unload_uri, s3)
        say(f"  deleted {deleted} intermediate file(s) under {unload_uri}/")
    import shutil

    shutil.rmtree(work, ignore_errors=True)
    say(f"build-dataset finished: {out}")
    return 0
