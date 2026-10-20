"""replay: the simulated OUTSIDE WORLD. Days 18-30 of PaySim arrive as live traffic through the real production path.

    s3://<ml>/replay/v1/day=<n>/part.parquet  (written once by the backfill)
        -> batches of 500 -> POST <app_url>/api/score -> the app -> the SageMaker endpoint -> Data Capture (S3)

Customers are not part of the system, and neither is this: it is the thing that makes the system busy. It prints a line per
day (rows, batches, seconds, flagged), asks for an explanation (`explain: true`) on 1 % of the batches only (it is slower),
and writes the features of the rows it sent to the Feature Store -- production features are recorded as they arrive, so the
next training run can find them. The true labels come later and separately (feedback.py): that delay is the point.

Needs pandas and pyarrow (requirements-sim.txt), nothing else beyond boto3.
"""

from __future__ import annotations

import io
import json
import time
from typing import Any

from errors import MlopsError, mask

REPLAY_PREFIX = "replay/v1"
SCORE_PATH = "/api/score"
INTEGRAL = ("step", "hour_of_day", "tx_index_for_origin", "customer_tx_count_so_far")


def read_day(ctx, bucket: str, day: int, prefix: str = REPLAY_PREFIX):
    import pandas as pd

    key = f"{prefix}/day={day}/part.parquet"
    try:
        body = ctx.s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as exc:  # noqa: BLE001
        raise MlopsError(f"s3://{bucket}/{key} is not readable ({type(exc).__name__}): has the backfill run? (backfill.yml)") from exc
    frame = pd.read_parquet(io.BytesIO(body))
    frame["_rid"] = pd.to_numeric(frame["row_id"], errors="coerce")
    return frame.sort_values(["step", "_rid"], kind="stable").drop(columns="_rid").reset_index(drop=True)


def instances(frame) -> list[dict[str, Any]]:
    """Exactly row_id + the model's features (the endpoint contract rejects anything else), as plain JSON numbers."""
    from finrisk_jobs.contract import FEATURE_COLUMNS

    out = []
    for record in frame[["row_id", *FEATURE_COLUMNS]].to_dict(orient="records"):
        row: dict[str, Any] = {"row_id": str(record["row_id"])}
        for key in FEATURE_COLUMNS:
            value = record[key]
            row[key] = value if isinstance(value, str) else (int(value) if key in INTEGRAL else float(value))
        out.append(row)
    return out


def post_score(ctx, app_url: str, rows: list[dict], explain: bool, retries: int = 12, wait: float = 15.0) -> dict:
    """One /api/score call. A 503 is a serverless/real-time endpoint that is still starting: wait and try again."""
    url = app_url.rstrip("/") + SCORE_PATH
    body = {"transactions": rows, "explain": explain}
    for attempt in range(1, retries + 1):
        status, payload = ctx.post(url, body, 90)
        if status == 200:
            if "results" not in payload:
                raise MlopsError(f"{mask(url)} answered 200 without 'results': {mask(str(payload)[:160])}")
            return payload
        if status == 503 and attempt < retries:
            ctx.out(f"    503 ({mask(str(payload)[:80])}); try {attempt}/{retries}, waiting {wait:.0f} s")
            ctx.sleep(wait)
            continue
        raise MlopsError(f"POST {mask(url)} -> HTTP {status}: {mask(json.dumps(payload)[:300])}")
    raise MlopsError(f"{mask(url)} stayed unavailable after {retries} tries")


def feature_group_rows(frame) -> list[tuple[str, ...]]:
    from finrisk_jobs.contract import FEATURE_GROUP_COLUMNS, INTEGRAL_FEATURES, STRING_FEATURES

    rows = []
    for record in frame[FEATURE_GROUP_COLUMNS].to_dict(orient="records"):
        rows.append(tuple(str(record[c]) if c in STRING_FEATURES else (str(int(record[c])) if c in INTEGRAL_FEATURES else repr(float(record[c])))
                          for c in FEATURE_GROUP_COLUMNS))
    return rows


def replay(ctx, app_url: str, from_day: int, to_day: int, batch: int, bucket: str, explain_every: int = 100,
           threads: int = 16) -> dict[str, Any]:
    from finrisk_jobs.contract import FEATURE_GROUP, FEATURE_GROUP_COLUMNS
    from finrisk_jobs.featurestore import write_records

    if not 1 <= batch <= 500:
        raise MlopsError("batch must be 1-500 (the endpoint takes at most 500 instances per request)")
    days = []
    totals = {"rows": 0, "batches": 0, "flagged": 0, "frauds": 0}
    versions: set[str] = set()
    started = ctx.clock()
    ctx.out(f"replay days {from_day}-{to_day} through {mask(app_url)}{SCORE_PATH} (batch {batch}, explain on every {explain_every}th batch)")
    for day in range(from_day, to_day + 1):
        frame = read_day(ctx, bucket, day)
        t0, flagged, batches = ctx.clock(), 0, 0
        for start in range(0, len(frame), batch):
            part = frame.iloc[start:start + batch]
            result = post_score(ctx, app_url, instances(part), explain=(batches % explain_every == explain_every - 1))
            flagged += sum(1 for r in result["results"] if r.get("flagged"))
            versions.add(str(result.get("model_version")))
            batches += 1
        seconds = ctx.clock() - t0
        frauds = int(frame["is_fraud"].sum())
        ctx.out(f"  day {day}: {len(frame):>7,} rows  {batches:>4} batches  {seconds:6.1f} s  flagged {flagged:>6,}  (true frauds in the day: {frauds:,})")
        stats = write_records(ctx.fs, FEATURE_GROUP, FEATURE_GROUP_COLUMNS, iter(feature_group_rows(frame)), threads=threads,
                              progress_every=10 ** 9, say=ctx.out, sleep=ctx.sleep, clock=ctx.clock)
        ctx.out(f"          features recorded: {stats.records:,} records, {stats.write_units:,} write units")
        days.append({"day": day, "rows": len(frame), "batches": batches, "seconds": round(seconds, 1), "flagged": flagged, "frauds": frauds})
        totals["rows"] += len(frame); totals["batches"] += batches; totals["flagged"] += flagged; totals["frauds"] += frauds
    ctx.out(f"replayed {totals['rows']:,} rows in {totals['batches']} calls, {ctx.clock() - started:.0f} s; model version(s) answering: {sorted(versions)}")
    return {"days": days, **totals, "model_versions": sorted(versions)}
