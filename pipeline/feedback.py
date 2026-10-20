"""feedback: the simulated ANALYSTS. They tell us, later, which of the replayed transactions really were fraud.

Labels are never available when a transaction is scored. They arrive days later (a customer disputes a charge, an analyst
confirms). This writes the true labels of the replayed days to

    s3://<ml>/labels/source=feedback/batch=<run id>/part.parquet     (columns = the Glue table finrisk_ml.labels)

with `label_time` = the last replayed event time + the analyst delay (2 days): the simulated "now". Two things read them
afterwards: the monitor (live PR-AUC and the live fraud rate) and the next training run (a point-in-time query only sees
labels whose label_time is before its cutoff).
"""

from __future__ import annotations

import io
from typing import Any

from errors import mask
from replay import read_day

LABELS_PREFIX = "labels"


def feedback(ctx, from_day: int, to_day: int, bucket: str, run_id: str, delay_days: int = 2, delay_minutes: float = 0.0) -> dict[str, Any]:
    import pandas as pd

    from finrisk_jobs.contract import label_time_of

    frames = [read_day(ctx, bucket, day) for day in range(from_day, to_day + 1)]
    frame = pd.concat(frames, ignore_index=True)
    if delay_minutes > 0:
        ctx.out(f"  the analysts need {delay_minutes:g} minute(s) ...")
        ctx.sleep(delay_minutes * 60)
    last_event = frame["event_time"].max()
    label_time = label_time_of(last_event, delay_days)
    out = pd.DataFrame({"row_id": frame["row_id"].astype(str), "is_fraud": frame["is_fraud"].astype(int), "event_time": frame["event_time"],
                        "label_time": label_time, "source": "feedback", "batch_id": run_id})
    buffer = io.BytesIO()
    out.to_parquet(buffer, index=False)
    key = f"{LABELS_PREFIX}/source=feedback/batch={run_id}/part.parquet"
    ctx.s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    summary = {"rows": len(out), "frauds": int(out["is_fraud"].sum()), "label_time": label_time, "key": key, "days": [from_day, to_day]}
    ctx.out(f"  wrote {summary['rows']:,} labels ({summary['frauds']:,} frauds) to s3://{mask(bucket)}/{key}")
    ctx.out(f"  label_time {label_time} (last replayed event {last_event} + {delay_days} days)")
    return summary
