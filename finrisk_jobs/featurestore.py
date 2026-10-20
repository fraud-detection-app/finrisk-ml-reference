"""Writing records to the Feature Store with BatchWriteRecord: threads, retries, write units.

Pure Python + a boto3 client (``sagemaker-featurestore-runtime``) passed in: no pandas. Used by the backfill job (history,
inside SageMaker) and by the replay (production features, from a GitHub runner).

BatchWriteRecord takes at most 25 records per call. A record is a list of {"FeatureName", "ValueAsString"}.
Writes cost 1 write unit per record of up to 1 KB (a record here is ~0.5 KB), so the UNIT COUNT reported is
``ceil(bytes / 1024)`` per record with bytes = the sum of name + value lengths.

Re-writing the same row_id with the same event_time is harmless: the offline store is append-only and the
point-in-time query keeps the latest write per record.
"""

from __future__ import annotations

import math
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Sequence

MAX_BATCH = 25
RETRY_CODES = {"ThrottlingException", "ServiceUnavailable", "InternalFailure", "InternalServerError", "RequestTimeout",
               "ThrottledException", "SlowDown", "ProvisionedThroughputExceededException"}


class FeatureStoreError(RuntimeError):
    pass


@dataclass
class WriteStats:
    records: int = 0
    calls: int = 0
    retries: int = 0
    write_units: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.records / self.seconds if self.seconds > 0 else 0.0

    def as_dict(self) -> dict:
        return {"records": self.records, "calls": self.calls, "retries": self.retries, "write_units": self.write_units,
                "seconds": round(self.seconds, 1), "records_per_second": round(self.rate, 1)}


def record_units(names: Sequence[str], values: Sequence[str]) -> int:
    size = sum(len(n) + len(v) for n, v in zip(names, values))
    return max(1, math.ceil(size / 1024))


def make_entry(group: str, names: Sequence[str], values: Sequence[str]) -> dict:
    return {"FeatureGroupName": group, "TargetStores": ["OfflineStore"],     # per ENTRY (botocore: BatchWriteRecord has no top-level TargetStores)
            "Record": [{"FeatureName": n, "ValueAsString": v} for n, v in zip(names, values)]}


def batches(rows: Iterable[Sequence[str]], size: int = MAX_BATCH) -> Iterator[list[Sequence[str]]]:
    chunk: list[Sequence[str]] = []
    for row in rows:
        chunk.append(row)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _code(exc: Exception) -> str:
    return (getattr(exc, "response", None) or {}).get("Error", {}).get("Code", "")


def write_batch(client, group: str, names: Sequence[str], rows: list[Sequence[str]], stats: WriteStats,
                lock: threading.Lock, sleep: Callable[[float], None] = time.sleep, max_attempts: int = 6) -> None:
    """One BatchWriteRecord call (<= 25 records); retries what AWS says it did not take, with growing waits.

    The response lists the entries that failed (``Errors[].Entry``) or were not processed (``UnprocessedEntries``); a retry
    sends only those, so a record is never counted twice."""
    entries = [make_entry(group, names, row) for row in rows]
    attempt = 0
    while True:
        attempt += 1
        try:
            response = client.batch_write_record(Entries=entries)
        except Exception as exc:  # noqa: BLE001
            if _code(exc) in RETRY_CODES and attempt < max_attempts:
                with lock:
                    stats.retries += 1
                sleep(min(30.0, 0.5 * 2 ** attempt))
                continue
            raise
        errors = list(response.get("Errors") or [])
        left = [e.get("Entry") for e in errors] + list(response.get("UnprocessedEntries") or [])
        left_idx = [i for i, entry in enumerate(entries) if entry in left]
        written = [r for i, r in enumerate(rows) if i not in set(left_idx)]
        with lock:
            stats.calls += 1
            stats.records += len(written)
            stats.write_units += sum(record_units(names, r) for r in written)
        if not left_idx:
            return
        if attempt >= max_attempts:
            first = errors[0] if errors else {}
            raise FeatureStoreError(f"BatchWriteRecord: {len(left_idx)} of {len(entries)} records still not written after "
                                    f"{attempt} tries: {first.get('ErrorCode', '')} {first.get('ErrorMessage', '')}".strip())
        entries = [entries[i] for i in left_idx]
        rows = [rows[i] for i in left_idx]
        with lock:
            stats.retries += 1
        sleep(min(30.0, 0.5 * 2 ** attempt))


def write_records(client, group: str, names: Sequence[str], rows: Iterable[Sequence[str]], threads: int = 32,
                  progress_every: int = 250_000, say: Callable[[str], None] = print,
                  sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                  limit: int | None = None, label: str = "") -> WriteStats:
    """Write every row (or stop once ``limit`` records are submitted) with a pool of threads; return the totals.

    ``rows`` should be an iterator: a second call continues where a limited first call stopped. Make ``limit`` a
    multiple of 25 (the batch size), or the rest of the last batch is dropped.

    Prints a line every ``progress_every`` records with the running throughput. Raises on the first batch that cannot
    be written (after its retries): a half-written history must not look finished."""
    stats = WriteStats()
    lock = threading.Lock()
    started = clock()
    next_report = progress_every
    submitted = 0
    pending: set = set()
    error: list[BaseException] = []

    def work(chunk: list[Sequence[str]]) -> None:
        write_batch(client, group, names, chunk, stats, lock, sleep=sleep)

    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        for chunk in batches(rows):
            submitted += len(chunk)
            pending.add(pool.submit(work, chunk))
            while len(pending) >= threads * 4:                 # a bounded queue: never hold the whole history in memory
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    if future.exception() and not error:
                        error.append(future.exception())
            if error or (limit is not None and submitted >= limit):
                break
            if stats.records >= next_report:
                elapsed = max(clock() - started, 1e-9)
                say(f"  {label}written {stats.records:>10,} records  {stats.records / elapsed:8,.0f} records/s  "
                    f"{stats.write_units:>10,} write units  ({stats.retries} retries)")
                next_report += progress_every
        for future in pending:
            if future.exception() and not error:
                error.append(future.exception())
    stats.seconds = clock() - started
    if error:
        raise error[0]
    return stats
