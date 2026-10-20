"""BatchWriteRecord: 25 per call, threads, retries, write units, the measurement limit."""

from __future__ import annotations

import pytest
from fakes import FakeFeatureStore

from finrisk_jobs import featurestore as fsm

NAMES = ["row_id", "event_time", "amount"]


def rows(n, start=0):
    return iter((str(i), "2026-09-01T00:00:00Z", f"{i}.5") for i in range(start, start + n))


def test_batches_of_25_and_every_record_arrives_once():
    fs = FakeFeatureStore()
    stats = fsm.write_records(fs, "finrisk-transactions", NAMES, rows(260), threads=4, say=lambda m: None)
    assert stats.records == 260 and len(fs.records) == 260 and fs.calls == 11          # ceil(260 / 25)
    assert stats.write_units == 260 and stats.retries == 0
    assert fs.records["7"] == {"row_id": "7", "event_time": "2026-09-01T00:00:00Z", "amount": "7.5"}


def test_a_throttled_call_is_retried_and_counted():
    fs = FakeFeatureStore()
    fs.throttle_first = 3
    waits = []
    stats = fsm.write_records(fs, "g", NAMES, rows(100), threads=1, say=lambda m: None, sleep=waits.append)
    assert stats.records == 100 and len(fs.records) == 100 and stats.retries == 3 and len(waits) == 3
    assert waits == sorted(waits) and waits[0] > 0                                       # growing waits


def test_entries_reported_failed_are_retried_alone():
    fs = FakeFeatureStore()
    fs.fail_entries_once = 5
    stats = fsm.write_records(fs, "g", NAMES, rows(50), threads=1, say=lambda m: None, sleep=lambda s: None)
    assert len(fs.records) == 50 and stats.records == 50 and stats.retries == 1 and fs.calls == 3     # 2 batches + 1 retry of 5


def test_a_batch_that_never_succeeds_raises():
    fs = FakeFeatureStore()
    fs.throttle_first = 99
    with pytest.raises(Exception, match="ThrottlingException"):
        fsm.write_records(fs, "g", NAMES, rows(25), threads=1, say=lambda m: None, sleep=lambda s: None)


def test_limit_stops_after_that_many_and_the_iterator_continues():
    fs, source = FakeFeatureStore(), rows(100)
    first = fsm.write_records(fs, "g", NAMES, source, threads=2, limit=50, say=lambda m: None)
    assert first.records == 50 and len(fs.records) == 50
    rest = fsm.write_records(fs, "g", NAMES, source, threads=2, say=lambda m: None)
    assert rest.records == 50 and len(fs.records) == 100


def test_write_units_grow_with_record_size():
    big = ("1", "x" * 3000, "1")
    assert fsm.record_units(NAMES, big) == 3 and fsm.record_units(NAMES, ("1", "t", "2")) == 1


def test_progress_lines_show_throughput():
    lines = []
    ticks = iter(range(0, 10_000))
    fsm.write_records(FakeFeatureStore(), "g", NAMES, rows(500), threads=1, progress_every=100, say=lines.append, clock=lambda: next(ticks) / 100)
    assert lines and "records/s" in lines[0] and "write units" in lines[0]
