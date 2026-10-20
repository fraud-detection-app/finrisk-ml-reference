"""Small real data for the job tests: the app's own demo generator + the backfill's own gold SQL (so the tests exercise the same path)."""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from finrisk_jobs import backfill, contract


@lru_cache(maxsize=4)
def demo_gold(rows: int = 30000, seed: int = 7) -> pd.DataFrame:
    """The gold table of `rows` demo transactions (all 31 days), with day, event_time, label_time, row_id as text."""
    from finrisk_jobs.common import add_app_package_to_path

    add_app_package_to_path()
    from financial_risk.data.paysim import generate_demo_data

    work = Path(tempfile.mkdtemp(prefix="finrisk-test-gold-"))
    csv = generate_demo_data(rows, work / "paysim.csv", seed=seed)
    gold, _ = backfill.build_gold(csv, work, 2, "2GB", "2026-09-01T00:00:00Z")
    con = backfill.open_gold(gold, "2026-09-01T00:00:00Z", 2, "2GB", work / "tmp", 2)
    frame = con.execute("SELECT * EXCLUDE (row_id_text), row_id_text AS row_id_s FROM g ORDER BY step, row_id").fetchdf()
    con.close()
    frame["row_id"] = frame["row_id_s"]
    return frame.drop(columns="row_id_s")


def history_tables(gold: pd.DataFrame, days=range(1, 18), write_time="2026-10-01T00:00:00Z") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(finrisk_ml.transactions, finrisk_ml.labels) as the Feature Store and the backfill would have left them."""
    part = gold[gold["day"].isin(list(days))]
    cols = contract.FEATURE_GROUP_COLUMNS
    tx = part[cols].copy()
    tx["write_time"] = pd.Timestamp(write_time)
    tx["api_invocation_time"] = pd.Timestamp(write_time)
    tx["is_deleted"] = False
    labels = pd.DataFrame({"row_id": part["row_id"], "is_fraud": part["is_fraud"].astype(int), "event_time": part["event_time"],
                           "label_time": part["label_time"], "source": "history", "batch_id": "history"})
    return tx, labels


def trained_package(tmp_path: Path, gold: pd.DataFrame, model_type="optimized", hyper=None) -> Path:
    """Run the real training job on days 1-13 / 14-15 and return the folder that holds the model package files."""
    from finrisk_jobs import train

    folders = {}
    for name, days in (("train", range(1, 14)), ("validation", range(14, 16))):
        folder = tmp_path / "in" / name
        folder.mkdir(parents=True, exist_ok=True)
        gold[gold["day"].isin(list(days))][["row_id", "event_time", "step", *contract.FEATURE_COLUMNS, contract.TARGET]].to_parquet(folder / "part-0.parquet")
        folders[name] = folder
    model_dir = tmp_path / "model"
    train.main({"SM_CHANNEL_TRAIN": str(folders["train"]), "SM_CHANNEL_VALIDATION": str(folders["validation"]), "SM_MODEL_DIR": str(model_dir),
                "SM_OUTPUT_DATA_DIR": str(tmp_path / "out"), "HYPERPARAMETERS": __import__("json").dumps({"model_type": model_type, **(hyper or {})}),
                "TRAINING_JOB_NAME": "finrisk-tune-20261016-001-abcd1234"})
    return model_dir


class FakeMlflow:
    """Just the part of the mlflow API that finrisk_jobs.common.Tracker uses; remembers everything it was told."""

    class _Run:
        class info:                                # noqa: N801
            run_id = "run-0001"

    def __init__(self, fail=None):
        self.fail, self.calls, self.params, self.metrics, self.tags, self.artifacts = fail, [], {}, {}, {}, []
        self.tracking_uri = self.experiment = self.run_name = None
        self.ended = None

    def _maybe(self, name):
        self.calls.append(name)
        if self.fail == name:
            raise RuntimeError(f"{name} exploded")

    def set_tracking_uri(self, uri):
        self._maybe("set_tracking_uri"); self.tracking_uri = uri

    def set_experiment(self, name):
        self._maybe("set_experiment"); self.experiment = name

    def start_run(self, run_name=None, tags=None):
        self._maybe("start_run"); self.run_name = run_name; self.tags.update(tags or {}); return self._Run()

    def log_params(self, values):
        self._maybe("log_params"); self.params.update(values)

    def log_metric(self, name, value, step=None):
        self._maybe("log_metric"); self.metrics[name] = value

    def set_tags(self, values):
        self._maybe("set_tags"); self.tags.update(values)

    def log_artifact(self, path, artifact_path=None):
        self._maybe("log_artifact"); self.artifacts.append((str(path), artifact_path))

    def end_run(self, status="FINISHED"):
        self._maybe("end_run"); self.ended = status
