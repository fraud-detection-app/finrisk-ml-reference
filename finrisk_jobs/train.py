"""train: runs INSIDE a SageMaker training job -- once for a plain job, once per trial of a hyperparameter-tuning job.

SageMaker's training contract is small. It starts our container with the argument `train` and says, with folders:

    /opt/ml/input/data/train/             the `train` channel (parquet; the Tuning step points it at datasets/<id>/train/)
    /opt/ml/input/data/validation/        the `validation` channel
    /opt/ml/input/config/hyperparameters.json   this trial's hyperparameters (values arrive as JSON-encoded strings)
    /opt/ml/model/                        put the model here; SageMaker tars it to model.tar.gz in S3
    /opt/ml/output/data/                  anything else worth keeping

and, with the log: lines shaped `name=value` become metrics (the Tuning step's regular expressions are built from
contract.METRIC_NAMES). The tuning objective is `validation:pr_auc`: Bayesian search maximises it.

This is NOT new modelling. It calls the app's own building blocks (`build_optimized_model`, `choose_threshold`,
`classification_metrics`): the image is built FROM the app image, so the feature code, the model code and the file format
are exactly what the endpoint loads. The tuned parameters are set on the app's pipeline with `set_params`.

The model package (model.tar.gz, flat) holds `model_v2.pkl` (`model_v1.pkl` for the baseline), `latest_model.joblib` and
`metrics.json`. It does NOT hold the feature table: six million rows do not belong in a model package.

On a laptop (no AWS):

    FINRISK_APP_SRC=../finrisk-app/src SM_CHANNEL_TRAIN=/tmp/in/train SM_CHANNEL_VALIDATION=/tmp/in/validation \\
        SM_MODEL_DIR=/tmp/m SM_OUTPUT_DATA_DIR=/tmp/o HYPERPARAMETERS='{"model_type":"baseline"}' python -m finrisk_jobs train
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import contract
from .common import JobError, Tracker, add_app_package_to_path, check_feature_contract, say, write_json

HYPERPARAMETERS_FILE = Path("/opt/ml/input/config/hyperparameters.json")
XGB_TYPES = {"max_depth": int, "learning_rate": float, "n_estimators": int, "subsample": float, "colsample_bytree": float,
             "min_child_weight": float}


def decode(value: Any) -> Any:
    """SageMaker stores hyperparameters as JSON-encoded strings: '"optimized"' -> optimized, '5' -> 5."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def read_hyperparameters(env: dict[str, str], path: Path = HYPERPARAMETERS_FILE) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    if env.get("HYPERPARAMETERS"):
        raw = json.loads(env["HYPERPARAMETERS"])
    elif path.exists():
        raw = json.loads(path.read_text())
    values = {k: decode(v) for k, v in raw.items()}
    if env.get("MODEL_TYPE"):
        values.setdefault("model_type", env["MODEL_TYPE"])
    model_type = str(values.get("model_type", "optimized")).strip()
    if model_type not in ("optimized", "baseline"):
        raise JobError(f"model_type must be 'optimized' or 'baseline', got {model_type!r}")
    values["model_type"] = model_type
    return values


def load_split(channel: Path, name: str):
    import pandas as pd

    if not channel.is_dir():
        raise JobError(f"the {name} channel folder is missing: {channel}")
    files = sorted(channel.glob("*.parquet"))
    if not files:
        raise JobError(f"no .parquet file in the {name} channel {channel}: found {sorted(p.name for p in channel.iterdir())}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def metric_lines(metrics: dict[str, Any]) -> list[str]:
    """``name=value`` in the exact shape the tuning job's regexes expect (plain decimals, never 1e-05)."""
    lines = []
    for name in contract.METRIC_NAMES:
        value = metrics.get(name)
        if value is None:
            continue
        lines.append(f"{name}={int(value)}" if name == "rows" else f"{name}={float(value):.6f}")
    return lines


def tuning_job_of(training_job: str | None) -> str | None:
    """A trial is called <tuning job>-<3 digits>-<8 hex>."""
    match = re.match(r"^(.*)-\d{3}-[0-9a-f]{8}$", training_job or "")
    return match.group(1) if match else None


def build_model(model_type: str, y_train, params: dict[str, Any]):
    from financial_risk.models.training import build_baseline_model, build_optimized_model

    if model_type == "baseline":
        return build_baseline_model()
    model = build_optimized_model(y_train)            # scale_pos_weight = negatives / positives of THIS train split
    available = model.get_params()
    chosen = {f"model__{k}": cast(params[k]) for k, cast in XGB_TYPES.items() if k in params}
    unknown = [k for k in chosen if k not in available]
    if unknown:
        say(f"  warning: this model has no parameter(s) {unknown} (xgboost missing?): ignored")
    model.set_params(**{k: v for k, v in chosen.items() if k in available}, **({"model__n_jobs": os.cpu_count() or 2}
                                                                              if "model__n_jobs" in available else {}))
    return model


def main(environ: dict[str, str] | None = None) -> dict[str, Any]:
    import joblib
    import numpy as np
    from sklearn.metrics import average_precision_score

    env = dict(os.environ if environ is None else environ)
    train_dir = Path(env.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    valid_dir = Path(env.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"))
    model_dir = Path(env.get("SM_MODEL_DIR", "/opt/ml/model"))
    output_dir = Path(env.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    job = env.get("TRAINING_JOB_NAME") or _job_from_training_env(env)

    src = add_app_package_to_path(env)
    check_feature_contract()
    from financial_risk.models.training import (FEATURE_COLUMNS, TARGET, artifact_metadata, choose_threshold,
                                                classification_metrics, predict_proba_positive)

    hyper = read_hyperparameters(env)
    model_type = hyper["model_type"]
    say(f"app package   : {src}")
    say(f"training job  : {job}   model type: {model_type}")
    say(f"hyperparameters: {json.dumps({k: v for k, v in hyper.items()}, default=str)}")

    train, valid = load_split(train_dir, "train"), load_split(valid_dir, "validation")
    for name, frame in (("train", train), ("validation", valid)):
        missing = [c for c in FEATURE_COLUMNS + [TARGET] if c not in frame.columns]
        if missing:
            raise JobError(f"the {name} data lacks {missing}: it was built with other features than this image")
    x_train, y_train = train[FEATURE_COLUMNS], train[TARGET].astype(int)
    x_valid, y_valid = valid[FEATURE_COLUMNS], valid[TARGET].astype(int)
    say(f"train {len(x_train):,} rows ({int(y_train.sum()):,} frauds)   validation {len(x_valid):,} rows ({int(y_valid.sum()):,} frauds)")
    if y_train.nunique() < 2 or y_valid.nunique() < 2:
        raise JobError("the train or validation split has one class only: PR-AUC is undefined. Check the cutoff / labels.")

    tracker = Tracker(contract.MLFLOW_EXPERIMENT_TRAIN, job or "train-local", {
        "job_type": "train", "model_type": model_type, "training_job": job or "local", "tuning_job": tuning_job_of(job) or "",
        "source": "finrisk-ml"}, tracking_uri=env.get("MLFLOW_TRACKING_ARN") or env.get("MLFLOW_TRACKING_URI")).start()
    tracker.params({k: v for k, v in hyper.items()})
    status = "FAILED"
    try:
        model = build_model(model_type, y_train, hyper)
        model.fit(x_train, y_train)
        train_pr_auc = float(average_precision_score(y_train, predict_proba_positive(model, x_train)))
        valid_score = predict_proba_positive(model, x_valid)
        threshold = choose_threshold(y_valid, valid_score)
        metrics = classification_metrics(y_valid, valid_score, threshold)
        metrics.update({"model_type": model_type, "train_rows": int(len(x_train)), "valid_rows": int(len(x_valid)),
                        "rows": int(len(x_train)), "train:pr_auc": train_pr_auc, "validation:pr_auc": metrics["pr_auc"],
                        "evaluated_on": "validation", "training_job": job, "hyperparameters": hyper,
                        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})

        model_dir.mkdir(parents=True, exist_ok=True)
        output_path = model_dir / ("model_v1.pkl" if model_type == "baseline" else "model_v2.pkl")
        joblib.dump({"model": model, "threshold": threshold, "feature_columns": FEATURE_COLUMNS, "target": TARGET,
                     "model_type": model_type, "metrics": metrics, "hyperparameters": hyper, **artifact_metadata()}, output_path)
        joblib.dump({"model_path": output_path.name}, model_dir / "latest_model.joblib")     # a file NAME, never a path
        for folder in (model_dir, output_dir):
            folder.mkdir(parents=True, exist_ok=True)
            write_json(folder / "metrics.json", metrics)
        say(f"model written : {output_path}")
        say(f"files in {model_dir} : {sorted(p.name for p in model_dir.iterdir())}")

        tracker.metrics({"train_pr_auc": train_pr_auc, "validation_pr_auc": metrics["pr_auc"], "roc_auc": metrics.get("roc_auc"),
                         "threshold": threshold, "precision": metrics["precision"], "recall": metrics["recall"]})
        tracker.artifact(model_dir / "metrics.json")
        status = "FINISHED"
    finally:
        tracker.end(status)

    # Last, so they are easy to find in the log. SageMaker reads these.
    for line in metric_lines(metrics):
        print(line, flush=True)
    return metrics


def _job_from_training_env(env: dict[str, str]) -> str | None:
    try:
        return json.loads(env.get("SM_TRAINING_ENV", "{}")).get("job_name")
    except ValueError:
        return None


def run(argv: list[str] | None = None) -> int:
    main()
    return 0
