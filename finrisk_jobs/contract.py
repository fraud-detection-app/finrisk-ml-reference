"""The contracts between the repos, in one pure-Python place (no pandas, no boto3).

* FEATURE_COLUMNS is the model's input, in order. It is a CONTRACT with finrisk-app (`financial_risk.models.training`):
  the endpoint, the replay and the monitor all use these names. tests/test_contract.py compares it with the app's
  list whenever the app's source is available, and the training job refuses to start if the two differ.
* The calendar maps PaySim's `step` (hour 1-743) onto 1-30 Sep 2026.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

CATEGORICAL_FEATURES = ["transaction_type"]
NUMERIC_FEATURES = [
    "amount", "log_amount", "origin_old_balance", "origin_new_balance", "destination_old_balance",
    "destination_new_balance", "origin_balance_delta", "destination_balance_delta", "amount_to_origin_balance",
    "hour_of_day", "tx_index_for_origin", "customer_tx_count_so_far", "customer_amount_so_far",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "is_fraud"

# The Feature Store record: identifier, event time, metadata, then the numeric model inputs.
RECORD_IDENTIFIER = "row_id"
EVENT_TIME = "event_time"
INTEGRAL_FEATURES = ("step", "hour_of_day", "tx_index_for_origin", "customer_tx_count_so_far")
FEATURE_GROUP_COLUMNS = ["row_id", "event_time", "step", "transaction_type", "origin_customer", *NUMERIC_FEATURES]
STRING_FEATURES = ("row_id", "event_time", "transaction_type", "origin_customer")

CALENDAR_START = datetime(2026, 9, 1, tzinfo=timezone.utc)      # step 1
LABEL_DELAY_DAYS = 2
EXCLUDED_DAYS = (31,)

FEATURE_GROUP = "finrisk-transactions"
GLUE_DATABASE = "finrisk_ml"
FEATURES_TABLE = "transactions"
LABELS_TABLE = "labels"
ATHENA_WORKGROUP = "finrisk"
MODEL_GROUP = "finrisk-fraud-model"
MLFLOW_EXPERIMENT_TRAIN = "finrisk-fraud"
MLFLOW_EXPERIMENT_MONITOR = "finrisk-monitoring"
CLOUDWATCH_NAMESPACE = "FinRisk/Model"
DEFAULT_REGION = "ap-southeast-2"
ENVIRONMENTS = ("dev", "sit", "pre-prod", "prod")

# The labels table (Glue, parquet). The column order is the table's.
LABEL_COLUMNS = ["row_id", "is_fraud", "event_time", "label_time", "source", "batch_id"]

# Metric names printed by the training job (name=value). The Tuning step turns two of them into HPO metrics.
HPO_OBJECTIVE = "validation:pr_auc"
METRIC_NAMES = ("train:pr_auc", "validation:pr_auc", "roc_auc", "threshold", "precision", "recall", "rows")

# The metric names the monitor puts into CloudWatch (namespace FinRisk/Model, dimension Env).
MONITOR_METRICS = ("MaxFeaturePSI", "DriftedFeatureShare", "ScorePSI", "FlaggedRate", "LiveFraudRate",
                   "FraudRateRatio", "LivePrAuc", "LiveRecall", "LabelledRows")


def day_of_step(step: int) -> int:
    """Calendar day 1-31 of a PaySim step: (step - 1) // 24 + 1."""
    return (int(step) - 1) // 24 + 1


def hour_of_step(step: int) -> int:
    return (int(step) - 1) % 24


def event_time_of_step(step: int) -> str:
    """step 1 -> 2026-09-01T00:00:00Z. One step is one hour. This exact text is the Feature Store's event time."""
    return format_time(CALENDAR_START + timedelta(hours=int(step) - 1))


def format_time(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def label_time_of(event_time: str, delay_days: int = LABEL_DELAY_DAYS) -> str:
    """When the analysts confirm the label: the event time plus the analyst delay."""
    return format_time(parse_time(event_time) + timedelta(days=delay_days))


def parse_days(text: str | int | list[int]) -> list[int]:
    """'1-17' -> [1..17]; '1,3,5-6' -> [1,3,5,6]; excluded days are NOT removed here (callers decide)."""
    if isinstance(text, int):
        return [text]
    if isinstance(text, (list, tuple)):
        return sorted({int(d) for d in text})
    days: set[int] = set()
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            if int(lo) > int(hi):
                raise ValueError(f"bad day range {part!r}: the first number must not be larger than the second")
            days.update(range(int(lo), int(hi) + 1))
        else:
            days.add(int(part))
    if not days:
        raise ValueError(f"no days in {text!r}")
    return sorted(days)


def endpoint_name(env: str) -> str:
    return f"finrisk-{env}-fraud"
