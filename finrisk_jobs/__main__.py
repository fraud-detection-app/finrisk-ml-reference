"""`python -m finrisk_jobs <command> [options]` -- the one entry point of the image.

    train          SageMaker TRAINING (also what every hyperparameter-tuning trial runs)      finrisk_jobs/train.py
    serve          SageMaker HOSTING: the app's own server (finrisk-app), on port 8080
    backfill       Processing job: raw data -> S3 -> Feature Store history + labels + production set
    build-dataset  Processing step 1 of finrisk-train: the point-in-time dataset (Athena) and its time splits
    evaluate       Processing step 3 of finrisk-train: candidate vs champion, SHAP, bias, model card, MLflow
    monitor        the only step of finrisk-monitor: drift + live quality from the captured traffic

SageMaker starts a training container with the argument `train` and a hosting container with `serve`; Processing jobs
pass the command and its options as ContainerArguments (pipeline/definition.py).
"""

from __future__ import annotations

import importlib
import os
import sys

COMMANDS = {
    "train": "finrisk_jobs.train",
    "backfill": "finrisk_jobs.backfill",
    "build-dataset": "finrisk_jobs.build_dataset",
    "evaluate": "finrisk_jobs.evaluate",
    "monitor": "finrisk_jobs.monitor",
}
SERVE_MODULE = "financial_risk.serving.sagemaker_server"


def serve() -> None:
    """Replace this process with the app's hosting server (so SageMaker's signals reach it directly)."""
    os.execvp(sys.executable, [sys.executable, "-m", SERVE_MODULE])


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 2
    command, rest = argv[0], argv[1:]
    if command == "serve":
        serve()
        return 0
    if command not in COMMANDS:
        print(f"error: unknown command {command!r}. Commands: serve, " + ", ".join(COMMANDS), file=sys.stderr)
        return 2
    from .common import JobError, say

    module = importlib.import_module(COMMANDS[command])
    try:
        code = module.run(rest)
    except JobError as exc:
        say(f"error: {exc}")
        return exc.code
    return int(code or 0)


if __name__ == "__main__":
    sys.exit(main())
