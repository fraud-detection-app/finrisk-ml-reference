"""evaluate: step 3 of the pipeline `finrisk-train`. A Processing job.

The candidate (the best trial of the tuning job) and the champion (the model version in /finrisk/prod/model, if any) are both
scored on the SAME test days. Writing, to the output `evaluation` (s3://<ml>/metrics/<execution id>/):

    evaluation.json      candidate / champion / gate / test window. The pipeline's Condition step reads it (the gate).
    explainability.json  global mean |SHAP| per feature (the SHAP library: what Clarify used) on a 20 000-row sample
    bias.json            the Clarify bias formulas per business segment (finrisk_jobs/bias.py)
    model_card.json      the Model Card content
    register_info.json   strings for the Register step: the card as text, lineage metadata (all <= 256 characters)

and one MLflow run in `finrisk-fraud`. This job NEVER fails because the candidate is bad: a bad candidate is a normal
outcome, decided by the Condition step. It fails only when it cannot measure (no test data, an unreadable champion).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from . import contract
from .bias import bias_report
from .card import build_model_card, card_content
from .common import (JobError, Tracker, add_app_package_to_path, aws, check_feature_contract, error_code, load_yaml, mask,
                     s3_download, say, workdir, write_json)
from .gate import champion_floor, checks_from_evaluation, gate_table

IN = Path("/opt/ml/processing/input")
OUT = Path("/opt/ml/processing/output/evaluation")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="finrisk_jobs evaluate", description=__doc__.split("\n\n")[0])
    p.add_argument("--candidate-job", required=True, help="the top tuning trial: its model.tar.gz is the candidate")
    p.add_argument("--tuning-job-arn", default="")
    p.add_argument("--gate-floor", type=float, default=0.70)
    p.add_argument("--gate-tolerance", type=float, default=0.02)
    p.add_argument("--champion-env", default="prod")
    p.add_argument("--train-image", default=os.environ.get("TRAIN_IMAGE", ""))
    p.add_argument("--trigger-reason", default="manual")
    p.add_argument("--execution-arn", default="")
    p.add_argument("--pipeline-name", default="finrisk-train")
    p.add_argument("--git-sha", default=os.environ.get("ML_GIT_SHA", "unknown"))
    p.add_argument("--ml-bucket")
    p.add_argument("--test-dir", default=str(IN / "test"))
    p.add_argument("--profile-dir", default=str(IN / "profile"))
    p.add_argument("--report-dir", default=str(IN / "report"))
    p.add_argument("--model-dir", default=str(IN / "model"), help="folder holding the candidate's model.tar.gz")
    p.add_argument("--output", default=str(OUT))
    p.add_argument("--shap-rows", type=int, default=20000)
    p.add_argument("--workdir")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------- loading models

def load_artifact(model_dir: Path, strict: bool = True) -> dict[str, Any]:
    """Open an unpacked model package with the app's own loader. ``strict=False`` (a champion trained on other features)
    falls back to the older loader and scores with the champion's OWN feature list."""
    add_app_package_to_path()
    from financial_risk.models.package import ModelPackageError, load_model_package

    try:
        return load_model_package(model_dir)
    except ModelPackageError as exc:
        if strict:
            raise JobError(f"cannot use the model in {model_dir.name}: {exc}") from exc
        from financial_risk.models.evaluation import load_latest_artifact

        say(f"  warning: {exc} -- scoring it with its own feature list")
        return load_latest_artifact(model_dir)


def unpack(tar_path: Path, folder: Path) -> Path:
    add_app_package_to_path()
    from financial_risk.models.package import ModelPackageError, unpack_model_package

    try:
        unpack_model_package(tar_path, folder)
    except ModelPackageError as exc:
        raise JobError(f"{tar_path.name}: {exc}") from exc
    return folder


def find_tarball(folder: Path) -> Path:
    tars = sorted(folder.glob("*.tar.gz")) or sorted(folder.rglob("*.tar.gz"))
    if not tars:
        raise JobError(f"no model.tar.gz in {folder}: the Evaluate step's `model` input is empty")
    return tars[0]


def score_and_measure(artifact: dict[str, Any], frame, label: str) -> tuple[dict[str, Any], Any]:
    from financial_risk.models.training import TARGET, classification_metrics, predict_proba_positive

    columns = list(artifact.get("feature_columns") or contract.FEATURE_COLUMNS)
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise JobError(f"the test data lacks {missing}, which the {label} model needs")
    score = predict_proba_positive(artifact["model"], frame[columns])
    metrics = classification_metrics(frame[TARGET].astype(int), score, float(artifact["threshold"]))
    keep = {k: metrics.get(k) for k in ("pr_auc", "roc_auc", "threshold", "precision", "recall", "f1")}
    keep["confusion_matrix"] = metrics.get("confusion_matrix")
    return keep, score


def find_champion(env: str, group: str, ssm: Any, sm: Any, sts: Any, region: str) -> dict[str, Any] | None:
    """{'version', 'arn', 'model_url'} for the model in /finrisk/<env>/model, or None (nothing promoted yet)."""
    name = f"/finrisk/{env}/model"
    try:
        value = ssm.get_parameter(Name=name)["Parameter"]["Value"]
    except Exception as exc:  # noqa: BLE001
        if error_code(exc) == "ParameterNotFound":
            say(f"  {name} does not exist: no champion")
            return None
        raise
    try:
        pointer = json.loads(value)
    except ValueError as exc:
        raise JobError(f"{name} does not hold JSON: {mask(value[:80])!r}") from exc
    if not isinstance(pointer, dict) or pointer.get("version") is None:
        say(f"  {name} has no version yet: no champion")
        return None
    version = int(pointer["version"])
    arn = pointer.get("model_package_arn") or f"arn:aws:sagemaker:{region}:{sts.get_caller_identity()['Account']}:model-package/{group}/{version}"
    say(f"  champion: version {version} ({mask(arn)})")
    detail = sm.describe_model_package(ModelPackageName=arn)
    return {"version": version, "arn": arn, "model_url": detail["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]}


# ---------------------------------------------------------------------------- explainability

def original_feature(column: str) -> str:
    """numeric__amount -> amount;  categorical__transaction_type_CASH_OUT -> transaction_type."""
    rest = column.split("__", 1)[1] if "__" in column else column
    for name in contract.CATEGORICAL_FEATURES:
        if rest == name or rest.startswith(name + "_"):
            return name
    return rest


def global_shap(model, sample, background, top: int = 10) -> dict[str, Any]:
    """Mean |SHAP| per ORIGINAL feature (one-hot columns are summed back per row first, then the absolute value)."""
    add_app_package_to_path()
    from financial_risk.models.explainability import compute_shap_values, make_explainer

    explainer = make_explainer(model, background)
    values = compute_shap_values(model, sample, None, explainer=explainer)
    grouped = values.T.groupby([original_feature(c) for c in values.columns]).sum().T
    importance = grouped.abs().mean().sort_values(ascending=False)
    ranked = [{"feature": str(k), "mean_abs_shap": float(v)} for k, v in importance.items()]
    return {"method": "SHAP (the library SageMaker Clarify used), TreeSHAP / linear SHAP on the candidate",
            "unit": "log-odds", "sample_rows": int(len(sample)), "background_rows": int(len(background)),
            "top": ranked[:top], "all": {r["feature"]: r["mean_abs_shap"] for r in ranked}}


# ---------------------------------------------------------------------------- the job

def fail_message(pr_auc: float, gate: dict, champion: dict | None) -> str:
    """The message the Fail step WOULD show if the gate said no. It is written into register_info.json whether or not the gate passed,
    so it states the rule and the candidate's number without claiming a verdict: 'GATE FAILED: candidate pr_auc X; needs >= floor 0.70
    and >= champion floor Y (champion v3 0.8800 - 0.02)'. (Only the Fail step shows it, and only when the Condition step said no.)"""
    text = f"GATE FAILED: candidate pr_auc {pr_auc:.4f}; needs >= floor {gate['floor']:.2f}"
    if champion:
        text += (f" and >= champion floor {gate['champion_floor']:.4f} "
                 f"(champion v{champion['version']} {champion['pr_auc']:.4f} - {gate['tolerance']:g})")
    else:
        text += " (no champion yet)"
    return text


def digest_of(image: str) -> str:
    return image.partition("@")[2] or image


def run(argv: list[str] | None = None) -> int:
    import pandas as pd

    a = parse_args(argv)
    cfg = load_yaml("train.yaml")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    work = workdir(a.workdir, "finrisk-evaluate-")
    group = cfg.get("registry", {}).get("group", contract.MODEL_GROUP)
    add_app_package_to_path()
    check_feature_contract()
    from financial_risk.models.training import FEATURE_COLUMNS, TARGET

    say(f"evaluate: candidate {a.candidate_job}; gate floor {a.gate_floor}, tolerance {a.gate_tolerance}")
    report = json.loads((Path(a.report_dir) / "dataset.json").read_text())
    test_files = sorted(Path(a.test_dir).glob("*.parquet"))
    if not test_files:
        raise JobError(f"no test data in {a.test_dir}")
    test = pd.concat([pd.read_parquet(f) for f in test_files], ignore_index=True)
    baseline = pd.read_parquet(sorted(Path(a.profile_dir).glob("*.parquet"))[0])
    splits = report["splits"]
    if test[TARGET].nunique() < 2:
        raise JobError("the test split has one class only: PR-AUC is undefined (cutoff too early?)")
    say(f"  test split: {len(test):,} rows, {int(test[TARGET].sum()):,} frauds, days {splits['test']['days']}")

    # 1. the candidate
    candidate_art = load_artifact(unpack(find_tarball(Path(a.model_dir)), work / "candidate"))
    candidate, cand_score = score_and_measure(candidate_art, test, "candidate")
    candidate["training_job"] = a.candidate_job
    candidate["validation_pr_auc"] = (candidate_art.get("metrics") or {}).get("pr_auc")
    candidate["model_type"] = candidate_art.get("model_type")
    say(f"  candidate  PR-AUC {candidate['pr_auc']:.4f}  ROC-AUC {candidate['roc_auc']:.4f}  "
        f"precision {candidate['precision']:.3f}  recall {candidate['recall']:.3f}  (threshold {candidate['threshold']:.4f})")

    # 2. the champion, on the SAME rows
    sm, ssm, sts, s3 = aws("sagemaker"), aws("ssm"), aws("sts"), aws("s3")
    found = find_champion(a.champion_env, group, ssm, sm, sts, os.environ.get("AWS_REGION", contract.DEFAULT_REGION))
    champion = None
    if found:
        tar = s3_download(found["model_url"], work / "champion.tar.gz", s3)
        art = load_artifact(unpack(tar, work / "champion"), strict=False)
        champion, _ = score_and_measure(art, test, f"champion v{found['version']}")
        champion.update({"version": found["version"], "model_type": art.get("model_type")})
        say(f"  champion v{found['version']}  PR-AUC {champion['pr_auc']:.4f} on the same test days")

    gate = {"floor": a.gate_floor, "tolerance": a.gate_tolerance,
            "champion_floor": champion_floor(champion["pr_auc"] if champion else None, a.gate_tolerance)}
    window = {"from": splits["test"]["from"], "to": splits["test"]["to"], "rows": int(len(test)), "frauds": int(test[TARGET].sum()),
              "days": splits["test"]["days"]}
    evaluation = {"candidate": candidate, "champion": champion, "gate": gate, "test_window": window}
    passed = candidate["pr_auc"] >= gate["floor"] and candidate["pr_auc"] >= gate["champion_floor"]
    for line in gate_table(a.candidate_job, candidate["pr_auc"], checks_from_evaluation(evaluation)):
        say(line)
    say(f"  the gate would {'PASS' if passed else 'FAIL'} (the pipeline's Condition step decides)")

    # 3. explainability and bias
    sample = test.sample(min(a.shap_rows, len(test)), random_state=42)
    background = baseline.sample(min(200, len(baseline)), random_state=42)
    explain = global_shap(candidate_art["model"], sample, background)
    say("  top features (mean |SHAP|): " + ", ".join(f"{t['feature']} {t['mean_abs_shap']:.3f}" for t in explain["top"][:5]))
    bias = bias_report(test.copy(), cand_score, float(candidate_art["threshold"]), baseline["amount"])
    for line in bias["highlights"]:
        say(f"  bias: {line}")

    # 4. MLflow
    tracker = Tracker(contract.MLFLOW_EXPERIMENT_TRAIN, f"evaluate-{report.get('execution_id', 'local')}", {
        "job_type": "evaluate", "candidate_job": a.candidate_job, "trigger_reason": a.trigger_reason, "cutoff": report["cutoff"],
        "pipeline_execution_arn": a.execution_arn, "gate": "PASS" if passed else "FAIL"}).start()
    tracker.params({"cutoff": report["cutoff"], "trigger_reason": a.trigger_reason, "candidate_job": a.candidate_job,
                    "gate_floor": a.gate_floor, "gate_tolerance": a.gate_tolerance, "train_rows": splits["train"]["rows"],
                    "test_days": splits["test"]["days"]})
    tracker.metrics({"candidate_pr_auc": candidate["pr_auc"], "candidate_roc_auc": candidate["roc_auc"],
                     "champion_pr_auc": champion["pr_auc"] if champion else None, "champion_floor": gate["champion_floor"],
                     "gate_passed": 1.0 if passed else 0.0})
    summary = describe_tuning(sm, a.tuning_job_arn)
    if summary:
        tracker.metrics({"tuning_trials": summary["trials"], "tuning_best_objective": summary.get("best_objective")})

    # 5. the files
    card = build_model_card(
        model_type=str(candidate.get("model_type")), training_job=a.candidate_job, image=a.train_image,
        model_artifact=f"s3://{a.ml_bucket or os.environ.get('ML_BUCKET', '<ml bucket>')}/models/{a.candidate_job}/output/model.tar.gz",
        hyperparameters=(candidate_art.get("hyperparameters") or {}), cutoff=report["cutoff"], dataset_uri=report["dataset_uri"],
        splits=splits, candidate=candidate, champion=champion, gate=gate, shap_top=explain["top"], bias_highlights=bias["highlights"],
        test_window=window, mlflow_run_id=tracker.run_id, execution_arn=a.execution_arn)
    content = card_content(card)
    info = {
        "fail_message": fail_message(candidate["pr_auc"], gate, champion),
        "model_card_content": content, "training_job": a.candidate_job, "train_image_digest": digest_of(a.train_image),
        "cutoff": report["cutoff"], "dataset_uri": report["dataset_uri"], "baseline_uri": report["baseline_uri"],
        "pipeline_execution_arn": a.execution_arn or "local", "ml_git_sha": a.git_sha, "trigger_reason": a.trigger_reason,
        "mlflow_run_id": tracker.run_id or "none", "champion_version": str(champion["version"]) if champion else "none",
        "pr_auc": f"{candidate['pr_auc']:.6f}", "champion_pr_auc": f"{champion['pr_auc']:.6f}" if champion else "none",
    }
    for key, value in info.items():
        if key != "model_card_content" and len(str(value)) > 256:
            raise JobError(f"register_info.{key} is {len(str(value))} characters; the registry allows 256")
    files = {"evaluation.json": evaluation, "explainability.json": explain, "bias.json": bias, "model_card.json": card,
             "register_info.json": info}
    for name, payload in files.items():
        write_json(out / name, payload)
        tracker.artifact(out / name, "evaluation")
    tracker.end("FINISHED")
    say(f"evaluate finished: {', '.join(files)} in {out}")
    return 0


def describe_tuning(sm: Any, arn: str) -> dict[str, Any] | None:
    """How the tuning went (best objective, number of trials). Best effort: a missing permission only skips the summary."""
    name = arn.rsplit("/", 1)[-1] if arn else ""
    if not name:
        return None
    try:
        d = sm.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=name)
        best = (d.get("BestTrainingJob") or {}).get("FinalHyperParameterTuningJobObjectiveMetric", {}).get("Value")
        counters = d.get("TrainingJobStatusCounters") or {}
        trials = sum(int(v) for v in counters.values()) if counters else 0
        say(f"  tuning job {name}: {trials} trial(s), best objective {best}")
        return {"name": name, "trials": trials, "best_objective": best, "status": d.get("HyperParameterTuningJobStatus")}
    except Exception as exc:  # noqa: BLE001
        say(f"  (tuning summary skipped: {type(exc).__name__})")
        return None
