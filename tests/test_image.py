"""The image: Dockerfile contents, the entry point's commands, the files the image copies, the pinned libraries."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from finrisk_jobs import __main__ as entry

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "train" / "Dockerfile").read_text()


def test_dockerfile_builds_from_the_pinned_base_and_sets_the_entrypoint():
    assert re.search(r"^ARG BASE_IMAGE$", DOCKERFILE, re.M) and re.search(r"^FROM \$\{BASE_IMAGE\}$", DOCKERFILE, re.M)
    assert 'ENTRYPOINT ["python", "-m", "finrisk_jobs"]' in DOCKERFILE and re.search(r"^CMD \[\]$", DOCKERFILE, re.M)


def test_dockerfile_copies_exactly_the_code_the_jobs_need():
    assert "COPY finrisk_jobs /opt/program/finrisk_jobs" in DOCKERFILE and "COPY config /opt/program/config" in DOCKERFILE
    assert "COPY pipeline/queries.py /opt/program/pipeline/queries.py" in DOCKERFILE
    assert "PYTHONPATH=/app/src:/opt/program:/opt/program/pipeline" in DOCKERFILE


def test_the_files_the_dockerfile_copies_exist_and_the_jobs_find_them_at_the_same_relative_places():
    from finrisk_jobs import common

    assert (ROOT / "finrisk_jobs" / "__main__.py").is_file() and (ROOT / "pipeline" / "queries.py").is_file()
    assert common.CONFIG_DIR == ROOT / "config" and common.PIPELINE_DIR == ROOT / "pipeline"           # code/config/pipeline side by side, here and in /opt/program
    for name in ("train.yaml", "data.yaml", "monitor.yaml", "smoke.json"):
        assert (ROOT / "config" / name).is_file()


def test_no_code_is_baked_under_opt_ml_which_sagemaker_mounts_over_at_run_time():
    copies = re.findall(r"^COPY\s+\S+\s+(\S+)", DOCKERFILE, re.M)
    assert copies and not [dest for dest in copies if dest.startswith("/opt/ml")], copies
    assert "WORKDIR /opt/ml" not in DOCKERFILE and ":/opt/ml/" not in DOCKERFILE.split("PYTHONPATH=", 1)[1].split("\n", 1)[0]


def test_the_three_extra_libraries_are_pinned_exactly_and_installed_under_a_constraints_file():
    pins = dict(re.findall(r"^([a-z-]+)==([\d.]+)$", (ROOT / "train" / "requirements-image.txt").read_text(), re.M))
    assert set(pins) == {"mlflow-skinny", "sagemaker-mlflow", "evidently"}   # the client only: full mlflow pins pyarrow<26, the app image has 26
    assert "-c /tmp/constraints.txt" in DOCKERFILE and "pip freeze" in DOCKERFILE and "import mlflow, evidently, sagemaker_mlflow" in DOCKERFILE


def test_the_old_trainer_is_gone_and_nothing_still_refers_to_it():
    assert not (ROOT / "train" / "sagemaker_train.py").exists()
    for path in list((ROOT / ".github").rglob("*.yml")) + [ROOT / "README.md", ROOT / "train" / "Dockerfile"]:
        assert "sagemaker_train" not in path.read_text(), path


def test_serve_hands_the_process_to_the_apps_server(monkeypatch):
    calls = []
    monkeypatch.setattr(entry.os, "execvp", lambda file, args: calls.append((file, args)))
    assert entry.main(["serve"]) == 0
    (file, args), = calls
    assert args[1:] == ["-m", "financial_risk.serving.sagemaker_server"] and file == args[0]


def test_unknown_and_missing_commands(capsys):
    assert entry.main(["nope"]) == 2 and "unknown command" in capsys.readouterr().err
    assert entry.main([]) == 2
    assert entry.main(["--help"]) == 0 and "build-dataset" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["train", "backfill", "build-dataset", "evaluate", "monitor"])
def test_every_command_module_has_a_run_function(command):
    import importlib

    module = importlib.import_module(entry.COMMANDS[command])
    assert callable(module.run)


def test_a_job_error_becomes_an_exit_code_and_one_line(monkeypatch, capsys):
    from finrisk_jobs import backfill, common

    monkeypatch.setattr(backfill, "run", lambda argv: (_ for _ in ()).throw(common.JobError("it broke", code=4)))
    assert entry.main(["backfill"]) == 4 and "error: it broke" in capsys.readouterr().out


def test_the_package_import_is_light():
    """pipeline/mlops.py runs on a runner with only boto3 and pyyaml: importing finrisk_jobs.gate/contract/featurestore must not need pandas."""
    import subprocess
    import sys

    code = ("import sys; sys.modules['pandas'] = None; sys.modules['numpy'] = None; "
            "import finrisk_jobs, finrisk_jobs.contract, finrisk_jobs.gate, finrisk_jobs.featurestore; print('ok')")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert done.stdout.strip() == "ok", done.stderr


def test_serve_and_train_are_programs_on_path_and_output_is_unbuffered():
    # serverless endpoints may start `serve` as a program instead of passing it to the ENTRYPOINT (live finding, 20 Oct 2026)
    assert "/usr/local/bin/serve" in DOCKERFILE and "/usr/local/bin/train" in DOCKERFILE
    assert "exec python -m finrisk_jobs serve" in DOCKERFILE and "exec python -m finrisk_jobs train" in DOCKERFILE
    assert "PYTHONUNBUFFERED=1" in DOCKERFILE and "o+rX" in DOCKERFILE


def test_the_model_image_drops_the_lambda_extension_it_inherits_from_the_app_release_image():
    # The app release image has the Lambda Web Adapter in /opt/extensions; a serverless endpoint (Lambda machinery) loads it
    # and fails with no log (live rehearsal, 20-21 Oct 2026). It must be removed after FROM and before anything else runs.
    assert "rm -rf /opt/extensions" in DOCKERFILE
    assert DOCKERFILE.index("rm -rf /opt/extensions") > DOCKERFILE.index("FROM ${BASE_IMAGE}")
    assert DOCKERFILE.index("rm -rf /opt/extensions") < DOCKERFILE.index("\nENTRYPOINT [")
