"""The eight workflows and the repo's own files: names, triggers, chain order, one role per Environment, no secrets echoed."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from conftest import GENERATED_DIRS, repo_files

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
NAMES = ["ml-ci", "backfill", "train", "approve-model", "promote-model", "replay", "monitor", "teardown"]
AWS_NAMES = [n for n in NAMES if n != "ml-ci"]
ENVS = ["dev", "sit", "pre-prod", "prod"]
LOGIN = "aws-actions/configure-aws-credentials"


def load(name: str) -> dict:
    data = yaml.safe_load((WORKFLOWS / f"{name}.yml").read_text())
    data["on"] = data.pop(True, data.get("on"))              # YAML reads a bare `on:` as the boolean True
    return data


def text(name: str) -> str:
    return (WORKFLOWS / f"{name}.yml").read_text()


def steps(job: dict) -> list[dict]:
    return job["steps"]


def run_text(job: dict) -> str:
    return "\n".join(s.get("run", "") for s in steps(job))


def step_index(job: dict, needle: str) -> int:
    for i, step in enumerate(steps(job)):
        if needle in step.get("name", "") or needle in step.get("run", ""):
            return i
    raise AssertionError(f"no step mentioning {needle!r}")


def login_step(job: dict) -> dict:
    return next(s for s in steps(job) if s.get("uses", "").startswith(LOGIN))


def test_exactly_these_workflows_exist_and_parse():
    assert sorted(p.stem for p in WORKFLOWS.glob("*.yml")) == sorted(NAMES)
    for name in NAMES:
        assert load(name)["jobs"]


def test_every_job_runs_on_the_pinned_runner_image():
    for name in NAMES:
        for job_id, job in load(name)["jobs"].items():
            assert job["runs-on"] == "ubuntu-24.04", f"{name} job {job_id}: pin the runner (ubuntu-24.04), not a moving label like ubuntu-latest"


# ---------------------------------------------------------------- ml-ci


def test_ml_ci_runs_pytest_in_a_job_called_test_on_prs_and_main():
    wf = load("ml-ci")
    assert set(wf["on"]) == {"pull_request", "push"} and wf["on"]["push"]["branches"] == ["main"]
    assert list(wf["jobs"]) == ["test"] and wf["jobs"]["test"]["name"] == "test"
    assert any("pytest" in s.get("run", "") for s in steps(wf["jobs"]["test"]))
    assert "id-token" not in str(wf["permissions"]) and wf["permissions"] == {"contents": "read"}   # CI touches no AWS


def test_ml_ci_checks_out_the_app_so_the_job_tests_can_run():
    job = load("ml-ci")["jobs"]["test"]
    repos = [s["with"].get("repository") for s in steps(job) if s.get("uses", "").startswith("actions/checkout") and "with" in s]
    assert "fraud-detection-app/finrisk-app" in repos
    pytest_step = next(s for s in steps(job) if "pytest" in s.get("run", ""))
    assert pytest_step["env"]["FINRISK_APP_SRC"].endswith("/finrisk-app/src")
    assert "requirements-dev.txt" in run_text(job)


# ---------------------------------------------------------------- train


def test_train_triggers_have_no_schedule_any_more():
    on = load("train")["on"]
    assert set(on) == {"workflow_dispatch", "push"}
    assert on["push"]["branches"] == ["main"] and on["push"]["paths"] == ["finrisk_jobs/**", "pipeline/**", "train/**", "config/**"]
    assert not re.search(r"^\s*schedule:", text("train"), re.M)             # the schedule lives in AWS (EventBridge Scheduler) now


def test_train_uses_the_ml_environment_and_its_role_secret():
    job = load("train")["jobs"]["train"]
    assert job["environment"] == "ml" and job["permissions"]["id-token"] == "write"
    assert login_step(job)["with"]["role-to-assume"] == "${{ secrets.AWS_ROLE_ARN }}" and login_step(job)["with"]["aws-region"] == "${{ vars.AWS_REGION }}"


def test_train_masks_the_account_id_right_after_login():
    job = load("train")["jobs"]["train"]
    login = next(i for i, s in enumerate(steps(job)) if s.get("uses", "").startswith(LOGIN))
    assert "::add-mask::" in steps(job)[login + 1]["run"] and "sts get-caller-identity" in steps(job)[login + 1]["run"]


def test_train_steps_run_in_order_image_then_upsert_then_run_and_nothing_overrides_the_gate():
    job = load("train")["jobs"]["train"]
    order = [step_index(job, n) for n in ("Training image", "mlops.py upsert", "mlops.py run")]
    assert order == sorted(order) and len(set(order)) == 3
    for step in steps(job)[order[1]:]:                         # a failed gate (exit 3) must stop the job: nothing may override it
        assert "continue-on-error" not in step and "if" not in step
    run = steps(job)[order[2]]["run"]
    assert "--trigger-reason" in run and "$GITHUB_STEP_SUMMARY" in run
    assert job["env"]["TRIGGER_REASON"] == "${{ github.event_name == 'push' && 'push' || 'manual' }}"


def test_train_passes_the_image_by_digest_and_builds_with_the_default_builder():
    job = load("train")["jobs"]["train"]
    assert '--image "$TRAIN_IMAGE"' in run_text(job) and "setup-buildx" not in text("train") and "docker/build-push-action" not in text("train")
    assert "scripts/ensure_train_image.sh" in run_text(job)


def test_the_image_script_builds_pushes_by_digest_and_refuses_the_placeholder():
    script = (ROOT / "scripts" / "ensure_train_image.sh").read_text()
    assert "describe-images" in script and "ImageNotFoundException" in script and "docker build" in script and "docker push" in script
    assert "sha-${GITHUB_SHA}" in script and "finrisk-train" in script and "ecr get-login-password" in script and "0" * 64 in script
    assert "TRAIN_IMAGE=${REPO_URI}@${DIGEST}" in script and "--platform linux/amd64" in script and "--provenance=false" in script
    assert "set -euo pipefail" in script and (ROOT / "scripts" / "ensure_train_image.sh").stat().st_mode & 0o111


# ---------------------------------------------------------------- backfill


def test_backfill_is_manual_runs_in_ml_and_passes_its_inputs_through_env():
    wf = load("backfill")
    assert set(wf["on"]) == {"workflow_dispatch"} and set(wf["on"]["workflow_dispatch"]["inputs"]) == {"days", "instance_count", "then_train"}
    job = wf["jobs"]["backfill"]
    assert job["environment"] == "ml" and job["env"]["DAYS"] == "${{ inputs.days }}" and job["env"]["INSTANCE_COUNT"] == "${{ inputs.instance_count }}"
    run = run_text(job)
    assert "mlops.py backfill" in run and "--instance-count" in run and "ensure_train_image.sh" in run
    then_train = wf["on"]["workflow_dispatch"]["inputs"]["then_train"]
    assert then_train["type"] == "boolean" and then_train["default"] is True and job["env"]["THEN_TRAIN"] == "${{ inputs.then_train }}"
    assert '[ "$THEN_TRAIN" = true ]' in run and "--then-train" in run
    assert step_index(job, "Training image") < step_index(job, "mlops.py backfill")


# ---------------------------------------------------------------- approve


def test_approve_model_inputs_environment_and_role():
    wf = load("approve-model")
    inputs = wf["on"]["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"version", "decision", "note"} and inputs["decision"]["options"] == ["approve", "reject"]
    job = wf["jobs"]["decide"]
    assert job["environment"] == "model-approval" and login_step(job)["with"]["role-to-assume"] == "${{ secrets.AWS_ROLE_ARN }}"
    run = run_text(job)
    assert "mlops.py approve" in run and "[0-9]+" in run     # the version is checked to be a number first


# ---------------------------------------------------------------- promote


def test_promote_model_chain_order_environments_and_per_env_roles():
    wf = load("promote-model")
    assert list(wf["jobs"]) == [f"promote-{e}" for e in ENVS]
    assert wf["on"]["workflow_dispatch"]["inputs"]["version"]["required"] is True
    previous = None
    for env in ENVS:
        job = wf["jobs"][f"promote-{env}"]
        assert job["environment"] == env and job.get("needs") == previous
        assert login_step(job)["with"]["role-to-assume"] == "${{ secrets.AWS_ROLE_ARN }}"
        assert f"mlops.py promote --env {env} --version" in run_text(job)
        assert job["timeout-minutes"] >= 45                  # endpoint InService budget (25 min) + smoke + the /ready wait (10 min)
        assert any("::add-mask::" in s.get("run", "") for s in steps(job))
        previous = f"promote-{env}"
    assert wf["concurrency"]["group"] == "promote-model" and wf["concurrency"]["cancel-in-progress"] is False


# ---------------------------------------------------------------- replay


def test_replay_runs_both_jobs_in_the_simulation_environment_traffic_first():
    wf = load("replay")
    assert list(wf["jobs"]) == ["traffic", "feedback"] and wf["jobs"]["feedback"]["needs"] == "traffic"
    assert set(wf["on"]["workflow_dispatch"]["inputs"]) == {"from_day", "to_day", "env", "batch", "label_delay"}
    for job in wf["jobs"].values():
        assert job["environment"] == "simulation" and login_step(job)["with"]["role-to-assume"] == "${{ secrets.AWS_ROLE_ARN }}"
        assert "requirements-sim.txt" in run_text(job)
    assert "mlops.py replay" in run_text(wf["jobs"]["traffic"]) and "mlops.py feedback" in run_text(wf["jobs"]["feedback"])
    assert "--label-delay-minutes" in run_text(wf["jobs"]["feedback"])
    assert wf["on"]["workflow_dispatch"]["inputs"]["env"]["options"] == ["prod"]           # the feedback role can read prod's app_url only


# ---------------------------------------------------------------- monitor, teardown


def test_monitor_runs_the_pipeline_in_the_ml_environment():
    job = load("monitor")["jobs"]["monitor"]
    assert job["environment"] == "ml" and "mlops.py monitor --env" in run_text(job) and "WINDOW_HOURS" in job["env"]
    assert load("monitor")["on"]["workflow_dispatch"]["inputs"]["env"]["options"] == ["prod", "pre-prod"]


def test_teardown_needs_the_environment_typed_and_uses_that_environments_role():
    wf = load("teardown")
    inputs = wf["on"]["workflow_dispatch"]["inputs"]
    assert inputs["env"]["options"] == ENVS and inputs["confirm"]["required"] is True
    job = wf["jobs"]["teardown"]
    assert job["environment"] == "${{ inputs.env }}"                                         # pre-prod and prod bring their reviewers with them
    assert step_index(job, "confirmation") < steps(job).index(login_step(job)) and '[ "$CONFIRM" = "$TARGET_ENV" ]' in steps(job)[0]["run"]
    assert 'mlops.py teardown --env "$TARGET_ENV" --confirm "$CONFIRM"' in run_text(job)
    assert wf["concurrency"]["group"] == "promote-model"                                     # never tear down during a promotion


# ---------------------------------------------------------------- every workflow


def test_every_role_assuming_job_has_an_environment_and_reads_the_one_secret_name():
    """The course rule: a job WITH `environment:` reads secrets.AWS_ROLE_ARN (an Environment secret)."""
    for name in AWS_NAMES:
        for job in load(name)["jobs"].values():
            logins = [s for s in steps(job) if s.get("uses", "").startswith(LOGIN)]
            assert len(logins) == 1 and job.get("environment"), name
            assert logins[0]["with"]["role-to-assume"] == "${{ secrets.AWS_ROLE_ARN }}"
            assert job["permissions"]["id-token"] == "write"
        assert set(re.findall(r"secrets\.(\w+)", text(name))) == {"AWS_ROLE_ARN"}
    assert not re.search(r"AWS_ML_\w*ROLE_ARN", "\n".join(text(n) for n in NAMES))      # the old per-role names are gone


def test_the_environments_used_are_the_documented_ones():
    used = {name: {str(j["environment"]) for j in load(name)["jobs"].values()} for name in AWS_NAMES}
    assert used == {"backfill": {"ml"}, "train": {"ml"}, "monitor": {"ml"}, "approve-model": {"model-approval"}, "replay": {"simulation"},
                    "promote-model": set(ENVS), "teardown": {"${{ inputs.env }}"}}


@pytest.mark.parametrize("name", NAMES)
def test_actions_are_pinned_to_the_course_majors(name):
    allowed = {"actions/checkout": "v7", "actions/setup-python": "v7", "aws-actions/configure-aws-credentials": "v6"}
    for action, version in re.findall(r"uses:\s*([\w./-]+)@([\w.]+)", text(name)):
        assert allowed.get(action) == version, (action, version)


@pytest.mark.parametrize("name", NAMES)
def test_no_secret_is_echoed_or_traced(name):
    body = text(name)
    for line in body.splitlines():
        code = line.split("#", 1)[0]
        if "secrets." in code:
            assert "role-to-assume" in code, f"a secret used somewhere other than role-to-assume: {line.strip()}"
    assert "set -x" not in body and "env | sort" not in body and "printenv" not in body
    assert not re.search(r"echo[^\n]*(SECRET|TOKEN|PASSWORD)", body, re.I)
    assert "AWS_SECRET_ACCESS_KEY" not in body and "AWS_ACCESS_KEY_ID" not in body   # OIDC only: no long-lived keys


@pytest.mark.parametrize("name", NAMES)
def test_user_input_never_goes_straight_into_a_shell_script(name):
    """`${{ inputs.x }}` inside run: is script injection. Inputs reach scripts through env: only."""
    for job in load(name)["jobs"].values():
        for step in steps(job):
            assert "${{" not in step.get("run", "") or "inputs." not in step["run"], step.get("name")
            assert "github.event" not in step.get("run", "")


@pytest.mark.parametrize("name", AWS_NAMES)
def test_the_secrets_table_at_the_top_lists_exactly_the_names_the_file_uses(name):
    body = text(name)
    header = "\n".join(line for line in body.splitlines() if line.startswith("#"))
    used = set(re.findall(r"secrets\.(\w+)", body)) | set(re.findall(r"vars\.(\w+)", body))
    for variable in used:
        assert re.search(rf"^#\s+{variable}\b", header, re.M), f"{variable} is used but not in the table at the top of {name}.yml"


@pytest.mark.parametrize("name", NAMES)
def test_workflows_use_bash_and_least_privilege(name):
    wf = load(name)
    assert wf["defaults"]["run"]["shell"] == "bash" and wf["permissions"] == {"contents": "read"}
    for job in wf["jobs"].values():
        assert job.get("permissions", {}).get("id-token", "none") in ("write", "none") and "timeout-minutes" in job


@pytest.mark.parametrize("name", AWS_NAMES)
def test_every_workflow_that_installs_python_packages_uses_a_requirements_file_of_this_repo(name):
    for job in load(name)["jobs"].values():
        for line in run_text(job).splitlines():
            if "pip install" in line:
                assert re.search(r"-r requirements(-sim)?\.txt", line), line


def test_every_mlops_command_the_workflows_call_exists():
    import mlops

    commands = set(mlops.COMMANDS)
    for name in AWS_NAMES:
        code = "\n".join(run_text(job) for job in load(name)["jobs"].values())
        for command in re.findall(r"mlops\.py (\w[\w-]*)", code):
            assert command in commands, (name, command)


# ---------------------------------------------------------------- the repo's own files


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", (ROOT / "VERSION").read_text().strip())


def test_base_image_pin_is_a_digest_with_no_registry_address():
    lines = [l for l in (ROOT / "train" / "base_image.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
    assert len(lines) == 1 and re.fullmatch(r"finrisk-app@sha256:[0-9a-f]{64}", lines[0])
    assert "pull request" in (ROOT / "train" / "base_image.txt").read_text().lower()


ACCOUNT_ID = re.compile(r"(?<!\d)\d{12}(?!\d)")
SECRET_KEY = re.compile(r"AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,}")
LONG_HEX = re.compile(r"[0-9a-f]{32,}")             # an image digest (sha256:<64 hex>) or a commit sha can contain 12 digits in a row


def leaks(root=ROOT):
    """(file, what) for every 12-digit number or key-like string in the files git tracks (or would track)."""
    found = []
    for path in repo_files(root):
        if path.suffix in (".parquet", ".pkl", ".joblib") or path.name in ("fakes.py", "helpers.py") or path.name.startswith("test_"):
            continue                                        # the tests use a fake 123456789012 on purpose
        content = path.read_text(errors="ignore")
        rel = path.relative_to(root).as_posix()
        if ACCOUNT_ID.search(LONG_HEX.sub("", content)):
            found.append((rel, "12-digit number"))
        if SECRET_KEY.search(content):
            found.append((rel, "key-like string"))
    return found


def test_no_account_id_or_key_is_committed_anywhere():
    """Only files git tracks (or would) are scanned: not .git/ (a runner's .git holds fetched refs), .venv/ or caches."""
    assert leaks() == []


def test_the_scan_ignores_generated_folders_and_digests_but_still_sees_a_leak(tmp_path):
    for folder in sorted(GENERATED_DIRS):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "FETCH_HEAD").write_text("123456789012\n")
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "base_image.txt").write_text("finrisk-app@sha256:ab123456789012cd" + "e" * 48 + "\n")   # 12 digits inside a digest
    (tmp_path / "finrisk-app" / ".git").mkdir(parents=True)                      # ml-ci checks finrisk-app out into a subfolder
    (tmp_path / "finrisk-app" / "artifact.py").write_text('ACCOUNT = "123456789012"\n')
    assert leaks(tmp_path) == []
    assert [p.name for p in repo_files(tmp_path)] == ["base_image.txt"]
    (tmp_path / "config.yaml").write_text("role: arn:aws:iam::123456789012:role/x\n")
    assert leaks(tmp_path) == [("config.yaml", "12-digit number")]


def test_gitignore_keeps_models_data_and_secrets_out():
    ignored = (ROOT / ".gitignore").read_text()
    for pattern in ("*.pkl", "*.parquet", ".env"):
        assert pattern in ignored


def test_the_smoke_file_is_json_not_ignored():
    assert (ROOT / "config" / "smoke.json").is_file() and "*.json" not in (ROOT / ".gitignore").read_text()
