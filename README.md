# finrisk-ml

The **model** side of the financial risk assistant, run as a full MLOps lifecycle on **SageMaker AI**, on the **full PaySim
dataset** (6.36 million transactions): load the data once, train with hyperparameter tuning, compare with the model in
production, gate it, register it with a Model Card, let a person approve it, serve it from endpoints, watch it, and start
over when it drifts. One of four repos (`finrisk-infra`, `finrisk-app`, **`finrisk-ml`**, `finrisk-knowledge`).

What runs in an environment is **three things**, each released by its own pipeline:

| thing | repo | how it is named | how the running app finds it |
| --- | --- | --- | --- |
| the app (code) | `finrisk-app` | image digest `sha256:...` | the service itself |
| the **model** | **this repo** | a **number**: model version 3 in the Model Registry | SSM `/finrisk/<env>/model` -> a SageMaker endpoint |
| the knowledge index | `finrisk-knowledge` | a version `YYYYMMDD-HHMMSS-<sha7>` | SSM `/finrisk/<env>/index` |

The app never restarts for a new model. It re-reads its pointer every minute, sees which **endpoint** serves the version,
and calls that endpoint (`/ready` says `source: endpoint`). That is why a model can be released and rolled back without
building an image.

## The lifecycle in one picture

```
 raw PaySim (S3, versioned, sha256 checked)                           MLflow App: every run, compared
   |  backfill job (Processing, DuckDB)                                    ^        ^            ^
   v                                                                       |        |            |
 Feature Store  finrisk-transactions (offline) --- build-dataset --- Tune --- Evaluate --- Gate --- Register (+ Model Card)
 labels table   finrisk_ml.labels                  (point-in-time)  (HPO)   (vs champion,  |       PendingManualApproval
   ^   ^                                                                    SHAP, bias)   |             | approve (a person)
   |   |                                          Pipeline  finrisk-train                 +-- GateFailed  v
   |   |                                                                                        promote: dev, sit = serverless endpoints
   |   |   feedback: the labels arrive late                                                     pre-prod, prod = real-time + Data Capture,
   |   +--------------------------------- replay: days 18-30 as live traffic                    blue/green with automatic rollback
   |                                       -> the app /api/score -> endpoint -> capture (S3)
   |       EventBridge  <-- CloudWatch alarm  <-- monitor job  (Pipeline  finrisk-monitor)  <-- capture + labels
   +-----  StartPipelineExecution(finrisk-train, TriggerReason=drift)      (+ weekly retrain by EventBridge Scheduler)
```

### Why we wrote our own monitor (and bias and explainability)

**SageMaker Model Monitor and SageMaker Clarify closed to new customers on 30 Jul 2026.** Our account never used them, so none
of their APIs is available. We build what AWS now tells customers to build:

* **Monitoring** = the endpoint's **Data Capture** (still available) + our **monitor job** (`finrisk_jobs/monitor.py`, PSI per
  feature and an Evidently drift report, plus model quality once the delayed labels arrive) in a SageMaker Processing job,
  metrics to **CloudWatch** and **MLflow**, **alarms**, and **EventBridge** to react.
* **Bias** = the standard Clarify formulas (CI, DPL, DPPL, DI, recall and specificity differences) computed in **pandas**
  (`finrisk_jobs/bias.py`). PaySim has no protected attribute, so the class sees the *mechanics* on business segments
  (transaction type, amount quartile).
* **Explainability** = the **SHAP** library, which is what Clarify itself used.

## Pipelines

### `finrisk-train` (SageMaker Pipelines; the definition is plain JSON built by `pipeline/definition.py`)

| step | type | what happens |
| --- | --- | --- |
| `BuildDataset` | Processing | Athena **point-in-time** query (features as known at the cutoff, labels that had arrived by then) -> time splits: test = the last 2 labelled days, validation = the 2 before, train = the rest; a stratified baseline sample; `dataset.json` |
| `Tune` | Tuning | Bayesian search, 8 trials, 4 at a time; maximise `validation:pr_auc`; ranges in `config/train.yaml`; every trial is an MLflow run |
| `Evaluate` | Processing | the best trial (the candidate) and the champion (the version in `/finrisk/prod/model`) scored on the **same test days**; `evaluation.json`, `explainability.json`, `bias.json`, `model_card.json`; one MLflow run |
| `Gate` | Condition | `candidate PR-AUC >= GateFloor` **and** `>= champion PR-AUC - GateTolerance` |
| `Register` | RegisterModel | version N of `finrisk-fraud-model`, `PendingManualApproval`, with the Model Card (Draft), the three reports and the lineage |
| `GateFailed` | Fail | `GATE FAILED: candidate pr_auc X < floor 0.7 or < champion v3 (Y) - 0.02` |

Parameters: `Cutoff` (`auto` = every label that has arrived), `TriggerReason` (`manual|push|schedule|drift`), `ModelType`
(`optimized|baseline`), `MaxTuningJobs`, `ParallelTuningJobs`, `GateFloor` (0.70), `GateTolerance` (0.02), `TrainImage`.

**Why JSON and not the SageMaker SDK?** A step's `Arguments` are the same request you would send to `CreateProcessingJob`,
`CreateHyperParameterTuningJob` or `CreateModelPackage`: "a pipeline is a list of AWS requests with arrows". Building it
with boto3 keeps this repo boto3-only and fully testable offline: `tests/test_definition.py` checks every step against
botocore's input shapes, and (when `sagemaker` 3.x is installed) against the SDK's own pipeline builder.

### `finrisk-monitor`

One Processing step, `Monitor`, started by `monitor.yml` (or by the EventBridge Scheduler rule `finrisk-hourly-monitor`,
disabled by default). Parameters `Env`, `WindowHours`, `Reference`.

### Who starts `finrisk-train`

`train.yml` (a merge that changes `finrisk_jobs/`, `pipeline/`, `train/` or `config/`, or *Run workflow*) -- the EventBridge
Scheduler schedule `finrisk-weekly-retrain` (`TriggerReason=schedule`) -- the EventBridge rule on the drift alarms
(`TriggerReason=drift`) -- a person in SageMaker Studio. There is no `schedule:` in GitHub any more: the schedule lives in AWS.

## The data

| what | where |
| --- | --- |
| raw CSV (sha256 checked, uploaded once) | `s3://<ml>/raw/paysim/v1/paysim.csv` + `.sha256`, `backfill_report.json` |
| history, days 1-17 (5,987,417 rows) | Feature Store group `finrisk-transactions` (offline store; Glue table `finrisk_ml.transactions`) |
| labels | `finrisk_ml.labels` = `s3://<ml>/labels/` (`source=history`, later `source=feedback`) |
| production set, days 18-30 | `s3://<ml>/replay/v1/day=<n>/part.parquet` (features **and** `is_fraud`; only `replay.yml` reads these) |
| training sets | `s3://<ml>/datasets/<execution id>/{train,validation,test,profile}/`, `dataset.json` |
| evaluation files | `s3://<ml>/metrics/<execution id>/` |
| captured traffic | `s3://<ml>/capture/<env>/` |
| monitor reports | `s3://<ml>/monitoring/<env>/<run>/{metrics.json, report.html}` |

`step` becomes a date: step 1 = 2026-09-01 00:00 UTC, one step = one hour; day = `(step - 1) // 24 + 1`. **Day 31 is excluded
everywhere** (272 rows, all fraud: an artefact). `row_id` is the row's 0-based position in the CSV. The model uses
`hour_of_day = (step - 1) % 24`, not `step`: a time index always drifts.

The history is days 1-17; from day 18 the volume falls 10-40 times while fraud stays ~250 a day, so the fraud rate rises
10-40 times. That is the go-live story: the replay sends days 18-22 through the app, the monitor sees `hour_of_day` move
(PSI ~0.5) and the fraud rate rise ~12x, two alarms go off, `finrisk-train` starts by itself -- and the gate and a person
still stand between the new model and prod.

### The Feature Store write (measurement first)

`backfill` writes the history with `BatchWriteRecord` (25 records per call, 32 threads, offline store only). It prints
progress and throughput every 250 000 records, **writes 100 000 first, times them and projects the rest**. If the whole history
would take more than 60 minutes it **stops** (exit code 4) and tells you to run `mlops.py backfill --instance-count N`, which
splits the days over N parallel jobs. It never writes fewer rows silently: `history_days` in `config/data.yaml` is the only way
to shrink the history, and the report says what was written. Cost: about USD 8.5 once per account (1 write unit per record).

### The offline store fills in the background: two waits

A record written with `BatchWriteRecord` becomes queryable in Athena **some minutes later (up to ~15)**. Two places wait for it,
so nothing ever works on a half-filled store:

| where | what it waits for | poll / budget | if it never happens |
| --- | --- | --- | --- |
| **end of the backfill job** (steps 6-7, with the role that can query Athena) | `count(DISTINCT row_id)` in `finrisk_ml.transactions` for its history days >= the records it wrote; prints `visible X of Y` | every 60 s, **30 min** | the job fails naming the delay; no report is written (the records are safe: run it again, it only waits). `backfill_report.json` is written **last**: its existence means "the data is complete and queryable" |
| **`build-dataset`** (the first pipeline step; also covers the drift retrain right after a replay) | labelled rows (`label_time <= cutoff`) that have **no feature row yet**: 0, or at most `completeness.max_missing_share` (0.1 %, `config/train.yaml`) of the labelled rows | every 60 s, **30 min** (`completeness.poll_seconds`, `wait_minutes`) | the step fails (exit 1): "the offline store is still filling: N labelled rows have no features yet; retry in a few minutes" |

## The order for a new account

1. **Pin PR**: put the app image digest in `train/base_image.txt` and merge. `train.yml` runs, builds the image and **upserts** the
   two pipelines -- but there is no data yet, so its run step prints `no data yet: run backfill.yml first` and ends **green**
   (push, schedule and drift triggers; a manual run fails with the same words).
2. **`backfill.yml`** with `then_train` (the default): the job loads the data (30-60 min, mostly the Feature Store write), waits
   until Athena sees it, writes `backfill_report.json`; then the workflow writes the pipelines again and starts `finrisk-train`
   with `TriggerReason=backfill`. It does not follow the run; the log prints the execution ARN, `python pipeline/mlops.py status <arn>`
   and where to look in SageMaker Studio.
3. **The first pipeline run** (~20-30 min) -> version 1 in the registry (`PendingManualApproval`) -> `approve-model.yml` ->
   `promote-model.yml`. Later runs (a merge, the weekly schedule, a drift alarm, or *Run workflow*) start straight away because the
   report exists.

## The gate

`config/train.yaml` holds the numbers. A candidate must pass **both** rules on the test days:

1. `PR-AUC >= 0.70` (the floor), and
2. `PR-AUC >= champion - 0.02`, where the **champion** is the model whose version is in `/finrisk/prod/model`, **scored on the
   same test days** (not the number it had when it was registered).

No champion yet: only the floor applies. A retrained model is **not** guaranteed to beat the champion: if the gate refuses
v2, that is the lesson, not a bug. Re-run with `ModelType=baseline` (a logistic regression) to watch it say no.

## Serving: endpoints

| env | endpoint | data capture | update |
| --- | --- | --- | --- |
| dev, sit | **serverless** 3072 MB (a new account's per-endpoint limit), concurrency 5 | no (serverless cannot) | replaced in place |
| pre-prod, prod | **real-time** `ml.t2.medium` x1 | 100 %, input and output, JSON -> `s3://<ml>/capture/<env>/` | **blue/green**, `ALL_AT_ONCE` with a 120 s **bake** while we run one instance (AWS refuses a `CANARY` on a single instance; with 2+ instances the config switches to canary 50 % for 120 s); old fleet kept 120 s; **automatic rollback** if `finrisk-<env>-endpoint-5xx` or `-latency` goes off during the bake |

`promote-model.yml` (`dev -> sit -> pre-prod -> prod`, reviewers on the last two) runs, per environment:

`CreateModel` -> `CreateEndpointConfig` -> `CreateEndpoint` (first time) or `UpdateEndpoint` -> wait `InService` (a rollback is
detected and reported) -> **smoke test** (the 3 rows in `config/smoke.json`: HTTP 200 and the version we just deployed) -> write
the pointer -> wait until the app's `/ready` reports `source: endpoint` and the version -> delete configs and models older than
the two newest (the rollback target stays). **Rolling back is the same workflow with an older version.**

### The pointer `/finrisk/<env>/model`

```json
{"version": 3, "serving": "endpoint", "endpoint": "finrisk-prod-fraud", "endpoint_config": "finrisk-prod-fraud-v3-20261021031500",
 "model_package_arn": "arn:aws:sagemaker:...:model-package/finrisk-fraud-model/3", "status": "Approved",
 "promoted_at": "2026-10-21T03:20:00+00:00", "run": "https://github.com/<org>/finrisk-ml/actions/runs/123"}
```

`{"version": null}` = nothing promoted: the app serves its baked model. The old shape (`model_data_url`, no `serving`) is rejected
by the app.

## The image: one image trains, evaluates, monitors and serves

`train/Dockerfile` starts FROM the app image pinned by digest in `train/base_image.txt` (bumping it is a pull request), so
training and serving share the feature and model code. It adds `finrisk_jobs/`, `config/`, `pipeline/queries.py` and three
pinned libraries (`train/requirements-image.txt`: mlflow, sagemaker-mlflow, evidently) installed under a constraints file made
from what the app image already has -- if they would change numpy, pandas or scikit-learn, the build **fails**.

`python -m finrisk_jobs <command>`:

| command | who calls it |
| --- | --- |
| `train` | SageMaker (every tuning trial; hyperparameters from SageMaker, prints `train:pr_auc=` and `validation:pr_auc=`) |
| `serve` | SageMaker hosting: replaced by `python -m financial_risk.serving.sagemaker_server` (finrisk-app, port 8080) |
| `backfill`, `build-dataset`, `evaluate`, `monitor` | Processing jobs |

The model package (`model.tar.gz`, flat) holds `model_v2.pkl` (`model_v1.pkl` for the baseline), `latest_model.joblib` and
`metrics.json`. **No feature table**: six million rows do not belong in a model package.

## The monitor

Reads the captured requests of the last `WindowHours`, compares each feature (PSI, quantile bins from the reference sample of
the serving model) and the score with the training data, joins the delayed labels, and puts nine metrics in CloudWatch
(`FinRisk/Model`, dimension `Env`): `MaxFeaturePSI`, `DriftedFeatureShare`, `ScorePSI`, `FlaggedRate`, `LiveFraudRate`,
`FraudRateRatio`, `LivePrAuc`, `LiveRecall`, `LabelledRows`. It exits 0 either way: drift is information, not a job failure.

The prod alarms (Terraform, `finrisk-infra`) are on the **largest** PSI, not on "the share of drifting features" (one feature of
fourteen moves in the simulated go-live; a share would never fire): `finrisk-prod-model-drift` (`MAX(MaxFeaturePSI, ScorePSI) >=
0.25`), `finrisk-prod-model-prior-shift` (`FraudRateRatio >= 5`), `finrisk-prod-model-quality` (live PR-AUC < 0.70, only with at
least 500 labelled rows). Thresholds: `config/monitor.yaml`.

Evidently is used for the HTML drift report when it is installed (it is, in the image); if not, the monitor writes its own table.

## Layout

```
VERSION
config/                      train.yaml (tuning ranges, instances, gate, splits, endpoints)   data.yaml (url, sha256, calendar, history days)
                             monitor.yaml (thresholds, window)                                 smoke.json (3 fixed rows: 1 fraud, 2 legit)
finrisk_jobs/                runs INSIDE SageMaker: __main__ (dispatch) backfill build_dataset train evaluate monitor bias card
                             common (S3, Athena, MLflow-safe logging) featurestore (BatchWriteRecord) contract (columns, calendar) gate
pipeline/                    mlops.py (the CLI) definition.py (the two pipelines) deploy.py (endpoints) queries.py (the SQL)
                             replay.py feedback.py errors.py
train/                       Dockerfile  base_image.txt  requirements-image.txt
scripts/ensure_train_image.sh   build + push the image unless this commit's image exists (train.yml, backfill.yml)
.github/workflows/           ml-ci  backfill  train  approve-model  promote-model  replay  monitor  teardown
tests/                       pytest with a fake AWS (no network, no cost); the SQL runs on DuckDB; the jobs run on small parquet files
```

## Run it yourself

Everything prints each AWS call before it makes it (account ids are shown as `<account-id>`).

```bash
pip install -r requirements.txt                    # boto3 >= 1.43.37 and pyyaml
python pipeline/mlops.py list                      # the versions in the registry (read-only)
python pipeline/mlops.py show --version 1
python pipeline/mlops.py upsert --image <repo>@sha256:...        # write the two pipelines
python pipeline/mlops.py backfill --image <digest> --then-train  # load the data, wait for Athena, upsert, start finrisk-train (not followed)
python pipeline/mlops.py run --trigger-reason manual             # start finrisk-train and follow it (needs the backfill report: exit 1 without it)
python pipeline/mlops.py status <execution id>
python pipeline/mlops.py approve --version 1 --decision approve
python pipeline/mlops.py promote --env dev --version 1
python pipeline/mlops.py mlflow-url                # a presigned link to the MLflow App UI
```

`run` first looks for `s3://<ml>/raw/paysim/v1/backfill_report.json`. Without it a `push`, `schedule` or `drift` trigger prints
`no data yet: run backfill.yml first` (in the log and the job summary) and exits **0** without starting anything; a `manual` run
exits **1** with the same words.

Commands: `backfill`, `upsert`, `run`, `status`, `list`, `show`, `approve`, `promote`, `replay`, `feedback`, `monitor`,
`mlflow-url`, `teardown`. Names come from the environment, then from SSM (`/finrisk/shared/*`), then from the naming convention
(`ML_BUCKET`, `SAGEMAKER_ROLE_ARN`, `MODEL_PACKAGE_GROUP`, `AWS_REGION`). Exit codes: `0` ok, `1` failed, `2` the model version
cannot be used here (not found, Rejected, or not Approved for pre-prod/prod), `3` the gate said no, `4` (backfill job only) too
slow for one instance.

### Who may promote what

| environment | accepts |
| --- | --- |
| `dev`, `sit` | any version that is not Rejected |
| `pre-prod`, `prod` | **Approved** versions only (exit code 2 otherwise) |

## GitHub setup (once; `finrisk-infra/scripts/github_bootstrap.sh` does it)

The repo is **public**, so values that contain the AWS account id are **secrets**; everything else is a variable. No AWS key
exists anywhere: each job exchanges GitHub's OIDC token for one role, and the role's trust names the exact Environment.

| name | kind | where | used by |
| --- | --- | --- | --- |
| `AWS_REGION` | variable | repository | all workflows (`ap-southeast-2`) |
| `AWS_ROLE_ARN` | secret | Environment `ml` | `backfill.yml`, `train.yml`, `monitor.yml` (role `finrisk-ml-train`) |
| `AWS_ROLE_ARN` | secret | Environment `model-approval` (reviewer) | `approve-model.yml` (role `finrisk-ml-approve`) |
| `AWS_ROLE_ARN` | secret | Environment `dev` / `sit` | `promote-model.yml`, `teardown.yml` (roles `finrisk-ml-promote-dev` / `-sit`) |
| `AWS_ROLE_ARN` | secret | Environment `pre-prod` / `prod` (reviewers) | the same two workflows (roles `finrisk-ml-promote-pre-prod` / `-prod`) |
| `AWS_ROLE_ARN` | secret | Environment `simulation` | `replay.yml` (role `finrisk-ml-feedback`: reads `replay/`, writes feedback labels and Feature Store records, nothing else) |

A job **with** `environment:` reads the Environment secret `AWS_ROLE_ARN`; every job in this repo has an Environment. Every
Environment accepts deployments from `main` only. `main` is protected: a pull request and a green `test` check (`ml-ci.yml`) are
required.

### Before the first run

`train/base_image.txt` holds a placeholder digest. Copy the digest of the `finrisk-app` image that passed `dev` (ECR ->
`finrisk-app` -> the image -> "Copy digest"), put it on the `finrisk-app@sha256:...` line and open a pull request. The workflows
refuse to run with the placeholder, and say so. The order is then: **pin PR -> `train.yml` (upserts only) -> `backfill.yml` with
`then_train` -> the first pipeline run** (see "The order for a new account" above).

## What this costs (Sydney, 20 Oct 2026 prices)

| item | estimate |
| --- | --- |
| Feature Store backfill, ~6.0 M records x 1 write unit | **~USD 8.5, once per account** (`history_days` can shrink it: a trade-off) |
| Backfill processing (`ml.m5.xlarge`, 30-60 min) | ~USD 0.15-0.30 |
| One `finrisk-train` run: dataset + 8 tuning trials (~5 min each) + evaluate | ~USD 0.25-0.35 |
| Real-time endpoints pre-prod + prod (`ml.t2.medium`, ~3 h, blue/green doubles for minutes) | ~USD 0.45 |
| Serverless dev + sit, monitor runs, Athena, MLflow (free) | cents |

Delete real-time endpoints when you are done: `teardown.yml` per environment, **then** the infra destroy.

## Tests

```bash
pip install -r requirements-dev.txt
FINRISK_APP_SRC=../finrisk-app/src python -m pytest     # most tests need the app's source (financial_risk); without it they skip
```

The tests need `boto3 >= 1.43.37` (the Feature Store `BatchWriteRecord`). Every AWS request the code builds is validated
against botocore's input shape by the fakes in `tests/fakes.py`. The Model Card is validated against AWS's own schema (`tests/fixtures/model_card_cfn_schema.json`, the CloudFormation schema of
`AWS::SageMaker::ModelCard`; needs `jsonschema`, a dev-only dependency): `CreateModelPackage` refuses an invalid card at the very end of a run.
Optional extras: `mlflow`/`evidently` (the versions in
`train/requirements-image.txt`; the matching tests skip or fall back without them) and the SageMaker SDK v3 (`sagemaker`,
`sagemaker-mlops`; set `FINRISK_SDK_PYTHON` to an interpreter that has it) for the pipeline-structure parity test.
