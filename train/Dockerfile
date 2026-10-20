# The finrisk-ml image = the app's own image + the jobs. ONE image trains, evaluates, monitors AND serves.
#
# Why start FROM the app image? Training, evaluation and serving must use the SAME feature code and the SAME model code, or a
# model that scores well in training can score differently in the endpoint. The app image already has financial_risk on
# PYTHONPATH (/app/src) and every library installed (pandas, scikit-learn, xgboost, shap, duckdb, fastapi ...).
# This file adds only what this repo owns: the job code, its config, the SQL, and three libraries.
#
# Build (the workflows do this; the build context is the repo root):
#   docker build -f train/Dockerfile \
#     --build-arg BASE_IMAGE=<registry>/$(grep -v '^#' train/base_image.txt | head -n 1) \
#     -t finrisk-train:local .
#
# BASE_IMAGE is pinned by DIGEST in train/base_image.txt (a tag can move, a digest cannot). Taking a newer app image is a
# pull request that changes that one line: reviewed, tested by the next pipeline run, and easy to revert.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# 1. The three libraries. A CONSTRAINTS file made from what the app image already has pins numpy, pandas, scikit-learn, xgboost,
#    shap ... where they are: if mlflow or evidently would need different versions, the build FAILS here, loudly, instead of
#    silently changing the libraries the model was trained with.
COPY train/requirements-image.txt /tmp/requirements-image.txt
RUN pip freeze | grep -i -E '^(numpy|pandas|scikit-learn|scipy|xgboost|shap|pyarrow|duckdb|joblib|boto3|botocore|pydantic|fastapi|starlette|uvicorn|httpx)==' > /tmp/constraints.txt \
 && pip install --no-cache-dir -c /tmp/constraints.txt -r /tmp/requirements-image.txt \
 && python -c "import mlflow, evidently, sagemaker_mlflow; print('mlflow', mlflow.__version__)" \
 && rm -f /tmp/requirements-image.txt /tmp/constraints.txt

# 2. The code, in /opt/program: NOT under /opt/ml, which SageMaker owns and mounts its own volumes over at run time (input,
#    model, output, processing/...): code baked there can vanish (found by a local run with /opt/ml mounted, 20 Oct 2026;
#    AWS's own bring-your-own-container examples use /opt/program for the same reason). finrisk_jobs/ is the package (python -m finrisk_jobs <command>), config/
#    holds the yaml files it reads, and pipeline/queries.py is the SQL (written once; mlops.py and the jobs share it).
COPY finrisk_jobs /opt/program/finrisk_jobs
COPY config /opt/program/config
COPY pipeline/queries.py /opt/program/pipeline/queries.py

# The app's package stays importable (/app/src); our code and the SQL join it.
ENV PYTHONPATH=/app/src:/opt/program:/opt/program/pipeline PYTHONUNBUFFERED=1
WORKDIR /opt/program

# NO LAMBDA EXTENSION in the model image. The app's release image carries the Lambda Web Adapter at /opt/extensions (the app
# itself needs it on its own Lambda in dev and sit). A SERVERLESS endpoint runs on Lambda's machinery too, and Lambda loads every
# file in /opt/extensions before the container starts: the adapter took over and every serverless endpoint failed after ~5
# minutes with only "Request to service failed" and no log at all, while the SAME image and model went InService on a
# real-time endpoint (live rehearsal, 20-21 Oct 2026). The same image without /opt/extensions went InService on serverless and
# answered in 1.0 s. The model image does not need the adapter: SageMaker talks to our server on port 8080 itself.
#
# Also: `serve` and `train` as real programs on PATH (either start form works: `docker run <image> serve` reaches our
# ENTRYPOINT, a runtime that starts `serve` itself finds the program), and every file readable by any user (o+rX).
RUN rm -rf /opt/extensions \
 && printf '#!/bin/sh\nexec python -m finrisk_jobs serve "$@"\n' > /usr/local/bin/serve \
 && printf '#!/bin/sh\nexec python -m finrisk_jobs train "$@"\n' > /usr/local/bin/train \
 && chmod 755 /usr/local/bin/serve /usr/local/bin/train \
 && chmod -R o+rX /opt/program /app

# SageMaker starts a training container with the argument `train` and a hosting container with `serve`; Processing jobs pass
# the command and its options as ContainerArguments (pipeline/definition.py). `serve` replaces itself with the app's own
# server (financial_risk.serving.sagemaker_server): the code that scores on a laptop scores on SageMaker.
ENTRYPOINT ["python", "-m", "finrisk_jobs"]
# The app image's CMD (start the web server) must not become arguments of our entrypoint.
CMD []
