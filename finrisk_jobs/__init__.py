"""Code that runs INSIDE SageMaker (training, processing and hosting jobs): `python -m finrisk_jobs <command>`.

Importing this package must stay cheap: no pandas, no AWS client. pipeline/mlops.py (which runs on a GitHub runner with
only boto3 and pyyaml) imports `finrisk_jobs.contract`, `finrisk_jobs.gate` and `finrisk_jobs.featurestore` from here.
"""
