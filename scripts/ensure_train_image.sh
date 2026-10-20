#!/usr/bin/env bash
# Build and push finrisk-train:sha-<this commit> FROM the app image pinned in train/base_image.txt -- unless this commit's
# image is already in ECR. Writes TRAIN_IMAGE=<repo>@<digest> to $GITHUB_ENV (and prints it) for the steps after it.
#
# Used by train.yml and backfill.yml (both run the same image). Needs: AWS credentials, docker, and these variables:
#   AWS_REGION  REGISTRY (<account>.dkr.ecr.<region>.amazonaws.com)  GITHUB_SHA  GITHUB_ENV  RUNNER_TEMP
set -euo pipefail

PIN="$(grep -v '^[[:space:]]*#' train/base_image.txt | grep -v '^[[:space:]]*$' | head -n 1)"
if [[ ! "$PIN" =~ ^[a-z0-9._/-]+@sha256:[0-9a-f]{64}$ ]]; then
  echo "::error title=Bad base image pin::train/base_image.txt must hold <repository>@sha256:<64 hex>, got '$PIN'"
  exit 1
fi
if [[ "$PIN" == *@sha256:0000000000000000000000000000000000000000000000000000000000000000 ]]; then
  echo "::error title=Base image not pinned yet::train/base_image.txt still has the placeholder digest. Copy the digest of the finrisk-app image from ECR and change the line with a pull request."
  exit 1
fi
TAG="sha-${GITHUB_SHA}"
REPO_URI="${REGISTRY}/finrisk-train"
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$REGISTRY"
if DIGEST="$(aws ecr describe-images --repository-name finrisk-train --image-ids "imageTag=$TAG" --query 'imageDetails[0].imageDigest' --output text 2>"$RUNNER_TEMP/ecr.err")"; then
  echo "finrisk-train:$TAG already exists ($DIGEST): not rebuilding"
elif grep -q ImageNotFoundException "$RUNNER_TEMP/ecr.err"; then
  echo "building finrisk-train:$TAG FROM $PIN"
  # linux/amd64: ml.m5.xlarge is an Intel machine. --provenance=false: one plain image manifest, no extras.
  docker build --platform linux/amd64 --provenance=false -f train/Dockerfile --build-arg "BASE_IMAGE=${REGISTRY}/${PIN}" -t "${REPO_URI}:${TAG}" .
  docker push "${REPO_URI}:${TAG}"
  DIGEST="$(aws ecr describe-images --repository-name finrisk-train --image-ids "imageTag=$TAG" --query 'imageDetails[0].imageDigest' --output text)"
else
  cat "$RUNNER_TEMP/ecr.err"
  exit 1
fi
echo "TRAIN_IMAGE=${REPO_URI}@${DIGEST}" >> "$GITHUB_ENV"
echo "training image: finrisk-train@${DIGEST}"
