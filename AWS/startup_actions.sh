#!/bin/bash
GITHUB_CONTAINER_PAT=
BUCKET_NAME=pepijn-test-bucket
BUCKET_MOUNTPOINT=${HOME}/data

# Select CUDA 12.1
export PATH=/usr/local/cuda/bin/:/usr/local/cuda/include:$PATH
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-12.1 /usr/local/cuda

# Pull experiments image from github container registry
GITHUB_CONTAINER_PAT=$(aws secretsmanager get-secret-value --region eu-central-1 --secret-id pepijn-test-gpu --query SecretString --output text | jq -r .ptemarvelde_github_container_pat)
echo "$GITHUB_CONTAINER_PAT" | docker login ghcr.io -u ptemarvelde --password-stdin
docker pull ghcr.io/ptemarvelde/amalur-cupy:latest &

# Mount Datasets using S3fs-Fuse
sudo apt update
sudo apt install -y s3fs
mkdir -p "$BUCKET_MOUNTPOINT"
s3fs "$BUCKET_NAME" "$BUCKET_MOUNTPOINT" -o iam_role="auto"