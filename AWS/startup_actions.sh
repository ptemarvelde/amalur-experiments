#!/bin/bash

GITHUB_CONTAINER_PAT=
BUCKET_NAME=
BUCKET_MOUNTPOINT=
S3_KEY="<access_key>:<secret_key>"
S3_KEYFILE=/etc/.passwd-s3fs

# Select CUDA 12.1
export PATH=/usr/local/cuda/bin/:/usr/local/cuda/include:$PATH
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-12.1 /usr/local/cuda

# Pull experiments image from github container registry
echo "$GITHUB_CONTAINER_PAT" | docker login ghcr.io -u ptemarvelde --password-stdin
docker pull ghcr.io/ptemarvelde/amalur-cupy:latest &

# Mount Datasets using S3fs-Fuse

apt update
apt install -y s3fs
echo $S3_KEY> $S3_KEYFILE
chmod 600 $S3_KEYFILE

s3fs "$BUCKET_NAME" "$BUCKET_MOUNTPOINT" -o passwd_file=$S3_KEYFILE