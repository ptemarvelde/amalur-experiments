#!/bin/bash
su ubuntu
BUCKET_NAME=amalur-thesis-pepijn
BUCKET_MOUNTPOINT=/home/ubuntu/data

# Select CUDA 12.1
export PATH=/usr/local/cuda/bin/:/usr/local/cuda/include:$PATH
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-12.1 /usr/local/cuda

# Pull experiments image from github container registry
GITHUB_CONTAINER_PAT=$(aws secretsmanager get-secret-value --region eu-central-1 --secret-id pepijn-amalur-thesis --query SecretString --output text | jq -r .ptemarvelde_github_container_pat)
echo "$GITHUB_CONTAINER_PAT" | docker login ghcr.io -u ptemarvelde --password-stdin
docker pull ghcr.io/ptemarvelde/amalur-daic:latest &

echo "alias run_experiment_example='docker run --env CUPY_PROFILE=True --rm --gpus all --cap-add SYS_ADMIN ghcr.io/ptemarvelde/amalur-daic:latest \
ncu --csv --profile-from-start no \
--metric "dram__bytes_read.sum","dram__bytes_write.sum" \
--section "MemoryWorkloadAnalysis" \
--section "SpeedOfLight" \
python3 /user/src/app/amalur-factorization/run_experiment.py flight inner 'LMM' 3 factorized'" >> /home/ubuntu/.bashrc

# Mount Datasets using S3fs-Fuse
sudo apt update
sudo apt install -y s3fs
mkdir -p "$BUCKET_MOUNTPOINT"
s3fs "$BUCKET_NAME" "$BUCKET_MOUNTPOINT" -o iam_role="auto" -o allow_other -o umask=000