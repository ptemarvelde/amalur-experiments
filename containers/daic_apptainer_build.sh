#!/bin/bash
# Conver the docker image to a singularity image and sync to daic
docker pull ghcr.io/ptemarvelde/amalur-daic:latest
dir=$(pwd)
cd ~/Documents/uni/y5/thesis/daic
apptainer build --force amalur_daic.sif docker-daemon://ghcr.io/ptemarvelde/amalur-daic:latest
du -sh ./*
echo "Syncing images dir"
paplay /usr/share/sounds/sound-icons/prompt.wav
rsync -rvh ~/Documents/uni/y5/thesis/daic/ daic:/tudelft.net/staff-umbrella/ThesisPepijnFactorizedML/images/
cd $dir