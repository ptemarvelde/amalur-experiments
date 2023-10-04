#!/bin/bash
docker build -f Dockerfile.daic -t amalur:daic .
dir=$(pwd)
cd ~/Documents/uni/y5/thesis/daic
apptainer build --force amalur_daic.sif docker-daemon://amalur:daic
du -sh ./*
echo "Syncing images dir"
paplay /usr/share/sounds/sound-icons/prompt.wav
rsync -rvh ~/Documents/uni/y5/thesis/daic/ daic:/tudelft.net/staff-umbrella/ThesisPepijnFactorizedML/images/
cd $dir