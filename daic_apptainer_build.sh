#!/bin/bash
docker build -f Dockerfile.daic -t amalur:daic .
dir=$(pwd)
cd ~/Documents/uni/y5/thesis/daic
apptainer build amalur_daic.sif docker-daemon://amalur:daic
echo "Syncing images dir"
rsync -rvh ~/Documents/uni/y5/thesis/daic/ daic:/tudelft.net/staff-umbrella/ThesisPepijnFactorizedML/images/
cd $dir