#!/bin/bash
apptainer shell -c --nv \
--bind \
/tudelft.net/staff-umbrella/ThesisPepijnFactorizedML/data:/mnt/data,\
"$PWD":"$HOME",\
"$PWD/experiment_config.py":/user/src/app/amalur-factorization/experiment_config.py \
--env USE_GPU=True \
--env USE_MKL=False \
--env NUM_REPEATS=30 \
--env DATA_GLOB_PATH=/mnt/data/synthetic/sigmod_extended/*/*/*/data/ \
/tudelft.net/staff-umbrella/ThesisPepijnFactorizedML/images/amalur_daic.sif


#Apptainer> python3 /user/src/app/amalur-factorization/experiment.py
apptainer shell -c --nv --bind /tudelft.net/staff-umbrella/ThesisPepijnFactorizedML/data:/mnt/data,/tudelft.net/staff-umbrella/ThesisPepijnFactorizedML/experiments/testing/:$HOME /tudelft.net/staff-umbrella/ThesisPepijnFactorizedML/images/amalur_daic.sif


#!/bin/bash
apptainer run -c --nv \
--bind \
/tudelft.net/staff-umbrella/ThesisPepijnFactorizedML/data:/mnt/data,\
"$PWD":"$HOME",\
"$PWD/experiment_config.py":/user/src/app/amalur-factorization/experiment_config.py \
--env USE_GPU=True \
--env USE_MKL=False \
--env NUM_REPEATS=30 \
--env DATA_GLOB_PATH=/mnt/data/synthetic/sigmod_extended/*/*/*/data/ \
/tudelft.net/staff-umbrella/ThesisPepijnFactorizedML/images/amalur_daic.sif python3 /user/src/app/amalur-factorization/experiment.py