for cores in 4 8 16 32 64
do
    echo "Starting amalur parallel with $cores cores"
    docker run --rm --name pepijn_amalur_parallel_cores_$cores --cpus="$cores.0" \
      --env NUM_CORES=$cores --env NUM_REPEATS=30 EXPERIMENT_LOG_LEVEL=INFO\
      -v /workspace/Pepijn/amalur_project/results/profile1:/user/src/app/amalur-factorization/results \
      ghcr.io/ptemarvelde/amalur-experiments:latest python profile_experiment.py
done