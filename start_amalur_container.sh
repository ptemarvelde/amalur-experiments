for cores in 8 16 32
do
    echo "Starting amalur parallel with $cores cores"
    docker run --rm --name pepijn_amalur_parallel_cores_$cores --cpus="$cores.0" --env NUM_CORES=$cores --env NUM_REPEATS=30 -v /workspace/Pepijn/amalur_project/results/run2:/user/src/app/amalur-factorization/results pepijn/amalur_parallel
done