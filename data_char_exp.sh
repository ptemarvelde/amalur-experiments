for cores in 4 8 16 32 64
do
    	echo "Starting amalur parallel with $cores cores"
 		docker run --rm --name pepijn_amalur_parallel_cores_$cores --cap-add SYS_ADMIN --cpus="$cores.0" \
     		--env NUM_CORES=$cores --env NUM_REPEATS=30 --env EXPERIMENT_LOG_LEVEL=INFO \
     		-v /workspace/Pepijn/amalur_project/results/gen_data/run2:/user/src/app/amalur-factorization/results \
		-v /workspace/Pepijn/amalur_generated_data/run1_corrected/:/user/data/generated/ \
     		ghcr.io/ptemarvelde/amalur-experiments:latest python experiment.py
done
