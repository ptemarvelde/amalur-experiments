CUPY_PROFILE=True ncu --csv -f --profile-from-start no \
--metric "dram__bytes_read.sum","dram__bytes_write.sum" \
--section "MemoryWorkloadAnalysis" \
--section "SpeedOfLight" \
python3 /user/src/app/amalur-factorization/run_experiment.py flight inner 'LMM' 3 factorized --resultfile resultfile.jsonl