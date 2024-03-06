Runtime results from synthetic, hamlet, tpcai datasets for:
- 1080
- 2080
- a40
- p100
- v100
- st4 for CPU
Experiments ran on DAIC, ST4


 - daic
    - runtime
        - cpu
            - synthetic/merged.jsonl 8,16,32 cores on ST4 synthetic (dataset manually fixed by `sed 's|/user/data/generated/|/mnt/data/synthetic/sigmod_extended/|g' -i *.log`)