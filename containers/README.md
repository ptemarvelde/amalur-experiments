# Amalur Docker files and misc container files
## Layout 
 - [`setup.sh`](./setup.sh) Gathers the necessary Amalur project and data files.
 - [`data_char_exp.sh`](./data_char_exp.sh) Shell script to start experiments with synthetic data.
   - Runs amalur-factorization/experiment.py with different number of cores
 - [`Dockerfile.daic`](./Dockerfile.daic) Dockerfile to build container with all Amalur components and CuPy installed. !Run [`setup.sh`](./setup.sh) first!
 - [`start_amalur_container.sh`](./start_amalur_container.sh) Script to start experiment.
