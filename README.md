# Wrapper around amalur project repos to ease running of experiments

## Repo layout
 - [`results`](./results) experiment results
 - [`Dockerfile`](./Dockerfile)
 - [`data_char_exp.sh`](./data_char_exp.sh) Shell script to start experiments with synthetic data.
   - Runs amalur-factorization/experiment.py with different number of cores
 - [`Dockerfile`](./Dockerfile) Dockerfile to build container with all Amalur components installed. !Run [`setup.sh`](./setup.sh) first!
 - [`setup.sh`](./setup.sh) Gathers the necessary Amalur project and data files.
 - [`start_amalur_container.sh`](./start_amalur_container.sh) Script to start experiment.

Github action pipeline is configured to build an amalur Docker image.
 - Created in github action flow by first running [`setup.sh`](./setup.sh) and then `docker build`

Image is tagged: ghcr.io/ptemarvelde/amalur-experiments-latest