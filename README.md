# Cost Estimation for Factorized Machine Learning Experiments
This repo is a wrapper containing configuration used for running the experiments for my thesis, [Cost Estimation for Factorized Machine Learning](http://resolver.tudelft.nl/uuid:b4255726-12e3-466d-96a2-0ab1421c20a3)
The LaTeX project can be found [here](https://github.com/ptemarvelde/MSc-CS-Thesis---Pepijn-te-Marvelde), other miscelleneous figures are grouped in a [third git repository](https://github.com/ptemarvelde/msc-thesis-figures).

:warning: This repo was not created with portability and collaboration in mind. However, I have tried to keep it somewhat logical, especially for reproducing the visualisations shown in my thesis.


## Repo layout
 - [`amalur-cost-estimation`](https://github.com/delftdata/amalur-cost-estimation) Amalur Cost Estimator (submodule for [amalur-cost-estimation](https://github.com/delftdata/amalur-cost-estimation))
 - [`amalur-data-generator`](https://github.com/delftdata/amalur-data-generato) Amalur Data Generator (sumbodule for [amalur-data-generator](https://github.com/delftdata/amalur-data-generator))
 - [`amalur-factorization`](https://github.com/delftdata/amalur-factorization) Amalur Factorization (submodule for [amalur-factorization](https://github.com/delftdata/amalur-factorization))
 - [`containers`](./containers) Dockerfiles and scripts to build image for running experiments
 - [`experiments`](./experiments) Misc files used for running the experiments
 - [`MorpheusPy`](./MorpheusPy) Used for Hamlet datasets
 - [`results`](./results/full_1) Experiment results. Also contains the full analysis and related jupyter notebooks
 - [`sparse_dot`](./sparse_dot) Used for parallel dot product
 
Github action pipeline is configured to build an amalur Docker image.
 - Created in github action flow by first running [`containers/setup.sh`](./conatiners/setup.sh) and then `docker build`

Image is tagged: `ghcr.io/ptemarvelde/amalur-daic:latest`
