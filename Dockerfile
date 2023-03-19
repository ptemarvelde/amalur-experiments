FROM continuumio/miniconda3:latest

RUN apt update
RUN apt-get update
RUN apt-get install -y git

RUN conda install -y Cython matplotlib numpy pandas plotly scipy setuptools tqdm
RUN conda install -y -c intel mkl

WORKDIR /user/src/app

# Move all files into place
COPY amalur-factorization amalur-factorization
RUN pip install ./amalur-factorization

COPY MorpheusPy/examples/data amalur-factorization/data/Hamlet

COPY amalur-cost-estimation amalur-cost-estimation
RUN pip install ./amalur-cost-estimation
RUN cp -r amalur-cost-estimation/estimator amalur-factorization/

COPY sparse_dot sparse_dot
RUN pip install ./sparse_dot

WORKDIR  /user/src/app/amalur-factorization
RUN mkdir results
CMD ["python", "hamlet_experiments_hardware.py"]

#docker build -t amalur_parallel .
#ssh wenbosun@st4.ewi.tudelft.nl