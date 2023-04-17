FROM continuumio/miniconda3:latest

RUN apt update && apt install linux-perf -y && cp /usr/bin/perf_5.10 /usr/bin/perf_5.4
RUN apt-get update
RUN apt-get install -y git

RUN conda install -y Cython matplotlib numpy pandas plotly scipy setuptools tqdm
RUN conda install -y -c intel mkl

WORKDIR /user/src/app

# Move all files into place

COPY MorpheusPy/examples/data amalur-factorization/data/Hamlet

COPY sparse_dot sparse_dot
RUN pip install ./sparse_dot


COPY amalur-factorization/requirements.txt requirements.txt
RUN pip install -r requirements.txt


COPY amalur-cost-estimation amalur-cost-estimation
RUN pip install ./amalur-cost-estimation
RUN cp -r amalur-cost-estimation/estimator amalur-factorization/

COPY amalur-data-generator amalur-data-generator
RUN pip install ./amalur-data-generator
RUN cp -r amalur-data-generator/generator amalur-factorization

COPY amalur-factorization amalur-factorization
RUN ls amalur-factorization
RUN pip install ./amalur-factorization

WORKDIR  /user/src/app/amalur-factorization
RUN mkdir results
CMD ["python", "hamlet_experiments_hardware.py"]

#docker build -t amalur_parallel .
#ssh wenbosun@st4.ewi.tudelft.nl