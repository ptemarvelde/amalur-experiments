FROM continuumio/miniconda3:latest

RUN apt update && apt install linux-perf -y && cp /usr/bin/perf_5.10 /usr/bin/perf_5.4
RUN apt-get update
RUN apt-get install -y git

RUN conda install -y Cython matplotlib numpy pandas plotly scipy setuptools tqdm
RUN conda install -y -c intel mkl

WORKDIR /user/src/app

# Move all files into place
COPY sparse_dot sparse_dot
RUN pip install ./sparse_dot

COPY amalur-cost-estimation amalur-cost-estimation
RUN pip install ./amalur-cost-estimation
RUN cp -r amalur-cost-estimation/estimator amalur-factorization/

COPY MorpheusPy/examples/data amalur-factorization/data/Hamlet

COPY amalur-factorization amalur-factorization
RUN pip install ./amalur-factorization

WORKDIR  /user/src/app/amalur-factorization
RUN pip install -r requirements.txt
RUN mkdir results
CMD ["python", "hamlet_experiments_hardware.py"]

#docker build -t amalur_parallel .
#ssh wenbosun@st4.ewi.tudelft.nl

#--table
#--repeat
#--no-aggr?
#-j (json)
#
#--metric-only
#--per-core
#
#
##stat record/statreport
#
#Some additional info on perf
#
#Use perf stat -a <pid> to measure a default list of counters including instructions and cycles (and their ratio, instructions per cycle: IPC).
#perf can also be used to measure many more counters. See this example to add measurement on cache misses, see also man perf-list and man perf-stat
#perf stat -a -e task-clock,cycles,instructions,branches,branch-misses -e stalled-cycles-frontend,stalled-cycles-backend -e cache-references,cache-misses -e LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses -e L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses -p <pid>
#
#
#perf stat -x "\t" -e task-clock,cycles,instructions,branches,branch-misses -e stalled-cycles-frontend,stalled-cycles-backend -e cache-references,cache-misses -e L1-dcache-loads,L1-dcache-load-misses -r 100 sleep 0.001
#perf stat --metric-only -x "\t" -e task-clock,cycles,instructions,branches,branch-misses -e stalled-cycles-frontend,stalled-cycles-backend -e cache-references,cache-misses -e L1-dcache-loads,L1-dcache-load-misses sleep 0.001
#perf stat -x '|' -e cache-references,cache-misses sleep 1