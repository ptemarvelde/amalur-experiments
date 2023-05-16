FROM cupy/cupy:v12.0.0

RUN apt update
RUN apt install -y git nano

RUN pip3 install Cython matplotlib numpy pandas plotly scipy setuptools tqdm
    # conda install -y -c intel mkl

WORKDIR /user/src/app

# Move all files into place

COPY MorpheusPy/examples/data amalur-factorization/data/Hamlet
ENV DATA_ROOT=/user/src/app/amalur-factorization/data/Hamlet

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
RUN mkdir -p results
CMD ["python", "hamlet_experiments_hardware.py"]
