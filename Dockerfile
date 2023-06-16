FROM cupy/cupy:v12.0.0

RUN apt update
RUN apt install -y git nano
ENV CUDA_VERSION=11.7
# install cuda nsight tools
RUN apt-get update && \
	apt-get install -y linux-headers-$(uname -r) wget && \
	wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
	dpkg -i cuda-keyring_1.0-1_all.deb && \
	apt-get update && \
	DEBIAN_FRONTEND=noninteractive apt install -y intel-mkl &&\
	DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-${CUDA_VERSION} || apt -y --fix-broken install

RUN update-alternatives --set cuda /usr/local/cuda-${CUDA_VERSION}
ENV LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
ENV PATH=/usr/local/cuda-${CUDA_VERSION}/bin${PATH:+:${PATH}}

RUN pip3 install Cython matplotlib numpy pandas plotly scipy setuptools tqdm mkl

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

ENV USE_GPU=True
ENV USE_MKL=False


CMD ["python", "hamlet_experiments_hardware.py"]
