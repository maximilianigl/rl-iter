FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --allow-unauthenticated --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

### From previous Docker template
# Ubuntu Packages
RUN apt-get update && apt-get update -y --allow-unauthenticated && \
    apt-get install -y --allow-unauthenticated \
    software-properties-common \
    apt-utils \
    nano \
    vim \
    man \
    build-essential \
    wget \
    sudo \
    git \
    mercurial \
    mpich \
    qt5-default \
    pkg-config \
    subversion && \
    rm -rf /var/lib/apt/lists/* \
    nvidia-profiler # --no-install-recommends

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH
RUN conda create -y --name pytorch-py36 python=3.6.3 numpy pyyaml scipy ipython mkl
RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

RUN pip install gym gym[atari] gym[mujoco] pandas hashfs pydevd remote_pdb rpdb matplotlib visdom \
    sacred GitPython pymongo tinydb tinydb-serialization pptree progressbar2 ipdb namedlist pyyaml cython \
    pyqt5 mpi4py joblib procgen plumbum pytest atari_py psutil pyprind plotly

RUN pip uninstall tensorflow
RUN pip install tensorflow==1.14.0 tensorflow-gpu==1.14.0
RUN pip install -e git+https://github.com/openai/baselines.git#egg=baselines

WORKDIR /project/
