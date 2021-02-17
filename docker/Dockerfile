FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
ARG PYTHON_VERSION=3.6

# Set the time zone correctly
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV SHELL /bin/bash

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libjpeg-dev \
         tmux \
         nano \
         xterm \
         rsync \
         zip \
         zsh \
         htop \
         screen \
         zlib1g-dev \
         libcurl3-dev \
         libfreetype6-dev \
         libpng12-dev \
         libzmq3-dev \
         libpng-dev \
         libglib2.0-0 \
         openssh-server \
         sudo \
         && apt-get clean \
         && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


RUN curl -o ~/miniconda.sh -L -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda100 && \
     /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH
RUN pip install ninja tqdm
RUN conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
RUN conda install numpy pyyaml scipy ipython mkl mkl-include cython typing
RUN pip install jupyterlab

RUN conda create -n pytorch_cpu python=3.6
RUN . /opt/conda/etc/profile.d/conda.sh && \
        conda activate pytorch_cpu && \
        conda install pytorch torchvision torchaudio cpuonly -c pytorch && \
        pip install tqdm pillow requests jupyterlab && \
        conda install numpy pyyaml scipy ipython mkl mkl-include cython typing

RUN pip install matplotlib
RUN  . /opt/conda/etc/profile.d/conda.sh && \
        conda activate pytorch_cpu && \
        pip install matplotlib

#RUN pip install tensorflow-gpu==1.15 lucid
#RUN  . /opt/conda/etc/profile.d/conda.sh && \
#        conda activate pytorch_cpu && \
#        pip install tensorflow==1.15 lucid

RUN pip install git+https://github.com/VLL-HD/FrEIA.git typing_extensions sklearn -q
RUN  . /opt/conda/etc/profile.d/conda.sh && \
        conda activate pytorch_cpu && \
        pip install git+https://github.com/VLL-HD/FrEIA.git typing_extensions sklearn -q

RUN conda install faiss-gpu -c pytorch
RUN  . /opt/conda/etc/profile.d/conda.sh && \
        conda activate pytorch_cpu && \
        conda install faiss-cpu -c pytorch


### add tmux config
COPY tmux.conf /etc/

RUN conda clean --all -y
RUN rm -rf ~/.cache/pip

# Enable passwordless sudo for all users
RUN echo '%sudo ALL=(ALL:ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
         && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Setup gosu (https://github.com/tianon/gosu)
# gosu is an improved version of su which behaves better inside docker
# we use it to dynamically switch to the desired user in the entrypoint
# (see below)
ENV GOSU_VERSION 1.10
RUN set -x \
 && dpkgArch="$(dpkg --print-architecture | awk -F- '{ print $NF }')" \
 && wget -O /usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-$dpkgArch" \
 && chmod +x /usr/local/bin/gosu \
 && gosu nobody true

COPY entrypoint.sh /usr/local/bin/
RUN chmod a+x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD ["/bin/bash"]