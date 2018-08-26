FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04
LABEL maintainer funkej@janelia.hhmi.org

# basic ubuntu packages

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        git \
        python-numpy \
        libmlpack-dev && \
    rm -rf /var/lib/apt/lists/*

# install conda (which is unfortunately required for z5)

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
ENV CONDA_SHLVL=1
ENV PATH=/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV CONDA_PYTHON_EXE=/opt/conda/bin/python
ENV CONDA_DEFAULT_ENV=base
ENV CONDA_EXE=/opt/conda/bin/conda
ENV CONDA_PREFIX=/opt/conda
ENV CONDA_PROMPT_MODIFIER=(base)

# install dependencies

RUN conda install -y tensorflow-gpu cython
RUN conda install -y -c conda-forge -c cpape \
  z5py \
  numpy \
  scipy \
  h5py \
  requests \
  retrying \
  scikit-image \
  dask distributed \
  luigi \
  mahotas \
  pymongo

# dependencies for lsd

ENV MALIS_ROOT=/src/malis
ENV MALIS_REPOSITORY=https://github.com/TuragaLab/malis.git
ENV MALIS_REVISION=2206fe01bd2d10c3bc6a861897820731d1ae131b

WORKDIR ${MALIS_ROOT}
RUN git clone ${MALIS_REPOSITORY} . && \
    git checkout ${MALIS_REVISION}
RUN python setup.py build_ext --inplace
ENV PYTHONPATH ${MALIS_ROOT}:$PYTHONPATH

ENV AUGMENT_ROOT=/src/augment
ENV AUGMENT_REPOSITORY=https://github.com/funkey/augment.git
ENV AUGMENT_REVISION=4a42b01ccad7607b47a1096e904220729dbcb80a 

WORKDIR ${AUGMENT_ROOT} 
RUN git clone ${AUGMENT_REPOSITORY} . && \
    git checkout ${AUGMENT_REVISION}
ENV PYTHONPATH ${AUGMENT_ROOT}:$PYTHONPATH

ENV DVISION_ROOT=/src/dvision
ENV DVISION_REPOSITORY=https://github.com/TuragaLab/dvision.git
ENV DVISION_REVISION=v0.1.1

WORKDIR ${DVISION_ROOT}
RUN git clone -b ${DVISION_REVISION} --depth 1 ${DVISION_REPOSITORY} .
ENV PYTHONPATH ${DVISION_ROOT}:$PYTHONPATH

ENV MALA_ROOT=/src/mala
ENV MALA_REPOSITORY=https://github.com/funkey/mala.git
ENV MALA_REVISION=465b02ffce4204a4c1922f2b316a8b447ec4eb9b

WORKDIR ${MALA_ROOT}
RUN git clone ${MALA_REPOSITORY} . && \
    git checkout ${MALA_REVISION}
RUN python setup.py build_ext --inplace
ENV PYTHONPATH ${MALA_ROOT}:$PYTHONPATH

ENV WATERZ_ROOT=/src/waterz
ENV WATERZ_REPOSITORY=https://github.com/funkey/waterz
ENV WATERZ_REVISION=f252c42afcc25e7f09398dd11fe11a534acc7c93

WORKDIR ${WATERZ_ROOT}
RUN git clone ${WATERZ_REPOSITORY} . && \
    git checkout ${WATERZ_REVISION}
RUN mkdir -p /.cython/inline
ENV PYTHONPATH ${WATERZ_ROOT}:$PYTHONPATH

ENV GUNPOWDER_ROOT=/src/gunpowder
ENV GUNPOWDER_REPOSITORY=https://github.com/funkey/gunpowder.git
ENV GUNPOWDER_REVISION=8ae0685793d728fee4b23744655e503ffd378cb3

WORKDIR ${GUNPOWDER_ROOT}
RUN git clone ${GUNPOWDER_REPOSITORY} . && \
    git checkout ${GUNPOWDER_REVISION}
RUN python setup.py build_ext --inplace
ENV PYTHONPATH ${GUNPOWDER_ROOT}:$PYTHONPATH

ENV DAISY_ROOT=/src/daisy
ENV DAISY_REPOSITORY=https://github.com/funkey/daisy
ENV DAISY_REVISION=415c707070d48403b9b514703d595430f291e24d

WORKDIR ${DAISY_ROOT}
RUN git clone ${DAISY_REPOSITORY} . && \
    git checkout ${DAISY_REVISION}
RUN python setup.py build_ext --inplace
ENV PYTHONPATH ${DAISY_ROOT}:$PYTHONPATH

# install lsd

# assumes that lsd package directory is in build context (the complementary
# Makefile ensures that)
ADD lsd /src/lsd/lsd
ADD requirements.txt /src/lsd/requirements.txt
ADD setup.py /src/lsd/setup.py
WORKDIR /src/lsd
RUN python setup.py build_ext --inplace
ENV PYTHONPATH /src/lsd:$PYTHONPATH
