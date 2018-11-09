FROM tensorflow/tensorflow:1.3.0-gpu-py3
LABEL maintainer funkej@janelia.hhmi.org

# basic ubuntu packages

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libmlpack-dev \
        liblzma-dev \
        python3-numpy \
        wget && \
    rm -rf /var/lib/apt/lists/*

# dependencies for lsd

ENV MALIS_ROOT=/src/malis
ENV MALIS_REPOSITORY=https://github.com/TuragaLab/malis.git
ENV MALIS_REVISION=beb4ee965acee89ab00a20a70205f51177003c69

RUN pip install cython
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
RUN pip install -r requirements.txt
ENV PYTHONPATH ${AUGMENT_ROOT}:$PYTHONPATH

ENV DVISION_ROOT=/src/dvision
ENV DVISION_REPOSITORY=https://github.com/TuragaLab/dvision.git
ENV DVISION_REVISION=v0.1.1

WORKDIR ${DVISION_ROOT}
RUN git clone -b ${DVISION_REVISION} --depth 1 ${DVISION_REPOSITORY} .
RUN pip install -r requirements.txt
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
ENV WATERZ_REVISION=8ccd0b308fed604d143577f128420da83ff444da

WORKDIR ${WATERZ_ROOT}
RUN git clone ${WATERZ_REPOSITORY} . && \
    git checkout ${WATERZ_REVISION}
RUN mkdir -p /.cython/inline
RUN python setup.py install

ENV NUMCODECS_ROOT=/src/numcodecs
ENV NUMCODECS_REPOSITORY=https://github.com/funkey/numcodecs
ENV NUMCODECS_REVISION=f950047d7b666723f81006cbdfd82c0e6705c39c

WORKDIR ${NUMCODECS_ROOT}
RUN git clone ${NUMCODECS_REPOSITORY} . && \
    git checkout ${NUMCODECS_REVISION} && \
    git submodule update --init --recursive
RUN pip install -r requirements.txt
RUN python setup.py install

ENV ZARR_ROOT=/src/zarr
ENV ZARR_REPOSITORY=https://github.com/funkey/zarr
ENV ZARR_REVISION=9ddf849a6e3329f5ff361ebf6156712926e2fdfe

WORKDIR ${ZARR_ROOT}
RUN git clone ${ZARR_REPOSITORY} . && \
    git checkout ${ZARR_REVISION}
RUN pip install -r requirements.txt
RUN python setup.py install

ENV GUNPOWDER_ROOT=/src/gunpowder
ENV GUNPOWDER_REPOSITORY=https://github.com/funkey/gunpowder.git
ENV GUNPOWDER_REVISION=92f1acfcb7ada08fbf1ac55cef160321d2236956

WORKDIR ${GUNPOWDER_ROOT}
RUN git clone ${GUNPOWDER_REPOSITORY} . && \
    git checkout ${GUNPOWDER_REVISION}
RUN pip install -r requirements.txt
RUN python setup.py build_ext --inplace
ENV PYTHONPATH ${GUNPOWDER_ROOT}:$PYTHONPATH

ENV DAISY_ROOT=/src/daisy
ENV DAISY_REPOSITORY=https://github.com/funkey/daisy
ENV DAISY_REVISION=dc14ee3d5395d9ec2bcaca032d7ed5c5d97f8c70

WORKDIR ${DAISY_ROOT}
RUN git clone ${DAISY_REPOSITORY} . && \
    git checkout ${DAISY_REVISION}
RUN pip install -r requirements.txt
RUN python setup.py build_ext --inplace
ENV PYTHONPATH ${DAISY_ROOT}:$PYTHONPATH

RUN pip install mahotas
RUN pip install pymongo
RUN pip install distributed --upgrade

# install lsd

# assumes that lsd package directory is in build context (the complementary
# Makefile ensures that)
ADD lsd /src/lsd/lsd
ADD requirements.txt /src/lsd/requirements.txt
ADD setup.py /src/lsd/setup.py
WORKDIR /src/lsd
RUN python setup.py build_ext --inplace
ENV PYTHONPATH /src/lsd:$PYTHONPATH
