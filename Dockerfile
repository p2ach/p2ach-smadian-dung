# Pull the base image with python 3.6 as a runtime for your Lambda
FROM public.ecr.aws/lambda/python:3.8

RUN yum -y install tar gzip zlib freetype-devel \
    gcc \
    ghostscript \
    lcms2-devel \
    libffi-devel \
    libimagequant-devel \
    libjpeg-devel \
    libraqm-devel \
    libtiff-devel \
    libwebp-devel \
    make \
    openjpeg2-devel \
    rh-python36 \
    rh-python36-python-virtualenv \
    sudo \
    tcl-devel \
    tk-devel \
    tkinter \
    which \
    xorg-x11-server-Xvfb \
    zlib-devel \
    && yum clean all

# Copy the earlier created requirements.txt file to the container
COPY requirements.txt ./

#RUN PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
#
#    rm -rf /var/lib/apt/lists/* \
#           /etc/apt/sources.list.d/cuda.list \
#           /etc/apt/sources.list.d/nvidia-ml.list && \

# Install the python requirements from requirements.txt
RUN python3 -m pip install -U pip
# ==================================================================
# pytorch
# ------------------------------------------------------------------

RUN python3 -m pip --no-cache-dir install --upgrade \
        future \
        numpy \
        protobuf \
        enum34 \
        pyyaml \
        typing \
        && \
    python3 -m pip --no-cache-dir install --upgrade \
        --pre torch torchvision torchaudio -f \
        https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html


RUN python3 -m pip --no-cache-dir install --upgrade -r requirements.txt
# ==================================================================
# torch
# ------------------------------------------------------------------


# Replace Pillow with Pillow-SIMD to take advantage of AVX2
#RUN pip uninstall -y pillow && CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
RUN mkdir -p /tmp
RUN chmod +w /tmp

WORKDIR /var/task

# Copy the earlier created app.py file to the container
#COPY ./ ./
COPY app.py ./
COPY network network
COPY model_weights model_weights
COPY build_data.py ./
COPY module_list.py ./
COPY svc_inference.py ./
COPY test_client.py ./


# Set the CMD to your handler
CMD ["app.lambda_handler"]