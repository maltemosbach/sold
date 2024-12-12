# Adapted from: https://github.com/nicklashansen/tdmpc2/blob/main/docker/Dockerfile

# Base image.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive

# Install required packages.
RUN apt-get -y update && \
    apt-get install -y --no-install-recommends build-essential git nano rsync vim tree curl wget \
    swig ffmpeg unzip htop tmux xvfb ca-certificates bash-completion libjpeg-dev libpng-dev \
    libssl-dev libcurl4-openssl-dev libopenmpi-dev zlib1g-dev qtbase5-dev qtdeclarative5-dev \
    libglib2.0-0 libglu1-mesa-dev libgl1-mesa-dev libvulkan1 libgl1-mesa-glx libosmesa6 \
    libosmesa6-dev libglew-dev mesa-utils && \
    apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir /root/.ssh

# Update and initialize Conda.
RUN conda update conda -y && \
    conda init bash && \
    . ~/.bashrc

# Clone multi-object-fetch repository and create Conda environment.
RUN git clone --depth 1 https://github.com/maltemosbach/multi-object-fetch.git && \
    cd multi-object-fetch && \
    ./create_conda_env.sh

# Compile mujoco_py.
RUN conda init bash && \
    . ~/.bashrc && \
    bash -c "source ~/.bashrc" && \
    conda activate mof && \
    python -c 'import mujoco_py'


# Install remaining SOLD dependencies.
COPY conda_env.yml /root
RUN conda env update -n mof --file /root/conda_env.yml --prune

# Activate the mof environment by default.
RUN echo "conda activate mof" >> ~/.bashrc

