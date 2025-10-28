# AutoQuake Development Container
# Based on Mambaforge for fast conda environment management
FROM condaforge/mambaforge:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install system dependencies for seismology and scientific computing
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    gfortran \
    # Git and version control
    git \
    git-lfs \
    # System utilities
    curl \
    wget \
    vim \
    nano \
    htop \
    tree \
    # Graphics and visualization dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Additional libraries for scientific computing
    libfftw3-dev \
    libsuitesparse-dev \
    liblapack-dev \
    libblas-dev \
    # Network tools for real-time data access
    libcurl4-openssl-dev \
    libssl-dev \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && mkdir -p /home/$USERNAME/.vscode-server /home/$USERNAME/.vscode-server-insiders \
    && chown ${USER_UID}:${USER_GID} /home/$USERNAME/.vscode-server*

# Create workspace directory
RUN mkdir -p /workspace && chown ${USER_UID}:${USER_GID} /workspace

# Copy environment files
COPY env.yml /tmp/env.yml
COPY requirements.txt /tmp/requirements.txt

# Create conda environment with mamba for faster installation
RUN mamba env create -f /tmp/env.yml && \
    mamba clean --all -f -y

# Initialize conda for the user
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /home/$USERNAME/.bashrc && \
    echo "conda activate AutoQuake_v0" >> /home/$USERNAME/.bashrc

# Switch to non-root user
USER $USERNAME

# Set working directory
WORKDIR /workspace

# Set the default shell to bash
SHELL ["/bin/bash", "-c"]

# Activate conda environment by default
RUN echo "conda activate AutoQuake_v0" >> ~/.bashrc

# Set PATH to include conda environment
ENV PATH="/opt/conda/envs/AutoQuake_v0/bin:$PATH"
