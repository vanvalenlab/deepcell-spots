# Use vanvalenlab/deepcell-tf as the base image
# Change the build arg to edit the deepcell version.
# Only supporting python3.
ARG DEEPCELL_VERSION=0.12.0-gpu

FROM vanvalenlab/deepcell-tf:${DEEPCELL_VERSION}

# https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212771
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# Install git for postcode installation
RUN apt-get update && apt-get install -y \
    git && \
    rm -rf /var/lib/apt/lists/*

# Copy the required setup files and install the deepcell-tf dependencies
COPY setup.py README.md requirements.txt /opt/deepcell-spots/

# Prevent reinstallation of tensorflow and install all other requirements.
RUN sed -i "/tensorflow>/d" /opt/deepcell-spots/requirements.txt && \
    pip install --no-cache-dir -r /opt/deepcell-spots/requirements.txt

# Copy the rest of the package code and its scripts
COPY deepcell_spots /opt/deepcell-spots/deepcell_spots

# Copy over deepcell notebooks
COPY notebooks/ /notebooks/
