# Use vanvalenlab/deepcell-tf as the base image
# Change the build arg to edit the tensorflow version.
# Only supporting python3.
ARG DEEPCELL_VERSION=0.11.0-gpu

FROM vanvalenlab/deepcell-tf:${DEEPCELL_VERSION}

# System maintenance
RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/* && \
    /usr/bin/python3 -m pip install --no-cache-dir --upgrade pip

# Copy the required setup files and install the deepcell-tf dependencies
COPY setup.py README.md requirements.txt /opt/deepcell-spots/

# Prevent reinstallation of tensorflow and install all other requirements.
RUN sed -i "/tensorflow>/d" /opt/deepcell-spots/requirements.txt && \
    pip install --no-cache-dir -r /opt/deepcell-spots/requirements.txt

# Copy the rest of the package code and its scripts
COPY deepcell_spots /opt/deepcell-spots/deepcell_spots

# Copy over deepcell notebooks
COPY notebooks/ /notebooks/
