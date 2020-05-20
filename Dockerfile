# Use tensorflow/tensorflow as the base image
# Change the build arg to edit the tensorflow version.
# Only supporting python3.
ARG TF_VERSION=1.14.0-gpu

FROM tensorflow/tensorflow:${TF_VERSION}-py3

# System maintenance
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-tk \
    libsm6 && \
    rm -rf /var/lib/apt/lists/* && \
    /usr/local/bin/pip install --upgrade pip

# installs git into the Docker image, as required by tox
RUN apt-get update && apt-get install git -y

WORKDIR /notebooks

# Copy the setup.py and requirements.txt and install the deepcell-spots dependencies
COPY requirements.txt /opt/deepcell-spots/

# Prevent reinstallation of tensorflow and install all other requirements.
RUN sed -i "/tensorflow/d" /opt/deepcell-spots/requirements.txt && \
    pip install -r /opt/deepcell-spots/requirements.txt

# Copy the rest of the package code and its scripts
#COPY deepcell /opt/deepcell-tf/deepcell
COPY deepcell_spots /opt/deepcell-spots/deepcell_spots

# Older versions of TensorFlow have notebooks, but they may not exist
RUN if [ -n "$(find /notebooks/ -prune)" ] ; then \
      mkdir -p /notebooks/intro_to_tensorflow && \
      ls -d /notebooks/* | grep -v intro_to_tensorflow | \
      xargs -r mv -t /notebooks/intro_to_tensorflow ; \
    fi

# Copy over deepcell-spots notebooks
COPY notebooks/ /notebooks/

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]