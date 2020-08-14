# Use tensorflow/tensorflow as the base image
# Change the build arg to edit the tensorflow version.
# Only supporting python3.
ARG TF_VERSION=1.15.0-gpu

FROM tensorflow/tensorflow:${TF_VERSION}-py3

# System maintenance
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-tk \
    libsm6 && \
    rm -rf /var/lib/apt/lists/* && \
    /usr/local/bin/pip install --upgrade pip

# installs git into the Docker image
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install git -y

WORKDIR /notebooks

# Copy the setup.py and requirements.txt and install the deepcell-spots dependencies
COPY requirements.txt /opt/deepcell-spots/

# Clone and install deepcell-tf (don't use requirements.txt to install it to prevent reinstallation of tensorflow)
RUN cd /opt && \        
    git clone https://github.com/vanvalenlab/deepcell-tf.git && \
    cd /opt/deepcell-tf

# Prevent reinstallation of tensorflow and install all other requirements.
RUN sed -i "/tensorflow/d" /opt/deepcell-tf/requirements.txt && \
    pip install -r /opt/deepcell-tf/requirements.txt

# Install deepcell via setup.py
RUN pip install /opt/deepcell-tf && \
    cd /opt/deepcell-tf && \
    python setup.py build_ext --inplace


# Prevent reinstallation of tensorflow and install all other requirements.
RUN sed -i "/tensorflow/d" /opt/deepcell-spots/requirements.txt && \
    pip install -r /opt/deepcell-spots/requirements.txt

# Copy the rest of the package code and its scripts
COPY deepcell_spots /opt/deepcell-spots/deepcell_spots

# Copy over deepcell-spots notebooks
COPY notebooks/ /notebooks/

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
