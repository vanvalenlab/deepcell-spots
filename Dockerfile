# Use tensorflow/tensorflow as the base image
# Change the build arg to edit the tensorflow version.
# Only supporting python3.
ARG TF_VERSION=2.4.1-gpu

FROM tensorflow/tensorflow:${TF_VERSION}

# System maintenance
RUN /usr/bin/python3 -m pip install --upgrade pip

# installs git into the Docker image
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install git -y

WORKDIR /notebooks

# Copy the required setup files and install the deepcell-tf dependencies
COPY setup.py README.md requirements.txt /opt/deepcell-spots/

# Prevent reinstallation of tensorflow and install all other requirements.
RUN sed -i "/tensorflow>/d" /opt/deepcell-spots/requirements.txt && \
    pip install -r /opt/deepcell-spots/requirements.txt

# Copy the rest of the package code and its scripts
COPY deepcell_spots /opt/deepcell-spots/deepcell_spots

# Copy over deepcell notebooks
COPY notebooks/ /notebooks/

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
