# DeepCell Spots

[![Build Status](https://github.com/vanvalenlab/deepcell-spots/workflows/build/badge.svg)](https://github.com/vanvalenlab/deepcell-spots/actions)
[![Documentation Status](https://readthedocs.org/projects/deepcell-spots/badge/?version=latest)](https://deepcell-spots.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/vanvalenlab/deepcell-spots/badge.svg)](https://coveralls.io/github/vanvalenlab/deepcell-spots)
[![Modified Apache 2.0](https://img.shields.io/badge/license-Modified%20Apache%202-blue)](https://github.com/vanvalenlab/deepcell-spots/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/DeepCell-Spots.svg)](https://badge.fury.io/py/DeepCell-Spots)
[![PyPi Monthly Downloads](https://img.shields.io/pypi/dm/deepcell-spots)](https://pypistats.org/packages/deepcell-spots)
[![Python Versions](https://img.shields.io/pypi/pyversions/deepcell-spots.svg)](https://pypi.org/project/deepcell-spots/)

`deepcell-spots` is a deep learning package for analyzing spatial transcriptomics data sets. Its application, Polaris, allows you to apply pre-existing models and train new deep learning models for fluorescent spot detection. Polaris also contains a probabilistic method for combinatorial barcode assignment. It is written in Python and leverages a number of packages, including: [TensorFlow](https://github.com/tensorflow/tensorflow), [Keras](https://www.tensorflow.org/guide/keras), [PyTorch](https://pytorch.org), and [DeepCell](https://github.com/vanvalenlab/deepcell-tf). More detailed documentation of `deepcell-spots` is available [here](https://deepcell-spots.readthedocs.io/).

# ![Spot Detection Example](/docs/images/spot_montage.png)

# Getting Started

## Install with pip
The simplest way to install `deepcell-spots` is with `pip`:

```bash
pip install deepcell-spots
```

## Install with Docker
The `deepcell-spots` Docker container is also available on DockerHub. For more details for running DeepCell Docker containers, please see the [deepcell-tf README](https://github.com/vanvalenlab/deepcell-tf/blob/master/README.md).


# DeepCell Spots Applications

`deepcell-spots` contains an applications that greatly simplify the implementation of machine learning models for spot detection and decoding. `SpotDetection` contains a pre-trained model for fluorescent spot detection on images derived from assays such as RNA FISH and in-situ sequencing. This model returns a list of coordinate locations for fluorescent spots detected in the input image. `SpotDecoding` fit a probabilistic model for spot decoding with SVI. It returns a gene identity for each detected spot for multiplex FISH data sets. `Polaris` pairs these two applications with [DeepCell](https://github.com/vanvalenlab/deepcell-tf) models for nuclear and cytoplasmic segmentation. For example implementation, please see the [example notebooks](https://github.com/vanvalenlab/deepcell-spots/blob/master/notebooks).

# DeepCell Spots for Developers

Build and run a local docker container, similarly to the instructions for deepcell-tf. The relevant parts are copied here with modifications to work for deepcell-spots. For further instructions, see the [deepcell-tf README](https://github.com/vanvalenlab/deepcell-tf/blob/master/README.md).

### Build a local docker container, specifying the deepcell version with DEEPCELL_VERSION

```bash
git clone https://github.com/vanvalenlab/deepcell-spots.git
cd deepcell-spots
docker build --build-arg DEEPCELL_VERSION=0.12.0-gpu -t $USER/deepcell-spots . 
```

### Run the new docker image

```bash
# '"device=0"' refers to the specific GPU(s) to run DeepCell-Spots on, and is not required
docker run --gpus '"device=0"' -it \
-p 8888:8888 \
$USER/deepcell-spots
```

It can also be helpful to mount the local copy of the repository and the notebooks to speed up local development.

```bash
# you can now start the docker image with the code mounted for easy editing
docker run --gpus '"device=0"' -it \
    -p 8888:8888 \
    -v $PWD/deepcell-spots/deepcell_spots:/usr/local/lib/python3.8/dist-packages/deepcell_spots \
    -v $PWD/notebooks:/notebooks \
    -v /$PWD:/data \
    $USER/deepcell-spots
```

## Copyright

Copyright © 2019-2023 [The Van Valen Lab](http://www.vanvalen.caltech.edu/) at the California Institute of Technology (Caltech), with support from the Shurl and Kay Curci Foundation, Google Research Cloud, the Paul Allen Family Foundation, & National Institutes of Health (NIH) under Grant U24CA224309-01.
All rights reserved.

## License

This software is licensed under a modified [APACHE2](https://github.com/vanvalenlab/deepcell-spots/blob/master/LICENSE). See [LICENSE](https://github.com/vanvalenlab/deepcell-spots/blob/master/LICENSE) for full details.

## Trademarks

All other trademarks referenced herein are the property of their respective owners.

## Credits

[![Van Valen Lab, Caltech](https://upload.wikimedia.org/wikipedia/commons/7/75/Caltech_Logo.svg)](http://www.vanvalen.caltech.edu/)
