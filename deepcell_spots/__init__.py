# Copyright 2019-2023 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-spots/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Package for fluorescent spot detection with convolutional neural networks"""

from deepcell_spots import applications
from deepcell_spots._version import __version__

from deepcell_spots import cluster_vis
from deepcell_spots import data_utils
from deepcell_spots import dotnet_losses
from deepcell_spots import dotnet
from deepcell_spots import image_alignment
from deepcell_spots import image_generators
from deepcell_spots import multiplex
from deepcell_spots import point_metrics
from deepcell_spots import postprocessing_utils
from deepcell_spots import preprocessing_utils
from deepcell_spots import simulate_data
from deepcell_spots import singleplex
from deepcell_spots import spot_em
from deepcell_spots import training
from deepcell_spots import utils
