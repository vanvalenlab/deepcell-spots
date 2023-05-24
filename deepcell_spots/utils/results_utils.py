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

"""Utility functions for processing and visualizing Polaris predictions"""

import skimage

import numpy as np


def filter_results(df_spots, batch_id=None, cell_id=None, gene_name=None, source=None):
    """Filter Pandas DataFrame output from Polaris application by batch ID, cell ID,
    predicted gene name, or prediction source. If filter arguments are None, that column
    will not be filtered.

    Args:
        df_spots (pandas.DataFrame): Polaris result, columns are `x`, `y`, `batch_id`, `cell_id`,
            `probability`, `predicted_id`, `predicted_name`, `spot_index`, and `source`.
        batch_id (list): List or array containing batch IDs to be included in the filtered result.
        cell_id (list): List or array containing cell IDs to be included in the filtered result.
        gene_name (list): List or array containing gene names to be included in the filtered
            result.
        source (list): List or array containing prediction sources to be included in the filtered
            result.

    Raises:
        ValueError: If defined, `batch_id` must be a list or array.
        ValueError: If defined, `cell_id` must be a list or array.
        ValueError: If defined, `gene_name` must be a list or array.
        ValueError: If defined, `source` must be a list or array.
    
    Returns:
        pandas.DataFrame: Filtered Polaris result, columns are `x`, `y`, `batch_id`, `cell_id`,
            `probability`, `predicted_id`, `predicted_name`, `spot_index`, and `source`.
    """
    
    output = df_spots.copy()
    
    if batch_id is not None:
        if not (type(batch_id) in [list, np.array]):
            raise ValueError('If defined, batch_id must be a list or array.')
        output = output.loc[output.batch_id.isin(batch_id)]
    
    if cell_id is not None:
        if not type(cell_id) in [list, np.array]:
            raise ValueError('If defined, cell_id must be a list or array.')
        output = output.loc[output.cell_id.isin(cell_id)]
    
    if gene_name is not None:
        if not type(gene_name) in [list, np.array]:
            raise ValueError('If defined, gene_name must be a list or array.')
        output = output.loc[output.predicted_name.isin(gene_name)]
    
    if source is not None:
        if not type(source) in [list, np.array]:
            raise ValueError('If defined, source must be a list or array.')
        output = output.loc[output.source.isin(source)]

    output = output.reset_index(drop=True)
        
    return(output)


def gene_visualization(df_spots, gene, image_dim, save_dir=None):
    """Construct an image where pixel values correspond with locations of decoded genes.
    Image can be saved to a defined directory.

    Args:
        df_spots (pandas.DataFrame): Polaris result, columns are `x`, `y`, `batch_id`, `cell_id`,
            `probability`, `predicted_id`, `predicted_name`, `spot_index`, and `source`.
        gene (str): Name of gene to be visualized. The value must match the gene's name in
            `df_spots`.
        image_dim (tuple): Dimensions `(x,y)` of image to be constructed, should match the dimensions
            of the image originally input to create predictions. 
        save_dir (str): Directory for saving image of gene expression visualization.
    """
    
    df_filter = filter_results(df_spots, gene_name=[gene])
    
    gene_im = np.zeros(image_dim)
    for i in range(len(df_filter)):
        x = df_filter.loc[i]['x']
        y = df_filter.loc[i]['y']
        gene_im[x,y] += 1
    
    if save_dir is not None:
        skimage.io.imsave(save_dir, gene_im)
        
    return(gene_im)
