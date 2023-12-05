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
import plotly.graph_objects as go
import plotly.express as px

import numpy as np
import pandas as pd

from scipy.spatial import distance
from tqdm import tqdm

from deepcell_spots.utils.preprocessing_utils import min_max_normalize


def get_cell_counts(df_spots):
    """Converts Polaris outputs into a DataFrame containing gene expression counts per cell.
    Detection assigned to the background (value of 0 in `segmentation_output`) are discarded.

    Args:
        df_spots (pandas.DataFrame): Polaris result, columns are `x`, `y`, `batch_id`, `cell_id`,
            `probability`, `predicted_id`, `predicted_name`, `spot_index`, `source`, and `masked`.

    Returns:
        pandas.DataFrame: Gene expression counts per cell, columns are `batch_id`, `cell_id`, and
            columns for each decoded gene in the sample.
    """
    genes = list(df_spots.predicted_name.unique())
    if 'Background' in genes:
        genes.remove('Background')
    if 'Unknown' in genes:
        genes.remove('Unknown')

    genes = [item for item in genes if not('Blank' in item)]
    df_cell_counts = pd.DataFrame(columns=['batch_id', 'cell_id'] + genes)

    for fov in tqdm(df_spots.batch_id.unique()):
        df_fov = df_spots.loc[df_spots.batch_id==fov]

        for cell in range(1,np.max(df_fov.cell_id.values)+1):
            df_cell = df_fov.loc[df_fov.cell_id==cell]
            counts = dict(df_cell.predicted_name.value_counts())
            data = {}
            data['batch_id'] = [fov]
            data['cell_id'] = [cell]

            for gene in genes:
                if gene in list(counts.keys()):
                    data[gene] = [counts[gene]]
                else:
                    data[gene] = [0]
            single_cell_counts = pd.DataFrame.from_dict(data)

            df_cell_counts = pd.concat([df_cell_counts, single_cell_counts], axis=0)

    df_cell_counts = df_cell_counts.reset_index(drop=True)
    return(df_cell_counts)


def assign_barcodes(df_spots, segmentation_results):
    """Assigns barcode identity to a cell for Polaris prediction for data from optical pooled
    screens. This function does not support multi-batch inputs.

    Args:
        df_spots (pandas.DataFrame): Polaris result, columns are `x`, `y`, `batch_id`, `cell_id`,
            `probability`, `predicted_id`, `predicted_name`, `spot_index`, `source`, and `masked`.
            `batch_id` should only have one unique value.
        segmentation_results (numpy.array): Segmentation result from Polaris with shape
            `(1,x,y,1)`. Pixel values should match `cell_id` values in `df_spots`. The background
            pixels are assumed to have the value 0.

    Returns:
        pandas.DataFrame: Barcode assignment for each cell, columns are `cell_id`, `x`, `y`,
            `predicted_name`, `predicted_id`, `spot_counts`, `spot_fraction`. `x` and `y` are the
            centroid of the cell with value `cell_id` in `segmentation_results`. `predicted_name`
            and `predicted_id` correspond to the assigned barcode. `spot_counts` is the number of
            spots detected in a cell with the assigned barcode. `spot_fraction` is the fraction
            of detections in a cell with the assigned barcode.
    """
    df_assignments = pd.DataFrame(columns=['cell_id', 'x', 'y', 'predicted_name', 'predicted_id',
                                           'spot_counts', 'spot_fraction'])
    
    if len(segmentation_results.shape) != 4:
        raise ValueError('Input data must have {} dimensions. '
                         'Input data only has {} dimensions'.format(
                         4, len(segmentation_results.shape)))
    if segmentation_results.shape[0] != 1:
        raise ValueError('Input data must have a batch dimension of size 1. '
                         'Input data only has a batch dimension of size {}.'.format(
                         segmentation_results.shape[0]))
    
    for i in tqdm(range(1,np.max(segmentation_results).astype(int)+1)):
        df_cell = df_spots.loc[df_spots.cell_id == i]
        df_cell = df_cell.loc[~df_cell.predicted_name.isin(['Background', 'Unknown'])]
        n_spots = len(df_cell)
        
        cell_pixels = np.argwhere(segmentation_results == i)
        x = np.mean(cell_pixels[:,1])
        y = np.mean(cell_pixels[:,2])

        if n_spots > 0:
            barcode_dict = {}
            for barcode in df_cell.predicted_name.unique():
                df_barcode = df_cell.loc[df_cell.predicted_name==barcode]
                barcode_dict[barcode] = sum(df_barcode.probability)
                assignment = max(barcode_dict, key=barcode_dict.get)
                df_correct = df_cell.loc[df_cell.predicted_name==assignment]
                assignment_id = df_correct.predicted_id.values[0]
                counts = len(df_correct)
                fraction = counts/n_spots
        else:
            assignment = 'None'
            assignment_id = -1
            counts = 0
            fraction = 0
            
        df_assignments.loc[len(df_assignments)] = [i, x, y, assignment, assignment_id, counts, fraction]
        
    return(df_assignments)


def filter_results(df_spots, batch_id=None, cell_id=None,
                   gene_name=None, source=None, masked=False):
    """Filter Pandas DataFrame output from Polaris application by batch ID, cell ID,
    predicted gene name, or prediction source. If filter arguments are None, that column
    will not be filtered.

    Args:
        df_spots (pandas.DataFrame): Polaris result, columns are `x`, `y`, `batch_id`, `cell_id`,
            `probability`, `predicted_id`, `predicted_name`, `spot_index`, `source`, and `masked`.
        batch_id (list): List or array containing batch IDs to be included in the filtered result.
            Defaults to None.
        cell_id (list): List or array containing cell IDs to be included in the filtered result.
            Defaults to None.
        gene_name (list): List or array containing gene names to be included in the filtered
            result. Defaults to None.
        source (list): List or array containing prediction sources to be included in the filtered
            result. Defaults to None.
        masked (bool): Whether to filter spots in regions of high background intensity. Defaults to
            False.

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

    if masked:
        output = output.loc[output.masked == 0]

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
        image_dim (tuple): Dimensions `(x,y)` of image to be constructed, should match the
            dimensions of the image originally input to create predictions.
        save_dir (str): Directory for saving image of gene expression visualization. Defaults to
            None.

    Returns:
        numpy.array: Array containing image where pixel values correspond with locations of decoded
            genes.
    """

    df_filter = filter_results(df_spots, gene_name=[gene])

    gene_im = np.zeros(image_dim)
    for i in range(len(df_filter)):
        x = df_filter.loc[i]['x']
        y = df_filter.loc[i]['y']
        gene_im[x, y] += 1

    if save_dir is not None:
        skimage.io.imsave(save_dir, gene_im)

    return(gene_im)


def gene_scatter(df_spots, gene_name=None, remove_errors=False):
    """Create scatter plot of locations of decoded genes.

    Args:
        df_spots (pandas.DataFrame): Polaris result, columns are `x`, `y`, `batch_id`, `cell_id`,
            `probability`, `predicted_id`, `predicted_name`, `spot_index`, and `source`. `source`
            columns must include `'prediction'` entries.
        gene_name (list): List of gene names to be included in scatter plot. If None, all gene
            names will be included.
        remove_errors (bool): Whether to exclude `'Background'` and `'Unknown'` assignments in
            scatter plot.

    Returns:
        plotly.graph_objects.Figure: Scatter plot of locations of decoded genes.
    """
    if gene_name is not None:
        if not type(gene_name) in [list, np.array]:
            raise ValueError('If defined, gene_name must be a list or array.')
        df_plot = filter_results(df_spots, gene_name=gene_name)
        
    elif remove_errors:
        genes = list(df_spots.predicted_name.unique())
        genes.remove('Background')
        genes.remove('Unknown')
        df_plot = filter_results(df_spots, gene_name=genes)
        
    else:
        df_plot = df_spots.copy()
        
    
    fig = px.scatter(df_plot,
                     x='y', y='x',
                     width=650, height=600,
                     hover_data=['spot_index'],
                     color='predicted_name',
                     title='Predicted gene locations')
    
    return(fig)


def spot_journey_plot(df_spots):
    """Plot Sankey diagram of predicted spot sources and assignments.

    Args:
        df_spots (pandas.DataFrame): Polaris result, columns are `x`, `y`, `batch_id`, `cell_id`,
            `probability`, `predicted_id`, `predicted_name`, `spot_index`, and `source`. `source`
            columns must include `'prediction'` entries.

    Returns:
        plotly.graph_objects.Figure: Sankey diagram of spot source and assignments.
    """
    label = ['All spots']

    df_copy = df_spots.copy()

    if 'masked' in list(df_copy.columns):
        all_spots = len(df_copy)
        df_copy = df_copy.loc[df_copy.masked == 0]
        masked = all_spots - len(df_copy)
    else:
        masked = 0

    sources = list(df_copy.source.unique())
    sources.remove('prediction')
    sources = ['prediction'] + sources
    s = len(sources)

    genes = list(df_copy.predicted_name.unique())
    if 'Background' in genes:
        genes.remove('Background')
    if 'Unknown' in genes:
        genes.remove('Unknown')

    source = np.zeros(s+s*3)
    target = np.zeros(s+s*3)
    target[:s] = np.arange(1, s+1)
    value = np.zeros(s+s*3)

    for i, item in enumerate(sources):
        source[s*(i+1):s*(i+2)] = i+1
        target[s*(i+1):s*(i+2)] = np.arange(s+1, 2*s+1)
        label.append(item)

        sub = len(filter_results(df_copy, source=[item]))
        value[i] = sub
        sub_genes = len(filter_results(df_copy, source=[item], gene_name=genes))
        value[s*(i+1)] = sub_genes
        sub_bkg = len(filter_results(df_copy, source=[item], gene_name=['Background']))
        value[s*(i+1)+1] = sub_bkg
        sub_unk = len(filter_results(df_copy, source=[item], gene_name=['Unknown']))
        value[s*(i+1)+2] = sub_unk

    label.extend(['genes', 'background', 'unknown', 'masked'])
    source = np.append(source, [0])
    target = np.append(target, [np.max(target)+1])
    value = np.append(value, [masked])

    fig = go.Figure(data=[go.Sankey(
        node=dict(
          pad=15,
          thickness=20,
          line=dict(color="black", width=0.5),
          label=label,
          color=['dodgerblue', 'green'] + ['gold']*(s-1) + ['green', 'coral', 'coral', 'coral']
        ),
        link=dict(
          source=source,
          target=target,
          value=value
        ))])

    fig.update_layout(title_text="Journey of detected spots", font_size=10)

    return(fig)


def expression_correlation(df_spots,
                           df_control,
                           expr_dict=None,
                           log=False,
                           exclude_genes=[],
                           exclude_zeros=False,
                           eps=0.001,
                           title=None,
                           xlabel=None,
                           ylabel=None):
    """Plot correlation between gene expression quantified by Polaris and a second control method.

    Args:
        df_spots (pandas.DataFrame): Polaris result, columns are `x`, `y`, `batch_id`, `cell_id`,
            `probability`, `predicted_id`, `predicted_name`, `spot_index`, and `source`.
        df_control (pandas.DataFrame): Control gene expression result, columns must include `gene`
            and `expression`.
        expr_dict (dict): Dictionary with gene names as keys and expression counts as values. This
            argument is an alternative format for inputting expression data to `df_spots`.
        log (bool): Whether to create the scatter plot in log space. Defaults to False.
        exclude_genes (list): List of outlier genes to exclude from scatter plot. Defaults as
            empty.
        exclude_zeros (bool): Whether to exclude genes with zero counts in control or FISH
            experiment. Defaults to False.
        eps (float): Small epsilon value added to counts to errors in logarithm calculation. The
            value should be small relative to the smallest count values. Defaults to 0.001.
        title (str): Title of plot.
        xlabel (str): Label for x-axis of plot.
        ylabel (str): Label for y-axis of plot.

    Returns:
        plotly.graph_objects.Figure: Scatter plot of gene expression from a control method vs.
            Polaris. Points are labeled with gene names. A fit line calculated with ordinary
            least squares is included.
    """
    if expr_dict is None:
        expr_dict = df_spots.predicted_name.value_counts()
        
    if title is None:
        title = 'Correlation with control counts'
    if xlabel is None:
        xlabel = 'Log(Control Counts)'
    if ylabel is None:
        ylabel = 'Log(FISH Counts)'

    correlation_df = pd.DataFrame(columns=['gene', xlabel, ylabel])

    for gene in df_control.gene:
        if gene in exclude_genes:
            continue

        control_expr = df_control.loc[df_control.gene == gene].expression.values[0]
        if exclude_zeros and control_expr==0:
            continue
        if gene in expr_dict.keys():
            fish_expr = expr_dict[gene]
        else:
            if exclude_zeros:
                continue
            else:
                fish_expr = 0

        if log:
            correlation_df.loc[len(correlation_df)] = [
                gene,
                np.log10(control_expr+eps),
                np.log10(fish_expr+eps)
            ]
        
        else:
            correlation_df.loc[len(correlation_df)] = [
                gene,
                control_expr+eps,
                fish_expr+eps
            ]

    fig = px.scatter(correlation_df,
                     x=xlabel, y=ylabel,
                     height=800, width=800,
                     log_x=(not log), log_y=(not log),
                     hover_data='gene', text='gene',
                     trendline='ols',
                     title=title)
    fig.update_traces(textposition='top center')
    fig.update_layout(font={'size':12})

    model = px.get_trendline_results(fig)
    rsq = model.iloc[0]["px_fit_results"].rsquared
    
    if log:
        x_loc = max((correlation_df[xlabel]))
        y_loc = min(correlation_df[ylabel])
    else:
        x_loc = np.log10(max(correlation_df[xlabel]))
        y_loc = np.log10(min(correlation_df[ylabel]))
    
    fig.add_annotation(
        x=x_loc,
        y=y_loc,
        text="r={}".format(np.round(np.sqrt(rsq), 3)),
        showarrow=False
    )

    return(fig)


def probability_hist(df_spots, gene_name=None):
    """Plot a histogram of the prediction probabilities for a subset of predicted genes or all
    genes to their predicted barcode.
    
    Args:
        df_spots (pandas.DataFrame): Polaris result, columns are `x`, `y`, `batch_id`, `cell_id`,
            `probability`, `predicted_id`, `predicted_name`, `spot_index`, and `source`.
        gene_name (list): List or array containing gene names to be included in the filtered
            result. Defaults to None.

    Returns:
        plotly.graph_objects.Figure: Histogram of the prediction probabilities.
    """

    labels = {
        'Polaris prob': 'Prediction probability'
    }
    title = 'Distribution of prediction probabilities'

    if gene_name is not None:
        if not type(gene_name) in [list, np.array]:
            raise ValueError('If defined, gene_name must be a list or array.')

        df_plot = filter_results(df_spots, gene_name=gene_name)
        df_plot = df_plot.rename({'probability': 'Polaris prob'}, axis=1)
        fig = px.histogram(df_plot, x='Polaris prob', color='predicted_name',
                           barmode='overlay', histnorm='probability', labels=labels,
                           title=title)

    else:
        df_plot = df_spots.copy()
        df_plot = df_plot.rename({'probability': 'Polaris prob'}, axis=1)
        fig = px.histogram(df_plot, x='Polaris prob', histnorm='probability',
                           labels=labels, title=title)

    fig.add_annotation(x=-0.65, y=0.02,
                       text="Probability of -1 results<br>from mixed rescue",
                       showarrow=False,
                       yshift=10)

    return(fig)

def mask_spots(background_image, mask_threshold):
    """Mask predicted spots in regions of high background intensity. If input background
    image contains more than one channel, background mask will be maximum intensity projected
    across channel axis.

    Args:
        spots_locations (list): A list of length `batch` containing arrays of spots
            coordinates with shape `[num_spots, 2]`.
        background_image (numpy.array): Input image for masking bright background objects with
            shape `[batch, x, y, channel]`.
        mask_threshold (float): Percentile of pixel values in background image used to
            create a mask for bright background objects.

    Returns:
        array: Array with values 0 and 1, whether predicted spot is within a masked backround
            object.
    """
    normalized_image = np.zeros(background_image.shape)
    for i in range(background_image.shape[0]):
        normalized_image[i] = min_max_normalize(background_image[i:i+1], clip=True)
    mask = normalized_image > mask_threshold
    mask = np.max(mask, axis=-1)

    return(mask)
