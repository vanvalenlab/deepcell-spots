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


def get_cell_counts(df_spots, segmentation_output):
    """Converts Polaris outputs into a DataFrame containing gene expression counts per cell.
    Detection assigned to the background (value of 0 in `segmentation_output`) are discarded.

    Args:
        df_spots (pandas.DataFrame): Polaris result, columns are `x`, `y`, `batch_id`, `cell_id`,
            `probability`, `predicted_id`, `predicted_name`, `spot_index`, `source`, and `masked`.
        segmentation_output (array): Polaris segmentation result, shape is `(b,x,y,c)`. `c` should
            equal one. `b` should be the number of fields of view (as defined by `batch_id` in
            `df_spots`).

    Returns:
        pandas.DataFrame: Gene expression counts per cell, columns are `batch_id`, `cell_id`, `x`,
            `y`, and columns for each decoded gene in the sample.
    """
    genes = list(df_spots.predicted_name.unique())
    if 'Background' in genes:
        genes.remove('Background')
    if 'Unknown' in genes:
        genes.remove('Unknown')
    genes = [item for item in genes if not('Blank' in item)]
    df_cell_counts = pd.DataFrame(columns=['batch_id', 'cell_id', 'x', 'y'] + genes)

    for fov in tqdm(df_spots.batch_id.unique()):
        df_fov = df_spots.loc[df_spots.batch_id==fov]
        seg = segmentation_output[fov,...,0]
        for cell in range(1,np.max(df_fov.cell_id.values)):
            cell_inds = np.argwhere(seg==cell)
            x = np.mean(cell_inds[:,0])
            y = np.mean(cell_inds[:,1])
            df_cell = df_fov.loc[df_fov.cell_id==cell]
            counts = dict(df_cell.predicted_name.value_counts())
            data = {}
            data['batch_id'] = [fov]
            data['cell_id'] = [cell]
            data['x'] = [x]
            data['y'] = [y]
            for gene in genes:
                if gene in list(counts.keys()):
                    data[gene] = [counts[gene]]
                else:
                    data[gene] = [0]
            single_cell_counts = pd.DataFrame.from_dict(data)

            df_cell_counts = pd.concat([df_cell_counts, single_cell_counts], axis=0)

    return(df_cell_counts)


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
    genes.remove('Background')
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


def hamming_dist_hist(df_spots, df_barcodes, gene_name=None):
    """Plot a histogram of the Hamming distance of pixel intensities for a subset of predicted
    genes or all genes to their predicted barcode.
    
    Args:
        df_spots (pandas.DataFrame): Polaris result, columns are `x`, `y`, `batch_id`, `cell_id`,
            `probability`, `predicted_id`, `predicted_name`, `spot_index`, and `source`.
        df_barcodes (pandas.DataFrame): Codebook, the first column is gene names (`'Gene'`),
                the rest are binary barcodes, encoded using 1 and 0. Index should start at 1.
                For exmaple, for a (rounds=10, channels=2) codebook, it should look like::
            
                Index:
                    RangeIndex (starting from 1)
                Columns:
                    Name: Gene, dtype: object
                    Name: r0c0, dtype: int64
                    Name: r0c1, dtype: int64
                    Name: r1c0, dtype: int64
                    Name: r1c1, dtype: int64
                    ...
                    Name: r9c0, dtype: int64
                    Name: r9c1, dtype: int64
        gene_name (list): List or array containing gene names to be included in the filtered
            result. Defaults to None.

    Returns:
        plotly.graph_objects.Figure: Histogram of the Hamming distances of pixels intensities to
            predicted barcodes.
    """

    labels = {
        'h_dist': 'Hamming distance'
    }
    title = 'Distribution of Hamming distances to assigned barcode'

    if gene_name is None:
        gene_name = list(df_spots.predicted_name.unique())
        if 'Unknown' in gene_name:
            gene_name.remove('Unknown')
        color = None

    else:
        if not type(gene_name) in [list, np.array]:
            raise ValueError('If defined, gene_name must be a list or array.')
        color = 'predicted_name'

    df_plot = filter_results(df_spots, gene_name=gene_name)

    dist_list = np.zeros(len(df_plot))
    for gene in gene_name:
        sub_df_plot = df_plot.loc[df_plot.predicted_name == gene]
        sub_indices = sub_df_plot.index
        sub_values = sub_df_plot.iloc[:, -20:].values
        sub_values = np.round(sub_values)
        barcode = df_barcodes.loc[df_barcodes.Gene == gene].values[0][1:]
        barcode_len = len(barcode)

        temp_dist_list = []
        for i in range(len(sub_df_plot)):
            temp_dist_list.append(distance.hamming(sub_values[i],
                                                   barcode))

        scaled_dist_list = np.array(temp_dist_list)*barcode_len
        dist_list[sub_indices] = scaled_dist_list

    df_plot['h_dist'] = dist_list

    fig = px.histogram(df_plot, x='h_dist', color=color,
                       barmode='overlay', histnorm='probability', labels=labels,
                       title=title)

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
