# Copyright 2019-2022 The Van Valen Lab at the California Institute of
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

import numpy as np
import pandas as pd
from scipy.spatial import distance
from skimage.feature import peak_local_max
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from deepcell_spots.singleplex import match_spots_to_cells
from deepcell_spots.point_metrics import (match_points_min_dist,
                                          match_points_mutual_nearest_neighbor)


def multiplex_match_spots_to_cells(coords_dict, cytoplasm_pred):
    """Deprecated. Function no longer maintained.

    Matches detected spots to labeled cell cytoplasms.

    Args:
        coords_dict (dict): Dictionary where keys are image IDs
            (`'readoutName'`) and values are coordinates of detected spots
        cytoplasm_pred (matrix): Image where pixel values are labels for
            segmented cell cytoplasms.

    Returns:
        dict: dict of dicts, keys are image IDs (`'readoutName'`),
            values are dictionaries where keys are cell cytoplasm labels
            and values are detected spots associated with that cell label,
            there is one item in list for each image in coords_dict.
    """
    coords = [item[0] for item in coords_dict.values()]
    keys = list(coords_dict.keys())

    spots_to_cells_dict = {}

    for i in range(len(coords)):
        matched_spots_dict = match_spots_to_cells(cytoplasm_pred, coords[i])
        spots_to_cells_dict[keys[i]] = matched_spots_dict

    return (spots_to_cells_dict)


def cluster_points(spots_to_cells_dict, cell_id, threshold=1.5, match_method='min_dist'):
    """Deprecated. Function no longer maintained.

    Clusters points between rounds with one of two methods: `'min_dist'` or
    `'mutual_nearest_neighbor'`.

    Args:
        spots_to_cells_dict (dict): Dict of dicts, keys are image IDs
            (`'readoutName'`), values are dictionaries where keys are cell
            cytoplasm labels and values are detected spots associated with
            that cell label, there is one item in list for each image in
            `coords_dict`.
        cell_id (int): Integer key in `spots_to_cells_dict`.
        threshold (float, optional): Distance threshold in pixels for matching
            points between rounds. Defaults to 1.5.
        match_method (str, optional): Method for matching spots between rounds.
            Options are `'min_dist'` and `'mutual_nearest_neighbor'`.
            Defaults to `'min_dist'`.
    """
    col_names = list(spots_to_cells_dict.keys())

    # Get up data frame of clustered points
    cluster_df = pd.DataFrame(columns=['centroids'] + col_names)

    # Add points from first round
    cluster_df['centroids'] = spots_to_cells_dict[col_names[0]][cell_id]
    cluster_df[col_names[0]] = spots_to_cells_dict[col_names[0]][cell_id]

    for i in range(1, len(col_names)):
        # Get points for current round
        cell_coords = spots_to_cells_dict[col_names[i]][cell_id][:]
        if len(cell_coords) == 0:
            continue

        # Match points with existing points
        if len(list(cluster_df['centroids'].values)) > 0:
            if match_method == 'min_dist':
                matches = match_points_min_dist(list(cluster_df['centroids'].values),
                                                cell_coords, threshold=threshold)
            if match_method == 'mutual_nearest_neighbor':
                matches = match_points_mutual_nearest_neighbor(list(cluster_df['centroids'].values),
                                                               cell_coords, threshold=threshold)
        curr = [np.nan] * len(cluster_df)
        for ii in range(np.shape(matches)[1]):
            # Set up to replace current column in clustering data frame
            curr[matches[0][ii]] = cell_coords[matches[1][ii]]

        # Replace column
        cluster_df[col_names[i]] = curr

        update_centroids = []
        # Recalculate centroid
        for ii in range(len(cluster_df)):
            # Exclude current centroid
            row_vals = cluster_df.iloc[ii].values[1:]
            # Drops NaNs
            row_vals = np.array(
                [item for item in row_vals if type(item) == np.ndarray])
            new_centroid = [np.mean(row_vals[:, 0]), np.mean(row_vals[:, 1])]
            update_centroids.append(new_centroid)

        # Replace column
        cluster_df['centroids'] = update_centroids

        # Add unmatched spots to data frame
        alr_matched = sorted(matches[1], reverse=True)
        for idx in alr_matched:
            cell_coords.pop(idx)

        temp_df = pd.DataFrame(columns=['centroids'] + col_names)
        temp_df[col_names[i]] = cell_coords
        temp_df['centroids'] = cell_coords

        cluster_df = pd.concat([cluster_df, temp_df])
        cluster_df = cluster_df.reset_index(drop=True)

    return (cluster_df)


def gene_counts(spots_to_cells_dict, codebook, threshold=1.5,
                match_method='min_dist', error_corr=True):
    """Deprecated. Function no longer maintained.

    Assigns combinatorial barcodes corresponding to gene identities.

    Matches spots between rounds with one of two methods:
    `'min_dist'` or `'mutual_nearest_neighbor'`.

    Args:
        spots_to_cells_dict (dict): Dict of dicts, keys are image IDs
            (`'readoutName'`), values are dictionaries where keys are cell
            cytoplasm labels and values are detected spots associated with
            that cell label, there is one item in list for each image in
            `coords_dict`.
        codebook (Pandas DataFrame): ``DataFrame`` with columns for each imaging
            round, rows are barcodes for genes values in data frame are 0 if
            that barcode includes that imaging round and 1 if the barcode does not
        threshold (float, optional): Distance threshold in pixels for matching
            points between rounds
        match_method (str, optional): Method for matching spots between rounds.
            Options are `'min_dist'` and `'mutual_nearest_neighbor'`.
            Defaults to `'min_dist'`.
        error_corr (bool, optional): Boolean that determines whether error
            correction is performed on barcodes that don't have an exact match.
            Defaults to ``True``.

    Returns:
        pandas.DateFrame: ``DataFrame`` containing gene counts for each cell.
    """
    gene_count_per_cell = {}
    # codebook = codebook[['name']+col_names]
    col_names = list(spots_to_cells_dict.keys())

    codebook_dict = {}
    for i in range(len(codebook)):
        codebook_dict[str(list(codebook.loc[i].values[1:-1]))
                      ] = codebook.loc[i].values[0]

    gene_counts_df = pd.DataFrame(
        columns=['cellID'] + list(codebook_dict.values()))

    cell_id_list = list(spots_to_cells_dict[col_names[0]].keys())
    for i_cell, cell_id in enumerate(tqdm(cell_id_list)):

        cluster_df = cluster_points(spots_to_cells_dict, cell_id, threshold,
                                    match_method=match_method)

        cluster_results = np.array(list(cluster_df.values))[:, 1:]

        barcodes = []
        for i, row in enumerate(cluster_results):
            barcodes.append([])
            for item in row:
                if type(item) == np.ndarray:
                    barcodes[i].append(1)
                else:
                    barcodes[i].append(0)

        filter_barcodes = [item for item in barcodes
                           if sum(item) == 4 or sum(item) == 3 or sum(item) == 5]

        temp_gene_counts_df = pd.DataFrame(
            columns=['cellID'] + list(codebook_dict.values()))
        temp_gene_counts_df['cellID'] = [cell_id]

        for barcode in filter_barcodes:
            try:
                gene = codebook_dict[str(barcode)]
                if type(temp_gene_counts_df.at[0, gene]) == int:
                    temp_gene_counts_df.at[0, gene] += 1
                else:
                    temp_gene_counts_df.at[0, gene] = 1

            except KeyError:
                if error_corr:
                    corrected_gene = error_correction(
                        str(barcode), codebook_dict)
                    if corrected_gene == 'No match':
                        continue
                    else:
                        if type(temp_gene_counts_df.at[0, corrected_gene]) == int:
                            temp_gene_counts_df.at[0, corrected_gene] += 1
                        else:
                            temp_gene_counts_df.at[0, corrected_gene] = 1
                else:
                    continue

        gene_counts_df = pd.concat([gene_counts_df, temp_gene_counts_df])

    return (gene_counts_df)


def gene_counts_DBSCAN(spots_to_cells_dict, codebook, threshold, error_corr=True):
    """Deprecated. Function no longer maintained.

    Assigns combinatorial barcodes corresponding to gene identities.
    Matches spots between rounds with DBSCAN clustering.

    Args:
        spots_to_cells_dict (dict): Dictionary of dictionaries, keys are image
            IDs (`'readoutName'`), values are dictionaries where keys are cell
            cytoplasm labels and values are detected spots associated with
            that cell label, there is one item in list for each image in
            `coords_dict`.
        codebook (pandas.DataFrame): ``DataFrame`` with columns for each imaging
            round, rows are barcodes for genes values in data frame are 0 if
            that barcode includes that imaging round and 1 if the barcode
            does not.
        threshold (float): Distance threshold in pixels for matching points
            between rounds.
        error_corr (bool, optional): Boolean that determines whether error
            correction is performed on barcodes that don't have an exact match.
            Defaults to ``True``.

    Returns:
        pandas.DateFrame: ``DataFrame`` containing gene counts for each cell.
    """
    # Codebook data frame to dictionary
    codebook_dict = {}
    for i in range(len(codebook)):
        codebook_dict[str(list(codebook.loc[i].values[1:-1]))
                      ] = codebook.loc[i].values[0]
    col_names = list(spots_to_cells_dict.keys())

    gene_counts_df = pd.DataFrame(
        columns=['cellID'] + list(codebook_dict.values()))

    # Iterate through cells
    cell_id_list = list(spots_to_cells_dict[col_names[0]].keys())
    for i_cell, cell_id in enumerate(tqdm(cell_id_list)):

        # Get cooridnates for all spots in a cell
        cell_coords = []
        for i in range(len(col_names)):
            # Get points for current round
            cell_coords.append(spots_to_cells_dict[col_names[i]][cell_id][:])

        # Flatten across rounds
        num_spots_list = [len(item) for item in cell_coords]
        running_total = [sum(num_spots_list[:(i + 1)])
                         for i in range(len(num_spots_list))]
        cell_coords_flat = np.vstack(cell_coords)

        # Cluster spots
        # TODO: DBSCAN is not imported anywhere??
        clustering = DBSCAN(eps=threshold, min_samples=2).fit(cell_coords_flat)
        labels = clustering.labels_

        # Data frame with gene counts for this cell
        temp_gene_counts_df = pd.DataFrame(
            columns=['cellID'] + list(codebook_dict.values()))
        temp_gene_counts_df['cellID'] = [cell_id]

        # Iterate through clusters
        for item in np.unique(labels):
            # Throw out noisy clusters
            if item == -1:
                continue

            # Get indices of all spots in a cluster
            spot_ids = np.argwhere(labels == item)

            # Throw out clusters with more than five spots
            if len(spot_ids) > 5:
                continue

            # Instantiate barcode
            barcode = np.zeros(10)

            # Create barcode
            for i in range(len(spot_ids)):

                counter = 0
                while spot_ids[i] > running_total[counter]:
                    counter += 1

                barcode[counter] += 1
            barcode_str = str(list(barcode.astype(int)))

            try:
                gene = codebook_dict[barcode_str]
                if type(temp_gene_counts_df.at[0, gene]) == int:
                    temp_gene_counts_df.at[0, gene] += 1
                else:
                    temp_gene_counts_df.at[0, gene] = 1

            except KeyError:
                if error_corr:
                    corrected_gene = error_correction(
                        barcode_str, codebook_dict)
                    if corrected_gene == 'No match':
                        continue
                    else:
                        if type(temp_gene_counts_df.at[0, corrected_gene]) == int:
                            temp_gene_counts_df.at[0, corrected_gene] += 1
                        else:
                            temp_gene_counts_df.at[0, corrected_gene] = 1
                else:
                    continue

        gene_counts_df = pd.concat([gene_counts_df, temp_gene_counts_df])

    return gene_counts_df


def error_correction(barcode, codebook_dict):
    """Deprecated. Function no longer maintained.

    Corrects barcodes that have no match in codebook.

    To be assigned, a barcode may have a maximum of one bit flipped
    (Hamming distance of one) from input barcode.

    Args:
        barcode (str): String of binary barcode list, where values are 1 or 0
            depending on whether transcripts with that barcode are labeled in
            a particular round.
        codebook_dict (dict): Codebook converted into a dictionary where the
            keys are the binary barcode and the values are the gene names.
    """
    for key in codebook_dict.keys():
        codebook_barcode = np.array(key.strip('][').split(', ')).astype(int)

        dist = distance.euclidean(codebook_barcode, barcode)
        if dist == 1:
            gene = codebook_dict[key]
            return gene

    return 'No match'


def extract_spots_prob_from_coords_maxpool(image, spots_locations, extra_pixel_num=1):
    """Perform a max pooling and extract the intensities for each spot.

    Args:
        image (numpy.array): Probability maps with shape ``[batch, x, y, channel]``.
        spots_locations (numpy.array): Coordiantes found by max projection, used as anchor 
            points for further max pooling operation. Shape ``[num_spots, 2]``.
        extra_pixel_num (int): Parameter for size of the pool. Defaults to 1, meaning 
            a pool with size=(1,0,-1)x(1,0,-1)

    Returns:
        list: Spots intensities, each entry in the list is a numpy.array with shape 
        ``[num_spots, channel]``.
    """

    spots_intensities = []
    for idx_batch in range(len(image)):
        image_slice = image[idx_batch]
        coords = spots_locations[idx_batch]

        num_spots = len(coords)
        img_boundary_x = image_slice.shape[0] - 1
        img_boundary_y = image_slice.shape[1] - 1

        intensity_d = np.zeros(
            ((extra_pixel_num*2+1)**2, num_spots, image_slice.shape[-1]))
        d = -1
        for dx in np.arange(-extra_pixel_num, extra_pixel_num + 1):
            for dy in np.arange(-extra_pixel_num, extra_pixel_num + 1):
                d = d + 1
                for ind_cr in range(image_slice.shape[-1]):
                    x_coord = np.maximum(0, np.minimum(
                        img_boundary_x, np.around(coords[:, 0]) + dx))  # (num_spots,)
                    y_coord = np.maximum(0, np.minimum(
                        img_boundary_y, np.around(coords[:, 1]) + dy))  # (num_spots,)

                    intensity_d[d, :, ind_cr] = image_slice[x_coord,
                                                            y_coord, ind_cr]
        # (num_spots, image_slice.shape[-1])
        intensity = np.max(intensity_d, axis=0)
        spots_intensities.append(intensity)

    return spots_intensities
