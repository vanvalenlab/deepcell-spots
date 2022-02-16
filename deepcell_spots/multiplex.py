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
from deepcell_spots.decoding_functions import (decoding_function,
                                               decoding_output_to_dataframe)
from scipy.spatial import distance
from skimage.feature import peak_local_max
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from deepcell_spots.singleplex import match_spots_to_cells
from deepcell_spots.point_metrics import (match_points_min_dist,
                                          match_points_mutual_nearest_neighbor)


def multiplex_match_spots_to_cells(coords_dict, cytoplasm_pred):
    """Matches detected spots to labeled cell cytoplasms.

    Args:
        coords_dict (dict): Dictionary where keys are image IDs
            ('readoutName') and values are coordinates of detected spots
        cytoplasm_pred (matrix): Image where pixel values are labels for
            segmented cell cytoplasms.

    Returns:
        dict: dict of dicts, keys are image IDs (readoutName),
            values are dictionaries where keys are cell cytoplasm labels
            and values are detected spots associated with that cell label,
            there is one item in list for each image in coords_dict.
    """
    coords = [item[0] for item in coords_dict.values()]
    keys = list(coords_dict.keys())

    spots_to_cells_dict = {}

    for i in range(len(coords)):
        matched_spot_dict = match_spots_to_cells(cytoplasm_pred, coords[i])
        spots_to_cells_dict[keys[i]] = matched_spot_dict

    return(spots_to_cells_dict)


def cluster_points(spots_to_cells_dict, cell_id, threshold=1.5, match_method='min_dist'):
    """Clusters points between rounds with one of two methods: 'min_dist' or
    'mutual_nearest_neighbor'.

    Args:
        spots_to_cells_dict (dict): Dict of dicts, keys are image IDs
            (readoutName), values are dictionaries where keys are cell
            cytoplasm labels and values are detected spots associated with
            that cell label, there is one item in list for each image in
            coords_dict
        cell_id (int): Integer key in spots_to_cells_dict
        threshold (float, optional): Distance threshold in pixels for matching
            points between rounds. Defaults to 1.5.
        match_method (str, optional): Method for matching spots between rounds.
            Options are 'min_dist' and 'mutual_nearest_neighbor'.
            Defaults to 'min_dist'.
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
            row_vals = np.array([item for item in row_vals if type(item) == np.ndarray])
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

    return(cluster_df)


def gene_counts(spots_to_cells_dict, codebook, threshold=1.5,
                match_method='min_dist', error_corr=True):
    """Assigns combinatorial barcodes corresponding to gene identities.

    Matches spots between rounds with one of two methods:
    'min_dist' or 'mutual_nearest_neighbor'.

    Args:
        spots_to_cells_dict (dict): Dict of dits, keys are image IDs
            (readoutName), values are dictionaries where keys are cell
            cytoplasm labels and values are detected spots associated with
            that cell label, there is one item in list for each image in
            coords_dict
        codebook (Pandas DataFrame): Data frame with columns for each imaging
            round, rows are barcodes for genes values in data frame are 0 if
            that barcode includes that imaging round and 1 if the barcode does not
        threshold (float, optional): Distance threshold in pixels for matching
            points between rounds
        match_method (str, optional): Method for matching spots between rounds.
            Options are 'min_dist' and 'mutual_nearest_neighbor'.
            Defaults to 'min_dist'.
        error_corr (bool, optional): Boolean that determines whether error
            correction is performed on barcodes that don't have an exact match.
            Defaults to True.

    Returns:
        pandas.DateFrame: DataFrame containing gene counts for each cell.
    """
    gene_count_per_cell = {}
    # codebook = codebook[['name']+col_names]
    col_names = list(spots_to_cells_dict.keys())

    codebook_dict = {}
    for i in range(len(codebook)):
        codebook_dict[str(list(codebook.loc[i].values[1:-1]))] = codebook.loc[i].values[0]

    gene_counts_df = pd.DataFrame(columns=['cellID'] + list(codebook_dict.values()))

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

        temp_gene_counts_df = pd.DataFrame(columns=['cellID'] + list(codebook_dict.values()))
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
                    corrected_gene = error_correction(str(barcode), codebook_dict)
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

    return(gene_counts_df)


def gene_counts_DBSCAN(spots_to_cells_dict, codebook, threshold, error_corr=True):
    """Assigns combinatorial barcodes corresponding to gene identities.
    Matches spots between rounds with DBSCAN clustering.

    Args:
        spots_to_cells_dict (dict): Dictionary of dictionaries, keys are image
            IDs (readoutName), values are dictionaries where keys are cell
            cytoplasm labels and values are detected spots associated with
            that cell label, there is one item in list for each image in
            coords_dict.
        codebook (pandas.DataFrame): DataFrame with columns for each imaging
            round, rows are barcodes for genes values in data frame are 0 if
            that barcode includes that imaging round and 1 if the barcode
            does not.
        threshold (float): Distance threshold in pixels for matching points
            between rounds.
        error_corr (bool, optional): Boolean that determines whether error
            correction is performed on barcodes that don't have an exact match.
            Defaults to True.

    Returns:
        pandas.DateFrame: DataFrame containing gene counts for each cell.
    """
    # Codebook data frame to dictionary
    codebook_dict = {}
    for i in range(len(codebook)):
        codebook_dict[str(list(codebook.loc[i].values[1:-1]))] = codebook.loc[i].values[0]
    col_names = list(spots_to_cells_dict.keys())

    gene_counts_df = pd.DataFrame(columns=['cellID'] + list(codebook_dict.values()))

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
        running_total = [sum(num_spots_list[:(i + 1)]) for i in range(len(num_spots_list))]
        cell_coords_flat = np.vstack(cell_coords)

        # Cluster spots
        # TODO: DBSCAN is not imported anywhere??
        clustering = DBSCAN(eps=threshold, min_samples=2).fit(cell_coords_flat)
        labels = clustering.labels_

        # Data frame with gene counts for this cell
        temp_gene_counts_df = pd.DataFrame(columns=['cellID'] + list(codebook_dict.values()))
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
                    corrected_gene = error_correction(barcode_str, codebook_dict)
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
    """Corrects barcodes that have no match in codebook.
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


def assign_gene_identities(cp_dict, dataorg, threshold, codebook):
    """Assigns gene identity to barcoded spots.

    Args:
        cp_dict (dict): Dictionary where keys are image IDs ('readoutName')
            and values are classification prediction output from the spot
            detection model.
        dataorg (pandas.DataFrame): Data frame containing information about
            organization of image files.
        threshold (float): value for the probability threshold a spot must
            exceed to be considered a spot.
        codebook (pandas.DataFrame): Data frame with columns for each imaging
            round, rows are barcodes for genes values in data frame are 0 if
            that barcode includes that imaging round and 1 if the barcode
            does not.

    Returns:
        pandas.DataFrame: Data frame with the spot locations, gene identity,
            and probability of assignment.
    """
    # Create array from classification prediction dictionary
    cp_array = np.array(list(cp_dict.values()))[:, 1, 0, :, :, 1]

    # Create maximum projection
    max_cp = np.max(cp_array, axis=0)
    # Convert classification prediction to list of points
    coords = peak_local_max(max_cp, threshold_abs=threshold)

    # Prepare spot intensities for postcode
    spots_s = []
    coords_list = []
    for c in coords:
        ints = cp_array[:, c[0], c[1]]
        coords_list.append([c[0], c[1]])
        spots_s.append(ints)

    spots_s = np.array(spots_s)
    coords_array = np.array(coords_list)

    r = len(dataorg['imagingRound'].unique())
    c = len(dataorg.loc[dataorg['readoutName'].str.contains('Spots')]['color'].unique())

    spots_s = np.reshape(spots_s, (np.shape(spots_s)[0], r, c))
    spots_s = np.swapaxes(spots_s, 1, 2)

    # Prepare codebook for postcode
    full_codebook = pd.DataFrame()
    full_codebook['name'] = codebook['name']

    for item in dataorg['readoutName']:
        if 'Spots' in item:
            if item in codebook.columns:
                full_codebook[item] = codebook[item]
            else:
                full_codebook[item] = np.zeros(len(full_codebook))

    barcodes_01 = np.reshape(full_codebook.values[:, 1:], (len(full_codebook), r, c))
    barcodes_01 = np.swapaxes(barcodes_01, 1, 2).astype(int)

    # Predict gene identities with postcode
    out = decoding_function(spots_s, barcodes_01, up_prc_to_remove=100,
                            print_training_progress=True)

    # Write results into pandas dataframe
    df_class_names = np.concatenate((codebook['name'].values, ['infeasible', 'background', 'nan']))
    df_class_codes = np.concatenate((np.arange(len(df_class_names)), ['inf', '0000', 'NA']))
    decoded_spots_df = decoding_output_to_dataframe(out, df_class_names, df_class_codes)
    decoded_spots_df['X'] = coords_array[:, 0]
    decoded_spots_df['Y'] = coords_array[:, 1]

    return decoded_spots_df


def assign_spots_to_cells(decoded_spots_df, cytoplasm_pred):
    """Adds column to spots DataFrame with identity of cell for each spot

    Args:
        decoded_spots_df (pandas.DataFrame): Data frame with the spot
            locations, gene identity, and probability of assignment
        cytoplasm_pred (array): Image where pixel values are labels for
            segmented cell cytoplasms.

    Returns:
        pandas.DataFrame: Data frame with the spot locations, gene identity,
            probability of assignment, and cell identity.
    """

    cell_list = []
    for i in range(len(decoded_spots_df)):
        cell_list.append(cytoplasm_pred[0, decoded_spots_df.iloc[i]['X'],
                         decoded_spots_df.iloc[i]['Y'], 0])

    decoded_spots_df['Cell'] = cell_list

    return decoded_spots_df
