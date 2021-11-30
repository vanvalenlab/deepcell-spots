# Copyright 2019-2021 The Van Valen Lab at the California Institute of
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
from skimage.feature import register_translation
from scipy.spatial import distance
import collections
import os


def read_images(root_dir, image_files, dataorg, verbose=True):
    """Reads in image files from given directories and parses them into dictionaries of different
    types.

    Args:
        root_dir (str): Directory containing all image files
        image_files (list): List of image names (str) in root directory. Paths must be to images
            must be saved in .npy format.
        dataorg (pandas.DataFrame): Data frame with required columns 'fileName' (item in
            image_files), 'readoutName' (unique ID name given to each channel in each image),
            'fiducialFrame' (frame number for image to be used for alignment), 'cytoplasmFrame'
            (frame number for image to be used for cell segmentation)
        verbose (bool, optional): Boolean determining if file names are printed as they are
            processed. Defaults to True.

    Returns:
        max_im_dict (dict): Dictionary where keys are image IDs ('readoutName') and values are
            maximum intensity projections of frames associated with that readout name
        fiducial_dict (dict): Dictionary where keys are image IDs ('readoutName') and values are
            fiducial channel (image used for alignment) for each readout name
            (multiple readout names may have the same)
        cytoplasm_dict (dict): Dictionary where keys are image IDs ('readoutName') and values are
            cytoplasm label image for each readout name (multiple readout names may have the same)
    """

    max_im_dict = {}
    fiducial_dict = {}
    cytoplasm_dict = {}

    for i in range(len(image_files)):

        # Slice data organization table to get information for this image stack
        round_df = dataorg.loc[dataorg['fileName'] == image_files[i]]
        image_file = os.path.join(root_dir, image_files[i])

        # Load image stack
        image_stack = np.load(image_file)

        # Grab ID names for each image in stack
        rounds = round_df['readoutName'].values
        rounds = [item for item in rounds if item in codebook.columns]

        for item in rounds:
            if verbose:
                print('Working on: {}'.format(item))

            # Get frames associated with a round in the image stack
            frame_list = round_df['frame'].loc[round_df['readoutName'] == item].values[0]
            frame_list = frame_list.strip('][').split(', ')
            frame_list = np.array(frame_list).astype(int)

            start_frame = frame_list[0]
            end_frame = frame_list[-1]

            # Maximum projection
            max_im = np.max(image_stack[:, :, start_frame:end_frame+1], axis=2)

            # Clip outlier high pixel values
            im = np.clip(max_im, np.min(max_im), np.percentile(max_im, 99.9))
            im = np.expand_dims(im, axis=[0, -1])

            max_im_dict[item] = im
            fiducial_dict[item] = np.expand_dims(
                                        image_stack[:, :, round_df['fiducialFrame'].values[0]],
                                        axis=[0, -1])
            cytoplasm_dict[item] = np.expand_dims(
                                        image_stack[:, :, round_df['cytoplasmFrame'].values[0]],
                                        axis=[0, -1])

    return(max_im_dict, fiducial_dict, cytoplasm_dict)


def align_images(image_dict, fiducial_dict):
    """Aligns images in given dictionary based on alignment learned from fiducial dictionary.

    Args:
        image_dict (dict): Dictionary (created by read_images) where keys are image IDs
            ('readoutName') and values are the images to be aligned
        fiducial_dict (dict): Dictionary (created by read_images) where keys are image IDs
            ('readoutName') and values are the images that are the fiducial channel (or images with
            objects that are to be used to learn the alignment transformation)

    Returns:
        aligned_dict (dict): Dictionary where keys are image IDs
            ('readoutName') and values are the images from image_dict after the alignment
            transformation has been applied
    """

    image_keys = list(image_dict.keys())
    num_images = len(image_keys)
    image_shifts = np.zeros((num_images, 2))

    # calculate image shifts
    for idx in range(num_images):
        if idx == 0:
            pass
        else:
            image = fiducial_dict[image_keys[idx-1]]
            offset_image = fiducial_dict[image_keys[idx]]

            shift, error, diffphase = register_translation(image, offset_image)
            image_shifts[idx, :] = image_shifts[idx-1, :] + shift[1:3]

    image_shifts = image_shifts.astype(int)
    aligned_dict = {}
    # apply image shifts
    for idx in range(num_images):
        im = image_dict[image_keys[idx]]
        non = lambda s: s if s < 0 else None
        mom = lambda s: max(0, s)
        padded = np.zeros_like(im)
        oy, ox = image_shifts[idx, :]
        padded[:, mom(oy):non(oy), mom(ox):non(ox)] = im[:, mom(-oy):non(-oy), mom(-ox):non(-ox)]
        aligned_dict[image_keys[idx]] = padded.copy()

    return aligned_dict


def crop_images(aligned_dict):
    """ Crops black space from edges of images after alignment has been applied.

    Args:
        aligned_dict (dict): Dictionary where keys are image IDs
            ('readoutName') and values are the images with black space to be cropped away

    Returns:
        crop_dict (dict): Dictionary where keys are image IDs ('readoutName') and values are
        cropped images

    """
    crop_dict = {}

    crop_bool = np.array(list(aligned_dict.values())) > 0
    crop_bool_all = np.min(crop_bool, axis=0)

    top = 0
    while np.array([crop_bool_all[0, :, :, 0][top] == 0]).any():
        top += 1
    bottom = np.shape(crop_bool_all)[1]-1
    while np.array([crop_bool_all[0, :, :, 0][bottom] == 0]).any():
        bottom -= 1

    left = 0
    while np.array([crop_bool_all[0, :, :, 0][:, left] == 0]).any():
        left += 1
    right = np.shape(crop_bool_all)[2]-1
    while np.array([crop_bool_all[0, :, :, 0][:, right] == 0]).any():
        right -= 1

    for item in aligned_dict.keys():
        crop_temp = aligned_dict[item] * crop_bool_all
        crop_dict[item] = crop_temp[:, top:bottom, left:right, :]

    return(crop_dict)


def multiplex_match_spots_to_cells(coords_dict, cytoplasm_pred):
    """Matches detected spots to labeled cell cytoplasms

    Args:
        coords_dict (dict): Dictionary where keys are image IDs
            ('readoutName') and values are coordinates of detected spots
        cytoplasm_pred (matrix): Image where pixel values are labels for segmented cell
            cytoplasms

    Returns:
        spots_to_cells_dict (dict): Dictionary of dictionaries, keys are image IDs (readoutName),
            values are dictionaries where keys are cell cytoplasm labels and values are detected
            spots associated with that cell label, there is one item in list for each image in
            coords_dict
    """
    coords = [item[0] for item in coords_dict.values()]
    keys = list(coords_dict.keys())

    spots_to_cells_dict = {}

    for i in range(len(coords)):
        matched_spot_dict = match_spots_to_cells(cytoplasm_pred, coords[i])
        spots_to_cells_dict[keys[i]] = matched_spot_dict

    return(spots_to_cells_dict)


def gene_count(spots_to_cells_dict, threshold, codebook):
    """Converts detected spot coordinates and codebook into a count of detected transcripts for
    each gene for a specified gene.

    Args:
        spots_to_cells_dict (list): List of dictionaries, dictionary keys are cell
            cytoplasm labels and values are detected spots associated with that cell label,
            there is one item in list for each image in coords_dict
        threshold (float): Distance in pixels within which detections will be connected during
            barcode assignment
        codebook (pandas DataFrame): Data frame with columns for each imaging round, rows are
            barcodes for genes values in data frame are 0 if that barcode includes that imaging
            round and 1 if the barcode does not

    Returns:
        gene_count_per_cell (dict): Dictionary of dictionaries where the keys are the cell IDs
            assigned during cytoplasmic segmentation, values are dictionaries where keys are
            names of genes and values are counts detected in that cell
    """
    gene_count_per_cell = {}
    col_names = list(spots_to_cells_dict.keys())
    codebook = codebook[['name']+col_names]

    for cell_id in spots_to_cells_list[col_names[0]].keys():
        # Get all detected spots for the given cell_id
        cell_coords = []
        for name in col_names:
            cell_coords.append(spots_to_cells_list[name][cell_id])

        # Pairwise distances
        try:
            flatten_cell_coords = np.vstack([item for item in cell_coords if len(item) > 0])
        except ValueError:
            return('No spots detected in this cell')
        distance_mat = distance.cdist(flatten_cell_coords, flatten_cell_coords, 'euclidean')

        # Find spots closer than threshold distance
        A = distance_mat < threshold
        A = np.array(A).astype(int)
        sum_A = np.sum(A, axis=1)

        # Get clusters with valid barcodes
        filter_A = np.squeeze(A[np.argwhere(sum_A == 4)], axis=1)
        num_coords = np.array([len(item) for item in cell_coords])
        running_total = [0] + [sum(num_coords[:i+1]) for i in range(len(num_coords))]

        # Assign barcodes
        barcodes = []
        for i in range(len(running_total)-1):
            barcodes.append(np.sum(filter_A[:, running_total[i]:running_total[i+1]], axis=1))
        barcodes = np.array(barcodes).T

        if len(barcodes) == 0:
            return('No spots detected in this cell')

        codebook_dict = {}
        for i in range(len(codebook)):
            codebook_dict[str(list(codebook.loc[i].values[1:]))] = codebook.loc[i].values[0]

        gene_count = {}
        for i in range(len(barcodes)):
            try:
                gene = codebook_dict[str(list(barcodes[i]))]

                if gene in gene_count:
                    gene_count[gene] += 1
                else:
                    gene_count[gene] = 1
            except KeyError:
                pass

        sorted_gene_count = collections.OrderedDict(sorted(gene_count.items()))

        gene_count_per_cell[cell_id] = sorted_gene_count

    return(gene_count_per_cell)

def assign_gene_identities(cp_dict, dataorg, threshold, codebook):
    # Create array from classification prediction dictionary
    cp_array = np.array(list(cp_dict.values()))[:,1,0,:,:,1]
    
    # Create maximum projection
    max_cp = np.max(cp_array, axis=0)
    # Convert classification prediction to list of points
    coords = peak_local_max(max_cp, threshold_abs=threshold)
    
    # Prepare spot intensities for postcode
    spots_s = []
    coords_list = []
    for c in coords:
        ints = cp_array[:,c[0],c[1]]
        coords_list.append([c[0],c[1]])
        spots_s.append(ints)

    spots_s = np.array(spots_s)
    coords_array = np.array(coords_list)
    
    r = dataorg['imagingRound'].unique()
    c = dataorg['color'].unique()
    
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
    
    barcodes_01 = np.reshape(full_codebook.values[:,1:], (len(full_codebook), r, c))
    barcodes_01 = np.swapaxes(barcodes_01, 1, 2).astype(int)
    
    # Predict gene identities with postcode
    out = decoding_function(spots_s, barcodes_01, up_prc_to_remove=100, print_training_progress=True)
    
    # Write results into pandas dataframe
    df_class_names = np.concatenate((codebook['name'].values,['infeasible','background','nan']))
    df_class_codes = np.concatenate((np.arange(len(df_class_names)),['inf','0000','NA']))
    decoded_spots_df = decoding_output_to_dataframe(out, df_class_names, df_class_codes)
    decoded_spots_df['X'] = coords_array[:,0]
    decoded_spots_df['Y'] = coords_array[:,1]
    
    return(decoded_spots_df)

def assign_spots_to_cells(decoded_spots_df, labeled_im_cyto):

    cell_list = []
    for i in range(len(decoded_spots_df)):
        cell_list.append(labeled_im_cyto[0,decoded_spots_df.iloc[i]['X'],decoded_spots_df.iloc[i]['Y'],0])

    decoded_spots_df['Cell'] = cell_list
    
    return(decoded_spots_df)
