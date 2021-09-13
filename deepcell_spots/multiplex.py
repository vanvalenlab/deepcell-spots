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

def read_images(root_dir, image_files, verbose=True):

    max_im_dict = {}
    fiducial_dict = {}
    cytoplasm_dict = {}

    for i in range(len(image_files)):

        # Slice data organization table to get information for this image stack
        round_df = dataorg.loc[dataorg['fileName'] == image_files[i]]
        image_file = os.path.join(root_dir, image_files[i])

        # Load image stack
        load_mat = scipy.io.loadmat(image_file)
        image_stack = load_mat['test_read_dax']

        rounds = round_df['readoutName'].values
        rounds = [item for item in rounds if item in codebook.columns]

        for item in rounds:
            if verbose:
                print('Working on: {}'.format(item))

            # Get frames of round in image stack
            frame_list = round_df['frame'].loc[round_df['readoutName'] == item].values[0]
            frame_list = frame_list.strip('][').split(', ')
            frame_list = np.array(frame_list).astype(int)

            start_frame = frame_list[0]
            end_frame = frame_list[-1]

            max_im = np.max(image_stack[:, :, start_frame:end_frame+1], axis=2)

            im = np.clip(max_im, np.min(max_im), np.percentile(max_im, 99.9))
            im = np.expand_dims(im, axis=[0, -1])

            max_im_dict[item] = im
            fiducial_dict[item] = np.expand_dims(image_stack[:, :, 29], axis=[0, -1])
            cytoplasm_dict[item] = np.expand_dims(image_stack[:, :, 24], axis=[0, -1])

    return(max_im_dict, fiducial_dict, cytoplasm_dict)


def align_images(image_dict, fiducial_dict):

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
    crop_dict = {}

    crop_bool = np.array(list(aligned_dict.values())) > 0
    crop_bool_all = np.min(crop_bool, axis=0)

    top = 0
    while np.array([crop_bool_all[0, :, :, 0][top] == 0]).all():
        top += 1
    bottom = np.shape(crop_bool_all)[1]-1
    while np.array([crop_bool_all[0, :, :, 0][bottom] == 0]).all():
        bottom -= 1

    left = 0
    while np.array([crop_bool_all[0, :, :, 0][:, left] == 0]).all():
        left += 1
    right = np.shape(crop_bool_all)[2]-1
    while np.array([crop_bool_all[0, :, :, 0][:, right] == 0]).all():
        right -= 1

    for item in aligned_dict.keys():
        crop_temp = aligned_dict[item] * crop_bool_all
        crop_dict[item] = crop_temp[:, top:bottom, left:right, :]

    return(crop_dict)


def gene_count(spots_to_cells_list, cell_id, threshold, codebook):
    # Get all detected spots for the given cell_id
    cell_coords = []
    for i in range(len(spots_to_cells_list)):
        cell_coords.append(spots_to_cells_list[i][cell_id])

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
        codebook_dict[str(list(codebook.loc[i].values[1:-1]))] = codebook.loc[i].values[0]

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

    return(sorted_gene_count)
