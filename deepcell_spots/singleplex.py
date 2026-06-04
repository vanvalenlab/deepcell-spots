"""Tools for analysis of singleplex FISH images"""

from collections import defaultdict

import numpy as np


def match_spots_to_cells(labeled_im, coords):
    """Assigns detected spots to regions of a labeled image.

    Returns a dictionary where keys are labeled regions of input image and
    values are spot coordinates corresponding with that labeled region.

    Args:
        labeled_im (array): Image output from segmentation algorithm with
            dimensions `(1,x,y,1)` where pixels values label regions of the
            image corresponding with objects of interest
            (nuclei, cytoplasm, etc.).
        coords (array): Array of coordinates for spot location with dimensions
            `(number of spots,2)`.

    Returns:
        dict: Dictionary where keys are labeled regions of input image and
        values are spot coordinates corresponding with that labeled region.
    """

    spot_dict = defaultdict(list)
    for i in range(np.shape(coords)[0]):
        assigned_cell = labeled_im[0, int(coords[i][0]), int(coords[i][1]), 0]

        if assigned_cell in spot_dict.keys():
            spot_dict[assigned_cell].append(coords[i])
        else:
            spot_dict[assigned_cell] = [coords[i]]

    return spot_dict


def match_spots_to_cells_as_vec(labeled_im, coords):
    """Assigns detected spots to regions of a labeled image.

    Returns a dictionary where keys are labeled regions of input image and
    values are spot coordinates corresponding with that labeled region.

    Args:
        labeled_im (array): Image output from segmentation algorithm with
            dimensions `(1,x,y,1)` where pixels values label regions of the
            image corresponding with objects of interest
            (nuclei, cytoplasm, etc.).
        coords (array): Array of coordinates for spot location with dimensions
            `(number of spots,2)`.

    Returns:
        array: Cell id for each spot.
    """
    coords = coords.astype(int)
    assigned_cell = labeled_im[0, coords[:, 0], coords[:, 1], 0]

    return assigned_cell


def match_spots_to_cells_as_vec_batched(segmentation_result, spots_locations):
    """Assigns detected spots to regions of a labeled image. For the whole batch.

    Args:
        segmentation_result (list): Each entry is a `(1,x,y,1)` image.
        spots_locations (list): Each entry is a `(number of spots,2)` array.

    Returns:
        spot_cell_assignments_vec (numpy.array): Array of cell ids, all batches
            concatenated.
    """
    spot_cell_assignments = []
    for i in range(len(spots_locations)):
        cell_id_list = match_spots_to_cells_as_vec(segmentation_result[i:i + 1],
                                                   spots_locations[i])
        spot_cell_assignments.append(cell_id_list)
    spot_cell_assignments_vec = np.concatenate(spot_cell_assignments)
    return spot_cell_assignments_vec


def process_spot_dict(spot_dict):
    """Processes spot dictionary into an array of coordinates and list of
    region labels for spots.

    Args:
        spot_dict (dict): Dictionary where keys are labeled regions of input
            image and values are spot coordinates corresponding with that
            labeled region.

    Returns:
        (array, list): (1) Array of coordinates for spot location with dimensions
        `(number of spots,2)`. Re-ordered to correspond with list of region
        labels. (2) List of region labels corresponding with coordinates.
        Intended to be used to color a ``cmap`` when visualizing spots.
    """
    coords = []
    cmap_list = []
    for key in spot_dict.keys():
        for item in spot_dict[key]:
            coords.append(item)
            cmap_list.append(key)

    coords = np.array(coords)
    return coords, cmap_list


def remove_nuc_spots_from_cyto(labeled_im_nuc, labeled_im_cyto, coords):
    """Removes spots in nuclear regions from spots assigned to cytoplasmic
    regions.

    Returns a dictionary where keys are labeled cytoplasmic regions of input
    image and values are spot coordinates corresponding with that labeled
    cytoplasm region.

    Args:
        labeled_im_nuc (array): Image output from segmentation algorithm with
            dimensions `(1,x,y,1)` where pixels values label nuclear regions.
        labeled_im_cyto (array): Image output from segmentation algorithm with
            dimensions `(1,x,y,1)` where pixels values label cytoplasmic regions.
        coords (array): Array of coordinates for spot location with dimensions
            `(number of spots,2)`.

    Returns:
        dict: Dictionary where keys are labeled regions of input image and
            values are spot coordinates corresponding with that labeled region
            (cytoplasm excluding nucleus).
    """
    # Match spots to nuclei and cytoplasms
    spot_dict_nuc = match_spots_to_cells(labeled_im_nuc, coords)
    spot_dict_cyto = match_spots_to_cells(labeled_im_cyto, coords)

    # Remove spots assigned to background in nuclei image
    del spot_dict_nuc[0]
    nuclear_spots = np.vstack(np.array(list(spot_dict_nuc.values())))

    # Fill new dictionary with spots outside of nuclei
    spot_dict_cyto_updated = defaultdict(list)
    for i in range(np.shape(coords)[0]):
        if coords[i] not in nuclear_spots:
            # Assign spots according to cytoplasm labels
            assigned_cell = labeled_im_cyto[0, int(
                coords[i][0]), int(coords[i][1]), 0]

        # Assign spots to background if inside nuclei
        else:
            assigned_cell = 0

        # Add to dictionary
        if assigned_cell in spot_dict_cyto_updated.keys():
            spot_dict_cyto_updated[assigned_cell].append(coords[i])
        else:
            spot_dict_cyto_updated[assigned_cell] = [coords[i]]

    return spot_dict_cyto_updated
