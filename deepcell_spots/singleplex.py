"""Tools for analysis of singleplex FISH images"""

from collections import defaultdict
import numpy as np

def match_spots_to_cells(labeled_im,coords):
    """Assigns detected spots to regions of a labeled image. 

    Returns a dictionary where keys are labeled regions of input image and values are spot coordinates 
    corresponding with that labeled region.

    Parameters:
    ------------
    labeled_im : array
        Image output from segmentation algorithm with dimensions (1,x,y,1) where pixels values label regions 
        of the image corresponding with objects of interest (nuclei, cytoplasm, etc.)
    coords : array
        Array of coordinates for spot location with dimensions (number of spots,2)

    Returns:
    -----------
    spot_dict: dictionary
        Dictionary where keys are labeled regions of input image and values are spot coordinates 
        corresponding with that labeled region.

    """
    spot_dict = defaultdict(list)
    for i in range(np.shape(coords)[0]):
        assigned_cell = labeled_im[0, int(coords[i][0]), int(coords[i][1]), 0]
        
        if assigned_cell in spot_dict.keys():
            spot_dict[assigned_cell].append(coords[i])
        else:
            spot_dict[assigned_cell] = [coords[i]]
            
    return spot_dict

def process_spot_dict(spot_dict):
    """Processes spot dictionary into an array of coordinates and list of region labels for spots.

    Parameters:
    ------------
    spot_dict: dictionary
        Dictionary where keys are labeled regions of input image and values are spot coordinates 
        corresponding with that labeled region.

    Returns:
    -----------
    coords: array
        Array of coordinates for spot location with dimensions (number of spots,2). 
        Re-ordered to correspond with list of region labels.
    cmap_list: list
        List of region labels corresponding with coordinates. Intended to be used to color a cmap when visualizing spots.
    """
    coords = []
    cmap_list = []
    for key in spot_dict.keys():
        for item in spot_dict[key]:
            coords.append(item)
            cmap_list.append(key)

    coords = np.array(coords)    
    return coords,cmap_list

def remove_nuc_spots_from_cyto(labeled_im_nuc,labeled_im_cyto,coords):
    """Removes spots in nuclear regions from spots assigned to cytoplasmic regions.

    Returns a dictionary where keys are labeled cytoplasmic regions of input image and values are spot coordinates 
    corresponding with that labeled cytoplasm region.

    Parameters:
    ------------
    labeled_im_nuc : array
        Image output from segmentation algorithm with dimensions (1,x,y,1) where pixels values label nuclear regions.
    labeled_im_cyto : array
        Image output from segmentation algorithm with dimensions (1,x,y,1) where pixels values label cytoplasmic regions.
    coords : array
        Array of coordinates for spot location with dimensions (number of spots,2)

    Returns:
    -----------
    """
    spot_dict_nuc = match_spots_to_cells(labeled_im_nuc,coords)
    spot_dict_cyto = match_spots_to_cells(labeled_im_cyto,coords)
    
    del spot_dict_nuc[0]
    
    nuclear_spots = np.vstack(np.array(list(spot_dict_nuc.values())))
    
    spot_dict_cyto_updated = defaultdict(list)
    for i in range(np.shape(coords)[0]):
        if coords[i] not in nuclear_spots:
            assigned_cell = labeled_im_cyto[0, int(coords[i][0]), int(coords[i][1]), 0]
                
        else:
            assigned_cell = 0

        if assigned_cell in spot_dict_cyto_updated.keys():
            spot_dict_cyto_updated[assigned_cell].append(coords[i])
        else:
            spot_dict_cyto_updated[assigned_cell] = [coords[i]]
    
    return spot_dict_cyto_updated