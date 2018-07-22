import mahotas
import numpy as np
import logging
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter, maximum_filter

logger = logging.getLogger(__name__)

def watershed(lsds, sigma, return_seeds=False, return_distances=False):
    '''Extract initial fragments from local shape descriptors ``lsds`` using a
    watershed transform. This assumes that the first three entries of
    ``lsds`` for each voxel are vectors pointing towards the center.'''

    boundary_distances = np.sum(lsds[0:3,:]**2, axis=0)
    boundary_distances = gaussian_filter(boundary_distances, sigma)
    boundary_distances = boundary_distances.max() - boundary_distances

    ret = watershed_from_boundary_distance(boundary_distances, return_seeds)

    if return_distances:
        ret = ret + (boundary_distances,)

    return ret

def watershed_from_affinities(affs, return_seeds=False, return_distances=False):
    '''Extract initial fragments from affinities using a watershed
    transform.'''

    boundary_mask = np.mean(affs, axis=0)>0.5
    boundary_distances = distance_transform_edt(boundary_mask)

    ret = watershed_from_boundary_distance(boundary_distances, return_seeds)

    if return_distances:
        ret = ret + (boundary_distances,)

    return ret

def watershed_from_boundary_distance(boundary_distances, return_seeds=False):

    max_filtered = maximum_filter(boundary_distances, 10)
    maxima = max_filtered==boundary_distances
    seeds, n = mahotas.label(maxima)

    logger.debug("Found %d fragments", n)

    if n == 0:
        return np.zeros(affs.shape[1:], dtype=np.uint64), 0

    fragments = mahotas.cwatershed(
        boundary_distances.max() - boundary_distances,
        seeds)

    ret = (fragments.astype(np.uint64), n)
    if return_seeds:
        ret = ret + (seeds.astype(np.uint64),)

    return ret
