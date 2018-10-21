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

def watershed_from_affinities(affs, fragments_in_xy=False, return_seeds=False):
    '''Extract initial fragments from affinities using a watershed
    transform. Returns the fragments and the maximal ID in it.'''

    if affs.dtype == np.uint8:
        max_affinity_value = 255.0
    else:
        max_affinity_value = 1.0

    if fragments_in_xy:

        mean_affs = 0.5*(affs[1] + affs[2])
        depth = mean_affs.shape[0]

        fragments = np.zeros(mean_affs.shape, dtype=np.uint64)
        if return_seeds:
            seeds = np.zeros(mean_affs.shape, dtype=np.uint64)

        id_offset = 0
        for z in range(depth):

            boundary_mask = mean_affs[z]>0.5*max_affinity_value
            boundary_distances = distance_transform_edt(boundary_mask)

            ret = watershed_from_boundary_distance(
                boundary_distances,
                return_seeds=return_seeds,
                id_offset=id_offset)

            fragments[z] = ret[0]
            if return_seeds:
                seeds[z] = ret[2]

            id_offset = ret[1]

        ret = (fragments, id_offset)
        if return_seeds:
            ret += (seeds,)

        return ret

    else:

        boundary_mask = np.mean(affs, axis=0)>0.5*max_affinity_value
        boundary_distances = distance_transform_edt(boundary_mask)

        return watershed_from_boundary_distance(
            boundary_distances,
            return_seeds)

def watershed_from_boundary_distance(
        boundary_distances,
        return_seeds=False,
        id_offset=0):

    max_filtered = maximum_filter(boundary_distances, 10)
    maxima = max_filtered==boundary_distances
    seeds, n = mahotas.label(maxima)

    logger.debug("Found %d fragments", n)

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds!=0] += id_offset

    fragments = mahotas.cwatershed(
        boundary_distances.max() - boundary_distances,
        seeds)

    ret = (fragments.astype(np.uint64), n + id_offset)
    if return_seeds:
        ret = ret + (seeds.astype(np.uint64),)

    return ret
