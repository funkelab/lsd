from __future__ import division
from .fragments import watershed_from_affinities
from funlib.segment.arrays import relabel, replace_values
from scipy.ndimage import measurements
import daisy
import logging
import numpy as np
import waterz

logger = logging.getLogger(__name__)

def upsample(a, factor):

    for d, f in enumerate(factor):
        a = np.repeat(a, f, axis=d)

    return a

def get_mask_data_in_roi(mask, roi, target_voxel_size):

    assert mask.voxel_size.is_multiple_of(target_voxel_size), (
        "Can not upsample from %s to %s" % (mask.voxel_size, target_voxel_size))

    aligned_roi = roi.snap_to_grid(mask.voxel_size, mode='grow')
    aligned_data = mask.to_ndarray(aligned_roi, fill_value=0)

    if mask.voxel_size == target_voxel_size:
        return aligned_data

    factor = mask.voxel_size/target_voxel_size

    upsampled_aligned_data = upsample(aligned_data, factor)

    upsampled_aligned_mask = daisy.Array(
        upsampled_aligned_data,
        roi=aligned_roi,
        voxel_size=target_voxel_size)

    return upsampled_aligned_mask.to_ndarray(roi)

def parallel_watershed(
        affs,
        rag_provider,
        block_size,
        context,
        fragments_out,
        num_workers,
        mask=None,
        fragments_in_xy=False,
        epsilon_agglomerate=0.0,
        filter_fragments=0.0):
    '''Extract fragments from affinities using watershed.

    Args:

        affs (`class:daisy.Array`):

            An array containing affinities.

        rag_provider (`class:SharedRagProvider`):

            A RAG provider to write nodes for extracted fragments to. This does
            not yet add adjacency edges, for that, an agglomeration method
            should be called after this function.

        block_size (``tuple`` of ``int``):

            The size of the blocks to process in parallel in world units.

        context (``tuple`` of ``int``):

            The context to consider for fragment extraction, in world units.

        fragments_out (`class:daisy.Array`):

            An array to store fragments in. Should be of ``dtype`` ``uint64``.

        num_workers (``int``):

            The number of parallel workers.

        mask (`class:daisy.Array`):

            A dataset containing a mask. If given, fragments are only extracted
            for masked-in (==1) areas.

        fragments_in_xy (``bool``):

            Whether to extract fragments for each xy-section separately.

        epsilon_agglomerate (``float``):

            Perform an initial waterz agglomeration on the extracted fragments
            to this threshold. Skip if 0 (default).

        filter_fragments (float):

            Filter fragments that have an average affinity lower than this
            value.

    Returns:

        True, if all tasks succeeded.
    '''

    assert fragments_out.data.dtype == np.uint64

    if context is None:
        context = daisy.Coordinate((0,)*affs.roi.dims())
    else:
        context = daisy.Coordinate(context)

    total_roi = affs.roi.grow(context, context)
    read_roi = daisy.Roi((0,)*affs.roi.dims(), block_size).grow(context, context)
    write_roi = daisy.Roi((0,)*affs.roi.dims(), block_size)

    num_voxels_in_block = (write_roi/affs.voxel_size).size()

    return daisy.run_blockwise(
        total_roi,
        read_roi,
        write_roi,
        lambda b: watershed_in_block(
            affs=affs,
            block=b,
            rag_provider=rag_provider,
            fragments_out=fragments_out,
            num_voxels_in_block=num_voxels_in_block,
            fragments_in_xy=fragments_in_xy,
            epsilon_agglomerate=epsilon_agglomerate,
            mask=mask,
            filtered_fragments=filtered_fragments),
        lambda b: block_done(b, rag_provider),
        num_workers=num_workers,
        read_write_conflict=False,
        fit='shrink')

def block_done(block, rag_provider):

    return rag_provider.num_nodes(block.write_roi) > 0

def watershed_in_block(
        affs,
        block,
        rag_provider,
        fragments_out,
        num_voxels_in_block,
        mask=None,
        fragments_in_xy=False,
        epsilon_agglomerate=0.0,
        filter_fragments=0.0):
    '''

    Args:

        filter_fragments (float):

            Filter fragments that have an average affinity lower than this
            value.
    '''

    total_roi = affs.roi

    logger.debug("reading affs from %s", block.read_roi)

    affs = affs.intersect(block.read_roi)
    affs.materialize()

    if affs.dtype == np.uint8:
        logger.info("Assuming affinities are in [0,255]")
        max_affinity_value = 255.0
        affs.data = affs.data.astype(np.float32)
    else:
        max_affinity_value = 1.0

    if mask is not None:

        logger.debug("reading mask from %s", block.read_roi)
        mask_data = get_mask_data_in_roi(mask, affs.roi, affs.voxel_size)
        logger.debug("masking affinities")
        affs.data *= mask_data

    # extract fragments
    fragments_data, _ = watershed_from_affinities(
        affs.data,
        max_affinity_value,
        fragments_in_xy=fragments_in_xy)

    if mask is not None:
        fragments_data *= mask_data.astype(np.uint64)

    if filter_fragments > 0:

        if fragments_in_xy:
            average_affs = np.mean(affs.data[0:2]/max_affinity_value, axis=0)
        else:
            average_affs = np.mean(affs.data/max_affinity_value, axis=0)

        filtered_fragments = []

        fragment_ids = np.unique(fragments_data)

        for fragment, mean in zip(
                fragment_ids,
                measurements.mean(
                    average_affs,
                    fragments_data,
                    fragment_ids)):
            if mean < filter_fragments:
                filtered_fragments.append(fragment)

        filtered_fragments = np.array(
            filtered_fragments,
            dtype=fragments_data.dtype)
        replace = np.zeros_like(filtered_fragments)
        replace_values(fragments_data, filtered_fragments, replace, inplace=True)

    if epsilon_agglomerate > 0:

        logger.info(
            "Performing initial fragment agglomeration until %f",
            epsilon_agglomerate)

        generator = waterz.agglomerate(
                affs=affs.data/max_affinity_value,
                thresholds=[epsilon_agglomerate],
                fragments=fragments_data,
                scoring_function='OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>',
                discretize_queue=256,
                return_merge_history=False,
                return_region_graph=False)
        fragments_data[:] = next(generator)

        # cleanup generator
        for _ in generator:
            pass

    fragments = daisy.Array(fragments_data, affs.roi, affs.voxel_size)

    # crop fragments to write_roi
    fragments = fragments[block.write_roi]
    fragments.materialize()
    max_id = fragments.data.max()

    # ensure we don't have IDs larger than the number of voxels (that would
    # break uniqueness of IDs below)
    if max_id > num_voxels_in_block:
        logger.warning(
            "fragments in %s have max ID %d, relabelling...",
            block.write_roi, max_id)
        fragments.data, max_id = relabel(fragments.data)

        assert max_id < num_voxels_in_block

    # ensure unique IDs
    id_bump = block.block_id*num_voxels_in_block
    logger.debug("bumping fragment IDs by %i", id_bump)
    fragments.data[fragments.data>0] += id_bump
    fragment_ids = range(id_bump + 1, id_bump + 1 + max_id)

    # store fragments
    logger.debug("writing fragments to %s", block.write_roi)
    fragments_out[block.write_roi] = fragments

    # following only makes a difference if fragments were found
    if max_id == 0:
        return

    # get fragment centers
    fragment_centers = {
        fragment: block.write_roi.get_offset() + affs.voxel_size*center
        for fragment, center in zip(
            fragment_ids,
            measurements.center_of_mass(fragments.data, fragments.data, fragment_ids))
        if not np.isnan(center[0])
    }

    # store nodes
    rag = rag_provider[block.write_roi]
    rag.add_nodes_from([
        (node, {
            'center_z': c[0],
            'center_y': c[1],
            'center_x': c[2]
            }
        )
        for node, c in fragment_centers.items()
    ])
    rag.write_nodes(block.write_roi)
