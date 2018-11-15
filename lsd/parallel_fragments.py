from __future__ import division
from .fragments import watershed_from_affinities
from .labels import relabel
from scipy.ndimage.measurements import center_of_mass
import logging
import numpy as np
import daisy

logger = logging.getLogger(__name__)

def parallel_watershed(
        affs,
        rag_provider,
        block_size,
        context,
        fragments_out,
        num_workers,
        fragments_in_xy=False,
        epsilon_agglomerate=0,
        mask=None):
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

        fragments_in_xy (``bool``):

            Whether to extract fragments for each xy-section separately.

        epsilon_agglomerate (``float``):

            Perform an initial waterz agglomeration on the extracted fragments
            to this threshold. Skip if 0 (default).

        mask (`class:daisy.Array`):

            A dataset containing a mask. If given, fragments are only extracted
            for masked-in (==1) areas.

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

    return daisy.run_blockwise(
        total_roi,
        read_roi,
        write_roi,
        lambda b: watershed_in_block(
            affs,
            b,
            rag_provider,
            fragments_out,
            fragments_in_xy,
            epsilon_agglomerate,
            mask),
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
        fragments_in_xy,
        epsilon_agglomerate,
        mask):

    total_roi = affs.roi

    logger.debug("reading affs from %s", block.read_roi)
    affs = affs.intersect(block.read_roi)
    affs.materialize()

    if mask is not None:

        logger.debug("reading mask from %s", block.read_roi)
        mask = mask.to_ndarray(affs.roi, fill_value=0)
        logger.debug("masking affinities")
        affs.data *= mask

    # extract fragments
    fragments_data, n = watershed_from_affinities(
        affs.data,
        fragments_in_xy=fragments_in_xy,
        epsilon_agglomerate=epsilon_agglomerate)
    if mask is not None:
        fragments_data *= mask.astype(np.uint64)
    fragments = daisy.Array(fragments_data, affs.roi, affs.voxel_size)

    # crop fragments to write_roi
    fragments = fragments[block.write_roi]
    fragments.materialize()

    # ensure we don't have IDs larger than the number of voxels (that would
    # break uniqueness of IDs below)
    max_id = fragments.data.max()
    if max_id > block.write_roi.size():
        logger.warning(
            "fragments in %s have max ID %d, relabelling...",
            block.write_roi, max_id)
        fragments.data, n = relabel(fragments.data)

    # ensure unique IDs
    size_of_voxel = daisy.Roi((0,)*affs.roi.dims(), affs.voxel_size).size()
    num_voxels_in_block = block.requested_write_roi.size()//size_of_voxel
    id_bump = block.block_id*num_voxels_in_block
    logger.debug("bumping fragment IDs by %i", id_bump)
    fragments.data[fragments.data>0] += id_bump
    fragment_ids = range(id_bump + 1, id_bump + 1 + n)

    # store fragments
    logger.debug("writing fragments to %s", block.write_roi)
    fragments_out[block.write_roi] = fragments

    # following only makes a difference if fragments were found
    if n == 0:
        return

    # get fragment centers
    fragment_centers = {
        fragment: block.write_roi.get_offset() + affs.voxel_size*center
        for fragment, center in zip(
            fragment_ids,
            center_of_mass(fragments.data, fragments.data, fragment_ids))
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
    rag.sync_nodes()
