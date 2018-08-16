from .fragments import watershed_from_affinities
from .labels import relabel
from scipy.ndimage.measurements import center_of_mass
import logging
import numpy as np
import peach

logger = logging.getLogger(__name__)

def parallel_watershed(
        affs,
        rag_provider,
        block_size,
        context,
        fragments_out,
        num_workers,
        fragments_in_xy=False):
    '''Extract fragments from affinities using watershed.

    Args:

        affs (array-like):

            A dataset that supports slicing to get affinities.

        rag_provider (`class:SharedRagProvider`):

            A RAG provider to write nodes for extracted fragments to. This does
            not yet add adjacency edges, for that, an agglomeration method
            should be called after this function.

        block_size (``tuple`` of ``int``):

            The size of the blocks to process in parallel in voxels.

        context (``tuple`` of ``int``):

            The context to consider for fragment extraction, in voxels.

        fragments_out (array-like):

            A dataset that supports slicing to store fragments in. Should be of
            ``dtype`` ``uint64``.

        num_workers (``int``):

            The number of parallel workers.

        fragments_in_xy (``bool``):

            Whether to extract fragments for each xy-section separately.

    Returns:

        True, if all tasks succeeded.
    '''

    assert fragments_out.dtype == np.uint64

    shape = affs.shape[1:]

    if context is None:
        context = peach.Coordinate((0,)*len(shape))
    else:
        context = peach.Coordinate(context)

    total_roi = peach.Roi((0,)*len(shape), shape).grow(context, context)
    read_roi = peach.Roi((0,)*len(shape), block_size).grow(context, context)
    write_roi = peach.Roi((0,)*len(shape), block_size)

    return peach.run_blockwise(
        total_roi,
        read_roi,
        write_roi,
        lambda b: watershed_in_block(
            affs,
            b,
            rag_provider,
            fragments_out,
            fragments_in_xy),
        lambda b: block_done(b, rag_provider),
        num_workers=num_workers,
        read_write_conflict=False,
        fit='shrink')

def block_done(block, rag_provider):

    rag = rag_provider[block.write_roi.to_slices()]
    logger.debug("%d nodes in %s", rag.number_of_nodes(), block.write_roi)
    return rag.number_of_nodes() > 0

def watershed_in_block(
        affs,
        block,
        rag_provider,
        fragments_out,
        fragments_in_xy):

    shape = affs.shape[1:]
    affs_roi = peach.Roi((0,)*len(shape), shape)

    # ensure read_roi is within bounds of affs.shape
    read_roi = affs_roi.intersect(block.read_roi)
    write_roi = block.write_roi

    logger.debug("reading affs from %s", read_roi)
    affs = affs[(slice(None),) + read_roi.to_slices()]
    fragments, n = watershed_from_affinities(affs, fragments_in_xy=fragments_in_xy)

    if write_roi != read_roi:

        # get fragments in write_roi
        write_in_read_roi = write_roi - read_roi.get_offset()
        logger.debug("cropping fragment array to %s", write_in_read_roi)
        fragments = fragments[write_in_read_roi.to_slices()]

        # ensure we don't have IDs larger than the number of voxels (that would
        # break uniqueness of IDs below)
        max_id = fragments.max()
        if max_id > write_roi.size():
            logger.warning(
                "fragments in %s have max ID %d, relabelling...",
                write_roi, max_id)
            fragments, n = relabel(fragments)

    # ensure unique IDs
    num_blocks = peach.Coordinate(shape)/write_roi.get_shape()
    block_index = write_roi.get_offset()/write_roi.get_shape()
    # id_bump ensures that IDs are unique, even if every voxel was a fragment
    id_bump = (
        block_index[0]*num_blocks[1]*num_blocks[2] +
        block_index[1]*num_blocks[2] +
        block_index[2])*write_roi.size()

    fragments[fragments>0] += id_bump
    fragment_ids = range(id_bump + 1, id_bump + 1 + n)

    # store fragments
    logger.debug("writing fragments to %s", write_roi)
    fragments_out[write_roi.to_slices()] = fragments

    # following only makes a difference if fragments were found
    if n == 0:
        return

    # get fragment centers
    fragment_centers = {
        fragment: write_roi.get_offset() + center
        for fragment, center in zip(
            fragment_ids,
            center_of_mass(fragments, fragments, fragment_ids))
        if not np.isnan(center[0])
    }

    # store nodes
    rag = rag_provider[write_roi.to_slices()]
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
