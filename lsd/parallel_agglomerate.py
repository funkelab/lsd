from agglomerate import LsdAgglomeration
from peach import Coordinate, Roi, process_blockwise
import logging
import luigi
import numpy as np

logger = logging.getLogger(__name__)

class ParallelLsdAgglomeration(object):
    '''Create a local shape descriptor agglomerator for parallel agglomeration.

    For large volumes, ``rag``, ``fragments``, and ``target_lsds`` should be
    held in an out-of-memory datastructure that supports slicing to read
    subgraphs/subarrays.

    Args:

        rag_provider (`class:SharedRagProvider`):

            A RAG provider to query for subgraphs. Results will be written back
            to this provider in mutually exclusive writes.

        fragments (array-like):

            Label array of the nodes in the RAG provided by ``rag_provider``.

        target_lsds (array-like):

            The local shape descriptors to match.

        lsd_extractor (``LsdExtractor``):

            The local shape descriptor object used to compute the difference
            between the segmentation and the target LSDs.

        block_write_size (``tuple`` of ``int``):

            The write size of the blocks processed in parallel in world units.

            Note that due to context needed to compute the LSDs the actual read
            size per block is larger. The necessary context is automatically
            determined from the given ``lsd_extractor``. Results will only be
            computed and written back for blocks that fit entirely into the
            total ROI.

        block_done_function (function):

            A function that is passed a `class:Roi` and should return ``True``
            if the block represented by this ROI was already processed. Only
            blocks for which this function returns ``False`` are processed.

        num_workers (``int``):

            The number of blocks to process in parallel.

        voxel_size (``tuple`` of ``int``, optional):

            The voxel size of ``fragments``. Defaults to 1.
    '''

    def __init__(
            self,
            rag_provider,
            fragments,
            target_lsds,
            lsd_extractor,
            block_write_size,
            block_done_function,
            num_workers,
            voxel_size=None):

        self.rag_provider = rag_provider
        self.fragments = fragments
        self.target_lsds = target_lsds
        self.lsd_extractor = lsd_extractor
        self.block_write_size = block_write_size
        self.block_done_function = block_done_function
        self.num_workers = num_workers
        self.voxel_size = voxel_size

    def merge_until(self, threshold, max_merges=-1):
        '''Merge until the given threshold. Since edges are scored by how much
        they decrease the distance to ``target_lsds``, a threshold of 0 should
        be optimal.'''

        logger.info("Merging until %f...", threshold)

        dims = len(self.fragments.shape)
        if not self.voxel_size:
            self.voxel_size = Coordinate((1,)*dims)

        voxel_size = self.voxel_size

        context = Coordinate(self.lsd_extractor.get_context())

        # assure that context is a multiple of voxel size
        one = Coordinate((1,)*len(context))
        context = ((context - one)/voxel_size + one)*voxel_size

        total_roi = Roi((0,)*dims, voxel_size*self.fragments.shape)
        write_roi = Roi((0,)*dims, self.block_write_size)
        read_roi = write_roi.grow(context, context)

        assert (write_roi/voxel_size)*voxel_size == write_roi, (
            "block_write_size needs to be a multiple of voxel_size")
        assert (write_roi/voxel_size)*voxel_size == write_roi, (
            "read_roi needs to be a multiple of voxel_size")

        process_blockwise(
            total_roi,
            read_roi,
            write_roi,
            lambda r, w: self.__agglomerate_block(r, w),
            self.block_done_function,
            self.num_workers)

    def __agglomerate_block(self, read_roi, write_roi):

        logger.info(
            "Agglomerating in block %s with context of %s",
            write_roi, read_roi)

        read_roi_voxels = read_roi/self.voxel_size
        write_roi_voxels = write_roi/self.voxel_size

        # get the subgraph to work on
        rag = self.rag_provider[read_roi.to_slices()]

        # get currently connected componets
        components = rag.get_connected_components()

        # replace each connected component by a single node
        component_nodes = self.__contract_rag(rag, components)

        # get fragments slice
        fragments = self.fragments[read_roi_voxels.to_slices()]

        # relabel fragments of the same connected components to match RAG
        fragments = self.__relabel(fragments, components, component_nodes)

        # get LSDs slice
        target_lsds = self.target_lsds[(slice(None),) + read_roi_voxels.to_slices()]

        logger.info("fragments: %s", fragments.shape)
        logger.info("target_lsds: %s", target_lsds.shape)

        # agglomerate on a copy of the original RAG (agglomeration changes the
        # RAG)
        agglomeration = LsdAgglomeration(
            fragments,
            target_lsds,
            self.lsd_extractor,
            voxel_size=self.voxel_size,
            rag=rag.copy())
        num_merged = agglomeration.merge_until(0)

        # TODO:
        # * get record of merged edges
        # * merge only edges inside write_roi
        #   OR
        #   set 'merged' attribute only for edges inside write_roi

        # for now, just mark the block as processed
        rag.set_edge_attributes('agglomerated', 1)

        logger.info("merged %d edges", num_merged)

        # write back results
        rag.sync()

    def __contract_rag(self, rag, components):
        '''Contract all nodes of one component into a single node, return the
        single node for each component.'''

        component_nodes = []

        for component in components:

            for i in range(1, len(component)):
                rag.merge_nodes(component[i - 1], component[i])

            component_nodes.append(component[-1])

        return component_nodes

    def __relabel(self, array, components, component_labels):

        values_map = np.arange(int(array.max() + 1), dtype=array.dtype)

        for component, label in zip(components, component_labels):
            for c in component:
                if c < len(values_map):
                    values_map[c] = label

        return values_map[array]
