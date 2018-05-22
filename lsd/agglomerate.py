from scipy.ndimage.measurements import find_objects
from skimage.future.graph import RAG, merge_hierarchical
import gunpowder as gp
import numpy as np
import logging

logger = logging.getLogger(__name__)

class LsdAgglomeration:
    '''Create a local shape descriptor agglomerator.

    Args:

        fragments (``np.ndarray``):

            Initial fragments.

        target_lsds (``np.ndarray``):

            The local shape descriptors to match.

        lsd_extractor (``LsdExtractor``):

            The local shape descriptor object used to compute the difference
            between the segmentation and the target LSDs.

        voxel_size (``tuple`` of ``int``, optional):

            The voxel size of ``fragments``. Defaults to 1.
    '''

    def __init__(
            self,
            fragments,
            target_lsds,
            lsd_extractor,
            voxel_size=None):

        self.segmentation = np.array(fragments)
        self.fragments = fragments
        self.target_lsds = target_lsds
        self.lsd_extractor = lsd_extractor

        if voxel_size is None:
            self.voxel_size = (1,)*len(fragments.shape)
        else:
            self.voxel_size = voxel_size

        self.__initialize_rag()

    def merge_until(self, threshold):
        '''Merge until the given threshold. Since edges are scored by how much
        they decrease the distance to ``target_lsds``, a threshold of 0 should
        be optimal.'''

        logger.info("Merging until %f...", threshold)

        merge_func = lambda _, src, dst: self.__update_lsds(src, dst)
        weight_func = lambda _g, _s, u, v: self.__score_merge(u, v)
        merge_hierarchical(
            self.fragments,
            self.rag,
            thresh=threshold,
            rag_copy=False,
            in_place_merge=True,
            merge_func=merge_func,
            weight_func=weight_func)

        logger.info("Finished merging")

    def get_segmentation(self):
        '''Return the segmentation obtained so far by calls to
        ``merge_until``.'''

        return self.segmentation

    def __initialize_rag(self):

        self.rag = RAG(self.fragments, connectivity=2)
        logger.info(
            "RAG contains %d nodes and %d edges",
            len(self.rag.nodes()),
            len(self.rag.edges()))

        logger.info("Computing LSDs for initial fragments...")

        for u in self.rag.nodes():

            logger.info("Initializing node %d", u)

            bb = find_objects(self.fragments==u)[0]
            self.rag.node[u]['roi'] = self.__slice_to_roi(bb)
            self.rag.node[u]['score'] = self.__compute_score(u)
            self.rag.node[u]['labels'] = [u] # needed by scikit

            logger.info("Node %d: %s", u, self.rag.node[u])

        logger.info("Scoring initial edges...")

        for (u, v) in self.rag.edges():
            score = self.__score_merge(u, v)
            self.rag[u][v]['weight'] = score['weight']

    def __update_lsds(self, src, dst):
        '''Callback for merge_hierarchical, called before src and dst are
        merged.'''

        # src and dst get merged into dst, update the stats for dst

        self.rag.node[dst]['score'] = self.__compute_score(
            src, dst,
            update_rois=True,
            update_segmentation=True)

        logger.debug(
            "Updated score of %d (merged with %d) to %f",
            dst, src, self.rag.node[dst]['score'])

    def __score_merge(self, u, v):
        '''Callback for merge_hierarchical, called to get the weight of a new
        edge.'''

        score_u = self.rag.node[u]['score']
        score_v = self.rag.node[v]['score']

        score_uv = self.__compute_score(u, v)

        weight = score_uv - (score_u + score_v)

        logger.debug("Scoring merge between %d and %d with %f", u, v, weight)

        return {'weight': weight}

    def __compute_score(self, u, v=None, update_rois=False, update_segmentation=False):
        '''Compute the LSDs score for either one (v=None) or two fragments
        together.

        If ``update_rois``, the ROI for v will be replaced with the ROI for
        u+v.

        If ``update_segmentation``, also replaces all occurences of u with v in
        ``self.segmentation``.'''

        # get ROI
        roi = self.rag.node[u]['roi']
        if v is not None:
            roi = roi.union(self.rag.node[v]['roi'])
        if update_rois:
            self.rag.node[v]['roi'] = roi

        # get slice of segmentation for dst roi
        roi_slice = self.segmentation[roi.get_bounding_box()]
        if not update_segmentation:
            roi_slice = np.array(roi_slice)

        if v is not None:
            # mark u as v
            roi_slice[roi_slice==u] = v
        else:
            v = u

        # get LSDs for u(+v)
        lsds = self.lsd_extractor.get_descriptors(
            roi_slice,
            labels=[v],
            voxel_size=self.voxel_size)

        # subtract from target LSDs
        diff = self.target_lsds[(slice(None),) + roi.get_bounding_box()] - lsds
        diff[:,roi_slice!=v] = 0

        return np.sum(diff**2)

    def __slice_to_roi(self, slices):

        offset = tuple(s.start for s in slices)
        shape = tuple(s.stop - s.start for s in slices)

        roi = gp.Roi(offset, shape)
        roi = roi.snap_to_grid((self.lsd_extractor.downsample,)*roi.dims())

        return roi
