from __future__ import absolute_import
from .labels import relabel
from scipy.ndimage.measurements import find_objects
import daisy
import logging
import math
import numpy as np

logger = logging.getLogger(__name__)

class RegionGrowing(object):
    '''Region growing on LSDs.

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
            voxel_size=None,
            log_prefix=''):

        self.lsds = np.zeros_like(target_lsds)
        self.fragments = fragments
        self.target_lsds = target_lsds
        self.lsd_extractor = lsd_extractor
        self.context = lsd_extractor.get_context()
        self.log_prefix = log_prefix

        if voxel_size is None:
            self.voxel_size = (1,)*len(fragments.shape)
        else:
            self.voxel_size = voxel_size

        self.__initialize_fragments()


    def __initialize_fragments(self):

        # get bounding box (+context) and size of each fragment
        # compute LSDs on these bounding boxes

        rois = self.__get_rois(self.fragments)
        ids, sizes = np.unique(self.fragments, return_counts=True)

        self.fragment_list = sorted([
            (
                size,
                rois[fragment_id].snap_to_grid((self.lsd_extractor.downsample,)*roi.dims())
                fragment_id
            )
            for size, fragment_id in zip(sizes, ids)
        ])

        self.processed = { fragment_id: False for fragment_id in ids }
        self.next_fragment_index = 0

    def __next_largest_fragment(self):
        '''Get the next largest fragment that is not processed, yet.'''

        while True:

            try:
                fragment = self.fragment_list[self.next_fragment_index]
            except:
                return None
            self.next_fragment_index += 1

            if self.processed[fragment[ID]]:
                continue

            return fragment

    def grow(self, threshold):

        while True:

            # pick largest fragment u not yet processed
            seed = self.__next_largest_fragment()

            # no more unprocessed fragments
            if seed is None:
                break

            self.__start_segment(seed)

            # u is both the ID of the seed and the segmentation
            u = seed[ID]

            # initialize open fragments with neighbors of u
            open_fragments = sorted(self.__neighors_of(u, processed=False))

            done = False
            while not done:

                # for each open fragment v, from large to small:
                for neighbor in open_fragments:

                    v = neighbor[ID]

                    # evaluate edge score (u, v)
                    score = self.__compute_edge_score(u, v)

                    if score < threshold:

                        # add v to segment
                        self.__add_fragment_to_segment(v, u)
                        grow_history.append({
                            'a': u,
                            'b': v,
                            'c': u,
                            'score': score
                        })

                        # add v's neighbors to list of open fragments
                        open_fragments = sorted(
                            open_fragments +
                            self.__neighors_of(v, processed=False))

                        # restart loop over open fragments
                        done = False
                        break

                # if we got here, there are no more open fragments below the
                # threshold
                done = True

        return grow_history

    def __compute_edge_score(self, u, v):
        '''Compute the LSDs score for an edge.

        The edge score is by how much the incident node scores would improve
        when merged (negative if the score decreases). More formally, it is:

            s(u + v) - (s(u) + s(v))

        where s(.) is the score of a node and (u + v) is a node obtained from
        merging u and v.
        '''

        (change_roi, context_roi) = self.__get_lsds_edge_rois(u, v)

        if change_roi is None:
            return 0

        # get slice of segmentation for context_roi (make a copy, since we
        # change it later)
        segmentation = self.segmentation[context_roi.to_slices()]
        segmentation = np.array(segmentation)

        # slices to cut change ROI from LSDs
        lsds_slice = (slice(None),) + change_roi.to_slices()

        # change ROI relative to context ROI
        change_in_context_roi = change_roi - context_roi.get_begin()

        # mask for voxels in u and v for change ROI
        not_uv_mask = np.logical_not(
            np.isin(
                segmentation[change_in_context_roi.to_slices()],
                [u, v]
            )
        )

        # get s(u) + s(v)
        lsds_separate = self.lsds[lsds_slice]
        diff = self.target_lsds[lsds_slice] - lsds_separate
        diff[:,not_uv_mask] = 0
        score_separate = np.sum(diff**2)

        # mark u as v in segmentation
        segmentation[segmentation==u] = v

        # get s(u + v)
        lsds_merged = self.lsd_extractor.get_descriptors(
            segmentation,
            roi=change_in_context_roi,
            labels=[v],
            voxel_size=self.voxel_size)
        diff = self.target_lsds[lsds_slice] - lsds_merged
        diff[:,not_uv_mask] = 0
        score_merged = np.sum(diff**2)

        assert lsds_separate.shape == lsds_merged.shape

        self.__log_debug(
            "Edge score for (%d, %d) is %f - %f = %f",
            u, v, score_merged, score_separate,
            score_merged - score_separate)

        return score_merged - score_separate

def get_rois(labels):

    # find_objects uses memory proportional to the max label in fragments,
    # therefore we relabel them here and use those
    relabelled, n, relabel_map = relabel(
        labels,
        return_backwards_map=True)

    bbs = find_objects(relabelled)

    return {
        relabel_map[i + 1]: __slice_to_roi(bbs[i])
        for i in range(len(bbs))
    }

def __slice_to_roi(slices):

    offset = tuple(s.start for s in slices)
    shape = tuple(s.stop - s.start for s in slices)

    roi = daisy.Roi(offset, shape)

    return roi
