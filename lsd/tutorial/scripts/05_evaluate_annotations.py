import daisy
import glob
import json
import logging
import multiprocessing as mp
import networkx
import numpy as np
import os
import sys
import time
from funlib.segment.arrays import replace_values
from funlib.evaluate import rand_voi, \
        expected_run_length, \
        get_skeleton_lengths, \
        split_graph
from pymongo import MongoClient

logging.basicConfig(level=logging.INFO)

class EvaluateAnnotations():

    def __init__(
            self,
            experiment,
            setup,
            iteration,
            fragments_file,
            fragments_dataset,
            edges_db_host,
            edges_db_name,
            edges_collection,
            scores_db_name,
            annotations_db_name,
            annotations_skeletons_collection_name,
            roi_offset,
            roi_shape,
            thresholds_minmax,
            thresholds_step,
            node_components,
            node_mask,
            method=None,
            run_type=None,
            scores_db_host=None,
            annotations_db_host=None,
            annotations_synapses_collection_name=None,
            compute_mincut_metric=False,
            **kwargs):


        '''

        Args:

            experiment (``string``):

                Name of the experiment (fib25, hemi, zfinch, ...).

            setup (``string``):

                Name of the setup to predict (setup01, setup02, ...).

            iteration (``int``):

                Training iteration.

            fragments_file (``string``):

                Path to file (zarr/n5) where fragments are stored.

            fragments_dataset (``string``):

                Name of fragments dataset (e.g `volumes/fragments`).

            edges_db_host (``string``):

                Name of MongoDB client containing region adjacency graph.

            edges_db_name (``string``):

                Name of MongoDB database containing region adjacency graph.

            edges_collection (``string``):

                Name of collection containing edges (e.g `edges_hist_quant_75`)

            scores_db_name (``string``):

                Name of MongoDB database to write scores to.

            annotations_db_name (``string``):

                Name of MongoDB database containing ground truth skeletons.

            annotations_skeletons_collection_name (``string``):

                Name of MongoDB collection containing relevant ground truth
                skeletons (e.g zebrafinch). Nodes and edges would then be
                contained inside collections (e.g zebrafinch.nodes).

            roi_offset (array-like of ``int``):

                The starting point (inclusive) of the ROI (in world units).

            roi_shape (array-like of ``int``):

                The shape of the ROI (in world units).

            thresholds_minmax (``list`` of ``int``):

                The lower and upper bound to use (i.e [0,1]) when generating
                thresholds.

            thresholds_step (``float``):

                The step size to use when generating thresholds between min/max.

            node_components(``string``):

                Components collection (e.g `zebrafinch_components`) to use.

            node_mask (``string``):

                Mask collection (e.g `zebrafinch_mask`) to use.

            method (``string``, optional):

                Name of network (e.g mtlsd, aclsd, etc) to write to scores database

            run_type (``string``, optional):

                Can be used to get luts from specific directory (e.g testing, validation,
                etc) and include in scores database.

            scores_db_host (``string``, optional):

                Name of MongoDB client to write scores to. If None, will write
                to same client as edges.

            annotations_db_host (``string``, optional):

                Name of MongoDB client containing ground truth skeletons. If
                None, will write to same client as edges.

            annotations_synapses_collection_name (``string``, optional):

                Collection containing synaptic sites to use for filtering
                skeletons by synapses.

            compute_mincut_metric (``bool``, optional):

                Recursively split a graph via min-cuts such that the given
                component nodes are separated. This can become very computationally
                expensive on larger rois.

        '''

        self.experiment = experiment
        self.setup = setup
        self.iteration = int(iteration)
        self.method = method
        self.fragments_file = fragments_file
        self.fragments_dataset = fragments_dataset
        self.edges_db_host = edges_db_host
        self.edges_db_name = edges_db_name
        self.edges_collection = edges_collection
        self.scores_db_name = scores_db_name
        self.annotations_db_name = annotations_db_name
        self.annotations_skeletons_collection_name = \
            annotations_skeletons_collection_name

        self.roi = daisy.Roi(roi_offset, roi_shape)
        self.thresholds_minmax = thresholds_minmax
        self.thresholds_step = thresholds_step
        self.node_mask = node_mask
        self.node_components = node_components
        self.run_type = run_type

        self.scores_db_host = edges_db_host if scores_db_host \
                is None else scores_db_host

        self.annotations_db_host = edges_db_host if annotations_db_host \
                is None else annotations_db_host

        self.annotations_synapses_collection_name = \
            annotations_synapses_collection_name

        self.compute_mincut_metric = compute_mincut_metric

        self.site_fragment_lut_directory = os.path.join(
            self.fragments_file,
            'luts/site_fragment')

        if self.run_type:
            logging.info(f"Run type set, evaluating on {self.run_type} dataset")
            self.site_fragment_lut_directory = os.path.join(
                    self.site_fragment_lut_directory,
                    self.run_type)

        logging.info("Path to site fragment luts: "
                f"{self.site_fragment_lut_directory}")

        try:
            self.fragments = daisy.open_ds(self.fragments_file,
                    self.fragments_dataset)
        except:
            self.fragments = daisy.open_ds(self.fragments_file,
                    self.fragments_dataset + '/s0')

    def store_lut_in_block(self, block):

        logging.info(f"Finding fragment IDs in block {block}")

        # get all skeleton nodes (which include synaptic sites)
        client = MongoClient(self.annotations_db_host)
        database = client[self.annotations_db_name]
        skeletons_collection = \
            database[self.annotations_skeletons_collection_name + '.nodes']

        bz, by, bx = block.read_roi.get_begin()
        ez, ey, ex = block.read_roi.get_end()

        site_nodes = skeletons_collection.find(
            {
                'z': {'$gte': bz, '$lt': ez},
                'y': {'$gte': by, '$lt': ey},
                'x': {'$gte': bx, '$lt': ex}
            })

        # get site -> fragment ID
        site_fragment_lut, num_bg_sites = get_site_fragment_lut(
            self.fragments,
            site_nodes,
            block.write_roi)

        if site_fragment_lut is None:
            return

        # store LUT
        block_lut_path = os.path.join(
            self.site_fragment_lut_directory,
            str(block.block_id) + '.npz')

        np.savez_compressed(
            block_lut_path,
            site_fragment_lut=site_fragment_lut,
            num_bg_sites=num_bg_sites)

    def prepare_for_roi(self):

        logging.info(f"Preparing evaluation for ROI {self.roi}...")

        self.skeletons = self.read_skeletons()

        # array with site IDs
        self.site_ids = np.array([
            n
            for n in self.skeletons.nodes()
        ], dtype=np.uint64)

        # array with component ID for each site
        self.site_component_ids = np.array([
            data['component_id']
            for _, data in self.skeletons.nodes(data=True)
        ])

        assert self.site_component_ids.min() >= 0

        self.site_component_ids = self.site_component_ids.astype(np.uint64)
        self.number_of_components = np.unique(self.site_component_ids).size

        if self.annotations_synapses_collection_name:
            # create a mask that limits sites to synaptic sites
            logging.info("Creating synaptic sites mask...")

            client = MongoClient(self.annotations_db_host)
            database = client[self.annotations_db_name]

            synapses_collection = \
                database[self.annotations_synapses_collection_name + '.edges']

            synaptic_sites = synapses_collection.find()

            synaptic_sites = np.unique([
                s
                for ss in synaptic_sites
                for s in [ss['source'], ss['target']]
            ])

            self.synaptic_sites_mask = np.isin(self.site_ids, synaptic_sites)

        logging.info("Calculating skeleton lengths...")
        start = time.time()
        self.skeleton_lengths = get_skeleton_lengths(
                self.skeletons,
                skeleton_position_attributes=['z', 'y', 'x'],
                skeleton_id_attribute='component_id',
                store_edge_length='length')

        self.total_length = np.sum([l for _, l in self.skeleton_lengths.items()])

    def prepare_for_fragments(self):
        '''Get the fragment ID for each site in site_ids.'''

        logging.info(f"Preparing evaluation for fragments in "
                f"{self.fragments_file}...")

        if not os.path.exists(self.site_fragment_lut_directory):

            logging.info("site-fragment LUT does not exist, creating it...")

            os.makedirs(self.site_fragment_lut_directory)
            daisy.run_blockwise(
                self.roi,
                daisy.Roi((0, 0, 0), (9000, 9000, 9000)),
                daisy.Roi((0, 0, 0), (9000, 9000, 9000)),
                lambda b: self.store_lut_in_block(b),
                num_workers=48,
                fit='shrink')

        else:

            logging.info("site-fragment LUT already exists, skipping preparation")

        logging.info("Reading site-fragment LUTs from "
                f"{self.site_fragment_lut_directory}...")

        lut_files = glob.glob(
            os.path.join(
                self.site_fragment_lut_directory,
                '*.npz'))

        site_fragment_lut = np.concatenate(
            [
                np.load(f)['site_fragment_lut']
                for f in lut_files
            ],
            axis=1)

        self.num_bg_sites = int(np.sum([np.load(f)['num_bg_sites'] for f in lut_files]))

        assert site_fragment_lut.dtype == np.uint64

        logging.info(f"Found {len(site_fragment_lut[0])} sites in site-fragment LUT")

        # convert to dictionary
        site_fragment_lut = {
            site: fragment
            for site, fragment in zip(
                site_fragment_lut[0],
                site_fragment_lut[1])
        }

        # create fragment ID array congruent to site_ids
        self.site_fragment_ids = np.array([
            site_fragment_lut[s] if s in site_fragment_lut else 0
            for s in self.site_ids
        ], dtype=np.uint64)

    def read_skeletons(self):

        if self.run_type:

            logging.info(f"Using components for {self.run_type} data")

            self.node_mask = self.node_mask + '_' + self.run_type
            self.node_components = self.node_components + '_' + self.run_type

        logging.info(f"Reading mask from {self.node_mask}")
        logging.info(f"Reading components from: {self.node_components}")

        # get all skeletons
        logging.info("Fetching all skeletons...")
        skeletons_provider = daisy.persistence.MongoDbGraphProvider(
            self.annotations_db_name,
            self.annotations_db_host,
            nodes_collection=self.annotations_skeletons_collection_name +
            '.nodes',
            edges_collection=self.annotations_skeletons_collection_name +
            '.edges',
            endpoint_names=['source', 'target'],
            position_attribute=['z', 'y', 'x'],
            node_attribute_collections={
                self.node_mask: ['masked'],
                self.node_components: ['component_id'],
            })

        skeletons = skeletons_provider.get_graph(
                self.roi,
                nodes_filter={'masked': True})
        logging.info(f"Found {skeletons.number_of_nodes()} skeleton nodes")

        # remove outside edges and nodes
        remove_nodes = []
        for node, data in skeletons.nodes(data=True):
            if 'z' not in data:
                remove_nodes.append(node)
            else:
                assert data['masked']
                assert data['component_id'] >= 0

        logging.info(f"Removing {len(remove_nodes)} nodes that were outside of ROI")

        for node in remove_nodes:
            skeletons.remove_node(node)

        return skeletons

    def evaluate(self):

        self.prepare_for_roi()

        self.prepare_for_fragments()

        thresholds = [round(i,2) for i in np.arange(
            float(self.thresholds_minmax[0]),
            float(self.thresholds_minmax[1]),
            self.thresholds_step)]

        procs = []

        logging.info("Evaluating thresholds...")

        for threshold in thresholds:
            proc = mp.Process(
                target=lambda: self.evaluate_threshold(threshold)
            )
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

    def get_site_segment_ids(self, threshold):

        # get fragment-segment LUT
        logging.info("Reading fragment-segment LUT...")
        start = time.time()

        fragment_segment_lut_dir = os.path.join(
                self.fragments_file,
                'luts/fragment_segment')

        if self.run_type:
            logging.info(f"Using lookup tables for {self.run_type} data")
            fragment_segment_lut_dir = os.path.join(
                    fragment_segment_lut_dir,
                    self.run_type)

        logging.info("Reading fragment segment luts from: "
                f"{fragment_segment_lut_dir}")

        fragment_segment_lut_file = os.path.join(
                fragment_segment_lut_dir,
                'seg_%s_%d.npz' % (self.edges_collection, int(threshold*100)))

        fragment_segment_lut = np.load(
            fragment_segment_lut_file)['fragment_segment_lut']

        assert fragment_segment_lut.dtype == np.uint64

        # get the segment ID for each site
        logging.info("Mapping sites to segments...")

        site_mask = np.isin(fragment_segment_lut[0], self.site_fragment_ids)
        site_segment_ids = replace_values(
            self.site_fragment_ids,
            fragment_segment_lut[0][site_mask],
            fragment_segment_lut[1][site_mask])

        return site_segment_ids, fragment_segment_lut

    def compute_expected_run_length(self, site_segment_ids):

        logging.info("Calculating expected run length...")

        node_segment_lut = {
            site: segment for site, segment in zip(
                self.site_ids,
                site_segment_ids)
        }

        erl, stats = expected_run_length(
                skeletons=self.skeletons,
                skeleton_id_attribute='component_id',
                edge_length_attribute='length',
                node_segment_lut=node_segment_lut,
                skeleton_lengths=self.skeleton_lengths,
                return_merge_split_stats=True)

        perfect_lut = {
                node: data['component_id'] for node, data in \
                        self.skeletons.nodes(data=True)
        }

        max_erl, _ = expected_run_length(
                skeletons=self.skeletons,
                skeleton_id_attribute='component_id',
                edge_length_attribute='length',
                node_segment_lut=perfect_lut,
                skeleton_lengths=self.skeleton_lengths,
                return_merge_split_stats=True)

        split_stats = [
            {
                'comp_id': int(comp_id),
                'seg_ids': [(int(a), int(b)) for a, b in seg_ids]
            }
            for comp_id, seg_ids in stats['split_stats'].items()
        ]

        merge_stats = [
            {
                'seg_id': int(seg_id),
                'comp_ids': [int(comp_id) for comp_id in comp_ids]
            }
            for seg_id, comp_ids in stats['merge_stats'].items()
        ]

        return erl, max_erl, split_stats, merge_stats

    def compute_splits_merges_needed(
            self,
            fragment_segment_lut,
            site_segment_ids,
            split_stats,
            merge_stats,
            threshold):

        total_splits_needed = 0
        total_additional_merges_needed = 0
        total_unsplittable_fragments = []

        logging.info("Computing min-cut metric for each merging segment...")

        for i, merge in enumerate(merge_stats):

            logging.info(f"Processing merge {i+1}/{len(merge_stats)}...")
            (
                splits_needed,
                additional_merges_needed,
                unsplittable_fragments) = self.mincut_metric(
                    fragment_segment_lut,
                    site_segment_ids,
                    merge['seg_id'],
                    merge['comp_ids'],
                    threshold)
            total_splits_needed += splits_needed
            total_additional_merges_needed += additional_merges_needed
            total_unsplittable_fragments += unsplittable_fragments

        total_merges_needed = 0
        for split in split_stats:
            total_merges_needed += len(split['seg_ids']) - 1
        total_merges_needed += total_additional_merges_needed

        return (
            total_splits_needed,
            total_merges_needed,
            total_unsplittable_fragments)

    def mincut_metric(
            self,
            fragment_segment_lut,
            site_segment_ids,
            segment_id,
            component_ids,
            threshold):

        # get RAG for segment ID
        rag = self.get_segment_rag(segment_id, fragment_segment_lut, threshold)

        logging.info("Preparing RAG for split_graph call")

        # replace merge_score with weight
        for _, _, data in rag.edges(data=True):
            data['weight'] = 1.0 - data['merge_score']

        # find fragments for each component in segment_id
        component_fragments = {}

        # True for every site that maps to segment_id
        segment_mask = site_segment_ids == segment_id

        for component_id in component_ids:

            # limit following to sites that are part of component_id and
            # segment_id

            component_mask = self.site_component_ids == component_id
            fg_mask = self.site_fragment_ids != 0
            mask = np.logical_and(np.logical_and(component_mask, segment_mask), fg_mask)
            site_ids = self.site_ids[mask]
            site_fragment_ids = self.site_fragment_ids[mask]

            component_fragments[component_id] = site_fragment_ids

            for site_id, fragment_id in zip(site_ids, site_fragment_ids):

                if fragment_id == 0:
                    continue

                # For each fragment containing a site, we need a position for
                # the split_graph call. We just take the position of the
                # skeleton node that maps to it, if there are several, we take
                # the last one.

                site_data = self.skeletons.nodes[site_id]
                fragment = rag.nodes[fragment_id]
                fragment['z'] = site_data['z']
                fragment['y'] = site_data['y']
                fragment['x'] = site_data['x']

                # Keep track of how many components share a fragment. If it is
                # more than one, this fragment is unsplittable.
                if 'component_ids' not in fragment:
                    fragment['component_ids'] = set()
                fragment['component_ids'].add(component_id)

        # find all unsplittable fragments...
        unsplittable_fragments = []
        for fragment_id, data in rag.nodes(data=True):
            if fragment_id == 0:
                continue
            if 'component_ids' in data and len(data['component_ids']) > 1:
                unsplittable_fragments.append(fragment_id)

        # ...and remove them from the component lists
        for component_id in component_ids:

            fragment_ids = component_fragments[component_id]
            valid_mask = np.logical_not(
                np.isin(
                    fragment_ids,
                    unsplittable_fragments))
            valid_fragment_ids = fragment_ids[valid_mask]
            if len(valid_fragment_ids) > 0:
                component_fragments[component_id] = valid_fragment_ids
            else:
                del component_fragments[component_id]

        logging.info(f"{len(unsplittable_fragments)} fragments are merging "
                "and can not be split")

        if len(component_fragments) <= 1:
            logging.info(
                "after removing unsplittable fragments, there is nothing to "
                "do anymore")
            return 0, 0, unsplittable_fragments

        # these are the fragments that need to be split
        split_fragments = list(component_fragments.values())

        logging.info(f"Splitting segment into {len(split_fragments)} components "
                f"with sizes {[len(c) for c in split_fragments]}")

        logging.info("Calling split_graph...")

        # call split_graph
        num_splits_needed = split_graph(
            rag,
            split_fragments,
            position_attributes=['z', 'y', 'x'],
            weight_attribute='weight',
            split_attribute='split')

        logging.info(f"{num_splits_needed} splits needed for segment "
                f"{segment_id}")

        # get number of additional merges needed after splitting the current
        # segment
        #
        # this is the number of split labels per component minus 1
        additional_merges_needed = 0
        for component, fragments in component_fragments.items():
            split_ids = np.unique([rag.node[f]['split'] for f in fragments])
            additional_merges_needed += len(split_ids) - 1

        logging.info(f"{additional_merges_needed} additional merges "
                "needed to join components again")

        return (
            num_splits_needed,
            additional_merges_needed,
            unsplittable_fragments)

    def get_segment_rag(self, segment_id, fragment_segment_lut, threshold):

        logging.info(f"Reading RAG for segment {segment_id}")

        rag_provider = daisy.persistence.MongoDbGraphProvider(
            self.edges_db_name,
            self.edges_db_host,
            mode='r',
            edges_collection=self.edges_collection,
            position_attribute=['z', 'y', 'x'])

        # get all fragments for the given segment
        segment_mask = fragment_segment_lut[1] == segment_id
        fragment_ids = fragment_segment_lut[0][segment_mask]

        # get the RAG containing all fragments
        nodes = [
            {'id': fragment_id, 'segment_id': segment_id}
            for fragment_id in fragment_ids
        ]
        edges = rag_provider.read_edges(self.roi, nodes=nodes)

        logging.info(f"RAG contains {len(nodes)} nodes & {len(edges)} edges")

        rag = networkx.Graph()

        node_list = [
            (n['id'], {'segment_id': n['segment_id']})
            for n in nodes
        ]

        edge_list = [
            (e['u'], e['v'], {'merge_score': e['merge_score']})
            for e in edges
            if e['merge_score'] <= threshold
        ]

        rag.add_nodes_from(node_list)
        rag.add_edges_from(edge_list)

        rag.remove_nodes_from([
            n
            for n, data in rag.nodes(data=True)
            if 'segment_id' not in data])

        logging.info("after filtering dangling nodes and unmerged edges, RAG "
                f"contains {rag.number_of_nodes()} nodes & "
                f"{rag.number_of_edges()} edges")

        return rag

    def compute_rand_voi(
            self,
            site_component_ids,
            site_segment_ids,
            return_cluster_scores=False):

        logging.info("Computing RAND and VOI...")

        rand_voi_report = rand_voi(
            np.array([[site_component_ids]]),
            np.array([[site_segment_ids]]),
            return_cluster_scores=return_cluster_scores)

        logging.info(f"VOI split: {rand_voi_report['voi_split']}")
        logging.info(f"VOI merge: {rand_voi_report['voi_merge']}")

        return rand_voi_report

    def evaluate_threshold(self, threshold):

        scores_client = MongoClient(self.scores_db_host)
        scores_db = scores_client[self.scores_db_name]
        scores_collection = scores_db['scores']

        site_segment_ids, fragment_segment_lut = \
            self.get_site_segment_ids(threshold)

        number_of_segments = np.unique(site_segment_ids).size

        erl, max_erl, split_stats, merge_stats = self.compute_expected_run_length(site_segment_ids)

        number_of_split_skeletons = len(split_stats)
        number_of_merging_segments = len(merge_stats)

        logging.info(f"ERL: {erl}")
        logging.info(f"Max ERL: {max_erl}")
        logging.info(f"Total path length: {self.total_length}")

        normalized_erl = erl/max_erl
        logging.info(f"Normalized ERL: {normalized_erl}")

        if self.compute_mincut_metric:

            splits_needed, merges_needed, unsplittable_fragments = \
                self.compute_splits_merges_needed(
                    fragment_segment_lut,
                    site_segment_ids,
                    split_stats,
                    merge_stats,
                    threshold)

            average_splits_needed = splits_needed/number_of_segments
            average_merges_needed = merges_needed/self.number_of_components

            logging.info(f"Number of splits needed: {splits_needed}")
            logging.info(f"Number of background sites: {self.num_bg_sites}")
            logging.info(f"Average splits needed: {average_splits_needed}")
            logging.info(f"Average merges needed: {average_merges_needed}")
            logging.info("Number of unsplittable fragments: "
                    f"{len(unsplittable_fragments)}")

        rand_voi_report = self.compute_rand_voi(
            self.site_component_ids,
            site_segment_ids,
            return_cluster_scores=True)

        if self.annotations_synapses_collection_name:
            synapse_rand_voi_report = self.compute_rand_voi(
                self.site_component_ids[self.synaptic_sites_mask],
                site_segment_ids[self.synaptic_sites_mask])

        report = rand_voi_report.copy()

        for k in {'voi_split_i', 'voi_merge_j'}:
            del report[k]

        if self.annotations_synapses_collection_name:
            report['synapse_voi_split'] = synapse_rand_voi_report['voi_split']
            report['synapse_voi_merge'] = synapse_rand_voi_report['voi_merge']

        report['expected_run_length'] = erl
        report['max_erl'] = max_erl
        report['total path length'] = self.total_length
        report['normalized_erl'] = normalized_erl
        report['number_of_segments'] = number_of_segments
        report['number_of_components'] = self.number_of_components
        report['number_of_merging_segments'] = number_of_merging_segments
        report['number_of_split_skeletons'] = number_of_split_skeletons

        if self.compute_mincut_metric:
            report['total_splits_needed_to_fix_merges'] = splits_needed
            report['average_splits_needed_to_fix_merges'] = average_splits_needed
            report['total_merges_needed_to_fix_splits'] = merges_needed
            report['average_merges_needed_to_fix_splits'] = average_merges_needed
            report['number_of_unsplittable_fragments'] = len(unsplittable_fragments)
            report['number_of_background_sites'] = self.num_bg_sites
            report['unsplittable_fragments'] = [int(f) for f in unsplittable_fragments]

        report['merge_stats'] = merge_stats
        report['split_stats'] = split_stats
        report['threshold'] = threshold
        report['experiment'] = self.experiment
        report['setup'] = self.setup
        report['iteration'] = self.iteration
        report['network_configuration'] = self.edges_db_name
        report['merge_function'] = self.edges_collection.strip('edges_')

        if self.method:
            report['method'] = self.method

        if self.run_type:
            report['run_type'] = self.run_type

        scores_collection.replace_one(
            filter={
                'network_configuration': report['network_configuration'],
                'merge_function': report['merge_function'],
                'threshold': report['threshold'],
                'run_type': report['run_type']
            },
            replacement=report,
            upsert=True)

        find_worst_split_merges(rand_voi_report)

def get_site_fragment_lut(fragments, sites, roi):
    #Get the fragment IDs of all the sites that are contained in the given ROI

    sites = list(sites)

    if len(sites) == 0:
        logging.info(f"No sites in {roi}, skipping")
        return None, None

    logging.info(f"Getting fragment IDs for {len(sites)} synaptic sites in {roi}...")

    # for a few sites, direct lookup is faster than memory copies
    if len(sites) >= 15:

        logging.info("Copying fragments into memory...")
        fragments = fragments[roi]
        fragments.materialize()

    logging.info(f"Getting fragment IDs for synaptic sites in {roi}...")

    fragment_ids = np.array([
        fragments[daisy.Coordinate((site['z'], site['y'], site['x']))]
        for site in sites
    ])
    site_ids = np.array(
        [site['id'] for site in sites],
        dtype=np.uint64)

    fg_mask = fragment_ids != 0
    fragment_ids = fragment_ids[fg_mask]
    site_ids = site_ids[fg_mask]

    lut = np.array([site_ids, fragment_ids])

    return lut, (fg_mask==0).sum()

def find_worst_split_merges(rand_voi_report):

    # get most severe splits/merges
    splits = sorted([
        (s, i)
        for (i, s) in rand_voi_report['voi_split_i'].items()
    ])
    merges = sorted([
        (s, j)
        for (j, s) in rand_voi_report['voi_merge_j'].items()
    ])

    logging.info("10 worst splits:")
    for (s, i) in splits[-10:]:
        logging.info(f"\tcomponent {i}\tVOI split {s}")

    logging.info("10 worst merges:")
    for (s, i) in merges[-10:]:
        logging.info(f"\tsegment {i}\tVOI merge {s}")

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    evaluate = EvaluateAnnotations(**config)
    evaluate.evaluate()
