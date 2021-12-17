import daisy
import json
import logging
import multiprocessing as mp
import numpy as np
import os
import sys
import time
from funlib.segment.graphs.impl import connected_components

logging.basicConfig(level=logging.INFO)

def find_segments(
        db_host,
        db_name,
        fragments_file,
        edges_collection,
        thresholds_minmax,
        thresholds_step,
        block_size,
        num_workers,
        fragments_dataset=None,
        run_type=None,
        roi_offset=None,
        roi_shape=None,
        **kwargs):

    '''

    Args:

        db_host (``string``):

            Name of MongoDB client.

        db_name (``string``):

            Name of MongoDB database to use (where the region adjacency graph is
            stored).

        fragments_file (``string``):

            Path to file (zarr/n5) containing fragments (supervoxels).

        edges_collection (``string``):

            The name of the MongoDB database edges collection to use.

        thresholds_minmax (``list`` of ``int``):

            The lower and upper bound to use (i.e [0,1]) when generating
            thresholds.

        thresholds_step (``float``):

            The step size to use when generating thresholds between min/max.

        block_size (``tuple`` of ``int``):

            The size of one block in world units (must be multiple of voxel
            size).

        num_workers (``int``):

            How many workers to use when reading the region adjacency graph
            blockwise.

        fragments_dataset (``string``, optional):

            Name of fragments dataset. Include if using full fragments roi, set
            to None if using a crop (roi_offset + roi_shape).

        run_type (``string``, optional):

            Can be used to direct luts into directory (e.g testing, validation,
            etc).

        roi_offset (array-like of ``int``, optional):

            The starting point (inclusive) of the ROI. Entries can be ``None``
            to indicate unboundedness.

        roi_shape (array-like of ``int``, optional):

            The shape of the ROI. Entries can be ``None`` to indicate
            unboundedness.

    '''

    start = time.time()

    logging.info(f"Reading graph from DB: {db_name} and collection: {edges_collection}")

    graph_provider = daisy.persistence.MongoDbGraphProvider(
        db_name,
        db_host,
        edges_collection=edges_collection,
        position_attribute=[
            'center_z',
            'center_y',
            'center_x'])

    if fragments_dataset:
        fragments = daisy.open_ds(fragments_file, fragments_dataset)
        roi = fragments.roi

    else:
        roi = daisy.Roi(
            roi_offset,
            roi_shape)

    node_attrs, edge_attrs = graph_provider.read_blockwise(
        roi,
        block_size=daisy.Coordinate(block_size),
        num_workers=num_workers)

    logging.info(f"Read graph in {time.time() - start}")

    if 'id' not in node_attrs:
        logging.info('No nodes found in roi %s' % roi)
        return

    nodes = node_attrs['id']
    edges = np.stack(
                [
                    edge_attrs['u'].astype(np.uint64),
                    edge_attrs['v'].astype(np.uint64)
                ],
            axis=1)

    scores = edge_attrs['merge_score'].astype(np.float32)

    logging.info(f"Complete RAG contains {len(nodes)} nodes, {len(edges)} edges")

    out_dir = os.path.join(
        fragments_file,
        'luts',
        'fragment_segment')

    if run_type:
        out_dir = os.path.join(out_dir, run_type)

    os.makedirs(out_dir, exist_ok=True)

    thresholds = [round(i,2) for i in np.arange(
        float(thresholds_minmax[0]),
        float(thresholds_minmax[1]),
        thresholds_step)]

    start = time.time()

    for threshold in thresholds:

        get_connected_components(
                nodes,
                edges,
                scores,
                threshold,
                edges_collection,
                out_dir)

        logging.info(f"Created and stored lookup tables in {time.time() - start}")

def get_connected_components(
        nodes,
        edges,
        scores,
        threshold,
        edges_collection,
        out_dir,
        **kwargs):

    logging.info(f"Getting CCs for threshold {threshold}...")
    components = connected_components(nodes, edges, scores, threshold)

    logging.info(f"Creating fragment-segment LUT for threshold {threshold}...")
    lut = np.array([nodes, components])

    logging.info(f"Storing fragment-segment LUT for threshold {threshold}...")

    lookup = f"seg_{edges_collection}_{int(threshold*100)}"

    out_file = os.path.join(out_dir, lookup)

    np.savez_compressed(out_file, fragment_segment_lut=lut)

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    start = time.time()
    find_segments(**config)

    logging.info('Took {time.time() - start} seconds to find segments and store LUTs')
