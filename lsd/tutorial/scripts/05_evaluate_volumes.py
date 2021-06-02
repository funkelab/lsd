import daisy
import json
import logging
import lsd
import numpy as np
import os
import sys
from funlib.evaluate import rand_voi
from funlib.segment.arrays import replace_values
from pymongo import MongoClient

logging.basicConfig(level=logging.INFO)

def evaluate(
        experiment,
        setup,
        iteration,
        gt_file,
        gt_dataset,
        fragments_file,
        fragments_dataset,
        db_host,
        rag_db_name,
        edges_collection,
        scores_db_name,
        thresholds_minmax,
        thresholds_step,
        num_workers,
        method=None,
        run_type=None,
        roi_offset=None,
        roi_shape=None):

    '''

    Args:

        experiment (``string``):

            Name of the experiment (fib25, hemi, zfinch, ...).

        setup (``string``):

            Name of the setup to predict (setup01, setup02, ...).

        iteration (``int``):

            Training iteration.

        gt_file (``string``):

            Path to file (zarr/n5) where ground truth volume is stored.

        gt_dataset (``string``):

            Name of ground truth dataset (e.g `volumes/labels/neuron_ids`).

        fragments_file (``string``):

            Path to file (zarr/n5) where fragments are stored.

        fragments_dataset (``string``):

            Name of fragments dataset (e.g `volumes/fragments`).

        db_host (``string``):

            Name of MongoDB client.

        rag_db_name (``string``):

            Name of MongoDB database containing region adjacency graph.

        edges_collection (``string``):

            Name of collection containing edges (e.g `edges_hist_quant_75`)

        scores_db_name (``string``):

            Name of MongoDB database to write scores to.

        thresholds_minmax (``list`` of ``int``):

            The lower and upper bound to use (i.e [0,1]) when generating
            thresholds.

        thresholds_step (``float``):

            The step size to use when generating thresholds between min/max.

        num_workers (``int``):

            How many blocks to run in parallel.

        method (``string``, optional):

            Name of network (e.g mtlsd, aclsd, etc) to write to scores database

        run_type (``string``, optional):

            Can be used to get luts from specific directory (e.g testing, validation,
            etc) and include in scores database.

        roi_offset (array-like of ``int``, optional):

            The starting point (inclusive) of the ROI. Entries can be ``None``
            to indicate unboundedness.

        roi_shape (array-like of ``int``, optional):

            The shape of the ROI. Entries can be ``None`` to indicate
            unboundedness.

    '''

    # open fragments

    logging.info(f"Reading fragments from {fragments_file}")
    logging.info(f"Reading gt from {gt_file}")

    fragments = open_ds(fragments_file, fragments_dataset)
    gt = open_ds(gt_file, gt_dataset)

    if roi_offset:
        common_roi = daisy.Roi(roi_offset, roi_shape)

    else:
        common_roi = fragments.roi.intersect(gt.roi)

    # evaluate only where we have both fragments and GT
    logging.info(f"Cropping fragments and GT to common ROI {common_roi}")
    fragments = fragments[common_roi]
    gt = gt[common_roi]

    logging.info("Converting fragments and gt to nd arrays...")
    fragments = fragments.to_ndarray()
    gt = gt.to_ndarray()

    thresholds = [round(i,2) for i in np.arange(
        float(thresholds_minmax[0]),
        float(thresholds_minmax[1]),
        thresholds_step)]

    logging.info("Evaluating thresholds...")
    for threshold in thresholds:

        segment_ids = get_segmentation(
                fragments,
                fragments_file,
                edges_collection,
                threshold,
                run_type)

        evaluate_threshold(
                experiment,
                setup,
                iteration,
                db_host,
                scores_db_name,
                edges_collection,
                segment_ids,
                gt,
                threshold,
                method,
                run_type)

def open_ds(f, ds):

    try:
        data = daisy.open_ds(f, ds)
    except:
        data = daisy.open_ds(f, ds + '/s0')

    return data

def get_segmentation(
        fragments,
        fragments_file,
        edges_collection,
        threshold,
        run_type):

    logging.info(f"Loading fragment - segment lookup table for threshold \
            {threshold}...")

    fragment_segment_lut_dir = os.path.join(
            fragments_file,
            'luts',
            'fragment_segment')

    if run_type:
        logging.info(f"Run type set, evaluating on {run_type} dataset")

        fragment_segment_lut_dir = os.path.join(
                    fragment_segment_lut_dir,
                    run_type)

    fragment_segment_lut_file = os.path.join(fragment_segment_lut_dir,
            f'seg_{edges_collection}_{int(threshold*100)}.npz')

    fragment_segment_lut = np.load(
            fragment_segment_lut_file)['fragment_segment_lut']

    assert fragment_segment_lut.dtype == np.uint64

    logging.info("Relabeling fragment ids with segment ids...")

    segment_ids = replace_values(fragments, fragment_segment_lut[0], fragment_segment_lut[1])

    return segment_ids

def evaluate_threshold(
        experiment,
        setup,
        iteration,
        db_host,
        scores_db_name,
        edges_collection,
        segment_ids,
        gt,
        threshold,
        method,
        run_type):

        #open score DB
        client = MongoClient(db_host)
        database = client[scores_db_name]
        score_collection = database['scores']

        #get VOI and RAND
        logging.info(f"Calculating VOI scores for threshold {threshold}...")

        logging.info(type(segment_ids))

        rand_voi_report = rand_voi(
                gt,
                segment_ids,
                return_cluster_scores=False)

        metrics = rand_voi_report.copy()

        for k in {'voi_split_i', 'voi_merge_j'}:
            del metrics[k]

        logging.info(f"Storing VOI values for threshold {threshold} in DB")

        metrics['threshold'] = threshold
        metrics['experiment'] = experiment
        metrics['setup'] = setup
        metrics['iteration'] = iteration
        metrics['merge_function'] = edges_collection.strip('edges_')

        if method is not None:
            metrics['method'] = method

        if run_type is not None:
            metrics['run_type'] = run_type

        logging.info(f"VOI split: {metrics['voi_split']}")
        logging.info(f"VOI merge: {metrics['voi_merge']}")

        logging.info(metrics)

        score_collection.replace_one(
                filter={
                    'method': metrics['method'],
                    'run_type': metrics['run_type'],
                    'merge_function': metrics['merge_function'],
                    'threshold': metrics['threshold']
                },
                replacement=metrics,
                upsert=True)

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    evaluate(**config)
