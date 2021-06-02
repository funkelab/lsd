import json
import logging
import numpy as np
import os
import pymongo
import sys

from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *

setup_dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(setup_dir, 'test_net.json'), 'r') as f:
    net_config = json.load(f)

# voxels
input_shape = Coordinate(net_config['input_shape'])
output_shape = Coordinate(net_config['output_shape'])

# nm
voxel_size = Coordinate((8, 8, 8))
input_size = input_shape*voxel_size
output_size = output_shape*voxel_size

def block_done_callback(
        db_host,
        db_name,
        worker_config,
        block,
        start,
        duration):

    print("Recording block-done for %s" % (block,))

    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    collection = db['blocks_predicted']

    document = dict(worker_config)
    document.update({
        'block_id': block.block_id,
        'read_roi': (block.read_roi.get_begin(), block.read_roi.get_shape()),
        'write_roi': (block.write_roi.get_begin(), block.write_roi.get_shape()),
        'start': start,
        'duration': duration
    })

    collection.insert(document)

    print("Recorded block-done for %s" % (block,))

def predict(
        iteration,
        raw_file,
        raw_dataset,
        auto_file,
        auto_dataset,
        out_file,
        out_dataset,
        db_host,
        db_name,
        worker_config,
        **kwargs):

    raw = ArrayKey('RAW')
    lsds = ArrayKey('PRETRAINED_LSDS')
    affs = ArrayKey('AFFS')

    print('Raw file is: %s, Raw dataset is: %s'%(raw_file, raw_dataset))
    print('Auto file is: %s, Auto dataset is: %s'%(auto_file, auto_dataset))

    chunk_request = BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(lsds, input_size)
    chunk_request.add(affs, output_size)

    pipeline = (
            (
                ZarrSource(
                    raw_file,
                    datasets = {
                        raw: raw_dataset
                    },
                    array_specs = {
                        raw: ArraySpec(interpolatable=True)
                    }
                ) +
                Pad(raw, size=None) +
                Normalize(raw) +
                IntensityScaleShift(raw, 2,-1),

                ZarrSource(
                    auto_file,
                    datasets = {
                        lsds: auto_dataset
                    },
                    array_specs = {
                        lsds: ArraySpec(interpolatable=True)
                    }
                ) +
                Pad(lsds, size=None) +
                Normalize(lsds)
            ) +
            MergeProvider() +
            Predict(
                checkpoint=os.path.join(
                    setup_dir,
                    'train_net_checkpoint_%d'%iteration),
                graph=os.path.join(setup_dir, 'test_net.meta'),
                max_shared_memory=(2*1024*1024*1024),
                inputs={
                    net_config['pretrained_lsd']: lsds,
                    net_config['raw']: raw
                },
                outputs={
                    net_config['affs']: affs
                }
            ) +
            IntensityScaleShift(affs, 255, 0) +
            ZarrWrite(
                dataset_names={
                    affs: out_dataset,
                },
                output_filename=out_file
            ) +
            PrintProfilingStats(every=10)+
            DaisyRequestBlocks(
                chunk_request,
                roi_map={
                    raw: 'read_roi',
                    lsds: 'read_roi',
                    affs: 'write_roi'
                },
                num_workers=worker_config['num_cache_workers'],
                block_done_callback=lambda b, s, d: block_done_callback(
                    db_host,
                    db_name,
                    worker_config,
                    b, s, d))
            )

    print("Starting prediction...")
    with build(pipeline):
        pipeline.request_batch(BatchRequest())
    print("Prediction finished")

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.getLogger('gunpowder.nodes.hdf5like_write_base').setLevel(logging.DEBUG)

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    predict(**run_config)
