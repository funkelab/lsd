import daisy
import datetime
import hashlib
import json
import logging
import numpy as np
import os
import pymongo
import sys
import time

logging.basicConfig(level=logging.INFO)

def predict_blockwise(
        base_dir,
        experiment,
        setup,
        iteration,
        raw_file,
        raw_dataset,
        out_base,
        file_name,
        num_workers,
        db_host,
        db_name,
        queue,
        auto_file=None,
        auto_dataset=None,
        singularity_image=None):

    '''

    Run prediction in parallel blocks. Within blocks, predict in chunks.


    Assumes a general directory structure:


    base
    ├── fib25 (experiment dir)
    │   │
    │   ├── 01_data (data dir)
    │   │   └── training_data (i.e zarr/n5, etc)
    │   │
    │   └── 02_train (train/predict dir)
    │       │
    │       ├── setup01 (setup dir - e.g baseline affinities)
    │       │   │
    │       │   ├── config.json (specifies meta data for inference)
    │       │   │
    │       │   ├── mknet.py (creates network, jsons to be used)
    │       │   │
    │       │   ├── model_checkpoint (saved network checkpoint for inference)
    │       │   │
    │       │   ├── predict.py (worker inference file - logic to be distributed)
    │       │   │
    │       │   ├── train_net.json (specifies meta data for training)
    │       │   │
    │       │   └── train.py (network training script)
    │       │
    │       ├──    .
    │       ├──    .
    │       ├──    .
    │       └── setup{n}
    │
    ├── hemi-brain
    ├── zebrafinch
    ├──     .
    ├──     .
    ├──     .
    └── experiment{n}

    Args:

        base_dir (``string``):

            Path to base directory containing experiment sub directories.

        experiment (``string``):

            Name of the experiment (fib25, hemi, zfinch, ...).

        setup (``string``):

            Name of the setup to predict (setup01, setup02, ...).

        iteration (``int``):

            Training iteration to predict from.

        raw_file (``string``):

            Path to raw file (zarr/n5) - can also be a json container
            specifying a crop, where offset and size are in world units:

                {
                    "container": "path/to/raw",
                    "offset": [z, y, x],
                    "size": [z, y, x]
                }

        raw_dataset (``string``):

            Raw dataset to use (e.g 'volumes/raw'). If using a scale pyramid,
            will try scale zero assuming stored in directory `s0` (e.g
            'volumes/raw/s0')

        out_base (``string``):

            Path to base directory where zarr/n5 should be stored. The out_file
            will be built from this directory, setup, iteration, file name

            **Note:

                out_dataset no longer needed as input, build out_dataset from config
                outputs dictionary generated in mknet.py (config.json for
                example)

        file_name (``string``):

            Name of output zarr/n5

        num_workers (``int``):

            How many blocks to run in parallel.

        db_host (``string``):

            Name of MongoDB client.

        db_name (``string``):

            Name of MongoDB database to use (for logging successful blocks in
            check function and DaisyRequestBlocks node inside worker predict
            script).

        queue (``string``):

            Name of gpu queue to run inference on (i.e gpu_rtx, gpu_tesla, etc)

        auto_file (``string``, optional):

            Path to zarr/n5 containing first pass predictions to use as input to
            autocontext network (i.e aclsd / acrlsd). None if not needed

        auto_dataset (``string``, optional):

            Input dataset to use if running autocontext (e.g 'volumes/lsds').
            None if not needed

        singularity_image (``string``, optional):

            Path to singularity image. None if not needed

    '''

    #get relevant dirs + files

    experiment_dir = os.path.join(base_dir, experiment)
    train_dir = os.path.join(experiment_dir, '02_train')
    network_dir = os.path.join(experiment, setup, str(iteration))

    raw_file = os.path.abspath(raw_file)
    out_file = os.path.abspath(os.path.join(out_base, setup, str(iteration), file_name))

    setup = os.path.abspath(os.path.join(train_dir, setup))

    # from here on, all values are in world units (unless explicitly mentioned)

    # get ROI of source
    try:
        source = daisy.open_ds(raw_file, raw_dataset)
    except:
        raw_dataset = raw_dataset + '/s0'
        source = daisy.open_ds(raw_file, raw_dataset)

    logging.info(f'Source shape: {source.shape}')
    logging.info(f'Source roi: {source.roi}')
    logging.info(f'Source voxel size: {source.voxel_size}')

    # load config
    with open(os.path.join(setup, 'config.json')) as f:
        net_config = json.load(f)

    outputs = net_config['outputs']

    # get chunk size and context for network (since unet has smaller output size
    # than input size
    net_input_size = daisy.Coordinate(net_config['input_shape'])*source.voxel_size
    net_output_size = daisy.Coordinate(net_config['output_shape'])*source.voxel_size

    context = (net_input_size - net_output_size)/2

    # get total input and output ROIs
    input_roi = source.roi.grow(context, context)
    output_roi = source.roi

    # create read and write ROI
    block_read_roi = daisy.Roi((0, 0, 0), net_input_size) - context
    block_write_roi = daisy.Roi((0, 0, 0), net_output_size)

    logging.info('Preparing output dataset...')

    # get output file(s) meta data from config.json, prepare dataset(s)
    for output_name, val in outputs.items():
        out_dims = val['out_dims']
        out_dtype = val['out_dtype']
        out_dataset = 'volumes/%s'%output_name

        ds = daisy.prepare_ds(
            out_file,
            out_dataset,
            output_roi,
            source.voxel_size,
            out_dtype,
            write_roi=block_write_roi,
            num_channels=out_dims,
            compressor={'id': 'gzip', 'level':5})

    logging.info('Starting block-wise processing...')

    # for logging successful blocks (see check_block function). if anything
    # fails, blocks which completed will be skipped when re-running

    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    if 'blocks_predicted' not in db.list_collection_names():
        blocks_predicted = db['blocks_predicted']
        blocks_predicted.create_index(
            [('block_id', pymongo.ASCENDING)],
            name='block_id')
    else:
        blocks_predicted = db['blocks_predicted']

    # process block-wise
    succeeded = daisy.run_blockwise(
        input_roi,
        block_read_roi,
        block_write_roi,
        process_function=lambda: predict_worker(
            experiment,
            setup,
            network_dir,
            iteration,
            raw_file,
            raw_dataset,
            auto_file,
            auto_dataset,
            out_file,
            out_dataset,
            db_host,
            db_name,
            queue,
            singularity_image),
        check_function=lambda b: check_block(
            blocks_predicted,
            b),
        num_workers=num_workers,
        read_write_conflict=False,
        fit='overhang')

    if not succeeded:
        raise RuntimeError("Prediction failed for (at least) one block")

def predict_worker(
        experiment,
        setup,
        network_dir,
        iteration,
        raw_file,
        raw_dataset,
        auto_file,
        auto_dataset,
        out_file,
        out_dataset,
        db_host,
        db_name,
        queue,
        singularity_image):

    # get the relevant worker script to distribute
    setup_dir = os.path.join('..', experiment, '02_train', setup)
    predict_script = os.path.abspath(os.path.join(setup_dir, 'predict.py'))

    if raw_file.endswith('.json'):
        with open(raw_file, 'r') as f:
            spec = json.load(f)
            raw_file = spec['container']

    worker_config = {
        'queue': queue,
        'num_cpus': 5,
        'num_cache_workers': 5
    }

    config = {
        'iteration': iteration,
        'raw_file': raw_file,
        'raw_dataset': raw_dataset,
        'auto_file': auto_file,
        'auto_dataset': auto_dataset,
        'out_file': out_file,
        'out_dataset': out_dataset,
        'db_host': db_host,
        'db_name': db_name,
        'worker_config': worker_config
    }

    # get a unique hash for this configuration
    config_str = ''.join(['%s'%(v,) for v in config.values()])
    config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))

    # get worker id
    worker_id = daisy.Context.from_env().worker_id

    output_dir = os.path.join('.predict_blockwise', network_dir)
    os.makedirs(output_dir, exist_ok=True)

    # pipe output
    config_file = os.path.join(output_dir, '%d.config'%config_hash)
    log_out = os.path.join(output_dir, 'predict_blockwise_%d.out'%worker_id)
    log_err = os.path.join(output_dir, 'predict_blockwise_%d.err'%worker_id)

    with open(config_file, 'w') as f:
        json.dump(config, f)

    logging.info('Running block with config %s...'%config_file)

    # create worker command
    command = [
        'bsub',
        '-n', str(worker_config['num_cpus']),
        '-o', f'{log_out}',
        '-gpu', 'num=1',
        '-q', worker_config['queue']
    ]

    if singularity_image is not None:
        command += [
                'singularity exec',
                '-B', '/groups',
                '--nv', singularity_image
            ]

    command += [
        'python -u %s %s'%(
            predict_script,
            config_file
        )]

    logging.info(f'Worker command: {command}')

    # call command
    daisy.call(command, log_out=log_out, log_err=log_err)

    logging.info('Predict worker finished')

    # if things went well, remove temporary files
    # os.remove(config_file)
    # os.remove(log_out)
    # os.remove(log_err)

def check_block(blocks_predicted, block):

    done = blocks_predicted.count({'block_id': block.block_id}) >= 1

    return done

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    start = time.time()

    predict_blockwise(**config)

    end = time.time()

    seconds = end - start
    logging.info(f'Total time to predict: {seconds}')

