from gunpowder import *
from lsd.train.gp import AddLocalShapeDescriptor
from skimage import data
from skimage.measure import label
import json
import logging
import numpy as np
import os
import pytest
import shutil
import sys
import zarr

logging.basicConfig(level=logging.INFO)


def test_pipeline():

    blobs = label(
        data.binary_blobs(length=1024, blob_size_fraction=0.01, volume_fraction=0.3)
    ).astype(np.uint64)

    out = zarr.open("synth.zarr", "a")

    out["labels"] = blobs
    out["labels"].attrs["offset"] = [0] * 2
    out["labels"].attrs["resolution"] = [1] * 2

    labels = ArrayKey("LABELS")
    lsds = ArrayKey("LSDS")

    shape = Coordinate((64,) * 2)

    request = BatchRequest()
    request.add(labels, shape)
    request.add(lsds, shape)

    source = ZarrSource(
        "synth.zarr", {labels: "labels"}, {labels: ArraySpec(interpolatable=False)}
    )

    source += RandomLocation()

    pipeline = source

    pipeline += AddLocalShapeDescriptor(labels, lsds, sigma=3, downsample=1)

    pipeline += Stack(10)

    pipeline += Snapshot({lsds: "lsds"}, every=1, output_filename="batch_{id}.zarr")

    with build(pipeline) as b:
        for i in range(1):
            b.request_batch(request)

    shutil.rmtree("synth.zarr")
    shutil.rmtree("snapshots")
