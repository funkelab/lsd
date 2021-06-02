import json
import logging
import math
import numpy as np
import os
import sys
import torch

from funlib.learn.torch.models import UNet
from gunpowder import *
from gunpowder.ext import torch
from gunpowder.torch import *

logging.basicConfig(level=logging.INFO)

torch.backends.cudnn.benchmark = True

data_dir = '../../01_data/training'

samples = [
    'trvol-250-1.zarr',
    'trvol-250-2.zarr',
    'tstvol-520-1.zarr',
    'tstvol-520-2.zarr',
]

neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

# needs to match order of samples (small to large)
probabilities = [0.05, 0.05, 0.45, 0.45]

class Convolve(torch.nn.Module):

    def __init__(
            self,
            model,
            in_channels,
            out_channels,
            kernel_size=(1,1,1)):

        super().__init__()

        self.model = model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        conv = torch.nn.Conv3d

        self.conv_pass = torch.nn.Sequential(
                            conv(
                                self.in_channels,
                                self.out_channels,
                                self.kernel_size),
                            torch.nn.Sigmoid())

    def forward(self, x):

        y = self.model.forward(x)

        return self.conv_pass(y)

def train_until(max_iteration):

    in_channels = 1
    num_fmaps = 12
    fmap_inc_factors = 6
    downsample_factors = [(2,2,2),(2,2,2),(3,3,3)]

    unet = UNet(
            in_channels,
            num_fmaps,
            fmap_inc_factors,
            downsample_factors)

    model = Convolve(unet, num_fmaps, 3)

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.5e-4,
            betas=(0.95,0.999))

    test_input_shape = Coordinate((196,)*3)
    test_output_shape = Coordinate((84,)*3)

    raw = ArrayKey('RAW')
    labels = ArrayKey('GT_LABELS')
    labels_mask = ArrayKey('GT_LABELS_MASK')
    affs = ArrayKey('PREDICTED_AFFS')
    gt_affs = ArrayKey('GT_AFFS')
    gt_affs_scale = ArrayKey('GT_AFFS_SCALE')
    affs_gradient = ArrayKey('AFFS_GRADIENT')

    voxel_size = Coordinate((8,)*3)
    input_size = Coordinate(test_input_shape) * voxel_size
    output_size = Coordinate(test_output_shape) * voxel_size

    #max labels padding calculated
    labels_padding = Coordinate((376,536,536))

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(gt_affs, output_size)
    request.add(gt_affs_scale, output_size)

    snapshot_request = BatchRequest({
        affs: request[gt_affs]
    })

    data_sources = tuple(
            ZarrSource(
                    os.path.join(data_dir, sample),
                    {
                        raw: 'volumes/raw',
                        labels: 'volumes/labels/neuron_ids',
                        labels_mask: 'volumes/labels/mask',
                    },
                    {
                        raw: ArraySpec(interpolatable=True),
                        labels: ArraySpec(interpolatable=False),
                        labels_mask: ArraySpec(interpolatable=False)
                    }
                ) +
            Normalize(raw) +
            Pad(raw, None) +
            Pad(labels, labels_padding) +
            Pad(labels_mask, labels_padding) +
            RandomLocation(min_masked=0.5, mask=labels_mask)
            for sample in samples
        )

    train_pipeline = data_sources

    train_pipeline += RandomProvider(probabilities=probabilities)

    train_pipeline += ElasticAugment(
            control_point_spacing=[40, 40, 40],
            jitter_sigma=[0, 0, 0],
            rotation_interval=[0,math.pi/2.0],
            prob_slip=0,
            prob_shift=0,
            max_misalign=0,
            subsample=8)

    train_pipeline += SimpleAugment()

    train_pipeline += ElasticAugment(
            control_point_spacing=[40,40,40],
            jitter_sigma=[2,2,2],
            rotation_interval=[0,math.pi/2.0],
            prob_slip=0.01,
            prob_shift=0.01,
            max_misalign=1,
            subsample=8)

    train_pipeline += IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1)

    train_pipeline += GrowBoundary(labels, labels_mask, steps=1)

    train_pipeline += AddAffinities(
            neighborhood,
            labels=labels,
            affinities=gt_affs)

    train_pipeline += BalanceLabels(
            gt_affs,
            gt_affs_scale)

    train_pipeline += IntensityScaleShift(raw, 2,-1)

    train_pipeline += Normalize(gt_affs)

    train_pipeline += Unsqueeze([raw, gt_affs])
    train_pipeline += Unsqueeze([raw])

    train_pipeline += PreCache(
            cache_size=40,
            num_workers=10)

    train_pipeline += Train(
            model=model,
            loss=loss,
            optimizer=optimizer,
            inputs={
                'x': raw
            },
            loss_inputs={
                0: affs,
                1: gt_affs
            },
            outputs={
                0: affs
            },
            save_every=1000,
            log_dir='log')

    train_pipeline += Squeeze([raw])
    train_pipeline += Squeeze([raw, gt_affs, affs])

    train_pipeline += IntensityScaleShift(raw, 0.5, 0.5)

    train_pipeline += Snapshot({
                raw: 'volumes/raw',
                labels: 'volumes/labels/neuron_ids',
                gt_affs: 'volumes/gt_affinities',
                affs: 'volumes/pred_affinities',
                labels_mask: 'volumes/labels/mask'
            },
            dataset_dtypes={
                labels: np.uint64,
                gt_affs: np.float32
            },
            every=1,
            output_filename='batch_{iteration}.zarr',
            additional_request=snapshot_request)

    train_pipeline += PrintProfilingStats(every=1)

    with build(train_pipeline) as b:
        for i in range(max_iteration):
            b.request_batch(request)

if __name__ == '__main__':

    iterations = 100
    train_until(iterations)

