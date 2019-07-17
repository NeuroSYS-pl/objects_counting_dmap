"""PyTorch dataset for HDF5 files generated with `get_data.py`."""
import os
from random import random
from typing import Optional

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class H5Dataset(Dataset):
    """PyTorch dataset for HDF5 files generated with `get_data.py`."""

    def __init__(self,
                 dataset_path: str,
                 horizontal_flip: float=0.0,
                 vertical_flip: float=0.0):
        """
        Initialize flips probabilities and pointers to a HDF5 file.

        Args:
            dataset_path: a path to a HDF5 file
            horizontal_flip: the probability of applying horizontal flip
            vertical_flip: the probability of applying vertical flip
        """
        super(H5Dataset, self).__init__()
        self.h5 = h5py.File(dataset_path, 'r')
        self.images = self.h5['images']
        self.labels = self.h5['labels']
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

    def __len__(self):
        """Return no. of samples in HDF5 file."""
        return len(self.images)

    def __getitem__(self, index: int):
        """Return next sample (randomly flipped)."""
        # if both flips probabilities are zero return an image and a label
        if not (self.horizontal_flip or self.vertical_flip):
            return self.images[index], self.labels[index]

        # axis = 1 (vertical flip), axis = 2 (horizontal flip)
        axis_to_flip = []

        if random() < self.vertical_flip:
            axis_to_flip.append(1)

        if random() < self.horizontal_flip:
            axis_to_flip.append(2)

        return (np.flip(self.images[index], axis=axis_to_flip).copy(),
                np.flip(self.labels[index], axis=axis_to_flip).copy())


# --- PYTESTS --- #

def test_loader():
    """Test HDF5 dataloader with flips on and off."""
    run_batch(flip=False)
    run_batch(flip=True)


def run_batch(flip):
    """Sanity check for HDF5 dataloader checks for shapes and empty arrays."""
    # datasets to test loader on
    datasets = {
        'cell': (3, 256, 256),
        'mall': (3, 480, 640),
        'ucsd': (1, 160, 240)
    }

    # for each dataset check both training and validation HDF5
    # for each one check if shapes are right and arrays are not empty
    for dataset, size in datasets.items():
        for h5 in ('train.h5', 'valid.h5'):
            # create a loader in "all flips" or "no flips" mode
            data = H5Dataset(os.path.join(dataset, h5),
                             horizontal_flip=1.0 * flip,
                             vertical_flip=1.0 * flip)
            # create dataloader with few workers
            data_loader = DataLoader(data, batch_size=4, num_workers=4)

            # take one batch, check samples, and go to the next file
            for img, label in data_loader:
                # image batch shape (#workers, #channels, resolution)
                assert img.shape == (4, *size)
                # label batch shape (#workers, 1, resolution)
                assert label.shape == (4, 1, *size[1:])

                assert torch.sum(img) > 0
                assert torch.sum(label) > 0

                break
