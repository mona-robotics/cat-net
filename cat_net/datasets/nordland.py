import torch.utils.data
from torchvision import transforms

import os.path
import glob
import random
import numpy as np

from collections import namedtuple
from PIL import Image

from .. import config
from .. import custom_transforms

class Dataset:
    """Load and parse data in Nordland format.

       Reference: https://nrkbeta.no/2013/01/15/nordlandsbanen-minute-by-minute-season-by-season/
    """

    def __init__(self, base_path, dataset, **kwargs):
        self.data_path = os.path.join(base_path, dataset)
        self.frames = kwargs.get('frames', None)
        self.rgb_dir = kwargs.get('rgb_dir', 'rgb')

        self.rgb_files = sorted(glob.glob(
            os.path.join(self.data_path, self.rgb_dir, '*.png')))

        if self.frames is not None:
            self.rgb_files = [self.rgb_files[i] for i in self.frames]

        self.num_frames = len(self.rgb_files)

    def __len__(self):
        return self.num_frames

    def get_rgb(self, idx):
        """Load RGB image from file."""
        return self._load_image(self.rgb_files[idx], mode='RGB', dtype=np.uint8)

    def get_gray(self, idx):
        """Load grayscale image from file."""
        return self._load_image(self.rgb_files[idx], mode='L', dtype=np.uint8)

    def get_depth(self, idx):
        """Load depth image from file."""
        return self._load_image(self.depth_files[idx],
                                mode='F', dtype=np.float, factor=5000)

    def _load_image(self, impath, mode='RGB', dtype=np.float, factor=1):
        """Load image from file."""
        im = Image.open(impath).convert(mode)
        return (np.array(im) / factor).astype(dtype)

class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, source_seq, target_seq, **kwargs):
        self.source = Dataset(config.data_dir, source_seq, **kwargs)
        self.target = Dataset(config.data_dir, target_seq, **kwargs)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        source = Image.fromarray(self.source.get_rgb(idx))
        target = Image.fromarray(self.target.get_rgb(idx))

        transform = transforms.Compose([
            transforms.Resize(min(config.image_load_size)),
            transforms.CenterCrop(config.image_load_size),
            custom_transforms.StatefulRandomCrop(
                config.image_final_size) if config.random_crop else transforms.Resize(config.image_final_size),
            transforms.ToTensor(),
            transforms.Normalize(config.image_mean, config.image_std)
        ])

        source = transform(source)
        target = transform(target)

        return source, target
