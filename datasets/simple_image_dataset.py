"""Loads all images in a folder while ignoring class information etc."""

import torch.utils.data
from typing import Optional, Callable
import glob
import os
import torchvision


class SimpleImageDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = root
        self.image_paths = list(sorted(list(glob.glob(os.path.join(root, "*.*")))))

        if transform is None:
            transform = lambda x: x
        self.transform = transform

        self.loader = torchvision.datasets.folder.pil_loader

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        assert 0 <= item < len(self)
        return self.transform(self.loader(self.image_paths[item]))
