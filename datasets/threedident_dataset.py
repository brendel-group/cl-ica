import torchvision
import torch.utils.data
import numpy as np
import os
import sys
import faiss
from latent_spaces import LatentSpace
from typing import Callable, Optional


class ThreeDIdentDataset(torch.utils.data.Dataset):
    """
    Samples latents according to a marginal and conditional distribution and then finds
    the closest latent representation in a previously rendered dataset and returns
    that latent and the according rendering.

    Args:
        root: Path to root folder of the dataset.
        latent_space: Space to sample the negative samples and positive pairs of latents from.
        transform: Transformation to apply to the images.
        approximate_mode: Use a faster approximate mode for the NN matching of the latents.
        use_gpu: Use the GPU for FAISS NN matching.
        loader: How to load the images.
        latent_dimensions_to_use: Which of the latent dimensions should be returned. None for all.
    """

    def __init__(
        self,
        root: str,
        latent_space: LatentSpace,
        transform: Optional[Callable] = None,
        approximate_mode: Optional[bool] = False,
        use_gpu: Optional[bool] = False,
        loader: Optional[Callable] = torchvision.datasets.folder.pil_loader,
        latent_dimensions_to_use=None,
    ):
        super(ThreeDIdentDataset, self).__init__()

        self.root = root
        self.latents = np.load(os.path.join(root, "raw_latents.npy"))
        self.unfiltered_latents = self.latents

        if latent_dimensions_to_use is not None:
            self.latents = np.ascontiguousarray(
                self.latents[:, latent_dimensions_to_use]
            )

        self.latent_space = latent_space
        dummy_sample = latent_space.sample_marginal(size=1, device="cpu")
        assert (
            dummy_sample.shape[1] == self.latents.shape[1]
        ), f"Shapes do not match, i.e. {dummy_sample.shape} vs. {self.latents.shape}"
        if transform is None:
            transform = lambda x: x
        self.transform = transform

        max_length = int(np.ceil(np.log10(len(self.latents))))
        self.image_paths = [
            os.path.join(root, "images", f"{str(i).zfill(max_length)}.png")
            for i in range(self.latents.shape[0])
        ]
        self.loader = loader

        if approximate_mode:
            self._index = faiss.index_factory(
                self.latents.shape[1], "IVF1024_HNSW32,Flat"
            )
            self._index.efSearch = 8
            self._index.nprobe = 10
        else:
            self._index = faiss.IndexFlatL2(self.latents.shape[1])

        if use_gpu:
            # make it an IVF GPU index
            self._index_cpu = self._index
            self._index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, self._index_cpu
            )

        if approximate_mode:
            self._index.train(self.latents)
        self._index.add(self.latents)

    def __len__(self) -> int:
        return sys.maxsize

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(len(self.latents))]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def __getitem__(self, item):
        del item

        # at first sample z, z~
        # then map them to the closes grid point for which we have images
        z = self.latent_space.sample_marginal(size=1, device="cpu")
        z_tilde = self.latent_space.sample_conditional(z, size=1, device="cpu")

        distance_z, index_z = self._index.search(z.numpy(), 1)
        distance_z_tilde, index_z_tilde = self._index.search(z_tilde.numpy(), 2)

        index_z = index_z[0, 0]

        # don't use the same sample for z, z~
        if index_z_tilde[0, 0] != index_z:
            index_z_tilde = index_z_tilde[0, 0]
        else:
            index_z_tilde = index_z_tilde[0, 1]

        z = self.latents[index_z]
        z_tilde = self.latents[index_z_tilde]

        path_z = self.image_paths[index_z]
        path_z_tilde = self.image_paths[index_z_tilde]

        x, x_tilde = self.transform(self.loader(path_z)), self.transform(
            self.loader(path_z_tilde)
        )

        return (z.flatten(), z_tilde.flatten()), (x, x_tilde)


class SequentialThreeDIdentDataset(torch.utils.data.Dataset):
    """
    Sequentially load all samples in the 3DIdent dataset.

    Args:
        root: Path to root folder of the dataset.
        transform: Transformation to apply to the images.
        loader: How to load the images.
        latent_dimensions_to_use: Which of the latent dimensions should be returned. None for all.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        loader: Optional[Callable] = torchvision.datasets.folder.pil_loader,
        latent_dimensions_to_use=None,
    ):
        super(SequentialThreeDIdentDataset, self).__init__()

        self.root = root
        self.latents = np.load(os.path.join(root, "raw_latents.npy"))
        self.unfiltered_latents = self.latents

        if latent_dimensions_to_use is not None:
            self.latents = np.ascontiguousarray(
                self.latents[:, latent_dimensions_to_use]
            )

        if transform is None:
            transform = lambda x: x
        self.transform = transform

        max_length = int(np.ceil(np.log10(len(self.latents))))
        self.image_paths = [
            os.path.join(root, "images", f"{str(i).zfill(max_length)}.png")
            for i in range(self.latents.shape[0])
        ]
        self.loader = loader

    def __len__(self) -> int:
        return len(self.latents)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(len(self.latents))]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def __getitem__(self, item):
        z = self.latents[item]
        path_z = self.image_paths[item]

        x = self.transform(self.loader(path_z))

        return z.flatten(), x
