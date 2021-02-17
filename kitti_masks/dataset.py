"""Modified https://github.com/bethgelab/slow_disentanglement/blob/master/scripts/dataset.py"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pickle
from matplotlib import pyplot as plt


class KittiMasks(Dataset):
    """
    latents encode:
    0: center of mass vertical position
    1: center of mass horizontal position
    2: area
    """

    def __init__(self, path="./data/kitti/", transform=None, max_delta_t=5):
        self.path = path
        self.data = None
        self.latents = None
        self.lens = None
        self.cumlens = None
        self.max_delta_t = max_delta_t
        self.fname = "kitti_peds_v2.pickle"
        self.url = (
            "https://zenodo.org/record/3931823/files/kitti_peds_v2.pickle?download=1"
        )

        if transform == "default":
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomAffine(
                        degrees=(2.0, 2.0), translate=(5 / 64.0, 5 / 64.0)
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    lambda x: x.numpy(),
                ]
            )
        else:
            self.transform = None

        self.load_data()

    def load_data(self):
        # download if not available
        file_path = os.path.join(self.path, self.fname)
        if not os.path.exists(file_path):
            os.makedirs(self.path, exist_ok=True)
            print(f"file not found, downloading from {self.url} ...")
            from urllib import request

            url = self.url
            request.urlretrieve(url, file_path)

        with open(file_path, "rb") as data:
            data = pickle.load(data)
        self.data = data["pedestrians"]
        self.latents = data["pedestrians_latents"]

        self.lens = [
            len(seq) - 1 for seq in self.data
        ]  # start image in sequence can never be starting point
        self.cumlens = np.cumsum(self.lens)

    def sample_observations(self, num, random_state, return_latents=False):
        """Sample a batch of observations X. Needed in dis. lib."""
        assert not (num % 2)
        batch_size = int(num / 2)
        indices = random_state.choice(self.__len__(), 2 * batch_size, replace=False)
        batch, latents = [], []
        for ind in indices:
            first_sample, second_sample, l1, l2 = self.__getitem__(ind)
            batch.append(first_sample)
            latents.append(l1)
        batch = np.stack(batch)
        if not return_latents:
            return batch
        else:
            return batch, np.stack(latents)

    def sample(self, num, random_state):
        # Sample a batch of factors Y and observations X
        x, y = self.sample_observations(num, random_state, return_latents=True)
        return y, x

    def __getitem__(self, index):
        sequence_ind = np.searchsorted(self.cumlens, index, side="right")
        if sequence_ind == 0:
            start_ind = index
        else:
            start_ind = index - self.cumlens[sequence_ind - 1]
        seq_len = len(self.data[sequence_ind])
        t_steps_forward = np.random.randint(1, self.max_delta_t + 1)
        end_ind = min(start_ind + t_steps_forward, seq_len - 1)

        first_sample = self.data[sequence_ind][start_ind].astype(np.uint8) * 255
        second_sample = self.data[sequence_ind][end_ind].astype(np.uint8) * 255

        latents1 = self.latents[sequence_ind][
            start_ind
        ]  # center of mass vertical, com hor, area
        latents2 = self.latents[sequence_ind][
            end_ind
        ]  # center of mass vertical, com hor, area

        if self.transform:
            stack = np.concatenate(
                [
                    first_sample[:, :, None],
                    second_sample[:, :, None],
                    np.ones_like(second_sample[:, :, None]) * 255,
                ],  # add ones to treat like RGB image
                axis=2,
            )
            samples = self.transform(stack)  # do same transforms to start and ending
            first_sample, second_sample = samples[0], samples[1]

        if len(first_sample.shape) == 2:  # set channel dim to 1
            first_sample = first_sample[None]
            second_sample = second_sample[None]

        if np.issubdtype(first_sample.dtype, np.uint8) or np.issubdtype(
            second_sample.dtype, np.uint8
        ):
            first_sample = first_sample.astype(np.float32) / 255.0
            second_sample = second_sample.astype(np.float32) / 255.0

        return first_sample, second_sample, latents1, latents2

    def __len__(self):
        return self.cumlens[-1]


def custom_collate(sample):
    inputs, labels = [], []
    for s in sample:
        inputs.append(s[0])
        inputs.append(s[1])
        labels.append(s[2])
        labels.append(s[3])
    return torch.tensor(np.stack(inputs)), torch.tensor(np.stack(labels))


def return_data(args):
    name = args.dataset
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    assert image_size == 64, "currently only image size of 64 is supported"
    # half batch_size for video couples
    assert not (batch_size % 2)
    batch_size = batch_size // 2
    num_channel = 1

    if name.lower() == "kittimasks":
        if args.evaluate:
            train_data = KittiMasks(max_delta_t=args.kitti_max_delta_t, transform=None)
        else:
            train_data = KittiMasks(max_delta_t=args.kitti_max_delta_t)
        num_channel = 1
    else:
        raise NotImplementedError

    return (
        DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=custom_collate,
        ),
        num_channel,
    )


def test_data(dset, plot=False):
    print(
        f"dataset {dset.data.shape}, min {np.min(dset.data)}, max {np.max(dset.data)} "
        f"type {type(dset.data)} {dset.data.dtype}, factors {dset.factor_sizes}"
    )

    print("Laplace")
    dset.prior = "laplace"
    dl = DataLoader(dset, shuffle=True, batch_size=32, collate_fn=custom_collate)
    for b, l in dl:
        print(b.shape, type(b), b.min().item(), b.max().item(), l.shape)
        if plot:
            plt.figure(figsize=(12, 12))
            for i in range(32):
                plt.subplot(8, 4, i + 1)
                if b.shape[1] == 1:
                    plt.imshow(b[i, 0])
                elif b.shape[1] == 3:
                    plt.imshow(np.transpose(b[i], (1, 2, 0)))
                plt.title(str(l[i]))
                plt.axis("off")
            plt.tight_layout()
            plt.show()
        break

    print("Uniform")
    dset.prior = "uniform"
    dl = DataLoader(dset, shuffle=True, batch_size=32, collate_fn=custom_collate)
    for b, l in dl:
        print(b.shape, type(b), b.min().item(), b.max().item(), l.shape)
        if plot:
            plt.figure(figsize=(12, 12))
            for i in range(32):
                plt.subplot(8, 4, i + 1)
                if b.shape[1] == 1:
                    plt.imshow(b[i, 0])
                elif b.shape[1] == 3:
                    plt.imshow(np.transpose(b[i], (1, 2, 0)))
                plt.title(str(l[i]))
                plt.axis("off")
            plt.tight_layout()
            plt.show()
        break
