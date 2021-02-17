"""Calculate mean and std of dataset."""

from datasets.simple_image_dataset import SimpleImageDataset
import torch.utils.data
import argparse
import torchvision.transforms
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--root-folder", required=True)
args = parser.parse_args()

dataset = SimpleImageDataset(
    args.root_folder, transform=torchvision.transforms.ToTensor()
)

full_loader = torch.utils.data.DataLoader(
    dataset, shuffle=True, num_workers=os.cpu_count(), batch_size=256
)

mean = torch.zeros(3)
std = torch.zeros(3)
print("==> Computing mean and std..")
for inputs in tqdm(full_loader):
    for i in range(3):
        mean[i] += inputs[:, i, :, :].mean(dim=(-1, -2)).sum(0)
        std[i] += inputs[:, i, :, :].std(dim=(-1, -2)).sum(0)
mean.div_(len(dataset))
std.div_(len(dataset))
print(mean, std)
