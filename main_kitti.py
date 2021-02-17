"""Modified https://github.com/bethgelab/slow_disentanglement/blob/master/main.py"""
import argparse
import shutil
import os
import json
import time
import numpy as np
import torch

os.system("pip3 install --upgrade pip")
os.system(
    "pip3 install h5py tensorboard==2.1.0 tensorflow==1.13.1 spriteworld gin-config disentanglement_lib"
)

from kitti_masks.solver import Solver
from kitti_masks.dataset import return_data
from kitti_masks.evaluate_disentanglement import main as eval_dis

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(args, data_loader=None):
    t0 = time.time()
    if not args.experiment_dir:
        dataset_param = ""
        if "kitti" in args.dataset:
            dataset_param = args.kitti_max_delta_t
        elif "natural" in args.dataset:
            dataset_param = args.natural_discrete
        else:
            dataset_param = args.data_distribution
        args.experiment_dir = os.path.join(
            f"{args.dataset}_{dataset_param}", f"{args.p}_{args.box_norm}"
        )
    args.output_dir = os.path.join(args.output_dir, args.experiment_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    existing = os.listdir(args.output_dir)
    if args.random_search or args.random_seeds:
        if str(args.seed) in existing:
            # search for unused hash
            while True:
                args.seed = randint(1000000, 9999999)
                if str(args.seed) not in existing:
                    break
    args.output_dir = os.path.join(args.output_dir, str(args.seed))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    args.ckpt_dir = os.path.join(args.ckpt_dir, args.experiment_dir, str(args.seed))
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir, exist_ok=True)
    if args.use_writer:
        from torch.utils.tensorboard import SummaryWriter

        args.log_dir = os.path.join(args.log_dir, args.experiment_dir, str(args.seed))
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
        for arg in vars(args):
            writer.add_text(arg, str(getattr(args, arg)))
    with open(os.path.join(args.output_dir, "args"), "w") as f:
        json.dump(args.__dict__, f)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.evaluate:
        eval_dis(args, data_loader.dataset)
    else:
        net = Solver(args, data_loader=data_loader)
        failure = net.train()
        if failure:
            print("failed in %.2fs" % (time.time() - t0))
            shutil.rmtree(args.output_dir)
        else:
            args.evaluate = True
            data_loader, num_channel = return_data(args)
            eval_dis(args, data_loader.dataset)
            print("done in %.2fs" % (time.time() - t0))
    # get original args back
    args = parser.parse_args()
    args.num_channel = num_channel
    return args


### For Random Search ###
def randint(low, high):
    return np.int(np.random.randint(low, high, 1)[0])


def uniform(low, high):
    return np.random.uniform(low, high, 1)[0]


def loguniform(low, high):
    return np.exp(np.random.uniform(np.log(low), np.log(high), 1))[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="slowVAE")
    parser.add_argument("--box-norm", type=int, default=0)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--experiment-dir", type=str, default="", help="specify path")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="evaluate instead of train",
    )
    parser.add_argument(
        "--specify",
        default="",
        type=str,
        help="use argument to only compute a subset of metrics",
    )
    parser.add_argument(
        "--random-search",
        action="store_true",
        default=False,
        help="whether to random search for params",
    )
    parser.add_argument(
        "--random-seeds",
        action="store_true",
        default=False,
        help="whether to go over random seeds with UDR params",
    )
    parser.add_argument("--seed", default=2, type=int, help="random seed")
    parser.add_argument("--beta", default=1, type=float, help="weight for kl to normal")
    parser.add_argument(
        "--gamma", default=10, type=float, help="weight for kl to laplace"
    )
    parser.add_argument(
        "--rate-prior",
        default=6,
        type=float,
        help="rate (or inverse scale) for prior laplace (larger -> sparser).",
    )
    parser.add_argument(
        "--data-distribution", default="laplace", type=str, help="(laplace, uniform)"
    )
    parser.add_argument(
        "--rate-data",
        default=1,
        type=float,
        help="rate (or inverse scale) for data laplace (larger -> sparser). (-1 = rand).",
    )
    parser.add_argument(
        "--data-k", default=-1, type=int, help="k for data uniform (-1 = rand)."
    )
    parser.add_argument(
        "--betavae",
        action="store_true",
        default=False,
        help="whether to do standard betavae training (gamma=0)",
    )
    parser.add_argument(
        "--search-beta",
        action="store_true",
        default=False,
        help="whether to do rand search over beta",
    )
    parser.add_argument(
        "--output-dir", default="outputs", type=str, help="output directory"
    )
    parser.add_argument("--log-dir", default="logs", type=str, help="log directory")
    parser.add_argument(
        "--ckpt-dir", default="checkpoints", type=str, help="checkpoint directory"
    )
    parser.add_argument(
        "--max-iter", default=300000, type=float, help="maximum training iteration"
    )
    parser.add_argument(
        "--dataset",
        default="kittimasks",
        type=str,
        help="dataset name (dsprites, cars3d,"
        "smallnorb, shapes3d, mpi3d, kittimasks, natural",
    )
    parser.add_argument("--batch-size", default=64, type=int, help="batch size")
    parser.add_argument(
        "--num-workers", default=2, type=int, help="dataloader num_workers"
    )
    parser.add_argument(
        "--image-size",
        default=64,
        type=int,
        help="image size. now only (64,64) is supported",
    )
    parser.add_argument(
        "--use-writer",
        action="store_true",
        default=False,
        help="whether to use a log writer",
    )
    parser.add_argument(
        "--z-dim", default=10, type=int, help="dimension of the representation z"
    )
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--beta1", default=0.9, type=float, help="Adam optimizer beta1")
    parser.add_argument(
        "--beta2", default=0.999, type=float, help="Adam optimizer beta2"
    )
    parser.add_argument(
        "--ckpt-name",
        default="last",
        type=str,
        help="load previous checkpoint. insert checkpoint filename",
    )
    parser.add_argument(
        "--log-step",
        default=1000,
        type=int,
        help="numer of iterations after which data is logged",
    )
    parser.add_argument(
        "--save-step",
        default=10000,
        type=int,
        help="number of iterations after which a checkpoint is saved",
    )
    parser.add_argument(
        "--kitti-max-delta-t",
        default=1,
        type=int,
        help="max t difference between frames sampled from " "kitti data loader.",
    )
    parser.add_argument(
        "--natural-discrete",
        action="store_true",
        default=False,
        help="discretize natural sprites",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="for evaluation"
    )
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument(
        "--num_runs", default=10, type=int, help="when searching over seeds, do 10"
    )

    args = parser.parse_args()

    assert not (args.random_search and args.betavae and not args.search_beta)
    assert not ((args.random_search or args.random_seeds) and args.evaluate)

    data_loader, num_channel = return_data(args)
    args.num_channel = num_channel
    if args.random_search:
        while True:
            args.seed = randint(1000000, 9999999)
            args.beta = uniform(1, 16) if args.search_beta else 1
            args.gamma = uniform(1, 16) if not args.betavae else 0
            args.rate_prior = uniform(1, 10) if not args.betavae else 1
            args = main(args, data_loader=data_loader)
    elif args.random_seeds:
        for run in range(args.num_runs):
            args.seed = randint(1000000, 9999999)
            args = main(args, data_loader=data_loader)
    else:
        args = main(args, data_loader=data_loader)
