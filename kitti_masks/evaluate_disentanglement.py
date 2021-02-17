"""Modified https://github.com/bethgelab/slow_disentanglement/blob/master/scripts/evaluate_disentanglement.py"""
import torch
import numpy as np
import gin.tf

gin.enter_interactive_mode()
import time
import os

# from scripts.model import reparametrize
from kitti_masks.model import reparametrize

# from scripts.model import BetaVAE_H as BetaVAE
from kitti_masks.model import BetaVAE_H as BetaVAE
from disentanglement_lib.utils import results

# needed later:


def main(args, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    net = BetaVAE
    net = net(args.z_dim, args.num_channel, args.box_norm).to(device)
    file_path = os.path.join(args.ckpt_dir, "last")
    checkpoint = torch.load(file_path)
    net.load_state_dict(checkpoint["model_states"]["net"])

    def mean_rep(x):
        distributions = net._encode(torch.from_numpy(x).float().to(device))
        # mu = distributions[:, :net.z_dim]
        # logvar = distributions[:, net.z_dim:]
        mu = distributions
        return np.array(mu.detach().cpu())

    def sample_rep(x):
        distributions = net._encode(torch.from_numpy(x).float().to(device))
        mu = distributions[:, : net.z_dim]
        logvar = distributions[:, net.z_dim :]
        return np.array(reparametrize(mu, logvar).detach().cpu())

    @gin.configurable("evaluation")
    def evaluate(
        post, output_dir, evaluation_fn=gin.REQUIRED, random_seed=gin.REQUIRED, name=""
    ):
        experiment_timer = time.time()
        assert post == "mean" or post == "sampled"
        results_dict = evaluation_fn(
            dataset,
            mean_rep if post == "mean" else sample_rep,
            random_state=np.random.RandomState(random_seed),
        )
        results_dict["elapsed_time"] = time.time() - experiment_timer
        results.update_result_directory(output_dir, "evaluation", results_dict)

    random_state = np.random.RandomState(0)
    config_dir = "metric_configs"
    eval_config_files = [
        f for f in os.listdir(config_dir) if not (f.startswith(".") or "others" in f)
    ]
    t0 = time.time()
    posts = ["mean"]
    for post in posts:
        for eval_config in eval_config_files:
            metric_name = os.path.basename(eval_config).replace(".gin", "")
            continuous = False
            if args.dataset == "kittimasks" or (
                args.dataset == "natural" and not args.natural_discrete
            ):
                continuous = True
            if continuous:
                if metric_name != "mcc":
                    continue
            contains = True
            if args.specify:
                contains = False
                for specific in args.specify.split("_"):
                    if specific in metric_name:
                        contains = True
                        break
            if contains:
                if args.verbose:
                    print("Computing metric '{}' on '{}'...".format(metric_name, post))
                eval_bindings = [
                    "evaluation.random_seed = {}".format(random_state.randint(2 ** 32)),
                    "evaluation.name = '{}'".format(metric_name),
                ]
                gin.parse_config_files_and_bindings(
                    [os.path.join(config_dir, eval_config)], eval_bindings
                )
                output_dir = os.path.join(
                    args.output_dir, "evaluation", args.ckpt_name, post, metric_name
                )
                evaluate(post, output_dir)
                gin.clear_config()
                if args.verbose:
                    print("took", time.time() - t0, "s")
            t0 = time.time()
