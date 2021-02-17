import numpy as np
import torch
import argparse
import losses
import spaces
import disentanglement_utils
import invertible_network_utils
import torch.nn.functional as F
import random
import os
import latent_spaces
import encoders

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = "cuda"
else:
    device = "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        "Disentanglement with InfoNCE/Contrastive Learning - Toy Examples"
    )
    parser.add_argument("--sphere-r", type=float, default=1.0)
    parser.add_argument(
        "--box-min",
        type=float,
        default=0.0,
        help="For box normalization only. Minimal value of box.",
    )
    parser.add_argument(
        "--box-max",
        type=float,
        default=1.0,
        help="For box normalization only. Maximal value of box.",
    )
    parser.add_argument(
        "--sphere-norm", action="store_true", help="Normalize output to a sphere."
    )
    parser.add_argument(
        "--box-norm", action="store_true", help="Normalize output to a box."
    )
    parser.add_argument(
        "--only-supervised", action="store_true", help="Only train supervised model."
    )
    parser.add_argument(
        "--only-unsupervised",
        action="store_true",
        help="Only train unsupervised model.",
    )
    parser.add_argument(
        "--more-unsupervised",
        type=int,
        default=3,
        help="How many more steps to do for unsupervised compared to supervised training.",
    )
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument(
        "--num-eval-batches",
        type=int,
        default=10,
        help="Number of batches to average evaluation performance at the end.",
    )
    parser.add_argument(
        "--rej-mult",
        type=float,
        default=1,
        help="Memory/CPU trade-off factor for rejection resampling.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--act-fct",
        type=str,
        default="leaky_relu",
        help="Activation function in mixing network g.",
    )
    parser.add_argument(
        "--c-param",
        type=float,
        default=0.05,
        help="Concentration parameter of the conditional distribution.",
    )
    parser.add_argument(
        "--m-param",
        type=float,
        default=1.0,
        help="Additional parameter for the marginal (only relevant if it is not uniform).",
    )
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument(
        "--n-mixing-layer",
        type=int,
        default=3,
        help="Number of layers in nonlinear mixing network g.",
    )
    parser.add_argument(
        "--n", type=int, default=10, help="Dimensionality of the latents."
    )
    parser.add_argument(
        "--space-type", type=str, default="box", choices=("box", "sphere", "unbounded")
    )
    parser.add_argument(
        "--m-p",
        type=int,
        default=0,
        help="Type of ground-truth marginal distribution. p=0 means uniform; "
        "all other p values correspond to (projected) Lp Exponential",
    )
    parser.add_argument(
        "--c-p",
        type=int,
        default=2,
        help="Exponent of ground-truth Lp Exponential distribution.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--p",
        type=int,
        default=2,
        help="Exponent of the assumed model Lp Exponential distribution.",
    )
    parser.add_argument("--batch-size", type=int, default=6144)
    parser.add_argument("--n-log-steps", type=int, default=250)
    parser.add_argument("--n-steps", type=int, default=100001)
    parser.add_argument("--resume-training", action="store_true")
    args = parser.parse_args()

    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")

    return args


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    if args.space_type == "box":
        space = spaces.NBoxSpace(args.n, args.box_min, args.box_max)
    elif args.space_type == "sphere":
        space = spaces.NSphereSpace(args.n, args.sphere_r)
    else:
        space = spaces.NRealSpace(args.n)
    if args.p:
        loss = losses.LpSimCLRLoss(
            p=args.p, tau=args.tau, simclr_compatibility_mode=True
        )
    else:
        loss = losses.SimCLRLoss(normalize=False, tau=args.tau)
    eta = torch.zeros(args.n)
    if args.space_type == "sphere":
        eta[0] = 1.0
    if args.m_p:
        if args.m_p == 1:
            sample_marginal = lambda space, size, device=device: space.laplace(
                eta, args.m_param, size, device
            )
        elif args.m_p == 2:
            sample_marginal = lambda space, size, device=device: space.normal(
                eta, args.m_param, size, device
            )
        else:
            sample_marginal = (
                lambda space, size, device=device: space.generalized_normal(
                    eta, args.m_param, p=args.m_p, size=size, device=device
                )
            )
    else:
        sample_marginal = lambda space, size, device=device: space.uniform(
            size, device=device
        )
    if args.c_p:
        if args.c_p == 1:
            sample_conditional = lambda space, z, size, device=device: space.laplace(
                z, args.c_param, size, device
            )
        elif args.c_p == 2:
            sample_conditional = lambda space, z, size, device=device: space.normal(
                z, args.c_param, size, device
            )
        else:
            sample_conditional = (
                lambda space, z, size, device=device: space.generalized_normal(
                    z, args.c_param, p=args.c_p, size=size, device=device
                )
            )
    else:
        sample_conditional = (
            lambda space, z, size, device=device: space.von_mises_fisher(
                z, args.c_param, size, device, rejection_multiplier=args.rej_mult
            )
        )
    latent_space = latent_spaces.LatentSpace(
        space=space,
        sample_marginal=sample_marginal,
        sample_conditional=sample_conditional,
    )

    def sample_marginal_and_conditional(size, device=device):
        z = latent_space.sample_marginal(size=size, device=device)
        z_tilde = latent_space.sample_conditional(z, size=size, device=device)

        return z, z_tilde

    g = invertible_network_utils.construct_invertible_mlp(
        n=args.n,
        n_layers=args.n_mixing_layer,
        act_fct=args.act_fct,
        cond_thresh_ratio=0.0,
        n_iter_cond_thresh=25000,
    )
    g = g.to(device)

    for p in g.parameters():
        p.requires_grad = False

    h_ind = lambda z: g(z)

    z_disentanglement = latent_space.sample_marginal(4096)

    (linear_disentanglement_score, _), _ = disentanglement_utils.linear_disentanglement(
        z_disentanglement, h_ind(z_disentanglement), mode="r2"
    )
    print(f"Id. Lin. Disentanglement: {linear_disentanglement_score:.4f}")
    (
        permutation_disentanglement_score,
        _,
    ), _ = disentanglement_utils.permutation_disentanglement(
        z_disentanglement,
        h_ind(z_disentanglement),
        mode="pearson",
        solver="munkres",
        rescaling=True,
    )
    print(f"Id. Perm. Disentanglement: {permutation_disentanglement_score:.4f}")

    def unpack_item_list(lst):
        if isinstance(lst, tuple):
            lst = list(lst)
        result_list = []
        for it in lst:
            if isinstance(it, (tuple, list)):
                result_list.append(unpack_item_list(it))
            else:
                result_list.append(it.item())
        return result_list

    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        torch.save(g.state_dict(), os.path.join(args.save_dir, "g.pth"))
    if args.only_unsupervised:
        test_list = [False]
    elif args.only_supervised:
        test_list = [True]
    else:
        test_list = [True, False]
    for test in test_list:
        print("supervised test: {}".format(test))

        def train_step(data, loss, optimizer):
            z1, z2_con_z1 = data
            z1 = z1.to(device)
            z2_con_z1 = z2_con_z1.to(device)

            # create random "negative" pairs
            # this is faster than sampling z3 again from the marginal distribution
            # and should also yield samples as if they were sampled from the marginal
            z3 = torch.roll(z1, 1, 0)

            optimizer.zero_grad()

            z1_rec = h(z1)
            z2_con_z1_rec = h(z2_con_z1)
            z3_rec = torch.roll(z1_rec, 1, 0)

            if test:
                total_loss_value = F.mse_loss(z1_rec, z1)
                losses_value = [total_loss_value]
            else:
                total_loss_value, _, losses_value = loss(
                    z1, z2_con_z1, z3, z1_rec, z2_con_z1_rec, z3_rec
                )

            total_loss_value.backward()
            optimizer.step()

            return total_loss_value.item(), unpack_item_list(losses_value)

        output_normalization = None
        if args.box_norm:
            output_normalization = "learnable_box"
        if args.sphere_norm:
            output_normalization = "learnable_sphere"
        else:
            if args.p == 0:
                output_normalization = "fixed_sphere"
            else:
                output_normalization = None
        f = encoders.get_mlp(
            n_in=args.n,
            n_out=args.n,
            layers=[
                args.n * 10,
                args.n * 50,
                args.n * 50,
                args.n * 50,
                args.n * 50,
                args.n * 10,
            ],
            output_normalization=output_normalization,
        )
        f = f.to(device)
        print("f: ", f)
        optimizer = torch.optim.Adam(f.parameters(), lr=args.lr)
        h = lambda z: f(g(z))
        if (
            "total_loss_values" in locals() and not args.resume_training
        ) or "total_loss_values" not in locals():
            individual_losses_values = []
            total_loss_values = []
            linear_disentanglement_scores = []
            permutation_disentanglement_scores = []

        global_step = len(total_loss_values) + 1
        while (
            global_step <= args.n_steps
            if test
            else global_step <= (args.n_steps * args.more_unsupervised)
        ):
            data = sample_marginal_and_conditional(size=args.batch_size)
            total_loss_value, losses_value = train_step(
                data, loss=loss, optimizer=optimizer
            )
            total_loss_values.append(total_loss_value)
            individual_losses_values.append(losses_value)
            if global_step % args.n_log_steps == 1 or global_step == args.n_steps:
                z_disentanglement = latent_space.sample_marginal(4096)
                (
                    linear_disentanglement_score,
                    _,
                ), _ = disentanglement_utils.linear_disentanglement(
                    z_disentanglement, h(z_disentanglement), mode="r2"
                )
                linear_disentanglement_scores.append(linear_disentanglement_score)
                (
                    permutation_disentanglement_score,
                    _,
                ), _ = disentanglement_utils.permutation_disentanglement(
                    z_disentanglement,
                    h(z_disentanglement),
                    mode="pearson",
                    solver="munkres",
                    rescaling=True,
                )
                permutation_disentanglement_scores.append(
                    permutation_disentanglement_score
                )

            else:
                linear_disentanglement_scores.append(linear_disentanglement_scores[-1])
                permutation_disentanglement_scores.append(
                    permutation_disentanglement_scores[-1]
                )
            if global_step % args.n_log_steps == 1 or global_step == args.n_steps:
                print(
                    f"Step: {global_step} \t",
                    f"Loss: {total_loss_value:.4f} \t",
                    f"<Loss>: {np.mean(np.array(total_loss_values[-args.n_log_steps:])):.4f} \t",
                    f"Lin. Disentanglement: {linear_disentanglement_score:.4f} \t",
                    f"Perm. Disentanglement: {permutation_disentanglement_score:.4f}",
                )
                if args.sphere_norm:
                    print(f"r: {f[-1].r}")
            global_step += 1
        if args.save_dir:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            torch.save(
                f.state_dict(),
                os.path.join(
                    args.save_dir, "{}_f.pth".format("sup" if test else "unsup")
                ),
            )
        torch.cuda.empty_cache()
    final_linear_scores = []
    final_perm_scores = []
    with torch.no_grad():
        for i in range(args.num_eval_batches):
            data = sample_marginal_and_conditional(args.batch_size)
            z1, z2_con_z1 = data
            z1 = z1.to(device)
            z2_con_z1 = z2_con_z1.to(device)
            z3 = torch.roll(z1, 1, 0)
            z1_rec = h(z1)
            z2_con_z1_rec = h(z2_con_z1)
            z3_rec = h(z3)
            (
                linear_disentanglement_score,
                _,
            ), _ = disentanglement_utils.linear_disentanglement(z1, z1_rec, mode="r2")
            (
                permutation_disentanglement_score,
                _,
            ), _ = disentanglement_utils.permutation_disentanglement(
                z1, z1_rec, mode="pearson", solver="munkres", rescaling=True
            )
            final_linear_scores.append(linear_disentanglement_score)
            final_perm_scores.append(permutation_disentanglement_score)
    print(
        "linear mean: {} std: {}".format(
            np.mean(final_linear_scores), np.std(final_linear_scores)
        )
    )
    print(
        "perm mean: {} std: {}".format(
            np.mean(final_perm_scores), np.std(final_perm_scores)
        )
    )


if __name__ == "__main__":
    main()
