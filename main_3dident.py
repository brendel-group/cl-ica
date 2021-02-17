import numpy as np
import torch
import faiss
from torch import nn
import argparse
import spaces
import encoders
import latent_spaces
import layers
import losses
import disentanglement_utils
from datasets.threedident_dataset import (
    ThreeDIdentDataset,
    SequentialThreeDIdentDataset,
)
from datetime import datetime
from torchvision import transforms
import os
from torchvision import models
import torch.utils.data
import invertible_network_utils
from infinite_iterator import InfiniteIterator

# color only means object color, background color and spot light color -> 2*n_objects+1 dim
# rotation ony means object rotation and spotlight rotation in euler angles -> 7*n_objects dim

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", default=512, type=int)
parser.add_argument("--n-eval-samples", default=4096, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--optimizer", default="adam", choices=("adam", "sgd"))
parser.add_argument(
    "--iterations", default=30000, type=int, help="How long to train the model"
)
parser.add_argument(
    "--n-log-steps",
    default=100,
    type=int,
    help="How often to calculate scores and print them",
)
parser.add_argument(
    "--load-model", default=None, type=str, help="Path from where to load the model"
)
parser.add_argument(
    "--save-model", default=None, type=str, help="Path where to save the model"
)
parser.add_argument(
    "--save-every",
    default=None,
    type=int,
    help="After how many steps to save the model (will always be saved at the end)",
)
parser.add_argument("--no-cuda", action="store_true")
parser.add_argument("--position-only", action="store_true")
parser.add_argument("--rotation-and-color-only", action="store_true")
parser.add_argument("--rotation-only", action="store_true")
parser.add_argument("--color-only", action="store_true")
parser.add_argument("--no-spotlight-position", action="store_true")
parser.add_argument("--no-spotlight-color", action="store_true")
parser.add_argument("--no-spotlight", action="store_true")
parser.add_argument("--non-periodic-rotation-and-color", action="store_true")
parser.add_argument("--dummy-mixing", action="store_true")
parser.add_argument("--identity-solution", action="store_true")
parser.add_argument("--identity-mixing-and-solution", action="store_true")
parser.add_argument("--approximate-dataset-nn-search", action="store_true")
parser.add_argument("--offline-dataset", type=str, required=True)
parser.add_argument("--faiss-omp-threads", type=int, default=16)
parser.add_argument(
    "--box-constraint",
    type=str,
    required=False,
    default=None,
    choices=(None, "fix", "learnable"),
)
parser.add_argument(
    "--sphere-constraint",
    type=str,
    required=False,
    default=None,
    choices=(None, "fix", "learnable"),
)
parser.add_argument(
    "--workers", default=0, type=int, help="Number of workers to use (0=#cpus)"
)
parser.add_argument(
    "--mode", default="supervised", choices=("supervised", "unsupervised", "test")
)
parser.add_argument("--supervised-loss", default="mse", type=str, choices=("mse", "r2"))
parser.add_argument(
    "--unsupervised-loss", default="l2", type=str, choices=("l1", "l2", "l3", "vmf")
)
parser.add_argument(
    "--non-periodical-conditional", default="l2", choices=("l1", "l2", "l3")
)
parser.add_argument(
    "--sigma",
    default=0.1,
    type=float,
    help="Sigma of the conditional distribution (for vMF: 1/kappa)",
)
parser.add_argument(
    "--encoder", default="rn18", choices=("rn18", "rn50", "rn101", "rn151")
)

args = parser.parse_args()

if args.no_spotlight:
    args.no_spotlight_color = True
    args.no_spotlight_position = True

print(args)

assert args.save_every is None or (args.save_every > 0)
assert not (
    args.save_model is None and args.save_every is not None
), "--save-every requires --save-model to be set"

assert not (
    args.position_only and args.rotation_and_color_only
), "Only one of these flags can be set."
assert not (
    args.position_only
    and (
        args.non_periodic_rotation_and_color
        or args.no_spotlight_color
        or args.no_spotlight_position
    )
)

assert not (args.box_constraint is not None and args.sphere_constraint is not None)

if args.save_model is not None:
    assert os.path.exists(
        os.path.dirname(args.save_model)
    ), f"Directory {os.path.dirname(args.save_model)} to save model does not exist"

np.set_printoptions(formatter={"float": lambda x: format(x, "1.5E")})


def setup_latent_space(args, n_objects=1):
    n_color_and_rotation_variables = (
        n_objects
        * (
            4
            + (0 if args.no_spotlight_color else 1)
            + (0 if args.no_spotlight_position else 1)
        )
        + 1
    )
    n_position_variables = n_objects * 3

    sigma = args.sigma

    if args.non_periodical_conditional == "l3":
        non_periodical_conditional = (
            lambda space, z, sigma, size, device: space.generalized_normal(
                z, lbd=sigma, p=3, size=size, device=device
            )
        )
    elif args.non_periodical_conditional == "l2":
        non_periodical_conditional = lambda space, z, sigma, size, device: space.normal(
            z, std=sigma, size=size, device=device
        )
    elif args.non_periodical_conditional == "l1":
        non_periodical_conditional = (
            lambda space, z, sigma, size, device: space.laplace(
                z, lbd=sigma, size=size, device=device
            )
        )

    position_space = latent_spaces.LatentSpace(
        spaces.NBoxSpace(n_position_variables),
        lambda space, size, device: space.uniform(size, device=device),
        lambda space, z, size, device="cpu": non_periodical_conditional(
            space, z, sigma=sigma, size=size, device=device
        ),
    )

    if args.non_periodic_rotation_and_color:
        rotation_and_color_space = latent_spaces.LatentSpace(
            spaces.NBoxSpace(
                n_objects
                * (
                    4
                    + (0 if args.no_spotlight_color else 1)
                    + (0 if args.no_spotlight_position else 1)
                    + 1
                )
            ),
            lambda space, size, device: space.uniform(size, device=device),
            lambda space, z, size, device="cpu": non_periodical_conditional(
                space, z, sigma=sigma, size=size, device=device
            ),
        )

        rotation_space = latent_spaces.LatentSpace(
            spaces.NBoxSpace(n_objects * 3 + (0 if args.no_spotlight_position else 1)),
            lambda space, size, device: space.uniform(size, device=device),
            lambda space, z, size, device="cpu": non_periodical_conditional(
                space, z, sigma=sigma, size=size, device=device
            ),
        )

        color_space = latent_spaces.LatentSpace(
            spaces.NBoxSpace(
                n_objects * (1 + (0 if args.no_spotlight_color else 1)) + 1
            ),
            lambda space, size, device: space.uniform(size, device=device),
            lambda space, z, size, device="cpu": non_periodical_conditional(
                space, z, sigma=sigma, size=size, device=device
            ),
        )
    else:
        rotation_and_color_space = latent_spaces.LatentSpace(
            spaces.NSphereSpace(n_color_and_rotation_variables + 1),
            lambda space, size, device: space.uniform(size, device=device),
            lambda space, z, size, device="cpu": space.von_mises_fisher(
                z, kappa=1 / sigma, size=size, device=device
            ),
        )

        rotation_space = latent_spaces.LatentSpace(
            spaces.NSphereSpace(n_objects * 3 + 1),
            lambda space, size, device: space.uniform(size, device=device),
            lambda space, z, size, device="cpu": space.von_mises_fisher(
                z, kappa=1 / sigma, size=size, device=device
            ),
        )

        color_space = latent_spaces.LatentSpace(
            spaces.NSphereSpace(n_objects * 3 + 1 + 1),
            lambda space, size, device: space.uniform(size, device=device),
            lambda space, z, size, device="cpu": space.von_mises_fisher(
                z, kappa=1 / sigma, size=size, device=device
            ),
        )

    if args.non_periodic_rotation_and_color:
        if args.rotation_and_color_only:
            n_non_angular_variables = rotation_and_color_space.dim
            n_angular_variables = 0
            latent_space = rotation_and_color_space
        elif args.position_only:
            raise ValueError()
        elif args.rotation_only:
            n_non_angular_variables = rotation_space.dim
            n_angular_variables = 0
            latent_space = rotation_space
        elif args.color_only:
            n_non_angular_variables = color_space.dim
            n_angular_variables = 0
            latent_space = color_space
        else:
            latent_space = latent_spaces.ProductLatentSpace(
                [position_space, rotation_and_color_space]
            )
            n_non_angular_variables = rotation_and_color_space.dim + position_space.dim
            n_angular_variables = 0
    else:
        if args.position_only:
            n_non_angular_variables = position_space.dim
            n_angular_variables = 0
            latent_space = position_space
        elif args.rotation_and_color_only:
            n_non_angular_variables = 0
            n_angular_variables = rotation_and_color_space.dim
            latent_space = rotation_and_color_space
        elif args.rotation_only:
            n_non_angular_variables = 0
            n_angular_variables = rotation_space.dim
            latent_space = rotation_space
        elif args.color_only:
            n_non_angular_variables = 0
            n_angular_variables = color_space.dim
            latent_space = color_space
        else:
            latent_space = latent_spaces.ProductLatentSpace(
                [position_space, rotation_and_color_space]
            )
            n_angular_variables = rotation_and_color_space.dim
            n_non_angular_variables = position_space.dim

    return latent_space, n_non_angular_variables, n_angular_variables


def setup_f(args, n_non_angular_latents, n_angular_latents):
    base_encoder_class = {
        "rn18": models.resnet18,
        "rn50": models.resnet50,
        "rn101": models.resnet101,
        "rn152": models.resnet152,
    }[args.encoder]

    if args.identity_solution:
        f = nn.Sequential(layers.Flatten())
        return f

    n_latents = n_non_angular_latents + n_angular_latents

    print(
        "#Latents:",
        n_latents,
        ", #Non-periodic latents:",
        n_non_angular_latents,
        ", #Periodic latents:",
        n_angular_latents,
    )

    periodic_rescale_layer = layers.RescaleLayer(fixed_r=False, mode="eq")
    if args.box_constraint is not None:
        non_periodic_rescale_layer = layers.SoftclipLayer(
            n=n_non_angular_latents, fixed_abs_bound=args.box_constraint == "fix"
        )
    else:
        # this doesnt make a lot of sense but we have to do this for the SimCLR loss
        if args.sphere_constraint is not None:
            non_periodic_rescale_layer = layers.RescaleLayer(
                fixed_r=args.sphere_constraint == "fix", mode="eq"
            )
        else:
            non_periodic_rescale_layer = layers.Lambda(lambda x: x)

    if args.position_only:
        # identity
        rescaling = non_periodic_rescale_layer
    elif args.rotation_and_color_only or args.rotation_only or args.color_only:
        if args.non_periodic_rotation_and_color:
            rescaling = non_periodic_rescale_layer
        else:
            rescaling = periodic_rescale_layer
    else:
        if args.non_periodic_rotation_and_color:
            rescaling = non_periodic_rescale_layer
        else:
            rescaling = layers.Lambda(
                lambda x: torch.cat(
                    (
                        non_periodic_rescale_layer(x[:, :n_non_angular_latents]),
                        # box_rescale_layer(x[:, :3*n_objects]),
                        periodic_rescale_layer(x[:, n_non_angular_latents:]),
                    ),
                    dim=1,
                )
            )

    if args.dummy_mixing:
        f = nn.Sequential(
            encoders.get_mlp(
                n_in=n_latents,
                n_out=n_latents,
                layers=[
                    n_latents * 10,
                    n_latents * 50,
                    n_latents * 50,
                    n_latents * 50,
                    n_latents * 50,
                    n_latents * 10,
                ],
                output_normalization=None,
            ),
            rescaling,
        )
    else:
        f = nn.Sequential(
            base_encoder_class(False, num_classes=n_latents * 10),
            nn.LeakyReLU(),
            nn.Linear(n_latents * 10, n_latents),
            rescaling,
        )

    f = torch.nn.DataParallel(f)

    if args.load_model is not None:
        print("device", device)
        f.load_state_dict(torch.load(args.load_model, map_location=device))
        print("Model loaded:", args.load_model)

    if device == "cpu":
        f = f.module

    f = f.to(device)

    return f


def test(args, test_loader):
    test_iterator = InfiniteIterator(test_loader)
    (
        permutation_disentanglement_score,
        linear_disentanglement_score,
        mse,
        linear_fit_mse,
    ) = evaluate(args, f, test_iterator, not args.identity_solution, True)

    print(
        f"Lin. Disentanglement: {linear_disentanglement_score}, MCC: {permutation_disentanglement_score}, MSE: {mse}, lin. fit MSE: {linear_fit_mse}"
    )


def train_unsupervised(args, train_loader):
    n_log_steps = args.n_log_steps
    n_steps = args.iterations
    evaluate_permutation_disentanglement = True  #  False and args.n <= 5

    spherical_loss = losses.SimCLRLoss(normalize=False, tau=1.0)

    if args.unsupervised_loss == "l2":
        nonspherical_loss = losses.LpSimCLRLoss(
            p=2, tau=1.0, simclr_compatibility_mode=True, pow=True
        )
    elif args.unsupervised_loss == "l1":
        nonspherical_loss = losses.LpSimCLRLoss(
            p=1, tau=1.0, simclr_compatibility_mode=True, pow=True
        )
    elif args.unsupervised_loss == "l3":
        nonspherical_loss = losses.LpSimCLRLoss(
            p=3, tau=1.0, simclr_compatibility_mode=True, pow=True
        )
    elif args.unsupervised_loss == "vmf":
        nonspherical_loss = losses.SimCLRLoss(normalize=True, tau=1.0)

    def loss(z1, z2_con_z1, z3, z1_rec, z2_con_z1_rec, z3_rec):
        nsl = nonspherical_loss(
            z1,
            z2_con_z1,
            z3,
            z1_rec[:, :n_non_angular_latents],
            z2_con_z1_rec[:, :n_non_angular_latents],
            z3_rec[:, :3],
        )
        sl = spherical_loss(
            z1,
            z2_con_z1,
            z3,
            z1_rec[:, n_non_angular_latents:],
            z2_con_z1_rec[:, n_non_angular_latents:],
            z3_rec[:, 3:],
        )
        return sl[0] + nsl[0], [(sl[0], sl[1])] + [(nsl[0], nsl[1])]

    if args.position_only:
        loss = nonspherical_loss
    elif args.rotation_and_color_only or args.rotation_only or args.color_only:
        loss = spherical_loss

    if args.non_periodic_rotation_and_color:
        loss = nonspherical_loss

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(f.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(f.parameters(), lr=args.lr)

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

    def train_step(data, loss, optimizer):
        (z1, z2_con_z1), (x1, x2_con_x1) = data

        if args.dummy_mixing:
            x1 = g(z1.to(device)).cpu()
            x2_con_x1 = g(z2_con_z1.to(device)).cpu()

        optimizer.zero_grad()

        if args.identity_mixing_and_solution:
            z1_rec = z1 * identity_scale
            z2_con_z1_rec = z2_con_z1 * identity_scale
        else:
            z1_rec = f(x1)
            z2_con_z1_rec = f(x2_con_x1)

        del x1, x2_con_x1

        # create random "negative" pairs
        # this is faster than sampling z3 again from the marginal distribution
        # and should also yield samples as if they were sampled from the marginal
        z3_rec = torch.roll(z1_rec, 1, 0)

        total_loss_value, total_loss_per_item_value, losses_value = loss(
            None, None, None, z1_rec, z2_con_z1_rec, z3_rec
        )
        # print(total_loss_value)

        if not args.identity_mixing_and_solution:
            total_loss_value.backward()
            optimizer.step()

        return (
            total_loss_value.item(),
            total_loss_per_item_value,
            unpack_item_list(losses_value),
        )

    individual_losses_values = []
    total_loss_values = []
    linear_disentanglement_scores = []
    permutation_disentanglement_scores = []

    train_iterator = InfiniteIterator(train_loader)

    if args.identity_mixing_and_solution:
        identity_scale = 1.0
    last_save_at_step = 0

    for global_step in range(n_steps):
        data = next(train_iterator)

        total_loss_value, total_loss_per_item_value, losses_value = train_step(
            data, loss=loss, optimizer=optimizer
        )
        total_loss_values.append(total_loss_value)
        individual_losses_values.append(losses_value)

        if global_step % n_log_steps == 0 or global_step == n_steps:
            (
                permutation_disentanglement_score,
                linear_disentanglement_score,
                mse,
                linear_fit_mse,
            ) = evaluate(args, f, train_iterator, evaluate_permutation_disentanglement)
        else:
            linear_disentanglement_score = np.inf
            permutation_disentanglement_score = np.inf

        linear_disentanglement_scores.append(linear_disentanglement_score)
        permutation_disentanglement_scores.append(permutation_disentanglement_score)

        if global_step % n_log_steps == 0 or global_step == n_steps:
            print(
                f"[{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}] \t",
                f"Step: {global_step + 1} \t",
                f"Loss: {total_loss_value:.6f} \t",
                f"sigma(loss): {torch.std(total_loss_per_item_value).item()} \t",
                f"<Loss>: {np.mean(np.array(total_loss_values[-n_log_steps:])):.6f} \t",
                f"sigma(<Loss>): {np.std(np.array(total_loss_values[-n_log_steps:])):.6f} \t",
                # f"Losses: {[it[0] for it in losses_value]} \t",
                # f"<Losses>: {[np.mean(lv) for lv in np.array([[i[0] for i in it] for it in individual_losses_values[-n_log_steps:]]).T]} \t",
                f"Lin. Disentanglement: {linear_disentanglement_score:.6f} \t",
                f"Perm. Disentanglement (MCC): {permutation_disentanglement_score:.4f}",
                f"L2: {mse}",
                f"lin. L2: {linear_fit_mse}",
            )

            if args.identity_mixing_and_solution:
                identity_scale = float(input("scale?: "))
                print("scale:", identity_scale)

        global_step += 1

        if args.save_every is not None:
            if global_step // args.save_every != last_save_at_step // args.save_every:
                last_save_at_step = global_step
                model_path = args.save_model + f".iteration_{global_step}"
                torch.save(f.state_dict(), model_path)
                print("Model saved as", model_path)


def train_supervised(args, train_loader):
    n_log_steps = args.n_log_steps
    n_steps = args.iterations
    evaluate_permutation_disentanglement = False and args.n <= 5

    if args.supervised_loss == "r2":
        loss = losses.R2Loss(reduction="mean", mode="negative_r2")
    elif args.supervised_loss == "mse":
        loss = torch.nn.MSELoss(reduction="mean")

    if not args.identity_solution:
        if args.optimizer == "adam":
            optimizer = torch.optim.Adam(f.parameters(), lr=args.lr)
        elif args.optimizer == "sgd":
            optimizer = torch.optim.SGD(f.parameters(), lr=args.lr)

    def train_step(data, loss, optimizer):
        (z1, _), (x1, _) = data

        if args.dummy_mixing:
            x1 = g(z1.to(device)).cpu()

        if not args.identity_solution:
            optimizer.zero_grad()

        hz1 = f(x1)
        total_loss_value = loss(hz1, z1.to(device))

        if not args.identity_solution:
            total_loss_value.backward()
            optimizer.step()

        return total_loss_value.item()

    total_loss_values = []
    linear_disentanglement_scores = []
    permutation_disentanglement_scores = []

    train_iterator = InfiniteIterator(train_loader)

    last_save_at_step = 0

    for global_step in range(n_steps):
        if global_step % n_log_steps == 0 or global_step == n_steps:
            (
                permutation_disentanglement_score,
                linear_disentanglement_score,
                mse,
                linear_fit_mse,
            ) = evaluate(args, f, train_iterator, evaluate_permutation_disentanglement)
        else:
            linear_disentanglement_score = np.inf
            permutation_disentanglement_score = np.inf
        linear_disentanglement_scores.append(linear_disentanglement_score)
        permutation_disentanglement_scores.append(permutation_disentanglement_score)

        data = next(train_iterator)

        if not args.identity_solution:
            total_loss_value = train_step(data, loss=loss, optimizer=optimizer)
        else:
            total_loss_value = np.inf

        total_loss_values.append(total_loss_value)

        if global_step % n_log_steps == 0 or global_step == n_steps:
            print(
                f"[{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}] \t"
                f"Step: {global_step} \t",
                f"Loss: {total_loss_value:.6f} \t",
                f"<Loss>: {np.mean(np.array(total_loss_values[-n_log_steps:])):.6f} \t",
                f"Lin. Disentanglement: {linear_disentanglement_score:.6f} \t",
                # f"Perm. Disentanglement: {permutation_disentanglement_score:.4f}",
                f"L2: {mse}",
                f"lin. L2: {linear_fit_mse}",
            )

        global_step += 1

        if args.save_every is not None:
            if global_step // args.save_every != last_save_at_step // args.save_every:
                last_save_at_step = global_step
                model_path = args.save_model + f".iteration_{global_step}"
                torch.save(f.state_dict(), model_path)
                print("Model saved as", model_path)


def evaluate(
    args, f, test_iterator, evaluate_permutation_disentanglement, no_pairs=False
):
    mse_distance = torch.nn.MSELoss(reduction="none")

    z_disentanglement, h_z_disentanglement = [], []
    with torch.no_grad():
        for batch_idx in range(args.n_eval_samples // args.batch_size):
            test_data = next(test_iterator)

            if no_pairs:
                batch_z_disentanglement, batch_x_disentanglement = test_data
            else:
                batch_z_disentanglement, batch_x_disentanglement = (
                    test_data[0][0],
                    test_data[1][0],
                )

            if args.dummy_mixing:
                batch_x_disentanglement = g(batch_z_disentanglement.to(device)).cpu()

            if args.identity_mixing_and_solution:
                batch_h_z_disentanglement = batch_z_disentanglement
            else:
                batch_h_z_disentanglement = (
                    f(batch_x_disentanglement)
                    .detach()
                    .to(batch_z_disentanglement.device)
                )

            z_disentanglement.append(batch_z_disentanglement)
            h_z_disentanglement.append(batch_h_z_disentanglement)

    if len(z_disentanglement) > 0:
        z_disentanglement = torch.cat(z_disentanglement, 0)
        h_z_disentanglement = torch.cat(h_z_disentanglement, 0)

        (linear_disentanglement_score, _), (
            test_z_disentanglement,
            linear_transformed_h_z_test_disentanglement,
        ) = disentanglement_utils.linear_disentanglement(
            z_disentanglement, h_z_disentanglement, mode="r2", train_test_split=True
        )

        if evaluate_permutation_disentanglement:
            (
                permutation_disentanglement_score,
                _,
            ), _ = disentanglement_utils.permutation_disentanglement(
                z_disentanglement,
                h_z_disentanglement,
                mode="pearson",
                solver="munkres",
                rescaling=True,
            )
        else:
            permutation_disentanglement_score = np.inf

        if not args.identity_solution:
            mse = (
                mse_distance(z_disentanglement, h_z_disentanglement)
                .mean(0)
                .detach()
                .cpu()
                .numpy()
            )
        else:
            mse = np.inf

        # use linear_transformed_h_z_disentanglement to get MSE b/w linearly transformed prediction and gt
        # exclude the samples that were used to train the linear fit
        linear_fit_mse = (
            mse_distance(
                torch.Tensor(test_z_disentanglement),
                torch.Tensor(linear_transformed_h_z_test_disentanglement),
            )
            .mean(0)
            .detach()
            .cpu()
            .numpy()
        )
    else:
        mse = np.inf
        linear_fit_mse = np.inf
        linear_disentanglement_score = np.inf
        permutation_disentanglement_score = np.inf

    return (
        permutation_disentanglement_score,
        linear_disentanglement_score,
        mse,
        linear_fit_mse,
    )


use_cuda = torch.cuda.is_available()
use_cuda = use_cuda and not args.no_cuda
if use_cuda:
    device = "cuda"
else:
    device = "cpu"

assert os.path.exists(args.offline_dataset)
print("Using dataset:", args.offline_dataset)

latent_space, n_non_angular_latents, n_angular_latents = setup_latent_space(args)
f = setup_f(args, n_non_angular_latents, n_angular_latents)

if args.dummy_mixing:
    g = invertible_network_utils.construct_invertible_mlp(
        n_angular_latents + n_non_angular_latents,
        n_layers=3,
        act_fct="leaky_relu",
        cond_thresh_ratio=0.0,
        n_iter_cond_thresh=25000,
    )
    g = g.to(device)

    for p in g.parameters():
        p.requires_grad = False

# set FAISS to single thread usage which makes sense as
# pytorch already uses multithreading to call FAISS
faiss.omp_set_num_threads(args.faiss_omp_threads)

if args.identity_mixing_and_solution:
    print("Using identity function for h(z)=f(g(z))")

if args.dummy_mixing or args.identity_mixing_and_solution:
    dataset_kwargs = dict(loader=lambda _: torch.ones(1), transform=None)
else:
    dataset_kwargs = dict(
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.3292, 0.3278, 0.3215], std=[0.0778, 0.0776, 0.0771]
                ),
            ]
        )
    )

latent_dimensions_to_use = None

if args.non_periodic_rotation_and_color:
    if args.rotation_and_color_only:
        latent_dimensions_to_use = [3, 4, 5, 6, 7, 8, 9]
    elif args.rotation_only:
        latent_dimensions_to_use = [3, 4, 5, 6]
    elif args.color_only:
        latent_dimensions_to_use = [7, 8, 9]
    elif args.position_only:
        raise ValueError("Not supported")
    else:
        latent_dimensions_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    if args.no_spotlight_position:
        latent_dimensions_to_use = [it for it in latent_dimensions_to_use if it != 6]
    if args.no_spotlight_color:
        latent_dimensions_to_use = [it for it in latent_dimensions_to_use if it != 8]
else:
    if args.position_only:
        latent_dimensions_to_use = [0, 1, 2]
    elif args.rotation_and_color_only:
        latent_dimensions_to_use = [3, 4, 5, 6, 7, 8, 9, 10]
    if args.no_spotlight_position:
        raise NotImplementedError(
            "This is only supported for non-periodic variables at the moment."
        )
    if args.no_spotlight_color:
        raise NotImplementedError(
            "This is only supported for non-periodic variables at the moment."
        )

print("Using latent dimensions:", latent_dimensions_to_use)

if latent_dimensions_to_use is not None:
    dataset_kwargs["latent_dimensions_to_use"] = latent_dimensions_to_use

if args.mode == "supervised" or args.mode == "unsupervised":
    train_dataset = ThreeDIdentDataset(
        args.offline_dataset,
        latent_space=latent_space,
        approximate_mode=args.approximate_dataset_nn_search,
        use_gpu=False,
        **dataset_kwargs,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )
else:
    test_dataset = SequentialThreeDIdentDataset(args.offline_dataset, **dataset_kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=True,
    )

if args.mode == "supervised":
    train_supervised(args, train_loader)
elif args.mode == "unsupervised":
    train_unsupervised(args, train_loader)
elif args.mode == "test":
    test(args, test_loader)
else:
    raise ValueError()

if args.save_model is not None:
    torch.save(f.state_dict(), args.save_model)
    print(f"Saving final model at: {args.save_model}")
