"""Create latents for 3DIdent dataset."""

import os
import numpy as np
import spaces
import latent_spaces
import argparse
import spaces_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-points", default=1000000, type=int)
    parser.add_argument("--n-objects", default=1, type=int)
    parser.add_argument("--output-folder", required=True, type=str)
    parser.add_argument("--position-only", action="store_true")
    parser.add_argument("--rotation-and-color-only", action="store_true")
    parser.add_argument("--rotation-only", action="store_true")
    parser.add_argument("--color-only", action="store_true")
    parser.add_argument("--fixed-spotlight", action="store_true")
    parser.add_argument("--non-periodic-rotation-and-color", action="store_true")

    args = parser.parse_args()

    print(args)

    assert not (
        args.position_only and args.rotation_and_color_only
    ), "Only either position-only or rotation-and-color-only can be set"

    os.makedirs(args.output_folder, exist_ok=True)

    """
    render internally assumes the variables form these value ranges:
    
    per object:
        0. x position in [-3, -3]
        1. y position in [-3, -3]
        2. z position in [-3, -3]
        3. alpha rotation in [0, 2pi]
        4. beta rotation in [0, 2pi]
        5. gamma rotation in [0, 2pi]
        6. theta spot light in [0, 2pi]
        7. hue object in [0, 2pi]
        8. hue spot light in [0, 2pi]
    
    per scene:
        9. hue background in [0, 2pi]
    """

    n_angular_variables = args.n_objects * 6 + 1
    n_non_angular_variables = args.n_objects * 3

    if args.non_periodic_rotation_and_color:
        s = latent_spaces.LatentSpace(
            spaces.NBoxSpace(n_non_angular_variables + n_angular_variables),
            lambda space, size, device: space.uniform(size, device=device),
            None,
        )

    else:
        s = latent_spaces.ProductLatentSpace(
            [
                latent_spaces.LatentSpace(
                    spaces.NBoxSpace(n_non_angular_variables),
                    lambda space, size, device: space.uniform(size, device=device),
                    None,
                ),
                latent_spaces.LatentSpace(
                    spaces.NSphereSpace(n_angular_variables + 1),
                    lambda space, size, device: space.uniform(size, device=device),
                    None,
                ),
            ]
        )

    raw_latents = s.sample_marginal(args.n_points, device="cpu").numpy()

    if args.position_only or args.rotation_and_color_only:
        assert args.n_objects == 1, "Only one object is supported for fixed variables"

    if args.non_periodic_rotation_and_color:
        if args.position_only:
            raw_latents[:, n_non_angular_variables:] = np.array(
                [-1, -0.66, -0.33, 0, 0.33, 0.66, 1]
            )
        if args.rotation_and_color_only or args.rotation_only or args.color_only:
            raw_latents[:, :n_non_angular_variables] = np.array([0, 0, 0])
        if args.rotation_only:
            # additionally fix color
            raw_latents[:, -3:] = np.array([-1, 0, 1.0])
        if args.color_only:
            # additionally fix rotation
            raw_latents[
                :, n_non_angular_variables : n_non_angular_variables + 4
            ] = np.array([-1, -0.5, 0.5, 1.0])

        if args.fixed_spotlight:
            # assert not args.rotation_only
            raw_latents[:, [-2, -4]] = np.array([0.0, 0.0])

        # the raw latents will later be used for the sampling process
        np.save(os.path.join(args.output_folder, "raw_latents.npy"), raw_latents)

        # get rotation and color latents from large vector
        rotation_and_color_latents = raw_latents[:, n_non_angular_variables:]
        rotation_and_color_latents *= np.pi / 2

        position_latents = raw_latents[:, :n_non_angular_variables]
        position_latents *= 3

    else:
        if args.position_only:
            spherical_fixed_angular_variables = np.array(
                [np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 2, np.pi / 2, 0, 1.5 * np.pi]
            )
            cartesian_fixed_angular_variables = spaces_utils.spherical_to_cartesian(
                1, spherical_fixed_angular_variables
            )
            raw_latents[:, n_non_angular_variables:] = cartesian_fixed_angular_variables
        if args.rotation_and_color_only:
            fixed_non_angular_variables = np.array([0, 0, 0])
            raw_latents[:, :n_non_angular_variables] = fixed_non_angular_variables

        np.save(os.path.join(args.output_folder, "raw_latents.npy"), raw_latents)

        # convert angular latents from cartesian to angular representation
        rotation_and_color_latents = spaces_utils.cartesian_to_spherical(
            raw_latents[:, n_non_angular_variables:]
        )[1]
        # map all but the last latent from [0,pi] to [0, 2pi]
        rotation_and_color_latents[:, :-1] *= 2

        position_latents = raw_latents[:, :n_non_angular_variables]
        # map z coordinate from -1,+1 to 0,+1
        position_latents[:, 2:n_non_angular_variables:3] = (
            position_latents[:, 2:n_non_angular_variables:3] + 1
        ) / 2.0
        position_latents *= 3

    latents = np.concatenate((position_latents, rotation_and_color_latents), 1)

    reordered_transposed_latents = []
    for n in range(args.n_objects):
        reordered_transposed_latents.append(latents.T[n * 3 : n * 3 + 3])
        reordered_transposed_latents.append(
            latents.T[
                n_non_angular_variables + n * 6 : n_non_angular_variables + n * 6 + 6
            ]
        )

    reordered_transposed_latents.append(latents.T[-1].reshape(1, -1))
    reordered_latents = np.concatenate(reordered_transposed_latents, 0).T

    # the latents will be used by the rendering process to generate the images
    np.save(os.path.join(args.output_folder, "latents.npy"), reordered_latents)


if __name__ == "__main__":
    main()
