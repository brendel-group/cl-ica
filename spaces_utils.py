"""Utility functions for spaces."""

import torch
import numpy as np
from typing import Callable


def spherical_to_cartesian(r, phi):
    """Convert spherical coordinates to cartesian coordinates."""

    must_convert_to_torch = False
    if isinstance(phi, np.ndarray):
        must_convert_to_torch = True
        phi = torch.Tensor(phi)

    if isinstance(r, (int, float)):
        r = torch.ones((len(phi))) * r

    must_flatten = False
    if len(phi.shape) == 1:
        phi = phi.reshape(1, -1)
        must_flatten = True

    a = torch.cat((torch.ones((len(phi), 1), device=phi.device) * 2 * np.pi, phi), 1)
    si = torch.sin(a)
    si[:, 0] = 1
    si = torch.cumprod(si, dim=1)
    co = torch.cos(a)
    co = torch.roll(co, -1, dims=1)

    result = si * co * r.unsqueeze(-1)

    if must_flatten:
        result = result[0]

    if must_convert_to_torch:
        result = result.numpy()

    return result


def cartesian_to_spherical(x):
    """Convert cartesian to spherical coordinates."""

    must_convert_to_torch = False
    if isinstance(x, np.ndarray):
        must_convert_to_torch = True
        x = torch.Tensor(x)

    must_flatten = False
    if len(x.shape) == 1:
        x = x.reshape(1, -1, 1)
        must_flatten = True

    T = np.triu(np.ones((x.shape[1], x.shape[1])))
    T = torch.Tensor(T).to(x.device)

    rs = torch.matmul(T, (x.unsqueeze(-1) ** 2)).reshape(x.shape)
    rs = torch.sqrt(rs)

    rs[rs == 0] = 1

    phi = torch.acos(torch.clamp(x / rs, -1, 1))[:, :-1]

    # if x.shape[-1] > 2:
    phi[:, -1] = phi[:, -1] + (2 * np.pi - 2 * phi[:, -1]) * (x[:, -1] <= 0).float()

    rs = rs[:, 0]

    if must_convert_to_torch:
        rs = rs.numpy()
        phi = phi.numpy()

    if must_flatten:
        result = rs[0], phi[0]
    else:
        result = rs, phi

    return result


def sample_generalized_normal(mean: torch.Tensor, lbd: float, p: int, shape):
    """Sample from a generalized Normal distribution.
    Modified according to:
    https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/GeneralizedNormal

    Args:
        mean: Mean of the distribution.
        lbd: Parameter controlling the standard deviation of the distribution.
        p: Exponent of the distribution.
        shape: Shape of the samples to generate.
    """

    assert isinstance(lbd, float)

    ipower = 1.0 / p
    gamma_dist = torch.distributions.Gamma(ipower, 1.0)
    gamma_sample = gamma_dist.rsample(shape)
    # could speed up operations, but doesnt....
    # gamma_sample = torch._standard_gamma(torch.ones(shape) * ipower)
    binary_sample = torch.randint(low=0, high=2, size=shape, dtype=mean.dtype) * 2 - 1
    sampled = binary_sample * torch.pow(torch.abs(gamma_sample), ipower)
    return mean + lbd * sampled.to(mean.device)


def truncated_rejection_resampling(
    sampler_fn: Callable,
    min_: float,
    max_: float,
    size: int,
    n: int,
    buffer_size_factor: int = 1,
    device: str = "cpu",
):
    """
    Args:
        sampler_fn:
        min_: Min value of the support.
        max_: Max value of the support.
        size: Number of samples to generate.
        n: Dimensionality of the samples.
        buffer_size_factor: How many more samples to generate
            first to select the feasible ones and samples from them.
        device: Torch device.
    """

    result = torch.ones((size, n), device=device) * np.nan
    finished_mask = ~torch.isnan(result)
    while torch.sum(finished_mask).item() < n * size:
        # get samples from sampler_fn w/o truncation
        buffer = sampler_fn(size * buffer_size_factor)
        buffer = buffer.view(buffer_size_factor, size, n)
        # check which samples are within the feasible set
        buffer_mask = (buffer >= min_) & (buffer <= max_)
        # calculate how many samples to use

        for i in range(buffer_size_factor):
            copy_mask = buffer_mask[i] & (~finished_mask)
            result[copy_mask] = buffer[i][copy_mask]
            finished_mask[copy_mask] = True

    return result
