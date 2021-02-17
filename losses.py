"""Definition of loss functions."""

from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from torch import nn


class CLLoss(ABC):
    """Abstract class to define losses in the CL framework that use one
    positive pair and one negative pair"""

    @abstractmethod
    def loss(self, z1, z2_con_z1, z3, z1_rec, z2_con_z1_rec, z3_rec):
        """
        z1_t = h(z1)
        z2_t = h(z2)
        z3_t = h(z3)
        and z1 ~ p(z1), z3 ~ p(z3)
        and z2 ~ p(z2 | z1)

        returns the total loss and componentwise contributions
        """
        pass

    def __call__(self, z1, z2_con_z1, z3, z1_rec, z2_con_z1_rec, z3_rec):
        return self.loss(z1, z2_con_z1, z3, z1_rec, z2_con_z1_rec, z3_rec)


class ConditionalPairCLLoss(ABC):
    """Abstract class to define losses in the CL framework that use one
    positive pair"""

    @abstractmethod
    def loss(self, z1_rec, z2_con_z1_rec):
        """
        z1_rec = h(z1)
        z2_con_z1_rec = h(z2_con_z1)
        and z2, z1 ~ p(z2_con_z1 | z1)
        """
        pass

    def __call__(self, z1_rec, z2_con_z1_rec):
        return self.loss(z1_rec, z2_con_z1_rec)


class MarginalPairCLLoss(ABC):
    """Abstract class to define losses in the CL framework that use one
    negative pair"""

    @abstractmethod
    def loss(self, z1_rec, z3_rec):
        """
        z1_rec = h(z1)
        z3_rec = h(z3)
        and z1 ~ p(z1), z3 ~ p(z3)
        """
        pass

    def __call__(self, z1_rec, z2_rec):
        return self.loss(z1_rec, z2_rec)


class SplitCombinedCLLoss(CLLoss):
    """Split the data into chunks and apply a different loss function to each.

    Args:
        losses_and_indices: Tuple that contains (1) the loss function,
            (2) the start and (3) the end index of the data chunk.
        weights: (Optional) Weights to combine the different losses.
    """

    def __init__(
        self, losses_and_indices: List[Tuple[CLLoss, int, int]], weights: List = None
    ):
        if weights is None:
            weights = torch.ones((len(losses_and_indices),))
        else:
            weights = torch.tensor(weights)
        assert len(weights) == len(losses_and_indices)

        self.weights = weights
        self.losses_and_indices = losses_and_indices

        for l in self.losses_and_indices:
            assert isinstance(l, (tuple, list))
            assert len(l) == 3
            assert isinstance(
                l[1], int
            ), "Second item of tuple must be index: loss[x[idx:]]"
            assert isinstance(
                l[2], int
            ), "Third item of tuple must be index: loss[x[:idx]]"

            assert issubclass(
                type(l[0]),
                (
                    MarginalPairCLLoss,
                    ConditionalPairCLLoss,
                    CLLoss,
                    MarginalSingleCLLoss,
                    CombinedCLLoss,
                ),
            ), f"Invalid class: {type(l[0])}"

    def loss(self, z1, z2_con_z1, z3, z1_rec, z2_con_z1_rec, z3_rec):
        loss_values = []
        loss_per_item_values = []
        individual_loss_values = []
        for (l, start_idx, end_idx), w in zip(self.losses_and_indices, self.weights):
            c_z1 = z1[:, start_idx:end_idx]
            c_z2_con_z1 = z2_con_z1[:, start_idx:end_idx]
            c_z3 = z3[:, start_idx:end_idx]
            c_z1_rec = z1_rec[:, start_idx:end_idx]
            c_z2_con_z1_rec = z2_con_z1_rec[:, start_idx:end_idx]
            c_z3_rec = z3_rec[:, start_idx:end_idx]

            if isinstance(l, MarginalPairCLLoss):
                tl, lpi, ils = l(c_z1_rec, c_z3_rec)
            elif isinstance(l, ConditionalPairCLLoss):
                tl, lpi, ils = l(c_z1_rec, c_z2_con_z1_rec)
            elif isinstance(l, CLLoss):
                tl, lpi, ils = l(
                    c_z1, c_z2_con_z1, c_z3, c_z1_rec, c_z2_con_z1_rec, c_z3_rec
                )
            elif isinstance(l, MarginalSingleCLLoss):
                tl, lpi, ils = l(c_z1)
            elif isinstance(l, CombinedCLLoss):
                tl, lpi, ils = l(
                    c_z1, c_z2_con_z1, c_z3, c_z1_rec, c_z2_con_z1_rec, c_z3_rec
                )
            else:
                raise ValueError("Invalid loss type found:", type(l))
            loss_values.append(tl)
            loss_per_item_values.append(lpi)
            individual_loss_values.append(ils)

        total_loss = torch.tensor(0).to(z1.device)
        for l, w in zip(loss_values, self.weights):
            total_loss = total_loss + w.to(z1.device) * l

        loss_per_item = torch.tensor(loss_per_item_values) * self.weights.view(-1, 1)
        loss_per_item = loss_per_item.sum(0)

        return (
            total_loss,
            loss_per_item,
            list(zip(loss_values, individual_loss_values, individual_loss_values)),
        )


class CombinedCLLoss(SplitCombinedCLLoss):
    """Apply different loss functions to the full data."""

    def __init__(self, losses, weights=None):
        losses_and_indices = [(l, 0, -1) for l in losses]
        super().__init__(losses_and_indices=losses_and_indices, weights=weights)


class SimCLRLoss(CLLoss):
    """InfoNCE loss function for L2 normalized representations
    as used for example by SimCLR.

    Args:
        normalize: Normalize vectors unit hypersphere.
        tau: Rescaling parameter of exponent.
        alpha: Weighting factor between the two summands.
    """

    def __init__(self, normalize: bool = False, tau: float = 1.0, alpha: float = 0.5):
        self.normalize = normalize
        self.tau = tau
        self.alpha = alpha

    def loss(self, z1, z2_con_z1, z3, z1_rec, z2_con_z1_rec, z3_rec):
        del z1, z2_con_z1, z3

        if self.normalize:
            z1_rec = z1_rec / torch.norm(z1_rec, p=2, dim=-1, keepdim=True)
            z2_con_z1_rec = z2_con_z1_rec / torch.norm(
                z2_con_z1_rec, p=2, dim=-1, keepdim=True
            )
            z3_rec = z3_rec / torch.norm(z3_rec, p=2, dim=-1, keepdim=True)

        neg = torch.einsum("ij,kj -> ik", z1_rec, z3_rec)
        pos = torch.einsum("ij,ij -> i", z1_rec, z2_con_z1_rec)

        neg_and_pos = torch.cat((neg, pos.unsqueeze(1)), dim=1)

        loss_pos = -pos / self.tau
        loss_neg = torch.logsumexp(neg_and_pos / self.tau, dim=1)

        loss_pos_mean = torch.mean(loss_pos)
        loss_neg_mean = torch.mean(loss_neg)

        loss = 2 * (self.alpha * loss_pos + (1.0 - self.alpha) * loss_neg)

        loss_mean = torch.mean(loss)

        return loss_mean, loss, [loss_pos_mean, loss_neg_mean]


class UniformityLoss(MarginalPairCLLoss):
    """Uniformity loss, i.e., loss over the negative pairs in L2 normalized InfoNCE."""

    def __init__(self, p: int = 2.0):
        self.p = p

    def loss(self, z1_rec, z3_rec):
        # calculate lp norm across all possible pairings
        deltas = z1_rec.unsqueeze(0) - z3_rec.unsqueeze(1)
        assert not torch.any(torch.isnan(deltas))

        lp = torch.sum(torch.pow(torch.abs(deltas), exponent=self.p), dim=-1)

        loss_per_item = _logmeanexp(-lp, dim=-1)
        loss = torch.mean(loss_per_item, dim=0)

        return loss, loss_per_item, [loss]


class AlignmentLoss(ConditionalPairCLLoss):
    """Alignment loss, i.e., loss over the positive pairs in L2 normalized InfoNCE."""

    def __init__(self, p: int = 2.0):
        self.p = p

    def loss(self, z1_rec, z2_rec):
        delta = torch.abs(z1_rec - z2_rec)

        assert not torch.any(torch.isnan(delta))

        lp = torch.sum(torch.pow(delta, exponent=self.p), -1)
        loss_per_item = lp
        loss = torch.mean(loss_per_item)

        return loss, loss_per_item, [loss]


class AlignmentUniformityLoss(CombinedCLLoss):
    """Convex combination of Alignment and Uniformity losses."""

    def __init__(self, alpha=0.5, p=2.0):
        assert 0 <= alpha <= 1

        super().__init__(
            [AlignmentLoss(p=p), UniformityLoss(p=p)], [1.0 - alpha, alpha]
        )


class MarginalSingleCLLoss(ABC):
    """Abstract class to define loss that uses neither positive nor
    negative pairs but just a single input"""

    @abstractmethod
    def loss(self, z1_rec):
        """
        z1_rec = h(z1)
        and z1 ~ p(z1)
        """
        pass

    def __call__(self, z1_rec):
        return self.loss(z1_rec)


class JacobianDeterminantLoss(MarginalSingleCLLoss):
    """Use the determinant of the Jacobian as an objective."""

    def __init__(self, h: nn.Module):
        self.h = h

    def loss(self, z1):
        assert len(z1.shape) == 2

        batch_size = len(z1)
        jacobian = torch.autograd.functional.jacobian(lambda z: self.h(z), z1)
        jacobian = jacobian[range(batch_size), :, range(batch_size), :]
        jacobian_det = torch.abs(torch.det(jacobian))
        jacobian_det = torch.mean(jacobian_det)

        loss = jacobian_det

        return loss, torch.ones(len(z1)) * np.nan, [loss]


class SlowVAELoss(CLLoss):
    """Loss function as used for training SlowVAE."""

    def __init__(
        self,
        dec_h: nn.Module = None,
        g: nn.Module = None,
        gamma: float = 10.0,
        beta: float = 1.0,
        rate_prior: float = 6.0,
        n: int = 1,
        decoder_dist="bernoulli",
        no_sigmoid: bool = False,
    ):
        import warnings

        warnings.filterwarnings("ignore")
        self.dec_h = dec_h
        device = dec_h.device
        self.g = g
        self.gamma = gamma
        self.beta = beta
        self.rate_prior = rate_prior * torch.ones(1, requires_grad=False, device=device)
        self.decoder_dist = decoder_dist
        self.n = n
        self.normal_dist = torch.distributions.normal.Normal(
            torch.zeros(self.n, device=device), torch.ones(self.n, device=device)
        )
        self.no_sigmoid = no_sigmoid

    def reconstruction_loss(self, x, x_recon, distribution):
        batch_size = x.size(0)
        assert batch_size != 0

        if distribution == "bernoulli":
            recon_loss = F.binary_cross_entropy_with_logits(
                x_recon, x, size_average=False
            ).div(batch_size)
        elif distribution == "gaussian":
            if not self.no_sigmoid:
                x_recon = F.sigmoid(x_recon)
            recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
        else:
            recon_loss = None

        return recon_loss

    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
        return mu + std * eps

    def compute_ent_normal(self, logvar):
        return 0.5 * (logvar + np.log(2 * np.pi * np.e))

    def compute_cross_ent_normal(self, mu, logvar):
        return 0.5 * (mu ** 2 + torch.exp(logvar)) + np.log(np.sqrt(2 * np.pi))

    def compute_cross_ent_laplace(self, mean, logvar, rate_prior):
        var = torch.exp(logvar)
        sigma = torch.sqrt(var)
        ce = (
            -torch.log(rate_prior / 2)
            + rate_prior
            * sigma
            * np.sqrt(2 / np.pi)
            * torch.exp(-(mean ** 2) / (2 * var))
            - rate_prior * mean * (1 - 2 * self.normal_dist.cdf(mean / sigma))
        )
        return ce

    def compute_cross_ent_combined(self, mu0, mu1, logvar0, logvar1):
        logvar = torch.cat([logvar0, logvar1])
        mu = torch.cat([mu0, mu1])
        normal_entropy = self.compute_ent_normal(logvar)
        cross_ent_normal = self.compute_cross_ent_normal(mu, logvar)
        # assuming couples, do Laplace both ways
        cross_ent_laplace = self.compute_cross_ent_laplace(
            mu0 - mu1, logvar0, self.rate_prior
        ) + self.compute_cross_ent_laplace(mu1 - mu0, logvar1, self.rate_prior)
        return [
            x.sum(1).mean(0, True)
            for x in [normal_entropy, cross_ent_normal, cross_ent_laplace]
        ]

    def loss(self, z1, z2_con_z1, z3, z1_rec, z2_con_z1_rec, z3_rec):
        n = len(z1[0])
        assert len(z1[0]) == self.n
        mu0 = z1_rec[:, : self.n]
        mu1 = z2_con_z1_rec[:, : self.n]
        logvar0 = z1_rec[:, self.n :]
        logvar1 = z2_con_z1_rec[:, self.n :]
        recon_loss = self.reconstruction_loss(
            self.g(torch.cat([z1, z2_con_z1])),
            self.dec_h(
                self.reparametrize(torch.cat([mu0, mu1]), torch.cat([logvar0, logvar1]))
            ),
            self.decoder_dist,
        )
        [
            normal_entropy,
            cross_ent_normal,
            cross_ent_laplace,
        ] = self.compute_cross_ent_combined(mu0, mu1, logvar0, logvar1)
        vae_loss = 2 * recon_loss
        kl_normal = cross_ent_normal - normal_entropy
        kl_laplace = cross_ent_laplace - normal_entropy
        vae_loss = vae_loss + self.beta * kl_normal
        vae_loss = vae_loss + self.gamma * kl_laplace
        return (
            vae_loss,
            torch.ones(len(z1)) * torch.nan,
            [recon_loss, kl_normal, kl_laplace],
        )


class LpSimCLRLoss(CLLoss):
    """Extended InfoNCE objective for non-normalized representations based on an Lp norm.

    Args:
        p: Exponent of the norm to use.
        tau: Rescaling parameter of exponent.
        alpha: Weighting factor between the two summands.
        simclr_compatibility_mode: Use logsumexp (as used in SimCLR loss) instead of logmeanexp
        pow: Use p-th power of Lp norm instead of Lp norm.
    """

    def __init__(
        self,
        p: int,
        tau: float = 1.0,
        alpha: float = 0.5,
        simclr_compatibility_mode: bool = False,
        pow: bool = True,
    ):
        self.p = p
        self.tau = tau
        self.alpha = alpha
        self.simclr_compatibility_mode = simclr_compatibility_mode
        self.pow = pow

    def loss(self, z1, z2_con_z1, z3, z1_rec, z2_con_z1_rec, z3_rec):
        del z1, z2_con_z1, z3

        if self.p < 1.0:
            # add small epsilon to make calculation of norm numerically more stable
            neg = torch.norm(
                torch.abs(z1_rec.unsqueeze(0) - z3_rec.unsqueeze(1) + 1e-12),
                p=self.p,
                dim=-1,
            )
            pos = torch.norm(
                torch.abs(z1_rec - z2_con_z1_rec) + 1e-12, p=self.p, dim=-1
            )
        else:
            # TODO: verify this
            # neg = torch.norm(z1_rec.unsqueeze(0) - z3_rec.unsqueeze(1), p=self.p, dim=-1)
            # pos = torch.norm(z1_rec - z2_con_z1_rec, p=self.p, dim=-1)
            neg = torch.norm(
                z1_rec.unsqueeze(1) - z3_rec.unsqueeze(0), p=self.p, dim=-1
            )
            pos = torch.norm(z1_rec - z2_con_z1_rec, p=self.p, dim=-1)

        if self.pow:
            neg = neg.pow(self.p)
            pos = pos.pow(self.p)

        # all = torch.cat((neg, pos.unsqueeze(1)), dim=1)

        if self.simclr_compatibility_mode:
            neg_and_pos = torch.cat((neg, pos.unsqueeze(1)), dim=1)

            loss_pos = pos / self.tau
            loss_neg = torch.logsumexp(-neg_and_pos / self.tau, dim=1)
        else:
            loss_pos = pos / self.tau
            loss_neg = _logmeanexp(-neg / self.tau, dim=1)

        loss = 2 * (self.alpha * loss_pos + (1.0 - self.alpha) * loss_neg)

        loss_mean = torch.mean(loss)
        loss_std = torch.std(loss)

        loss_pos_mean = torch.mean(loss_pos)
        loss_neg_mean = torch.mean(loss_neg)

        # print(loss_std)

        return loss_mean, loss, [loss_pos_mean, loss_neg_mean]


class R2Loss:
    """(Negative) R2 score"""

    def __init__(self, reduction="none", mode="negative_r2"):
        assert mode in ("negative_r2", "r2")

        self.mode = mode
        self.reduction = reduction

    def forward(self, y_pred, y):
        var_y = torch.var(y, unbiased=False, dim=0)
        r2 = 1.0 - F.mse_loss(y_pred, y, reduction="none").mean(0) / var_y
        if self.reduction == "mean":
            r2 = torch.mean(r2)
        elif self.reduction == "sum":
            r2 = torch.sum(r2)

        if self.mode == "r2":
            return r2
        else:
            return -r2

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _logmeanexp(x, dim):
    # do the -log thing to use logsumexp to calculate the mean and not the sum
    # as log sum_j exp(x_j - log N) = log sim_j exp(x_j)/N = log mean(exp(x_j)
    N = torch.tensor(x.shape[dim], dtype=x.dtype, device=x.device)
    return torch.logsumexp(x, dim=dim) - torch.log(N)
