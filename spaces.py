"""Definition of topological/mathematical spaces with probability densities defined on."""

from abc import ABC, abstractmethod
import torch
import numpy as np
import vmf
import spaces_utils as sut


class Space(ABC):
    """Base class."""

    @abstractmethod
    def uniform(self, size, device):
        pass

    @abstractmethod
    def normal(self, mean, std, size, device):
        pass

    @abstractmethod
    def laplace(self, mean, std, size, device):
        pass

    @abstractmethod
    def generalized_normal(self, mean, lbd, p, size, device):
        pass

    @property
    @abstractmethod
    def dim(self):
        pass


class NRealSpace(Space):
    """Unconstrained space over the real numbers, i.e., R^N."""

    def __init__(self, n):
        self.n = n

    @property
    def dim(self):
        return self.n

    def uniform(self, size, device="cpu"):
        raise NotImplementedError("Not defined on R^n")

    def normal(self, mean, std, size, device="cpu"):
        """Sample from a Normal distribution in R^N.

        Args:
            mean: Value(s) to sample around.
            std: Concentration parameter of the distribution (=standard deviation).
            size: Number of samples to draw.
            device: torch device identifier
        """

        if len(mean.shape) == 1 and mean.shape[0] == self.n:
            mean = mean.unsqueeze(0)
        if not torch.is_tensor(std):
            std = torch.ones(self.n) * std
        if len(std.shape) == 1 and std.shape[0] == self.n:
            std = std.unsqueeze(0)
        assert len(mean.shape) == 2
        assert len(std.shape) == 2

        if torch.is_tensor(mean):
            mean = mean.to(device)
        if torch.is_tensor(std):
            std = std.to(device)

        return torch.randn((size, self.n), device=device) * std + mean

    def laplace(self, mean, lbd, size, device="cpu"):
        """Sample from a Laplace distribution in R^N.

        Args:
            mean: Value(s) to sample around.
            lbd: Concentration parameter of the distribution.
            size: Number of samples to draw.
            device: torch device identifier
        """

        if len(mean.shape) == 1 and mean.shape[0] == self.n:
            mean = mean.unsqueeze(0)
        assert len(mean.shape) == 2
        assert isinstance(lbd, float)

        mean = mean.to(device)

        return (
            torch.distributions.Laplace(torch.zeros(self.n), lbd)
            .rsample(sample_shape=(size,))
            .to(device)
            + mean
        )

    def generalized_normal(self, mean, lbd, p, size, device=None):
        """Sample from a Generalized Normal distribution in R^N.

        Args:
            mean: Value(s) to sample around.
            lbd: Concentration parameter of the distribution.
            p: Exponent of the distribution.
            size: Number of samples to draw.
            device: torch device identifier
        """

        if len(mean.shape) == 1 and mean.shape[0] == self.n:
            mean = mean.unsqueeze(0)
        assert len(mean.shape) == 2
        assert isinstance(lbd, float)

        result = sut.sample_generalized_normal(mean, lbd, p, (size, self.n))

        if device is not None:
            result = result.to(device)

        return result


class NSphereSpace(Space):
    """N-dimensional hypersphere, i.e. {x | |x| = r and x € R^N}."""

    def __init__(self, n, r=1):
        self.n = n
        self._n_sub = n - 1
        self.r = r

    @property
    def dim(self):
        return self.n

    def uniform(self, size, device="cpu"):
        x = torch.randn((size, self.n), device=device)
        x /= torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))

        return x

    def normal(self, mean, std, size, device="cpu"):
        """Sample from a Normal distribution in R^N and then project back on the sphere.

        Args:
            mean: Value(s) to sample around.
            std: Concentration parameter of the distribution (=standard deviation).
            size: Number of samples to draw.
            device: torch device identifier
        """

        assert len(mean.shape) == 1 or (len(mean.shape) == 2 and len(mean) == size)
        assert mean.shape[-1] == self.n

        if len(mean.shape) == 1:
            mean = mean.unsqueeze(0)

        mean = mean.to(device)
        if not torch.is_tensor(std):
            std = torch.ones(self.n) * std
        std = std.to(device)

        assert mean.shape[1] == self.n
        assert torch.allclose(
            torch.sqrt((mean ** 2).sum(-1)), torch.Tensor([self.r]).to(device)
        )

        result = torch.randn((size, self.n), device=device) * std + mean
        # project back on sphere
        result /= torch.sqrt(torch.sum(result ** 2, dim=-1, keepdim=True))

        return result

    def laplace(self, mean, lbd, size, device="cpu"):
        """Sample from a Laplace distribution in R^N and then project back on the sphere.

        Args:
            mean: Value(s) to sample around.
            lbd: Concentration parameter of the distribution.
            size: Number of samples to draw.
            device: torch device identifier
        """

        assert len(mean.shape) == 1 or (len(mean.shape) == 2 and len(mean) == size)
        assert mean.shape[-1] == self.n

        if len(mean.shape) == 1:
            mean = mean.unsqueeze(0)

        mean = mean.to(device)

        assert mean.shape[1] == self.n
        assert torch.allclose(
            torch.sqrt((mean ** 2).sum(-1)), torch.Tensor([self.r]).to(device)
        )

        result = NRealSpace(self.n).laplace(mean, lbd, size, device)
        # project back on sphere
        result /= torch.sqrt(torch.sum(result ** 2, dim=-1, keepdim=True))

        return result

    def generalized_normal(self, mean, lbd, p, size, device="cpu"):
        """Sample from a Generalized Normal distribution in R^N and then project back on the sphere.

        Args:
            mean: Value(s) to sample around.
            lbd: Concentration parameter of the distribution.
            p: Exponent of the distribution.
            size: Number of samples to draw.
            device: torch device identifier
        """

        assert len(mean.shape) == 1 or (len(mean.shape) == 2 and len(mean) == size)
        assert mean.shape[-1] == self.n

        if len(mean.shape) == 1:
            mean = mean.unsqueeze(0)

        mean = mean.to(device)

        assert mean.shape[1] == self.n
        assert torch.allclose(
            torch.sqrt((mean ** 2).sum(-1)), torch.Tensor([self.r]).to(device)
        )

        result = NRealSpace(self.n).generalized_normal(
            mean=mean, lbd=lbd, p=p, size=size, device=device
        )
        # project back on sphere
        result /= torch.sqrt(torch.sum(result ** 2, dim=-1, keepdim=True))

        return result

    def von_mises_fisher(self, mean, kappa, size, device="cpu"):
        """Sample from a von Mises-Fisher distribution (=Normal distribution on a hypersphere).

        Args:
            mean: Value(s) to sample around.
            kappa: Concentration parameter of the distribution.
            size: Number of samples to draw.
            device: torch device identifier
        """

        assert len(mean.shape) == 1 or (len(mean.shape) == 2 and len(mean) == size)
        assert mean.shape[-1] == self.n

        mean = mean.cpu().detach().numpy()

        if len(mean.shape) == 1:
            mean = np.repeat(np.expand_dims(mean, 0), size, axis=0)

        assert mean.shape[1] == self.n
        assert np.allclose(np.sqrt((mean ** 2).sum(-1)), self.r)

        samples_np = vmf.sample_vMF(mean, kappa, size)
        samples = torch.Tensor(samples_np).to(device)

        return samples


class NBoxSpace(Space):
    """Constrained box space in R^N, i.e. {x | a <= x_i <= b and x € R^N} for
    lower and upper limit a, b"""

    def __init__(self, n, min_=-1, max_=1):
        self.n = n
        self.min_ = min_
        self.max_ = max_

    @property
    def dim(self):
        return self.n

    def uniform(self, size, device="cpu"):
        return (
            torch.rand(size=(size, self.n), device=device) * (self.max_ - self.min_)
            + self.min_
        )

    def normal(self, mean, std, size, device="cpu"):
        """Sample from a Normal distribution in R^N and then restrict the samples to a box.

        Args:
            mean: Value(s) to sample around.
            std: Concentration parameter of the distribution (=standard deviation).
            size: Number of samples to draw.
            device: torch device identifier
        """

        assert len(mean.shape) == 1 or (len(mean.shape) == 2 and len(mean) == size)
        assert mean.shape[-1] == self.n

        if len(mean.shape) == 1:
            mean = mean.unsqueeze(0)

        mean = mean.to(device)

        sampler = lambda s: torch.randn((s, self.n), device=device) * std + mean
        values = sut.truncated_rejection_resampling(
            sampler, self.min_, self.max_, size, self.n, device=device
        )

        return values.view((size, self.n))

    def laplace(self, mean, lbd, size, device="cpu"):
        """Sample from a Laplace distribution in R^N and then restrict the samples to a box.

        Args:
            mean: Value(s) to sample around.
            lbd: Concentration parameter of the distribution.
            size: Number of samples to draw.
            device: torch device identifier
        """

        assert len(mean.shape) == 1 or (len(mean.shape) == 2 and len(mean) == size)
        assert mean.shape[-1] == self.n

        if len(mean.shape) == 1:
            mean = mean.unsqueeze(0)

        sampler = lambda s: torch.distributions.Laplace(
            torch.zeros(self.n), lbd
        ).rsample(sample_shape=(s,)).to(device) + mean.to(device)
        values = sut.truncated_rejection_resampling(
            sampler, self.min_, self.max_, size, self.n, device=device
        )

        return values.view((size, self.n))

    def generalized_normal(self, mean, lbd, p, size, device=None):
        """Sample from a Generalized Normal distribution in R^N and then restrict the samples to a box.

        Args:
            mean: Value(s) to sample around.
            lbd: Concentration parameter of the distribution.
            p: Exponent of the distribution.
            size: Number of samples to draw.
            device: torch device identifier
        """

        assert len(mean.shape) == 1 or (len(mean.shape) == 2 and len(mean) == size)
        assert mean.shape[-1] == self.n

        if len(mean.shape) == 1:
            mean = mean.unsqueeze(0)

        sampler = lambda s: sut.sample_generalized_normal(mean, lbd, p, (s, self.n))
        values = sut.truncated_rejection_resampling(
            sampler, self.min_, self.max_, size, self.n, device=device
        )

        return values.view((size, self.n))
