"""Additional layers not included in PyTorch."""

import torch
from torch import nn
import torch.nn.modules.conv as conv
from typing import Optional
from typing_extensions import Literal


class PositionalEncoding(nn.Module):
    """Add a positional encoding as two additional channels to the data."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pos = torch.stack(
            torch.meshgrid(
                torch.arange(x.shape[-2], dtype=torch.float, device=x.device),
                torch.arange(x.shape[-1], dtype=torch.float, device=x.device),
            ),
            0,
        )
        pos /= torch.max(pos) + 1e-12
        pos = torch.repeat_interleave(pos.unsqueeze(0), len(x), 0)

        return torch.cat((pos, x), 1)


class Lambda(nn.Module):
    """Apply a lambda function to the input."""

    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs)


class Flatten(Lambda):
    """Flatten the input data after the batch dimension."""

    def __init__(self):
        super().__init__(lambda x: x.view(len(x), -1))


class RescaleLayer(nn.Module):
    """Normalize the data to a hypersphere with fixed/variable radius."""

    def __init__(
        self, init_r=1.0, fixed_r=False, mode: Optional[Literal["eq", "leq"]] = "eq"
    ):
        super().__init__()
        self.fixed_r = fixed_r
        assert mode in ("leq", "eq")
        self.mode = mode
        if fixed_r:
            self.r = torch.ones(1, requires_grad=False) * init_r
        else:
            self.r = nn.Parameter(torch.ones(1, requires_grad=True) * init_r)

    def forward(self, x):
        if self.mode == "eq":
            x = x / torch.norm(x, dim=-1, keepdim=True)
            x = x * self.r.to(x.device)
        elif self.mode == "leq":
            norm = torch.norm(x, dim=-1, keepdim=True)
            x[norm > self.r] /= torch.norm(x, dim=-1, keepdim=True) / self.r

        return x


class SoftclipLayer(nn.Module):
    """Normalize the data to a hyperrectangle with fixed/learnable size."""

    def __init__(self, n, init_abs_bound=1.0, fixed_abs_bound=True):
        super().__init__()
        self.fixed_abs_bound = fixed_abs_bound
        if fixed_abs_bound:
            self.max_abs_bound = torch.ones(n, requires_grad=False) * init_abs_bound
        else:
            self.max_abs_bound = nn.Parameter(
                torch.ones(n, requires_grad=True) * init_abs_bound
            )

    def forward(self, x):
        x = torch.sigmoid(x)
        x = x * self.max_abs_bound.to(x.device).unsqueeze(0)

        return x
