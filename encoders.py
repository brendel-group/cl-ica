from torch import nn
import layers as ls
from typing import List, Union
from typing_extensions import Literal


__all__ = ["get_mlp"]


def get_mlp(
    n_in: int,
    n_out: int,
    layers: List[int],
    layer_normalization: Union[None, Literal["bn"], Literal["gn"]] = None,
    output_normalization: Union[
        None,
        Literal["fixed_sphere"],
        Literal["learnable_sphere"],
        Literal["fixed_box"],
        Literal["learnable_box"],
    ] = None,
    output_normalization_kwargs=None,
):
    """
    Creates an MLP.

    Args:
        n_in: Dimensionality of the input data
        n_out: Dimensionality of the output data
        layers: Number of neurons for each hidden layer
        layer_normalization: Normalization for each hidden layer.
            Possible values: bn (batch norm), gn (group norm), None
        output_normalization: (Optional) Normalization applied to output of network.
        output_normalization_kwargs: Arguments passed to the output normalization, e.g., the radius for the sphere.
    """
    modules: List[nn.Module] = []

    def add_module(n_layer_in: int, n_layer_out: int, last_layer: bool = False):
        modules.append(nn.Linear(n_layer_in, n_layer_out))
        # perform normalization & activation not in last layer
        if not last_layer:
            if layer_normalization == "bn":
                modules.append(nn.BatchNorm1d(n_layer_out))
            elif layer_normalization == "gn":
                modules.append(nn.GroupNorm(1, n_layer_out))
            modules.append(nn.LeakyReLU())

        return n_layer_out

    if len(layers) > 0:
        n_out_last_layer = n_in
    else:
        assert n_in == n_out, "Network with no layers must have matching n_in and n_out"
        modules.append(layers.Lambda(lambda x: x))

    layers.append(n_out)

    for i, l in enumerate(layers):
        n_out_last_layer = add_module(n_out_last_layer, l, i == len(layers) - 1)

    if output_normalization_kwargs is None:
        output_normalization_kwargs = {}

    if output_normalization == "fixed_sphere":
        modules.append(ls.RescaleLayer(fixed_r=True, **output_normalization_kwargs))
    elif output_normalization == "learnable_sphere":
        modules.append(ls.RescaleLayer(init_r=1.0, fixed_r=False))
    elif output_normalization == "fixed_box":
        modules.append(
            ls.SoftclipLayer(
                n=n_out, fixed_abs_bound=True, **output_normalization_kwargs
            )
        )
    elif output_normalization == "learnable_box":
        modules.append(
            ls.SoftclipLayer(
                n=n_out, fixed_abs_bound=False, **output_normalization_kwargs
            )
        )
    elif output_normalization is None:
        pass
    else:
        raise ValueError("output_normalization")

    return nn.Sequential(*modules)


def get_flow(
    n_in: int,
    n_out: int,
    init_identity: bool = False,
    coupling_block: Union[Literal["gin", "glow"]] = "gin",
    num_nodes: int = 8,
    node_size_factor: int = 1,
):
    """
    Creates an flow-based network.

    Args:
        n_in: Dimensionality of the input data
        n_out: Dimensionality of the output data
        init_identity: Initialize weights to identity network.
        coupling_block: Coupling method to use to combine nodes.
        num_nodes: Depth of the flow network.
        node_size_factor: Multiplier for the hidden units per node.
    """

    # do lazy imports here such that the package is only
    # required if one wants to use the flow mixing
    import FrEIA.framework as Ff
    import FrEIA.modules as Fm

    def _invertible_subnet_fc(c_in, c_out, init_identity):
        subnet = nn.Sequential(
            nn.Linear(c_in, c_in * node_size),
            nn.ReLU(),
            nn.Linear(c_in * node_size, c_in * node_size),
            nn.ReLU(),
            nn.Linear(c_in * node_size, c_out),
        )
        if init_identity:
            subnet[-1].weight.data.fill_(0.0)
            subnet[-1].bias.data.fill_(0.0)
        return subnet

    assert n_in == n_out

    if coupling_block == "gin":
        block = Fm.GINCouplingBlock
    else:
        assert coupling_block == "glow"
        block = Fm.GLOWCouplingBlock

    nodes = [Ff.InputNode(n_in, name="input")]

    for k in range(num_nodes):
        nodes.append(
            Ff.Node(
                nodes[-1],
                block,
                {
                    "subnet_constructor": lambda c_in, c_out: _invertible_subnet_fc(
                        c_in, c_out, init_identity
                    ),
                    "clamp": 2.0,
                },
                name=f"coupling_{k}",
            )
        )

    nodes.append(Ff.OutputNode(nodes[-1], name="output"))
    return Ff.ReversibleGraphNet(nodes, verbose=False)
