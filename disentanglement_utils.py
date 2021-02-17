"""Disentanglement evaluation scores such as R2 and MCC."""

from sklearn import metrics
from sklearn import linear_model
import torch
import numpy as np
import scipy as sp
from munkres import Munkres
from typing import Union
from typing_extensions import Literal

__Mode = Union[
    Literal["r2"], Literal["adjusted_r2"], Literal["pearson"], Literal["spearman"]
]


def _disentanglement(z, hz, mode: __Mode = "r2", reorder=None):
    """Measure how well hz reconstructs z measured either by the Coefficient of Determination or the
    Pearson/Spearman correlation coefficient."""

    assert mode in ("r2", "adjusted_r2", "pearson", "spearman")

    if mode == "r2":
        return metrics.r2_score(z, hz), None
    elif mode == "adjusted_r2":
        r2 = metrics.r2_score(z, hz)
        # number of data samples
        n = z.shape[0]
        # number of predictors, i.e. features
        p = z.shape[1]
        adjusted_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
        return adjusted_r2, None
    elif mode in ("spearman", "pearson"):
        dim = z.shape[-1]

        if mode == "spearman":
            raw_corr, pvalue = sp.stats.spearmanr(z, hz)
        else:
            raw_corr = np.corrcoef(z.T, hz.T)
        corr = raw_corr[:dim, dim:]

        if reorder:
            # effectively computes MCC
            munk = Munkres()
            indexes = munk.compute(-np.absolute(corr))

            sort_idx = np.zeros(dim)
            hz_sort = np.zeros(z.shape)
            for i in range(dim):
                sort_idx[i] = indexes[i][1]
                hz_sort[:, i] = hz[:, indexes[i][1]]

            if mode == "spearman":
                raw_corr, pvalue = sp.stats.spearmanr(z, hz_sort)
            else:
                raw_corr = np.corrcoef(z.T, hz_sort.T)

            corr = raw_corr[:dim, dim:]

        return np.diag(np.abs(corr)).mean(), corr


def linear_disentanglement(z, hz, mode: __Mode = "r2", train_test_split=False):
    """Calculate disentanglement up to linear transformations.

    Args:
        z: Ground-truth latents.
        hz: Reconstructed latents.
        mode: Can be r2, pearson, spearman
        train_test_split: Use first half to train linear model, second half to test.
            Is only relevant if there are less samples then latent dimensions.
    """

    if torch.is_tensor(hz):
        hz = hz.detach().cpu().numpy()
    if torch.is_tensor(z):
        z = z.detach().cpu().numpy()

    assert isinstance(z, np.ndarray), "Either pass a torch tensor or numpy array as z"
    assert isinstance(hz, np.ndarray), "Either pass a torch tensor or numpy array as hz"

    # split z, hz to get train and test set for linear model
    if train_test_split:
        n_train = len(z) // 2
        z_1 = z[:n_train]
        hz_1 = hz[:n_train]
        z_2 = z[n_train:]
        hz_2 = hz[n_train:]
    else:
        z_1 = z
        hz_1 = hz
        z_2 = z
        hz_2 = hz

    model = linear_model.LinearRegression()
    model.fit(hz_1, z_1)

    hz_2 = model.predict(hz_2)

    inner_result = _disentanglement(z_2, hz_2, mode=mode, reorder=False)

    return inner_result, (z_2, hz_2)


def permutation_disentanglement(
    z,
    hz,
    mode="r2",
    rescaling=True,
    solver: Union[Literal["naive", "munkres"]] = "naive",
    sign_flips=True,
    cache_permutations=None,
):
    """Measure disentanglement up to permutations by either using the Munkres solver
    or naively trying out every possible permutation.
    Args:
        z: Ground-truth latents.
        hz: Reconstructed latents.
        mode: Can be r2, pearson, spearman
        rescaling: Rescale every individual latent to maximize the agreement
            with the ground-truth.
        solver: How to find best possible permutation. Either use Munkres algorithm
            or naively test every possible permutation.
        sign_flips: Only relevant for `naive` solver. Also include sign-flips in
            set of possible permutations to test.
        cache_permutations: Only relevant for `naive` solver. Cache permutation matrices
            to allow faster access if called multiple times.
    """

    assert solver in ("naive", "munkres")
    if mode == "r2" or mode == "adjusted_r2":
        assert solver == "naive", "R2 coefficient is only supported with naive solver"

    if cache_permutations and not hasattr(
        permutation_disentanglement, "permutation_matrices"
    ):
        permutation_disentanglement.permutation_matrices = dict()

    if torch.is_tensor(hz):
        hz = hz.detach().cpu().numpy()
    if torch.is_tensor(z):
        z = z.detach().cpu().numpy()

    assert isinstance(z, np.ndarray), "Either pass a torch tensor or numpy array as z"
    assert isinstance(hz, np.ndarray), "Either pass a torch tensor or numpy array as hz"

    def test_transformation(T, reorder):
        # measure the r2 score for one transformation

        Thz = hz @ T
        if rescaling:
            assert z.shape == hz.shape
            # find beta_j that solve Y_ij = X_ij beta_j
            Y = z
            X = hz

            beta = np.diag((Y * X).sum(0) / (X ** 2).sum(0))

            Thz = X @ beta

        return _disentanglement(z, Thz, mode=mode, reorder=reorder), Thz

    def gen_permutations(n):
        # generate all possible permutations w/ or w/o sign flips

        def gen_permutation_single_row(basis, row, sign_flips=False):
            # generate all possible permutations w/ or w/o sign flips for one row
            # assuming the previous rows are already fixed
            basis = basis.clone()
            basis[row] = 0
            for i in range(basis.shape[-1]):
                # skip possible columns if there is already an entry in one of
                # the previous rows
                if torch.sum(torch.abs(basis[:row, i])) > 0:
                    continue
                signs = [1]
                if sign_flips:
                    signs += [-1]

                for sign in signs:
                    T = basis.clone()
                    T[row, i] = sign

                    yield T

        def gen_permutations_all_rows(basis, current_row=0, sign_flips=False):
            # get all possible permutations for all rows

            for T in gen_permutation_single_row(basis, current_row, sign_flips):
                if current_row == len(basis) - 1:
                    yield T.numpy()
                else:
                    # generate all possible permutations of all other rows
                    yield from gen_permutations_all_rows(T, current_row + 1, sign_flips)

        basis = torch.zeros((n, n))

        yield from gen_permutations_all_rows(basis, sign_flips=sign_flips)

    n = z.shape[-1]
    # use cache to speed up repeated calls to the function
    if cache_permutations and not solver == "munkres":
        key = (rescaling, n)
        if not key in permutation_disentanglement.permutation_matrices:
            permutation_disentanglement.permutation_matrices[key] = list(
                gen_permutations(n)
            )
        permutations = permutation_disentanglement.permutation_matrices[key]
    else:
        if solver == "naive":
            permutations = list(gen_permutations(n))
        elif solver == "munkres":
            permutations = [np.eye(n, dtype=z.dtype)]

    scores = []

    # go through all possible permutations and check r2 score
    for T in permutations:
        scores.append(test_transformation(T, solver == "munkres"))

    return max(scores, key=lambda x: x[0][0])
