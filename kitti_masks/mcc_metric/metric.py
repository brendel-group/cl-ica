"""Mean Correlation Coefficient from Hyvarinen & Morioka
"""
from absl import logging
from disentanglement_lib.evaluation.metrics import utils
import numpy as np
import gin.tf
import scipy as sp
from kitti_masks.mcc_metric.munkres import Munkres


def correlation(x, y, method="Pearson"):
    """Evaluate correlation
    Args:
        x: data to be sorted
        y: target data
    Returns:
        corr_sort: correlation matrix between x and y (after sorting)
        sort_idx: sorting index
        x_sort: x after sorting
        method: correlation method ('Pearson' or 'Spearman')
    """

    print("Calculating correlation...")

    x = x.copy()
    y = y.copy()
    dim = x.shape[0]

    # Calculate correlation -----------------------------------
    if method == "Pearson":
        corr = np.corrcoef(y, x)
        corr = corr[0:dim, dim:]
    elif method == "Spearman":
        corr, pvalue = sp.stats.spearmanr(y.T, x.T)
        corr = corr[0:dim, dim:]

    # Sort ----------------------------------------------------
    munk = Munkres()
    indexes = munk.compute(-np.absolute(corr))

    sort_idx = np.zeros(dim)
    x_sort = np.zeros(x.shape)
    for i in range(dim):
        sort_idx[i] = indexes[i][1]
        x_sort[i, :] = x[indexes[i][1], :]

    # Re-calculate correlation --------------------------------
    if method == "Pearson":
        corr_sort = np.corrcoef(y, x_sort)
        corr_sort = corr_sort[0:dim, dim:]
    elif method == "Spearman":
        corr_sort, pvalue = sp.stats.spearmanr(y.T, x_sort.T)
        corr_sort = corr_sort[0:dim, dim:]

    return corr_sort, sort_idx, x_sort


@gin.configurable(
    "mcc",
    blacklist=[
        "ground_truth_data",
        "representation_function",
        "random_state",
        "artifact_dir",
    ],
)
def compute_mcc(
    ground_truth_data,
    representation_function,
    random_state,
    artifact_dir=None,
    num_train=gin.REQUIRED,
    correlation_fn=gin.REQUIRED,
    batch_size=16,
):
    """Computes the mean correlation coefficient.

    Args:
      ground_truth_data: GroundTruthData to be sampled from.
      representation_function: Function that takes observations as input and
        outputs a dim_representation sized representation for each observation.
      random_state: Numpy random state used for randomness.
      artifact_dir: Optional path to directory where artifacts can be saved.
      num_train: Number of points used for training.
      batch_size: Batch size for sampling.

    Returns:
      Dict with mcc stats
    """
    del artifact_dir
    logging.info("Generating training set.")
    mus_train, ys_train = utils.generate_batch_factor_code(
        ground_truth_data, representation_function, num_train, random_state, batch_size
    )
    assert mus_train.shape[1] == num_train
    return _compute_mcc(mus_train, ys_train, correlation_fn, random_state)


def _compute_mcc(mus_train, ys_train, correlation_fn, random_state):
    """Computes score based on both training and testing codes and factors."""
    score_dict = {}
    result = np.zeros(mus_train.shape)
    result[: ys_train.shape[0], : ys_train.shape[1]] = ys_train

    for i in range(len(mus_train) - len(ys_train)):
        result[ys_train.shape[0] + i, :] = random_state.normal(size=ys_train.shape[1])

    corr_sorted, sort_idx, mu_sorted = correlation(
        mus_train, result, method=correlation_fn
    )
    score_dict["meanabscorr"] = np.mean(np.abs(np.diag(corr_sorted)[: len(ys_train)]))

    for i in range(len(corr_sorted)):
        for j in range(len(corr_sorted[0])):
            score_dict["corr_sorted_{}{}".format(i, j)] = corr_sorted[i][j]

    for i in range(len(sort_idx)):
        score_dict["sort_idx_{}".format(i)] = sort_idx[i]

    return score_dict
