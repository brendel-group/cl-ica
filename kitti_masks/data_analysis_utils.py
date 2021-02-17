import csv
from collections import defaultdict
import ast
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import collections
import warnings
from sklearn.feature_selection import mutual_info_regression
import random

# for youtube
name_list = (
    "person giant_panda lizard parrot skateboard sedan ape dog snake monkey hand rabbit duck cat cow "
    "fish train horse turtle bear motorbike giraffe leopard fox deer owl surfboard airplane truck zebra "
    "tiger elephant snowboard boat shark mouse frog eagle earless_seal tennis_racket".split(
        " "
    )
)


def load_csv(csv_file, sequence=2):
    csv_reader = csv.reader(csv_file, delimiter=",")
    next(csv_reader)
    data = defaultdict(list)
    for i, row in enumerate(csv_reader):
        for j in range(2, len(row)):
            considered_columns = row[j : j + sequence]
            if all(considered_columns):
                temp = defaultdict(list)
                for column in considered_columns:
                    val_list = ast.literal_eval(column)
                    for i, val in enumerate(val_list):
                        if val:
                            temp["pos"].append(i)
                            temp["y"].append(val[0])
                            temp["x"].append(val[1])
                            temp["area"].append(val[2])
                for i in range(len(val_list)):
                    if temp["pos"].count(i) == sequence:
                        data["id"].append(int(row[0]))
                        data["category_id"].append(int(row[1]))
                        data["area"].append(
                            [
                                temp["area"][j]
                                for j in range(len(temp["pos"]))
                                if temp["pos"][j] == i
                            ]
                        )
                        data["x"].append(
                            [
                                temp["x"][j]
                                for j in range(len(temp["pos"]))
                                if temp["pos"][j] == i
                            ]
                        )
                        data["y"].append(
                            [
                                temp["y"][j]
                                for j in range(len(temp["pos"]))
                                if temp["pos"][j] == i
                            ]
                        )
                        for k in range(1, sequence):
                            data["area_diff{}".format(k if k > 1 else "")].append(
                                data["area"][-1][k] - data["area"][-1][k - 1]
                            )
                            data["x_diff{}".format(k if k > 1 else "")].append(
                                data["x"][-1][k] - data["x"][-1][k - 1]
                            )
                            data["y_diff{}".format(k if k > 1 else "")].append(
                                data["y"][-1][k] - data["y"][-1][k - 1]
                            )
                    else:
                        assert temp["pos"].count(i) < sequence
    return data


def load_data(path):
    with open(path + ".pkl", "rb") as data:
        data_dict = pickle.load(data)
    return data_dict


def plot_bar(data, key):
    bars = np.bincount(data[key])
    _ = plt.bar(range(len(bars)), bars)


def plot_type(data, x, type_="all", semilogy=False):
    if type_ == "all":
        _ = plt.hist(x, bins=100)
        if semilogy:
            _ = plt.semilogy()
    else:
        fig, axes = plt.subplots(
            len(name_list) // 8,
            len(name_list) // 5,
            figsize=((len(name_list) // 5) * 10, (len(name_list) // 8) * 10),
        )
        data_per_cat = collections.defaultdict(list)
        for j in range(len(data["id"])):
            data_per_cat[data["category_id"][j]].append(x[j])
        for i in range(len(name_list)):
            _ = axes[i % (len(name_list) // 8), i % (len(name_list) // 5)].hist(
                data_per_cat[i + 1], bins=100
            )
            _ = axes[i % (len(name_list) // 8), i % (len(name_list) // 5)].set_title(
                name_list[i]
            )
        plt.show()


def plot_val(data, key, low=0.0, high=1.0, type_="all"):
    x = pd.Series(np.array(data[key])[:, 0])
    print("orig_min", x.min())
    print("orig_max", x.max())
    x = x[x.between(x.quantile(low), x.quantile(high))]
    print("{}_min".format(low), x.min())
    print("{}_max".format(high), x.max())
    plot_type(data, x, type_)


def plot_diff(data, key, semilogy=True, type_="all"):
    plot_type(data, data[key + "_diff"], type_, semilogy)


def visualize_mask(mask):
    _ = plt.imshow(mask)


def generate_dataframe(data, type_="all", mi=False, mi_samples=20000):
    stats = collections.defaultdict(list)
    warnings.filterwarnings(action="ignore")
    distributions = [scipy.stats.gennorm, scipy.stats.norm, scipy.stats.laplace]
    for i in range((0 if type_ == "all" else np.max(data["category_id"])) + 1):
        if i == 0:
            stats["category"].append("all")
            stats["N"].append(len(data["id"]))
            area_val = data["area_diff"]
            x_val = data["x_diff"]
            y_val = data["y_diff"]
        else:
            stats["category"].append(name_list[i - 1])
            stats["N"].append(
                np.count_nonzero(np.array(data["category_id"]).astype(int) == i)
            )
            area_val = [
                data["area_diff"][j]
                for j in range(len(data["area_diff"]))
                if i == data["category_id"][j]
            ]
            x_val = [
                data["x_diff"][j]
                for j in range(len(data["x_diff"]))
                if i == data["category_id"][j]
            ]
            y_val = [
                data["y_diff"][j]
                for j in range(len(data["y_diff"]))
                if i == data["category_id"][j]
            ]
        vals = {"area": area_val, "x": x_val, "y": y_val}
        for x in vals.keys():
            stats["kurtosis_" + x].append("%.2f" % scipy.stats.kurtosis(vals[x]))
            for distribution in distributions:
                params = distribution.fit(vals[x])
                stats["{}_{}".format(distribution.name, x)].append(
                    ["%.2e" % x for x in params]
                )
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                stats["ll_{}_{}".format(distribution.name, x)].append(
                    "%.2e" % distribution.logpdf(vals[x], *params).sum()
                )
                stats["ks_{}_{}".format(distribution.name, x)].append(
                    "%.2e"
                    % scipy.stats.kstest(
                        vals[x],
                        lambda x: distribution.cdf(x, loc=loc, scale=scale, *arg),
                    )[1]
                )
        stats["pearson_area_x"].append(
            ["%.2f" % x for x in scipy.stats.pearsonr(vals["area"], vals["x"])]
        )
        stats["pearson_area_y"].append(
            ["%.2f" % x for x in scipy.stats.pearsonr(vals["area"], vals["y"])]
        )
        stats["pearson_x_y"].append(
            ["%.2f" % x for x in scipy.stats.pearsonr(vals["x"], vals["y"])]
        )
        if mi:
            indices = random.sample(
                range(len(vals["area"])), min(mi_samples, stats["N"][-1])
            )
            stats["mi_area_x"].append(
                "%.2f"
                % mutual_info_regression(
                    np.array(vals["area"]).reshape(-1, 1)[indices],
                    np.array(vals["x"])[indices],
                )
            )
            stats["mi_area_y"].append(
                "%.2f"
                % mutual_info_regression(
                    np.array(vals["area"]).reshape(-1, 1)[indices],
                    np.array(vals["y"])[indices],
                )
            )
            stats["mi_x_y"].append(
                "%.2f"
                % mutual_info_regression(
                    np.array(vals["x"]).reshape(-1, 1)[indices],
                    np.array(vals["y"])[indices],
                )
            )
    return pd.DataFrame.from_dict(stats).sort_values("N", ascending=True)


def find_best(df, criterion="ll"):
    best_df = pd.concat(
        [
            df["category"],
            df["N"],
            df[[*[col for col in df.columns if ("area" in col and criterion in col)]]]
            .astype(np.float64)
            .idxmax(axis=1),
            df[[*[col for col in df.columns if ("x" in col and criterion in col)]]]
            .astype(np.float64)
            .idxmax(axis=1),
            df[[*[col for col in df.columns if ("y" in col and criterion in col)]]]
            .astype(np.float64)
            .idxmax(axis=1),
        ],
        axis=1,
    )
    return best_df.sort_values("N", ascending=False)
