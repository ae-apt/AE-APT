import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RepeatedStratifiedKFold,
)


def get_loss_fl(model, data):
    rec = model.predict(data)
    loss = tf.reduce_sum(tf.math.abs(tf.cast(data, tf.float32) - rec), axis=1).numpy()
    return rec, loss


def discounted_cumulative_gain(ranks):
    dcg = 0.0
    for rank in ranks:
        dcg = dcg + 1.0 / np.log2(rank + 1)
    return dcg


# Calculate max possible DCG and ratio
def normalized_discounted_cumulative_gain(ranks, num_gt):
    dcg = discounted_cumulative_gain(ranks)
    maxdcg = 0.0
    for i in range(1, num_gt + 1):
        maxdcg = maxdcg + 1.0 / np.log2(i + 1)
    return dcg / maxdcg


def list_to_txt(arr, file):
    for i in arr:
        file.write(str(i) + "\n")
    file.close()


def get_reshape_pair(shape):
    divisors = [i for i in range(1, int(shape ** (1 / 2)) + 1) if shape % i == 0]
    return divisors[-1], int(shape / divisors[-1])


def plot_data_points(data, nrows, ncols, filepath, title=""):
    if nrows * ncols > data.shape[0]:
        print("ERR: Size of figures exceeds size of data.")
        return
    if data.shape[1] > 299:
        print("ERR: Data too high dimensional (>299) to visualize effectively.")
        return
    fig_size = (5 * nrows, 4 * ncols)
    # reshape_size = get_reshape_pair(data.shape[1])
    reshape_size = (13, 23)
    rand_indices = random.sample(range(data.shape[0]), nrows * ncols)

    cmap = ["#f5f5f5", "#fb7d74"]
    kwargs = {"linewidth": 0.05, "linecolor": "white", "cmap": cmap}

    f, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=fig_size, tight_layout=True, dpi=150)
    for i in range(nrows):
        for j in range(ncols):
            idx = rand_indices[ncols * i + j]
            x = data[idx].numpy()
            if x.shape[0] < 299:
                padding_size = 299 - x.shape[0]
                pads = np.empty(padding_size)
                pads[:] = np.nan
                x = np.concatenate((x, pads), axis=None)
            x = x.reshape(reshape_size)
            sns.heatmap(x, cbar=False, ax=ax[i][j], **kwargs)
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
    plt.suptitle(title, fontsize=16)
    plt.savefig(f"{filepath}.png")
    plt.close()