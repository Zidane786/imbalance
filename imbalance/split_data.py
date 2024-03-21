import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from math import ceil
from utils import _valid_perc
from logger import _logger


def segment_data_splitter(
    *, data_to_split: pd.DataFrame | list[dict], n: int, rest_perc: float | int = 0.10
) -> list[pd.DataFrame] | list[list[dict]]:
    """Responsible for splitting the data in seemingly n equal segments

    Args:
        data_to_split (pd.DataFrame | list): data to split
        n (int): to split in seemingly n equal segments
        rest_perc (float | int): use to check if last batch is less then rest_perc. Defaults to 0.10.

    Returns:
        list[pd.DataFrame] | list[list[dict]]: list of splitted data
    """

    rest_perc = _valid_perc(rest_perc)

    data_list = []
    if isinstance(data_to_split, (pd.DataFrame, list)):
        length = len(data_to_split)
        start = 0
        batch = length // n
        while start < length:
            end = min(start + batch, length)
            # if less than rest_perc data remains then let end=length
            if len(data_to_split[end:]) <= ceil(batch * rest_perc):
                end = length
            data_list.append(data_to_split[start:end])
            start = end

    return data_list


def _split_data_on_label(*, data, labels):
    """Split the dataset based on label

    Args:
        data (array-like): data to split
        labels (array-like): labels to split data on

    Returns:
        array-like: split data
    """
    split_data = list()
    data["label"] = labels
    for label in np.unique(labels):
        cluster_data = data[data["label"] == label]
        cluster_data.drop("label", inplace=True, axis=1)
        split_data.append(cluster_data)

    return split_data


def kmeans_data_splitter(
    *, data_to_split: pd.DataFrame | list[dict], additional_params: dict = {}
) -> list[pd.DataFrame] | list[list[dict]]:
    """Responsible for splitting the data_to_split in n cluster using k-mean cluster

    Args:
        data_to_split (pd.DataFrame | list): data_to_split to split

    Returns:
        list[pd.DataFrame] | list[list[dict]]: list of splitted data_to_split
    """
    _logger.log(f"Kmeans model params:- {additional_params}")

    labels = KMeans(**additional_params).fit(data_to_split).labels_
    labels = labels.astype(int)

    return _split_data_on_label(data=data_to_split, labels=labels)
