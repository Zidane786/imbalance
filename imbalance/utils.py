from sklearn.metrics import (
    accuracy_score,
    precision_score,
    confusion_matrix,
    recall_score,
    f1_score,
)
import pandas as pd
import numpy as np
from path import find_closest
import os
from logger import _logger


def _valid_perc(perc: int | float) -> float:
    """check if perc is between 0% to 100%

    Args:
        perc (int | float): perc to check

    Raises:
        ValueError: perc value should be between 0% to 100%

    Returns:
        float: returns valid percentage
    """
    if isinstance(perc, int):
        if perc < 0 or perc > 100:
            raise ValueError(
                f"perc value should be between 0% to 100%. you passed {perc}"
            )
        # converting into float
        perc /= 100

    if isinstance(perc, float):
        if perc < 0 or perc > 1:
            raise ValueError(
                f"perc value should be between 0% to 100%. you passed {perc}"
            )

    return perc


def generate_model_report(y_actual, y_predicted) -> dict:
    """this function is reponsible for calculating matrix of our model
    which can help understand performance of our model.

    Args:
        y_actual (array-like): correct target value
        y_predicted (array-like): predicted target value

    Returns:
        dict: metrics result
    """
    metrics = {
        "Confusion Matrix": confusion_matrix(y_actual, y_predicted),
        "Accuracy": accuracy_score(y_actual, y_predicted),
        "Precision": precision_score(y_actual, y_predicted),
        "Recall": recall_score(y_actual, y_predicted),
        "F1 Score": f1_score(y_actual, y_predicted),
    }

    return metrics


def generate_predicted_label_cost_report(
    *, y_actual, y_predicted, amount_col, target: str, label: str | int
) -> dict:
    """this function generate the cost report where it should how much was actual cost to capture
    and how much is been captured by predicted values. In Amount and Percentage.

    Args:
        y_actual (array-like): correct target value
        y_predicted (array-like): predicted target value
        amount_col (array-like): amount column
        target (str): target/class column name
        label (str | int): label for which to predict amount for

    Returns:
        dict: cost report
    """
    print(type(amount_col), type(y_actual), type(y_predicted))

    if isinstance(y_predicted, np.ndarray):
        y_predicted = pd.Series(y_predicted, name=target)

    label_acutal_sum = (
        pd.concat([amount_col, y_actual], axis=1)
        .groupby(target)
        .sum()
        .loc[label]
        .values[0]
    )

    label_predicted_sum = (
        pd.concat([amount_col, y_predicted], axis=1)
        .groupby(target)
        .sum()
        .loc[label]
        .values[0]
    )

    amount_metrics = {
        "label_to_calculate_amount_for": label,
        "label_actual_amount_sum": label_acutal_sum,
        "label_predicted_amount_sum": label_predicted_sum,
        "amount_predicted_correctly_in_precentage": f"{(label_predicted_sum/label_acutal_sum)*100}%",
    }

    return amount_metrics


def flatten_dict(d, parent_key="", sep="_"):
    """
    Flatten a nested dictionary by concatenating keys of nested dictionaries
    with their parent keys.

    Args:
    - d: The dictionary to flatten.
    - parent_key: The key of the parent dictionary (used for recursion).
    - sep: The separator to use when concatenating keys.

    Returns:
    - A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def list_of_dict_to_excel(
    data: list[dict],
    path: str = None,
    file_name: str = "imbalance_model_metrics.xlsx",
    flattened_nested_dict: bool = True,
    sheet_name: str = "model_metrics",
) -> None:
    """dump list of dictionary to the excel

    Args:
        data (list[dict]): data to dump into excel
        path (str, optional): path/directory of excel. if none  or Wrong Path will consider project root directory path. Defaults to None.
        file_name (str, optional): excel file name. Defaults to "imbalance_model_metrics.xlsx".
        flattened_nested_dict (bool, optional): True for flattened nested dictionary else False. Defaults to True.
        sheet_name (str): sheet name to dump data into .Defaults to model_metrics
    """

    if flattened_nested_dict == True:
        # Flattening the nested dictionary in the list of dictionary
        data = [flatten_dict(d=item) for item in data]

    if path is not None and os.path.exists(path) and os.path.isdir(path):
        full_path = os.path.join(path, file_name)
    else:
        full_path = find_closest(filename=file_name)
    df = pd.DataFrame(data)
    df.to_excel(full_path, sheet_name=sheet_name)
