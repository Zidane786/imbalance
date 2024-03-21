import pandas as pd
from path import find_closest
from utils import _valid_perc
from split_data import segment_data_splitter, kmeans_data_splitter
from logger import _logger
from exceptions import ObjectCannotBeCreatedException
from utils import (
    generate_model_report,
    generate_predicted_label_cost_report,
)
from typing import Any


def is_imbalance(
    data: pd.DataFrame | list[dict], target: str, treshold: int | float = 0.20
) -> bool:
    """Will check if the classification dataset is imbalance or not
    Note:- it will work for binary classification

    Args:
        data (pd.DataFrame | list[dict]): classification dataset to check if its imbalance or not
        target (str): target column name of classification data set
        treshold (int | float): treshold for minority classes

    Returns:
        bool: return True if data is imbalance or False if not
    """
    try:
        treshold = _valid_perc(treshold)
        # will try to convert non dataframe into a dataframe
        data = pd.DataFrame(data)  # if data is already a df it will return a copy of it

    except:
        TypeError(f"{type(data)} can not be converted into pandas dataframe")

    class_count = data[target].value_counts()
    _logger.log(f"classes:val_count :- \n {class_count}")
    num_classes = len(class_count)
    if num_classes < 2:
        Warning("only 1 class present in dataset please check if passed correct data")
        return False  # If there is only one class, it's not imbalanced

    # Calculate the ratio between the number of samples in the majority class
    # and the number of samples in each minority class
    majority_class_count = class_count.max()
    minority_class_count = class_count.min()
    imbalance_perc = (minority_class_count / majority_class_count) * 100
    _logger.log(f"imbalance percentage:-{imbalance_perc}")
    _logger.log(f"is data imbalance:- {imbalance_perc < treshold}")
    return imbalance_perc < treshold


class ImbalanceHandler:

    def __init__(
        self,
        *,
        data: pd.DataFrame,
        target: str,
        bypass_imbalance_check: bool = False,
        is_imbalance_treshold: int | float = 0.2,
    ) -> None:
        self.data = data
        _logger.log(f"total dataset length :- {len(data)}")
        if not (
            (not bypass_imbalance_check)
            and is_imbalance(data, target, is_imbalance_treshold)
        ):
            ObjectCannotBeCreatedException(
                f"{self.__class__.__name__} Object cannot be created since imbalance check got failed(i.e data is balanced)."
            )
        self.target = target
        class_count = data[target].value_counts()
        self.majority_class = class_count.idxmax()
        self.minority_class = class_count.idxmin()

        _logger.log(
            f"majority_class:-{self.majority_class}, \n minority_class:- {self.minority_class}"
        )

        self.majority_data = data[data[target] == self.majority_class].copy()
        self.minority_data = data[data[target] == self.minority_class].copy()

        _logger.log(
            f"majority_data:-{len(self.majority_data)}, \n minority_data:- {len(self.minority_data)}"
        )

    def fit_resample(
        self,
        *,
        n_majority_split: int = 5,
        split_mode: str = "segment",
        cluster_algo: str = "kmeans",
        additional_params: dict = {},
        rest_perc: int | float = 0.1,
    ):
        results = list()
        if split_mode == "segment":
            self.imbalancer_name = f"{split_mode.capitalize()}({n_majority_split})"
            majority_split_data = segment_data_splitter(
                data_to_split=self.majority_data,
                n=n_majority_split,
                rest_perc=rest_perc,
            )
        elif split_mode == "cluster" and cluster_algo == "kmeans":
            self.imbalancer_name = f"{cluster_algo.capitalize()}({n_majority_split})"
            kmeans_params = additional_params.get("kmeans", {})
            kmeans_params["n_clusters"] = n_majority_split
            if kmeans_params:
                majority_split_data = kmeans_data_splitter(
                    data_to_split=self.majority_data, additional_params=kmeans_params
                )

        for i, majority_sample in enumerate(majority_split_data):
            results.append(pd.concat([majority_sample, self.minority_data]))
            _logger.log(f"split_{i} length is :- {len(results[i])}")
            _logger.log(
                f"split_{i} value counts:- \n{results[i][self.target].value_counts()}"
            )

        return results

    def final_prediction(self, y_df: pd.DataFrame, pred_columns: list, treshold: int):
        def _final_pred(x):
            if x > treshold:
                return 1
            else:
                return 0

        y_df["model_sum_pred"] = y_df.loc[:, pred_columns].apply(
            lambda x: sum(x), axis=1
        )
        y_df["final_pred"] = y_df["model_sum_pred"].apply(lambda x: _final_pred(x))

        return y_df


def imbalance_poc_simulation(
    *,
    Model,
    model_params: dict,
    fit_resample_params: dict,
    final_pred_treshold: int,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    X_test_amount: pd.DataFrame,
    target: str,
    dataset_repr: str = "test",
    metrics_verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]] | pd.DataFrame:
    """This function will run the full simulation for our Imbalance POC and also calculate the matrix

    Args:
        Model (ML Model): ML Model to fit on
        model_params (dict): model parameter for ML Model
        fit_resample_params (dict): imbalance_handler fit resample params
        final_pred_treshold (int): final prediction treshold
        X_train (pd.DataFrame): X_train dataset
        y_train (pd.DataFrame): y_train dataset
        X_test (pd.DataFrame): X_test dataset
        y_test (pd.DataFrame): y_test dataset
        X_test_amount (pd.DataFrame): X_test amount dataset
        target (str): Target column name which we are predicting
        dataset_repr (str, optional): dataset representation i.e what data we are predicting on. Defaults to "test".
        metrics_verbose (bool, optional): Return metrics . Defaults to True.

    Returns:
        tuple[pd.DataFrame, dict[str, Any]] | pd.DataFrame: return final_prediction df or metrics
    """
    test_all_pred_columns = []
    x_columns = X_train.columns
    train_dataset = pd.concat([X_train, y_train], axis=1)
    imb_handler = ImbalanceHandler(data=train_dataset, target=target)
    split_data: list = imb_handler.fit_resample(**fit_resample_params)
    y_test_actual = {"y_test_actual": y_test.values}
    test_n_model_predict_df = pd.DataFrame(y_test_actual)

    for indx, data in enumerate(split_data):
        test_pred_column = f"model_{indx}_test_pred"
        test_all_pred_columns.append(test_pred_column)
        # since we already split train and test data in start data lying in split_data is trained data
        X_split_train = data.loc[:, x_columns]
        y_split_train = data[target]
        clf = Model(**model_params).fit(X_split_train, y_split_train)
        test_pred = clf.predict(X_test)
        test_pred_df = pd.DataFrame({test_pred_column: test_pred})

        test_n_model_predict_df = pd.concat(
            [test_n_model_predict_df, test_pred_df], axis=1
        )

        _logger.log(
            f"{test_pred_column} model summary :- \n {generate_model_report(y_test, test_pred)}"
        )

    final_test_pred = imb_handler.final_prediction(
        test_n_model_predict_df, test_all_pred_columns, final_pred_treshold
    )

    final_model_summary = generate_model_report(y_test, final_test_pred.final_pred)
    final_amount_summary = generate_predicted_label_cost_report(
        y_actual=y_test,
        y_predicted=final_test_pred.final_pred.to_numpy(),
        amount_col=X_test_amount,
        target=target,
        label=1,
    )
    _logger.log(
        f"final test prediction  model summary for imbalance handler:- \n {final_model_summary}"
    )
    _logger.log(
        f"final test amount summary for inbalance handler:- :- \n {final_amount_summary}"
    )

    final_summary = {
        "Model": Model,
        "imbalancer_name": imb_handler.imbalancer_name,
        "dataset_repr": dataset_repr,
        "model_summary": final_model_summary,
        "amount_summary": final_amount_summary,
        "final_pred_treshold": final_pred_treshold,
    }

    if metrics_verbose == True:
        return final_test_pred, final_summary

    return final_test_pred


if __name__ == "__main__":
    """For Debugging"""
    df = pd.read_csv(r"datasets\creditcard.csv")
    target = "Class"
    imb_handler = ImbalanceHandler(
        data=df,
        target=target,
    )
    result = imb_handler.fit_resample(n_majority_split=100)

    for i, item in enumerate(result):
        _logger.log(f"split_{i} value counts:- \n{item[target].value_counts()}")
