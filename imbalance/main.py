"""
Performing Imbalance POC
"""

import pandas as pd
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imbalance_handler import imbalance_poc_simulation
from logger import _logger
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import math
from utils import (
    generate_model_report,
    generate_predicted_label_cost_report,
    list_of_dict_to_excel,
)
from imblearn.over_sampling import SMOTE


df = pd.read_csv(r"datasets\creditcard.csv")
# C:\Users\ZidaneSunesara\Documents\Development\imbalance\imbalance\datasets\creditcard.csv

ALL_MODEL_SUMMARY = []
# Set Constants
VALID_SIZE = 0.20  # simple validation using train_test_split
TEST_SIZE = 0.20  # test size using_train_test_split
target = "Class"

df["normAmount"] = StandardScaler().fit_transform(df["Amount"].values.reshape(-1, 1))
df = df.drop(["Time"], axis=1)

X = df.loc[:, "V1":"normAmount"].drop(target, axis=1)
Y = df[target]

## train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, random_state=52, test_size=TEST_SIZE, shuffle=True
)
X_test_amount = pd.DataFrame(X_test["Amount"])
X_test.drop(["Amount"], inplace=True, axis=1)

# splitting the training data again in 80-20 ratio
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, random_state=52, test_size=VALID_SIZE, shuffle=True
)
X_train.drop(["Amount"], inplace=True, axis=1)
X_valid_amount = pd.DataFrame(X_valid["Amount"])
X_valid.drop(["Amount"], inplace=True, axis=1)


additional_params = {
    "kmeans": {"random_state": 52},
}
# "prediction_addition_metrics": {
#         "amount_perc": {
#             # "calculate_amount": True,
#             # "amount_col": "Amount",
#             # "label": 1,
#             # "x_test_amount": X_test_amount,
#             # "x_valid_amount": X_valid,
#         }
#     },
model_params = {
    "max_depth": 5,
    "n_estimators": 20,
    "random_state": 52,
}  # Random Forest Classifier
# model_params = {"random_state": 52}
MODEL = RandomForestClassifier

PARAMS = {
    "MODEL": str(MODEL),
    "model_parms": model_params,
    "additional_params": additional_params,
}

_logger.log(f"PARAMS:- \n {PARAMS}")


segment_5_fit_resample_params = {
    "n_majority_split": 5,
    "split_mode": "segment",
}

TRESHOLD_PERC = 0.40
FINAL_PRED_TRESHOLD = int(
    math.floor(segment_5_fit_resample_params["n_majority_split"] * TRESHOLD_PERC)
)


segment_5_test_pred, segment_5_test_summary = imbalance_poc_simulation(
    Model=MODEL,
    model_params=model_params,
    fit_resample_params=segment_5_fit_resample_params,
    final_pred_treshold=FINAL_PRED_TRESHOLD,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    X_test_amount=X_test_amount,
    target=target,
    metrics_verbose=True,
    dataset_repr="test",
)
_logger.log(segment_5_test_summary)
ALL_MODEL_SUMMARY.append(segment_5_test_summary)

segment_5_valid_pred, segment_5_valid_summary = imbalance_poc_simulation(
    Model=MODEL,
    model_params=model_params,
    fit_resample_params=segment_5_fit_resample_params,
    final_pred_treshold=FINAL_PRED_TRESHOLD,
    X_train=X_train,
    y_train=y_train,
    X_test=X_valid,
    y_test=y_valid,
    X_test_amount=X_valid_amount,
    target=target,
    dataset_repr="valid",
    metrics_verbose=True,
)
_logger.log(segment_5_valid_summary)
ALL_MODEL_SUMMARY.append(segment_5_valid_summary)

cluster_5_fit_resample_params = {
    "n_majority_split": 5,
    "split_mode": "cluster",
    "cluster_algo": "kmeans",
    "additional_params": additional_params,
}

TRESHOLD_PERC = 0.60
FINAL_PRED_TRESHOLD = int(
    math.floor(cluster_5_fit_resample_params["n_majority_split"] * TRESHOLD_PERC)
)

cluster_5_test_pred, cluster_5_test_summary = imbalance_poc_simulation(
    Model=MODEL,
    model_params=model_params,
    fit_resample_params=cluster_5_fit_resample_params,
    final_pred_treshold=FINAL_PRED_TRESHOLD,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    X_test_amount=X_test_amount,
    target=target,
    dataset_repr="test",
    metrics_verbose=True,
)
_logger.log(cluster_5_test_summary)
ALL_MODEL_SUMMARY.append(cluster_5_test_summary)


cluster_5_valid_pred, cluster_5_valid_summary = imbalance_poc_simulation(
    Model=MODEL,
    model_params=model_params,
    fit_resample_params=cluster_5_fit_resample_params,
    final_pred_treshold=FINAL_PRED_TRESHOLD,
    X_train=X_train,
    y_train=y_train,
    X_test=X_valid,
    y_test=y_valid,
    X_test_amount=X_valid_amount,
    target=target,
    dataset_repr="valid",
    metrics_verbose=True,
)
_logger.log(cluster_5_valid_summary)
ALL_MODEL_SUMMARY.append(cluster_5_valid_summary)


segment_10_fit_resample_params = {
    "n_majority_split": 10,
    "split_mode": "segment",
}

TRESHOLD_PERC = 0.40
FINAL_PRED_TRESHOLD = int(
    math.floor(segment_10_fit_resample_params["n_majority_split"] * TRESHOLD_PERC)
)

segment_10_test_pred, segment_10_test_summary = imbalance_poc_simulation(
    Model=MODEL,
    model_params=model_params,
    fit_resample_params=segment_10_fit_resample_params,
    final_pred_treshold=FINAL_PRED_TRESHOLD,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    X_test_amount=X_test_amount,
    target=target,
    dataset_repr="test",
    metrics_verbose=True,
)
_logger.log(segment_10_test_summary)
ALL_MODEL_SUMMARY.append(segment_10_test_summary)


segment_10_valid_pred, segment_10_valid_summary = imbalance_poc_simulation(
    Model=MODEL,
    model_params=model_params,
    fit_resample_params=segment_10_fit_resample_params,
    final_pred_treshold=FINAL_PRED_TRESHOLD,
    X_train=X_train,
    y_train=y_train,
    X_test=X_valid,
    y_test=y_valid,
    X_test_amount=X_valid_amount,
    target=target,
    dataset_repr="valid",
    metrics_verbose=True,
)

_logger.log(segment_10_valid_summary)
ALL_MODEL_SUMMARY.append(segment_10_valid_summary)


cluster_10_fit_resample_params = {
    "n_majority_split": 10,
    "split_mode": "cluster",
    "cluster_algo": "kmeans",
    "additional_params": additional_params,
}

TRESHOLD_PERC = 0.60
FINAL_PRED_TRESHOLD = int(
    math.floor(cluster_10_fit_resample_params["n_majority_split"] * TRESHOLD_PERC)
)

cluster_10_test_pred, cluster_10_test_summary = imbalance_poc_simulation(
    Model=MODEL,
    model_params=model_params,
    fit_resample_params=cluster_10_fit_resample_params,
    final_pred_treshold=FINAL_PRED_TRESHOLD,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    X_test_amount=X_test_amount,
    target=target,
    dataset_repr="test",
    metrics_verbose=True,
)
_logger.log(cluster_10_test_summary)
ALL_MODEL_SUMMARY.append(cluster_10_test_summary)

cluster_10_valid_pred, cluster_10_valid_summary = imbalance_poc_simulation(
    Model=MODEL,
    model_params=model_params,
    fit_resample_params=cluster_10_fit_resample_params,
    final_pred_treshold=FINAL_PRED_TRESHOLD,
    X_train=X_train,
    y_train=y_train,
    X_test=X_valid,
    y_test=y_valid,
    X_test_amount=X_valid_amount,
    target=target,
    dataset_repr="valid",
    metrics_verbose=True,
)
_logger.log(cluster_10_valid_summary)
ALL_MODEL_SUMMARY.append(cluster_10_valid_summary)


# normal model prediction
clf = MODEL(**model_params).fit(X_train, y_train)
y_test_pred = clf.predict(X_test)
normal_test_model_summary = generate_model_report(y_test, y_test_pred)
normal_test_amount_summary = generate_predicted_label_cost_report(
    y_actual=y_test,
    y_predicted=y_test_pred,
    amount_col=X_test_amount,
    target=target,
    label=1,
)

normal_test_summary = {
    "Model": MODEL,
    "imbalancer_name": "Normal/Single Model",
    "dataset_repr": "test",
    "model_summary": normal_test_model_summary,
    "amount_summary": normal_test_amount_summary,
}
_logger.log(normal_test_summary)
ALL_MODEL_SUMMARY.append(normal_test_summary)


_logger.log(f"normal test prediction model summary:- \n {normal_test_model_summary}")
_logger.log(f"normal test amount summary :- :- \n {normal_test_amount_summary}")
y_valid_pred = clf.predict(X_valid)
normal_valid_model_summary = generate_model_report(y_valid, y_valid_pred)
normal_valid_amount_summary = generate_predicted_label_cost_report(
    y_actual=y_valid,
    y_predicted=y_valid_pred,
    amount_col=X_valid_amount,
    target=target,
    label=1,
)
_logger.log(f"normal valid prediction model summary:- \n {normal_valid_model_summary}")
_logger.log(f"normal valid amount summary :- :- \n {normal_valid_amount_summary}")
normal_valid_summary = {
    "Model": MODEL,
    "imbalancer_name": "Normal/Single Model",
    "dataset_repr": "valid",
    "model_summary": normal_valid_model_summary,
    "amount_summary": normal_valid_amount_summary,
}
_logger.log(normal_valid_summary)
ALL_MODEL_SUMMARY.append(normal_valid_summary)

# smote model prediction
X_train_sm, y_train_sm = SMOTE(random_state=12).fit_resample(X_train, y_train)
clf_sm = MODEL(**model_params).fit(X_train_sm, y_train_sm)
y_test_pred_sm = clf_sm.predict(X_test)
smote_test_model_summary = generate_model_report(y_test, y_test_pred_sm)
smote_test_amount_summary = generate_predicted_label_cost_report(
    y_actual=y_test,
    y_predicted=y_test_pred_sm,
    amount_col=X_test_amount,
    target=target,
    label=1,
)
_logger.log(f"smote test  prediction model summary:- \n {smote_test_model_summary}")
_logger.log(f"smote test amount summary :- :- \n {smote_test_amount_summary}")

smote_test_summary = {
    "Model": MODEL,
    "imbalancer_name": "Smote",
    "dataset_repr": "test",
    "model_summary": smote_test_model_summary,
    "amount_summary": smote_test_amount_summary,
}
_logger.log(smote_test_summary)
ALL_MODEL_SUMMARY.append(smote_test_summary)

y_valid_pred_sm = clf_sm.predict(X_valid)
smote_valid_model_summary = generate_model_report(y_valid, y_valid_pred_sm)
smote_valid_amount_summary = generate_predicted_label_cost_report(
    y_actual=y_valid,
    y_predicted=y_valid_pred_sm,
    amount_col=X_valid_amount,
    target=target,
    label=1,
)
_logger.log(f"smote valid prediction model summary:- \n {smote_valid_model_summary}")
_logger.log(f"smote valid amount summary :- :- \n {smote_valid_amount_summary}")

smote_valid_summary = {
    "Model": MODEL,
    "imbalancer_name": "Smote",
    "dataset_repr": "valid",
    "model_summary": smote_valid_model_summary,
    "amount_summary": smote_valid_amount_summary,
}

_logger.log(smote_valid_summary)
ALL_MODEL_SUMMARY.append(smote_valid_summary)


list_of_dict_to_excel(
    ALL_MODEL_SUMMARY,
)
