import pytest
import pandas as pd


@pytest.fixture(scope="session")
def imbalance_df():
    df = pd.read_csv(r".\tests\data\creditcard.csv")

    return df


@pytest.fixture(scope="session")
def balance_df(target_column):
    df = pd.read_csv(r".\tests\data\creditcard.csv")
    classes = df[target_column].value_counts()
    # making majority count same as minority
    new_majority_df = df[df[target_column] == classes.idxmax()][: classes.min()]
    minority_df = df[df[target_column] == classes.idxmin()]
    balanced_df = pd.concat([new_majority_df, minority_df])

    return balanced_df


@pytest.fixture(scope="session")
def target_column():
    return "Class"
