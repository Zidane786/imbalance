from imbalance.imbalance_handler import is_imbalance


def test_is_imbalance(imbalance_df, target_column):
    result = is_imbalance(data=imbalance_df, target=target_column)
    assert result == True


def test_is_balance(balance_df, target_column):
    result = is_imbalance(data=balance_df, target=target_column)
    assert result == False
