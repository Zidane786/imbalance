from imbalance.split_data import segment_data_splitter


def test_df_segment_data_split(imbalance_df):
    output = segment_data_splitter(data_to_split=imbalance_df, n=5, rest_perc=0.10)
    sums = 0
    for out in output:
        sums += len(out)

    # check if every data is included in split by checking the length count
    assert len(imbalance_df) == sums


def test_list_segment_data_split(imbalance_df):
    records = imbalance_df.to_dict(orient="records")
    output = segment_data_splitter(data_to_split=records, n=5)
    sums = 0
    for out in output:
        sums += len(out)

    # check if every data is included in split by checking the length count
    assert len(records) == sums
