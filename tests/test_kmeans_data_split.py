from imbalance.split_data import kmeans_data_splitter


additional_params = {"random_state": 52, "n_clusters": 5}


def test_df_kmeans_data_split(imbalance_df):
    output = kmeans_data_splitter(
        data_to_split=imbalance_df, additional_params=additional_params
    )
    sums = 0
    for out in output:
        sums += len(out)

    # check if every data is included in split by checking the length count
    assert len(imbalance_df) == sums
