import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def prepare_features():
    # read complete dataset
    input_data = utils.read_data_file(utils.INPUT_FILE_LOCATION)

    # boolean to int (one-hot-encoded)
    for b_feat in utils.BOOLEAN_FEATURES:
        input_data[b_feat] = input_data[b_feat].astype(int)

    input_data.to_csv(utils.DATASET_FILE_LOCATION, index=False)
