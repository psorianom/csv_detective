
"""
Here we train a machine learning model to detect the type of column. We need :
1. Load the annotation csv /home/pavel/etalab/code/csv_true_detective/data/columns_annotation.csv
    a. Get the filenames found in this file
2. Start csv_detective routine for each filename
3. Get the
"""
from collections import Counter
# from csv_detective.machine_learning.train_model_cli import RESOURCE_ID_COLUMNS

from csv_detective.detection import detect_encoding, detect_separator, detect_headers, detect_heading_columns, \
    detect_trailing_columns, parse_table, detect_ints_as_floats

import numpy as np
import pandas as pd


def features_cell(value):
    pass


def features_cell_wise(column:pd.Series):




    pass


def features_column_wise(column:pd.Series):
    """Extract features of the column (np.array)

    """

    features = {}

    column = column.dropna()
    if column.empty:
        return {"empty_column": 1}


    # number of chars
    features["length_avg"] = np.mean(column.apply(len))
    features["length_min"] = np.min(column.apply(len))
    features["length_max"] = np.max(column.apply(len))
    features["length_std"] = np.std(column.apply(len))

    # type of chars
    features["chars_num_unique"] = len(Counter(column.to_string(header=False, index=False).replace("\n", "").replace(" ", "")))
    features["chars_avg_num_digits"] = np.mean(column.apply(lambda x: sum(1 for c in x if c.isdigit())))
    features["chars_avg_num_letters"] = np.mean(column.apply(lambda x: sum(1 for c in x if c.isalpha())))
    features["chars_avg_num_lowercase"] = np.mean(column.apply(lambda x: sum(1 for c in x if c.islower())))
    features["chars_avg_num_uppercase"] = np.mean(column.apply(lambda x: sum(1 for c in x if c.isupper())))

    features["values_nunique"] = column.nunique()
    return features


def load_file(file_path, true_labels, num_rows=50):
    with open(file_path, mode='rb') as binary_file:
        encoding = detect_encoding(binary_file)['encoding']

    with open(file_path, 'r', encoding=encoding) as str_file:
        sep = detect_separator(str_file)
        header_row_idx, header = detect_headers(str_file, sep)
        if header is None:
            return_dict = {'error': True}
            return return_dict
        elif isinstance(header, list):
            if any([x is None for x in header]):
                return_dict = {'error': True}
                return return_dict

    table, total_lines = parse_table(
        file_path,
        encoding,
        sep,
        header_row_idx,
        num_rows
    )

    if table.empty:
        print("Could not read {}".format(file_path))
        return

    assert table.shape[1] == len(true_labels), "Annotated number of columns does not match the number of columns in" \
                                               " file {}".format(file_path)
    return table


def extract_text_features(file_path, true_labels, num_rows=50):
    '''Returns a dict with information about the csv table and possible
    column contents
    '''
    resource_df = load_file(file_path, true_labels, num_rows=num_rows)

    if resource_df is None:
        return None

    resource_list = []
    for j in range(len(resource_df.columns)):
        resource_list.append(resource_df.iloc[:, j].dropna().values)

    assert(len(resource_list) == len(true_labels))  # Assert we have the same number of annotated columns and columns
    expanded_labels = []
    expanded_rows = []

    for i, l in enumerate(true_labels):
        expanded_labels.extend([l] * len(resource_list[i]))
        expanded_rows.extend(resource_list[i])

    return expanded_rows, expanded_labels


def extract_aggregated_features(file_path, true_labels, num_rows=50):
    '''Returns a dict with information about the csv table and possible
    column contents
    '''
    import os
    resource_id = os.path.basename(file_path)[:-4]
    resource_df = load_file(file_path, true_labels, num_rows=50)

    features_dict = list(resource_df.apply(lambda column: features_column_wise(column)).to_dict().values())
    return {resource_id: features_dict}


if __name__ == '__main__':
    file_path = "/data/datagouv/csv_top/edf158f9-bdde-4e6e-b92c-c156c9316383.csv"

    features_dict = extract_aggregated_features(file_path)

    pass