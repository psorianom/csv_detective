'''Loads an annotated file and extracts features and tagged types for each resource id

Usage:
    train_model.py <i> <p> [options]

Arguments:
    <i>                                An input file or directory (if dir it will convert all txt files inside).
    <p>                                Path where to find the resource's CSVs
    --output FOLDER                    Folder where to store the output structures [default: "."]
    --num_files NFILES                 Number of files (CSVs) to work with [default: 10:int]
    --num_rows NROWS                   Number of rows per file to use [default: 200:int]
    --cores=<n> CORES                  Number of cores to use [default: 2:int]
'''
import glob
# import logging
from itertools import chain

import numpy as np
import pandas as pd
from argopt import argopt
from joblib import Parallel, delayed
from tqdm import tqdm

from csv_detective.detection import detect_encoding, detect_separator, detect_headers, parse_table
from csv_detective.machine_learning.training import train_model2, create_data_matrix, features_cell, explain_parameters, \
    explore_features
from csv_detective.machine_learning import logger

# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler())


def features_wrap(file_path, true_labels, num_rows=10):
    file_labels = true_labels[extract_id(file_path)]
    logger.info("Extracting features from file {}".format(file_path))
    try:
        features = extract_features(file_path, true_labels=file_labels, num_rows=num_rows)
        return features
    except Exception as e:
        print("Could not read {}".format(file_path))
        print(e)
        logger.error(e)


def get_files(data_path, ext="csv"):
    return glob.glob(data_path + "/*.{}".format(ext))


def extract_id(file_path):
    import os
    resource_id = os.path.basename(file_path)[:-4]
    return resource_id


def cols2features(list_files, true_labels, num_rows=10, num_jobs=None):
    if num_jobs and num_jobs > 1:
        features = Parallel(n_jobs=num_jobs)(
            delayed(features_wrap)(file_path, true_labels, num_rows) for file_path in tqdm(list_files))
    else:
        features = [features_wrap(f, true_labels=true_labels, num_rows=num_rows) for f in tqdm(list_files)]

    features = [d for d in features if d]
    features_docs, labels, columns_names, additional_features = zip(*features)

    features_docs = list(chain.from_iterable(features_docs))
    labels = list(chain.from_iterable(labels))
    columns_names = list(chain.from_iterable(columns_names))
    additional_features = list(chain.from_iterable(additional_features))

    return features_docs, labels, columns_names, additional_features


def load_annotations_ids(tagged_file_path, num_files=None):
    df_annotation = pd.read_csv(tagged_file_path)
    # df_annotation = df_annotation.sample(frac=1)
    csv_ids = df_annotation.id.unique()
    dict_ids_labels = {}
    if num_files:
        csv_ids = csv_ids[:num_files]
        df_annotation = df_annotation[df_annotation.id.isin(csv_ids)]

    df_annotation.human_detected = df_annotation.human_detected.fillna("O")
    for i in range(df_annotation.shape[0]):
        dict_ids_labels.setdefault(df_annotation.iloc[i].id, []).append(df_annotation.iloc[i].human_detected)
    y_true = np.array(list(chain.from_iterable(dict_ids_labels.values())))
    num_annotations_per_resource = df_annotation.groupby("id", sort=False).count()["columns"].to_dict()
    assert (all(True if list(dict_ids_labels.keys())[i] == list(num_annotations_per_resource.keys())[i] else False
                for i in range(len(num_annotations_per_resource))))
    return y_true, num_annotations_per_resource, dict_ids_labels


def create_list_files(csv_folder, list_resources_ids):
    csv_paths = ["{0}/{1}.csv".format(csv_folder, resource_id) for resource_id in sorted(list_resources_ids)]
    return csv_paths


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
        num_rows,
        random_state=42
    )

    if table.empty:
        print("Could not read {}".format(file_path))
        return

    assert table.shape[1] == len(true_labels), "Annotated number of columns does not match the number of columns in" \
                                               " file {}".format(file_path)
    return table


def extract_features(file_path, true_labels, num_rows=50):
    '''Returns a dict with information about the csv table and possible
    column contents
    '''
    resource_df = load_file(file_path, true_labels, num_rows=num_rows)

    if resource_df is None:
        return None

    resource_list = []
    expanded_col_names = []
    for j in range(len(resource_df.columns)):
        temp_list = resource_df.iloc[:, j].dropna().to_list()
        resource_list.append(temp_list)
        expanded_col_names.extend([resource_df.columns[j].lower()] * len(temp_list))

    assert len(resource_list) == len(true_labels)  # Assert we have the same number of annotated columns and columns

    expanded_labels = []
    expanded_rows = []

    for i, l in enumerate(true_labels):
        expanded_labels.extend([l] * len(resource_list[i]))
        expanded_rows.extend(resource_list[i])

    additional_features = features_cell(expanded_rows, expanded_labels)

    assert len(expanded_rows) == len(expanded_labels) == len(expanded_col_names) == len(additional_features)

    return expanded_rows, expanded_labels, expanded_col_names, additional_features


if __name__ == '__main__':
    # try_detective(begin_from="DM1_2018_EHPAD")
    parser = argopt(__doc__).parse_args()
    tagged_file_path = parser.i
    csv_folder_path = parser.p
    output_path = parser.output or "."
    num_files = parser.num_files
    num_rows = parser.num_rows

    n_cores = int(parser.cores)
    y_true, resource_ncolumns, dict_ids_labels = load_annotations_ids(tagged_file_path, num_files=num_files)

    csv_path_list = create_list_files(csv_folder_path, list(resource_ncolumns.keys()))

    list_documents, list_labels, list_columns_names, list_additional_features = cols2features(csv_path_list,
                                                                                              true_labels=dict_ids_labels,
                                                                                              num_rows=num_rows,
                                                                                              num_jobs=n_cores)

    X_all, cell_cv, header_cv, extra_dv = create_data_matrix(list_documents, list_columns_names,
                                                             list_additional_features, list_labels)
    clf = train_model2(X_all, list_labels, [cell_cv, extra_dv])
    # explore_features("adresse", list_labels, cell_cv, X_all)
    # explain_parameters(clf=clf, label_id=1, vectorizers=[extra_dv], features_names=list_labels, n_feats=10)
    pass
