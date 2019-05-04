'''Loads an annotated file and extracts features and tagged types for each resource id

Usage:
    train_model.py <i> <p> [options]

Arguments:
    <i>                                An input file or directory (if dir it will convert all txt files inside).
    <p>                                Path where to find the resource's CSVs
    --output FOLDER                    Folder where to store the output structures [default:"."]
    --cores=<n> CORES                  Number of cores to use [default: 2]
'''
import glob
from collections import defaultdict

import pandas as pd
from argopt import argopt
from joblib import Parallel, delayed
from functools import partial

from sklearn.feature_extraction import DictVectorizer
from tqdm import tqdm
import logging
from itertools import chain
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


from csv_detective.machine_learning.training import train_routine

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def run_csv_detective(file_path):
    logger.info("Extracting features from file {}".format(file_path))
    try:
        feat_dict = train_routine(file_path, num_rows=500)
        return feat_dict
    except Exception as e:
        print("Could not read {}".format(file_path))
        print(e)
        logger.error(e)



def get_files(data_path, ext="csv"):
    return glob.glob(data_path + "/*.{}".format(ext))


def get_csv_detective_analysis_single(list_files, begin_from=None, n_datasets=None):
    if n_datasets:
        list_files = list_files[:n_datasets]

    if begin_from:
        indx_begin = [i for i, path in enumerate(list_files) if begin_from in path]
        if indx_begin:
            list_files = list_files[indx_begin[0]:]



    list_dict_result = []
    for f in tqdm(list_files):
        output_csv_detective = run_csv_detective(f)
        list_dict_result.append(output_csv_detective)

    list_dict_result = [d for d in list_dict_result if d]
    return list_dict_result


def get_csv_detective_analysis(list_files, begin_from=None, n_datasets=None, n_jobs=None):
    if n_datasets:
        list_files = list_files[:n_datasets]

    if begin_from:
        indx_begin = [i for i, path in enumerate(list_files) if begin_from in path]
        if indx_begin:
            list_files = list_files[indx_begin[0]:]

    if n_jobs and n_jobs > 1:
        run_csv_detective_p = partial(run_csv_detective)
        list_dict_result = Parallel(n_jobs=n_jobs)(
            delayed(run_csv_detective_p)(file_path) for file_path in tqdm(list_files))
    else:
        list_dict_result = [run_csv_detective(f) for f in tqdm(list_files)]


    list_dict_result = [d for d in list_dict_result if d]
    return list_dict_result


def load_annotations_ids(tagged_file_path, num_files=None):
    df_annotation = pd.read_csv(tagged_file_path)
    csv_ids = df_annotation.id.unique()
    dict_ids_tags = defaultdict(list)
    if num_files:
        csv_ids = csv_ids[:num_files]
        df_annotation = df_annotation[df_annotation.id.isin(csv_ids)]
    df_annotation.human_detected = df_annotation.human_detected.fillna("O")
    for i, row in df_annotation.iterrows():
        dict_ids_tags[row.id].append(row.human_detected)

    y_true = np.array(list(chain.from_iterable(dict_ids_tags.values())))

    return y_true, csv_ids, df_annotation.groupby("id").count()["columns"].to_dict()


def train_model(list_features_dict, y_true):
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(list_features_dict)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

    indices = list(sss.split(X, y_true))
    train_indices, test_indices = indices[0][0], indices[0][1]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y_true[train_indices], y_true[test_indices]

    clf = SVC(kernel="rbf")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_true=y_test, y_pred=y_pred))



def create_list_files(csv_folder, list_resources_ids):
    csv_paths = ["{0}/{1}.csv".format(csv_folder, resource_id) for resource_id in sorted(list_resources_ids)]
    return csv_paths

RESOURCE_ID_COLUMNS = None

if __name__ == '__main__':
    # try_detective(begin_from="DM1_2018_EHPAD")
    parser = argopt(__doc__).parse_args()
    tagged_file_path = parser.i
    csv_folder_path = parser.p
    output_path = parser.output or "."
    n_cores = int(parser.cores)
    y_true, csv_ids, RESOURCE_ID_COLUMNS = load_annotations_ids(tagged_file_path, num_files=False)

    csv_path_list = create_list_files(csv_folder_path, csv_ids)

    list_features_dict = get_csv_detective_analysis(csv_path_list, begin_from=None, n_datasets=None, n_jobs=n_cores)

    # Transform to dict to keep better order
    features_dict =  {}
    for dico in list_features_dict:
        key = list(dico.keys())[0]
        features_dict[key] = dico[key]

    # assert len(list_features_dict) == len(y_true)
    not_same_n_columns = {}
    print(len(RESOURCE_ID_COLUMNS))
    print(len(list_features_dict))
    for k, v in RESOURCE_ID_COLUMNS.items():
        if len(features_dict[k]) > v:
            not_same_n_columns[k] = (len(features_dict[k]), v)
            features_dict[k] = features_dict[k][: v]
    print(not_same_n_columns)

    list_features_dict = list(chain.from_iterable(features_dict.values()))

    # HORRIBLE HACK! Adding a new siren instance bc there is only one in the dataset
    id_siren = np.where(y_true == "siren")[0][0]
    print(id_siren)
    list_features_dict.append(list_features_dict[id_siren])
    y_true = y_true.tolist()
    y_true.append("siren")
    y_true = np.array(y_true)

    not_bools = np.where(y_true != "booleen")[0]
    y_true[not_bools] = "O"


    train_model(list_features_dict, y_true)