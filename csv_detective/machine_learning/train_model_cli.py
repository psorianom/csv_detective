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
import logging
from itertools import chain

import numpy as np
import pandas as pd
from argopt import argopt
from joblib import Parallel, delayed
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from tqdm import tqdm

from csv_detective.machine_learning.training import extract_text_features

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def extract_features(file_path, true_labels):
    file_labels = true_labels[extract_id(file_path)]
    logger.info("Extracting features from file {}".format(file_path))
    try:
        features = extract_text_features(file_path, true_labels=file_labels, num_rows=100)
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


def get_features(list_files, true_labels, begin_from=None, n_datasets=None, n_jobs=None):
    if n_datasets:
        list_files = list_files[:n_datasets]

    if begin_from:
        indx_begin = [i for i, path in enumerate(list_files) if begin_from in path]
        if indx_begin:
            list_files = list_files[indx_begin[0]:]

    if n_jobs and n_jobs > 1:
        features = Parallel(n_jobs=n_jobs)(
            delayed(extract_features)(file_path, true_labels) for file_path in tqdm(list_files))
    else:
        features = [extract_features(f, true_labels=true_labels) for f in tqdm(list_files)]

    features = [d for d in features if d]
    return features


def cells2docs(list_files, true_labels, begin_from=None, n_datasets=None, n_jobs=None):
    if n_datasets:
        list_files = list_files[:n_datasets]

    if begin_from:
        indx_begin = [i for i, path in enumerate(list_files) if begin_from in path]
        if indx_begin:
            list_files = list_files[indx_begin[0]:]

    if n_jobs and n_jobs > 1:
        documents = Parallel(n_jobs=n_jobs)(
            delayed(extract_features)(file_path, true_labels) for file_path in tqdm(list_files))
    else:
        documents = [extract_features(f, true_labels=true_labels) for f in tqdm(list_files)]

    documents = [d for d in documents if d]
    documents, labels = zip(*documents)

    documents = chain.from_iterable(documents)
    labels = chain.from_iterable(labels)

    return documents, labels


def load_annotations_ids(tagged_file_path, num_files=None):
    df_annotation = pd.read_csv(tagged_file_path)
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


def create_data_matrix(documents):
    vect = CountVectorizer(ngram_range=(1, 3), analyzer="char", max_df=0.8, min_df=2)
    X = vect.fit_transform(documents)
    return X


def train_model2(X, y_true):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    y_true = np.array(list(y_true))
    indices = list(sss.split(X, y_true))
    train_indices, test_indices = indices[0][0], indices[0][1]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y_true[train_indices], y_true[test_indices]

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_true=y_test, y_pred=y_pred))
    show_confusion_matrix(y_true=y_test, y_pred=y_pred, labels=np.unique(y_true))
    pass


def show_confusion_matrix(y_true, y_pred, labels):
    import seaborn as sns
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=np.unique(y_true))

    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt="g", cmap='Greens')  # annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labels[::-1], rotation=90)
    ax.yaxis.set_ticklabels(labels, rotation=0)

    plt.show()

if __name__ == '__main__':
    # try_detective(begin_from="DM1_2018_EHPAD")
    parser = argopt(__doc__).parse_args()
    tagged_file_path = parser.i
    csv_folder_path = parser.p
    output_path = parser.output or "."
    n_cores = int(parser.cores)
    y_true, resource_ncolumns, dict_ids_labels = load_annotations_ids(tagged_file_path, num_files=10)

    csv_path_list = create_list_files(csv_folder_path, list(resource_ncolumns.keys()))

    list_documents, list_labels = cells2docs(csv_path_list, true_labels=dict_ids_labels,
                                             begin_from=None, n_datasets=None, n_jobs=n_cores)

    X = create_data_matrix(list_documents)
    train_model2(X, list_labels)
    pass

    # # Transform to dict to keep better order
    # features_dict = {}
    # for dico in list_documents:
    #     key = list(dico.keys())[0]
    #     features_dict[key] = dico[key]
    #
    # # assert len(list_features_dict) == len(y_true)
    # not_same_n_columns = {}
    # print(len(RESOURCE_ID_COLUMNS))
    # print(len(list_documents))
    # for k, v in RESOURCE_ID_COLUMNS.items():
    #     if len(features_dict[k]) > v:
    #         not_same_n_columns[k] = (len(features_dict[k]), v)
    #         features_dict[k] = features_dict[k][: v]
    # print(not_same_n_columns)
    #
    # list_documents = list(chain.from_iterable(features_dict.values()))
    #
    # # HORRIBLE HACK! Adding a new siren instance bc there is only one in the dataset
    # id_siren = np.where(y_true == "siren")[0][0]
    # print(id_siren)
    # list_documents.append(list_documents[id_siren])
    # y_true = y_true.tolist()
    # y_true.append("siren")
    # y_true = np.array(y_true)
    #
    # not_bools = np.where(y_true != "booleen")[0]
    # y_true[not_bools] = "O"
    #
    # train_model(list_documents, y_true)
