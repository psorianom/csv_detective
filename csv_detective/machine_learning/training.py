
"""
Here we train a machine learning model to detect the type of column. We need :
1. Load the annotation csv /home/pavel/etalab/code/csv_true_detective/data/columns_annotation.csv
    a. Get the filenames found in this file
2. Start csv_detective routine for each filename
3. Get the
"""
import string
from collections import Counter
# from csv_detective.machine_learning.train_model_cli import RESOURCE_ID_COLUMNS
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import SVC
from scipy.sparse import vstack, hstack


import numpy as np
import pandas as pd


def features_cell(rows: list):

    list_features = []
    features = {}
    for value in rows:

        # num chars
        features["num_chars"] = len(value)

        # num numeric
        features["num_numeric"] = sum(1 for c in value if c.isnumeric())

        # num alpha
        features["num_alpha"] = sum(1 for c in value if c.isalpha())

        # num distinct chars
        features["num_unique_chars"] = len(set(value))

        # num white spaces
        features["num_spaces"] = value.count(" ")

        # num of special chars
        features["num_special_chars"] = sum(1 for c in value if c in string.punctuation)

        list_features.append(features)

    return list_features



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


def train_model(list_features_dict, y_true):
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(list_features_dict)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

    indices = list(sss.split(X, y_true))
    train_indices, test_indices = indices[0][0], indices[0][1]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y_true[train_indices], y_true[test_indices]

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_true=y_test, y_pred=y_pred))


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
    ax.xaxis.set_ticklabels(labels, rotation=90)
    ax.yaxis.set_ticklabels(labels, rotation=0)
    plt.show()


def train_model2(X, y_true):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    y_true = np.array(list(y_true))
    indices = list(sss.split(X, y_true))
    train_indices, test_indices = indices[0][0], indices[0][1]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y_true[train_indices], y_true[test_indices]

    clf = SVC(kernel="linear")
    # clf = LogisticRegression()
    clf = MLPClassifier(hidden_layer_sizes=(50,50), activation="relu")
    # clf = ExtraTreeClassifier(class_weight="balanced")
    # clf = RandomForestClassifier(n_estimators=200, n_jobs=5, class_weight="balanced_subsample")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_true=y_test, y_pred=y_pred))
    show_confusion_matrix(y_true=y_test, y_pred=y_pred, labels=np.unique(y_true))
    pass


def create_data_matrix(documents, columns_names, extra_features):

    # Text from the cell value itself
    cell_cv = CountVectorizer(ngram_range=(1, 3), analyzer="char_wb", max_df=0.8, min_df=2)
    X_cell = cell_cv.fit_transform(documents)


    # Text from the header
    header_cv = CountVectorizer(ngram_range=(1, 3), analyzer="char")
    X_header = header_cv.fit_transform(columns_names)

    # Hand-crafted features
    extra_dv = DictVectorizer()
    X_extra = extra_dv.fit_transform(extra_features)

    X_all = hstack([X_cell, X_extra], format="csr")

    return X_all, cell_cv, header_cv, extra_dv


if __name__ == '__main__':
    file_path = "/data/datagouv/csv_top/edf158f9-bdde-4e6e-b92c-c156c9316383.csv"

