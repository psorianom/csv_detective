import os
import string
from collections import defaultdict, Counter
from itertools import chain

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from tqdm import tqdm
from xgboost import XGBClassifier
import re

from scipy.sparse import vstack

from csv_detective.detection import detect_encoding, detect_separator, detect_headers, parse_table
from csv_detective.machine_learning import logger
from csv_detective.machine_learning.utils import visualize_matrices, visualize_multivariate


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class ColumnInfoExtractor(BaseEstimator, TransformerMixin):
    """Extract the subject & body from a usenet post in a single pass.

    Takes a sequence of strings and produces a dict of sequences.  Keys are
    `subject` and `body`.
    """

    def __init__(self, n_files=None, n_rows=200, n_jobs=1, train_size=0.7, save_dataset=False):

        self.n_rows = n_rows
        self.n_files = n_files
        self.n_jobs = n_jobs
        self.save_dataset = save_dataset
        self.train_size = train_size
        self._file_idx = {}

    def fit(self, X, y=None):
        return self

    def _load_file(self, file_path, n_rows):
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
            n_rows,
            random_state=42
        )

        if table.empty:
            print("Could not read {}".format(file_path))
            return
        return table

    def _load_annotations_info(self, annotations_file):
        df_annotation = pd.read_csv(annotations_file)
        # df_annotation = df_annotation.sample(frac=1)
        csv_ids = df_annotation.id.unique()
        dict_ids_labels = {}
        if self.n_files:
            csv_ids = csv_ids[:self.n_files]
            df_annotation = df_annotation[df_annotation.id.isin(csv_ids)]

        df_annotation.human_detected = df_annotation.human_detected.fillna("O")

        for i in range(df_annotation.shape[0]):
            dict_ids_labels.setdefault(df_annotation.iloc[i].id, []).append(df_annotation.iloc[i].human_detected)

        num_annotations_per_resource = df_annotation.groupby("id", sort=False).count()["columns"].to_dict()
        assert (all(True if list(dict_ids_labels.keys())[i] == list(num_annotations_per_resource.keys())[i] else False
                    for i in range(len(num_annotations_per_resource))))

        self._dict_ids_labels = dict_ids_labels
        return dict_ids_labels

    def _extract_columns(self, file_path):
        csv_id = os.path.basename(file_path)[:-4]
        file_labels = self._dict_ids_labels[csv_id]

        csv_df = self._load_file(file_path=file_path, n_rows=self.n_rows)

        if csv_df is None:
            return None

        if csv_df.shape[1] != len(file_labels):
            print("Annotated number of columns does not match the number of columns in file {}. The csv parsing"
                  "does not match with that of the annotation file "
                  "(more or less columns found in the annotation file).".format(file_path))
            return

        # df_idx = np.arange(len(csv_df))
        df_idx = sorted(csv_df.index.values)
        # train_idx = np.random.choice(df_idx, size=int(len(df_idx) * self.train_size), replace=True)
        train_idx = df_idx[:int(len(df_idx) * self.train_size)]
        test_idx = np.setdiff1d(df_idx, train_idx)
        self._file_idx[file_path] = train_idx
        if len(test_idx):
            dataset_type = {"train": csv_df.loc[train_idx], "test": csv_df.loc[test_idx]}
        else:
            dataset_type = {"train": csv_df.loc[train_idx]}

        datasets_info = {}
        for ds_type, df in dataset_type.items():
            file_columns = []
            columns_names = []
            for j in range(len(df.columns)):
                # Get all values of the column j and clean it a little bit
                temp_list = df.iloc[:, j].dropna().apply(lambda x: x.replace(" ", "")).to_list()
                file_columns.append(temp_list)
                columns_names.extend([df.columns[j].lower()] * len(temp_list))

            rows_labels = []
            rows_values = []

            # Get both lists of labels and values-per-column in a single flat huge list
            for i, l in enumerate(file_labels):
                rows_labels.extend([l] * len(file_columns[i]))
                rows_values.extend(file_columns[i])

            assert len(rows_values) == len(rows_labels) == len(columns_names)
            datasets_info[ds_type] = {"all_columns": rows_values, "y": rows_labels, "all_headers": columns_names,
                                      "per_file_labels": [file_labels], "per_file_rows": [file_columns]}

        return datasets_info

    def _extract_columns_selector(self, csv_folder):
        list_files = ["{0}/{1}.csv".format(csv_folder, resource_id) for resource_id in sorted(self._dict_ids_labels)]

        if self.n_jobs and self.n_jobs > 1:
            csv_info = Parallel(n_jobs=self.n_jobs)(
                delayed(self._extract_columns)(file_path)
                for file_path in tqdm(list_files))
        else:
            csv_info = [self._extract_columns(f)
                        for f in tqdm(list_files)]

        dataset_items = defaultdict(lambda: defaultdict(list))
        for datasets in csv_info:
            if not datasets:
                continue
            for ds_type in datasets:
                for k, v in datasets[ds_type].items():
                    if not v:
                        continue
                    dataset_items[ds_type][k].extend(v)

        if self.save_dataset:
            import json
            with open("{0}_{1}rows.json".format("./csv_detective/machine_learning/data/out/dataset",
                                               self.n_rows), "w") as filo:
                json.dump(dataset_items, filo)

        return [dataset_items[k] for k in dataset_items]

    def transform(self, annotations_file, csv_folder):
        self._load_annotations_info(annotations_file)
        columns_info = self._extract_columns_selector(csv_folder)
        if len(columns_info) < 2:
            return columns_info[0], None

        return columns_info


class CustomFeatures(BaseEstimator, TransformerMixin):
    """Extract the subject & body from a usenet post in a single pass.

    Takes a sequence of strings and produces a dict of sequences.  Keys are
    `subject` and `body`.
    """

    def __init__(self, n_jobs=1):

        self.n_jobs = n_jobs

    def transform(self, rows_values: list):
        """
        rows_values is a list of lists, one list per csv, containing the n rows per column
        :param rows_values:
        :return:
        """
        if self.n_jobs and self.n_jobs > 1:
            features = Parallel(n_jobs=self.n_jobs)(
                delayed(self._extract_custom_features)(file_path)
                for file_path in tqdm(rows_values))
        else:
            features = [self._extract_custom_features(f)
                        for f in tqdm(rows_values)]

        features = list(chain.from_iterable(features))
        return features

    def _extract_custom_features(self, rows_values):
        list_features = []

        def is_float(x):
            return re.sub(r'[.,]', '', x, 1).isdigit() and len(re.findall(r"[.,]", x)) > 0

        for j, rows in enumerate(rows_values):
            numeric_col = np.array([float(f.replace(",", ".")) for f in rows if f.isdigit() or is_float(f)], dtype=float)
            for i, value in enumerate(rows):
                # Add column features if existent
                features = {}
                if len(numeric_col):
                    features["num_unique"] = len(set("".join(rows)))
                    features["col_sum"] = 1 if sum(numeric_col) < len(numeric_col) else 0
                    # features["num_unique"] = len(np.unique(numeric_col))
                    # features["col_sum"] = 1 if sum(numeric_col) < len(numeric_col) else 0


                # if j > 0:
                #     column_prev = columns[j - 1][:]
                #     np.random.shuffle(column_prev)
                    # features[str(hash("".join(column_prev)) % (10 ** 2))] = 1
                # elif j + 1 < len(columns):
                #     column_next = columns[j + 1][:]
                #     np.random.shuffle(column_next)
                    # features[str(hash("".join(column_next)) % (10 ** 2))] = 1
                #
                columns_copy = rows_values[j][:]
                np.random.shuffle(columns_copy)

                features[str(hash("".join(columns_copy)) % (10 ** 3))] = 1

                # features["is_numeric"] = 1 if value.isnumeric() or is_float(value) else 0
                # features["single_char"] = 1 if len(value.strip()) == 1 else 0
                # if features["is_numeric"]:
                #
                #     try:
                #         numeric_value = int(value)
                #     except:
                #         numeric_value = float(value.replace(",", "."))
                #
                #     if numeric_value < 0:
                #         features["<0"] = 1
                #     if 0 <= numeric_value < 2:
                #         features[">=0<2"] = 1
                #     elif 2 <= numeric_value < 500:
                #         features[">=2<500"] = 1
                #     elif 500 <= numeric_value < 1000:
                #         features[">=500<1000"] = 1
                #     elif 1000 <= numeric_value < 10000:
                #         features[">=1k<10k"] = 1
                #     elif 10000 <= numeric_value:
                #         features[">=10k<100k"] = 1
                #
                # num lowercase
                features["num_lower"] = sum(1 for c in value if c.islower())

                # num uppercase
                features["num_upper"] = sum(1 for c in value if c.isupper())


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
                #
                list_features.append(features)
                # print(value, str(features))


        return list_features

    def fit(self, x, y=None):
        return self


if __name__ == '__main__':

    # return {"all_columns": rows_values, "y": rows_labels, "all_headers": columns_names,
    #         "per_file_labels": file_labels, "per_file_rows": file_columns}

    pipeline = Pipeline([
        # Extract column info information from csv

        # Use FeatureUnion to combine the features from subject and body
        ('union', FeatureUnion(
            transformer_list=[

                # Pipeline for pulling custom features from the columns
                ('custom_features', Pipeline([
                    ('selector', ItemSelector(key='per_file_rows')),
                    ('customfeatures', CustomFeatures(n_jobs=1)),
                    ("customvect", DictVectorizer())
                ])),

                # Pipeline for standard bag-of-words model for cell values
                # ('cell_features', Pipeline([
                #     ('selector', ItemSelector(key='all_columns')),
                #     ('count', CountVectorizer(ngram_range=(1, 3), analyzer="char_wb", binary=False, max_features=2000)),
                # ])),

                # Pipeline for standard bag-of-words model for header values
                # ('header_features', Pipeline([
                #     ('selector', ItemSelector(key='all_headers')),
                #     ('count', CountVectorizer(ngram_range=(1, 3), analyzer="char_wb", binary=False, max_features=2000)),
                # ])),

            ],

            # weight components in FeatureUnion
            # transformer_weights={
            #     'column_custom': 1.0,
            #     'cell_bow': 1.0,
            #     # 'header_bow': 1.0,
            # },
            verbose=True
        )),

        # Use a SVC classifier on the combined features
        # ('LR', LogisticRegression(multi_class="ovr", n_jobs=-1, solver="lbfgs")),
        ('XG', XGBClassifier(n_jobs=5)),
    ])

    annotations_file = "./csv_detective/machine_learning/data/columns_annotation.csv"
    csv_folder = "/data/datagouv/csv_top/"


    train, test = ColumnInfoExtractor(n_files=20, n_rows=100, train_size=.7, n_jobs=1).transform(
        annotations_file=annotations_file,
        csv_folder=csv_folder)


    debug = False

    if test is None: # good performance
        pipelinem1 = Pipeline(pipeline.steps[:-1])
        X = pipelinem1.fit_transform(train)

        y_true = np.array(train["y"])
        print(pipelinem1.named_steps["union"].transformer_list[0][1].named_steps["customvect"].get_feature_names())

        sss = StratifiedShuffleSplit(n_splits=1, train_size=.7, random_state=42)
        indices = list(sss.split(X, train["y"]))
        train_indices, test_indices = sorted(indices[0][0]), sorted(indices[0][1])
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y_true[train_indices], y_true[test_indices]

        visualize_matrices([X_train], names=["X_train"])
        X2 = vstack([X_train, X_test])
        visualize_matrices([X2], names=["X2"])
        # visualize_matrices([X_train, X_test], names=["X_train", "X_test"])

        clf = XGBClassifier(n_jobs=5)
        # clf = LogisticRegression()

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred=y_pred))

        exit(0)

    else: # shitty performance
        pipelinem1 = Pipeline(pipeline.steps[:-1])
        pepe = Pipeline(pipeline.steps[:-1])

        X_train = pipelinem1.fit_transform(train)
        # X_test_pp = pepe.fit_transform(test)

        X_test = pipelinem1.transform(test)
        X_test = X_test[:, range(X_test.shape[1])]
        sss = StratifiedShuffleSplit(n_splits=1, train_size=.7, random_state=42)
        y_true = np.array(train["y"])

        indices = list(sss.split(X_train, y_true))
        train_indices, test_indices = sorted(indices[0][0]), sorted(indices[0][1])
        X_train2, X_test2 = X_train[train_indices], X_train[test_indices]
        y_train, y_test = y_true[train_indices], y_true[test_indices]

        # visualize_matrices([X_train], names=["X_train"])
        # visualize_multivariate(X_train2, y_train)
        # visualize_distros(X_test2, y_test)
        # print(pipelinem1.named_steps["union"].transformer_list[0][1].named_steps["customvect"].get_feature_names())
        # visualize_matrices([X_train, X_test], names=["X_train", "X_test"])
        clf = XGBClassifier(n_jobs=5)
        clf2 = XGBClassifier(n_jobs=5)

        clf.fit(X_train2, y_train)
        clf2.fit(X_train, y_true)
        y_pred = clf.predict(X_test2)
        y_pred2 = clf2.predict(X_test)
        # pipeline.fit(train, train["y"])
        # y_pred = pipeline.predict(test)
        # visualize_distros(X_train, train["y"])
        # visualize_distros(X_test, test["y"])

        print("Good clf with good test")
        print(classification_report(y_test, y_pred=y_pred))
        print("Bad clf with bad test")
        print(classification_report(test["y"], y_pred=y_pred2))

        print("Good clf with bad test")
        print(classification_report(test["y"], y_pred=clf.predict(X_test)))


    if debug:
        X_test = Pipeline(pipeline.steps[:-1]).transform(test)

        import pickle
        # clf = pickle.load(open("clf_training", "rb"))
        # y_pred = clf.predict(X_t)
        X_train2, y_train2 = pickle.load(open("xytrain", "rb"))
        clf2 = XGBClassifier(n_jobs=5)
        clf2.fit(X_train2, y_train2)
        y_pred = clf2.predict(X_test)
