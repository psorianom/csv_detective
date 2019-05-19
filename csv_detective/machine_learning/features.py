import os
import string
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from tqdm import tqdm

from csv_detective.detection import detect_encoding, detect_separator, detect_headers, parse_table


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

    def __init__(self, n_files=None, n_rows=200, n_jobs=1, save_dataset=False):

        self.n_rows = n_rows
        self.n_files = n_files
        self.n_jobs = n_jobs
        self.save_dataset = save_dataset

    def fit(self, X, y=None):
        return self

    def _load_file(self, file_path):
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
            self.n_rows,
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

        csv_df = self._load_file(file_path=file_path)

        if csv_df is None:
            return None

        if csv_df.shape[1] != len(file_labels):
            print("Annotated number of columns does not match the number of columns in file {}. The csv parsing"
                  "does not match with that of the annotation file "
                  "(more or less columns found in the annotation file).".format(file_path))
            return

        file_columns = []
        columns_names = []
        for j in range(len(csv_df.columns)):
            # Get all values of the column j and clean it a little bit
            temp_list = csv_df.iloc[:, j].dropna().apply(lambda x: x.replace(" ", "")).to_list()
            file_columns.append(temp_list)
            columns_names.extend([csv_df.columns[j].lower()] * len(temp_list))

        assert len(file_columns) == len(file_labels)  # Assert we have the same number of annotated columns and columns

        rows_labels = []
        rows_values = []

        # Get both lists of labels and values-per-column in a single flat huge list
        for i, l in enumerate(file_labels):
            rows_labels.extend([l] * len(file_columns[i]))
            rows_values.extend(file_columns[i])

        assert len(rows_values) == len(rows_labels) == len(columns_names)
        return {"all_columns": rows_values, "y": rows_labels, "all_headers": columns_names,
                "per_file_labels": [file_labels], "per_file_rows": [file_columns]}

    def _extract_columns_selector(self, csv_folder):
        list_files = ["{0}/{1}.csv".format(csv_folder, resource_id) for resource_id in sorted(self._dict_ids_labels)]

        if self.n_jobs and self.n_jobs > 1:
            csv_info = Parallel(n_jobs=self.n_jobs)(
                delayed(self._extract_columns)(file_path)
                for file_path in tqdm(list_files))
        else:
            csv_info = [self._extract_columns(f)
                        for f in tqdm(list_files)]

        csv_info = [d for d in csv_info if d]
        csv_info_items = defaultdict(list)
        for csv_dict in csv_info:
            for k, v in csv_dict.items():
                csv_info_items[k].extend(v)

        if self.save_dataset:
            dataset_df = pd.DataFrame({"rows_values": csv_info_items["all_columns"],
                                       "rows_labels": csv_info_items["y"], "columns_names": csv_info_items["all_headers"]})
            dataset_df.to_csv("csv_detective/machine_learning/data/out/per_line_dataset.csv", index=False)


        return csv_info_items

    def transform(self, annotations_file, csv_folder):
        self._load_annotations_info(annotations_file)
        columns_info = self._extract_columns_selector(csv_folder)
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
                delayed(self._extract_custom_features)(file_path, self._dict_ids_labels)
                for file_path in tqdm(rows_values))
        else:
            features = [self._extract_custom_features(f)
                        for f in tqdm(rows_values)]

        features = [d for d in features if d]
        return features

    def _extract_custom_features(self, rows_values):
        list_features = []

        def is_float(x):
            return x.replace('.', '', 1).isdigit() and "." in x

        for i, value in enumerate(rows_values):
            features = {}
            value = value.replace(" ", "")
            features["is_numeric"] = 1 if value.isnumeric() or is_float(value) else 0
            # features["single_char"] = 1 if len(value.strip()) == 1 else 0
            if features["is_numeric"]:
                try:
                    numeric_value = int(value)
                except Exception as e:
                    numeric_value = float(value)

                if numeric_value < 0:
                    features["<0"] = 1
                if 0 <= numeric_value < 2:
                    features[">=0<2"] = 1
                elif 2 <= numeric_value < 1000:
                    features[">=2<1000"] = 1
                elif 1000 <= numeric_value < 10000:
                    features[">=1k<10k"] = 1
                elif 10000 <= numeric_value:
                    features[">=10k<100k"] = 1

            # "contextual" features

            # if i > 0:
            #     features["is_numeric-1"] = 1 if rows[i-1].isnumeric() or is_float(rows[i-1]) else 0
            #     features["num_chars_-1"] = len(rows[i-1])
            #     if i > 1:
            #         features["is_numeric-2"] = 1 if rows[i-2].isnumeric() or is_float(rows[i-2]) else 0
            #         features["num_chars_-2"] = len(rows[i-2])
            # if i <= len(rows) - 2:
            #     features["is_numeric+1"] = 1 if rows[i+1].isnumeric() or is_float(rows[i+1]) else 0
            #     features["num_chars_+1"] = len(rows[i+1])
            #     if i <= len(rows) - 3:
            #         features["is_numeric+2"] = 1 if rows[i+2].isnumeric() or is_float(rows[i+2]) else 0
            #         features["num_chars_+2"] = len(rows[i+2])

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

            list_features.append(features)

        return list_features

    def fit(self, x, y=None):
        return self


if __name__ == '__main__':
    annotations_file_train = "path-to-annotations-train"
    annotations_file_test = "path-to-annotations-test"
    csv_folder = "folder-of-csvs"
    train = ColumnInfoExtractor(n_files=100, n_rows=200, n_jobs=10).transform(annotations_file=annotations_file_train,
                                                                              csv_folder=csv_folder)
    test = ColumnInfoExtractor(n_files=100, n_rows=200, n_jobs=10).transform(annotations_file=annotations_file_test,
                                                                             csv_folder=csv_folder)

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
                    ('customfeatures', CustomFeatures()),
                    ("customvect", DictVectorizer())
                ])),

                # Pipeline for standard bag-of-words model for cell values
                ('cell_features', Pipeline([
                    ('selector', ItemSelector(key='all_columns')),
                    ('count', CountVectorizer(ngram_range=(1, 3), analyzer="char_wb", binary=False, max_features=2000)),
                ])),

                # Pipeline for standard bag-of-words model for header values
                ('header_features', Pipeline([
                    ('selector', ItemSelector(key='all_headers')),
                    ('count', CountVectorizer(ngram_range=(1, 3), analyzer="char_wb", binary=False, max_features=2000)),
                ])),

            ],

            # weight components in FeatureUnion
            transformer_weights={
                'column_custom': 1.0,
                'cell_bow': 0.0,
                'header_bow': 1.0,
            },
        )),

        # Use a SVC classifier on the combined features
        # clf = LogisticRegression(multi_class="ovr", n_jobs=-1, solver="lbfgs")
        ('LR', LogisticRegression(multi_class="ovr", n_jobs=-1, solver="lbfgs")),
    ])

    pipeline.fit(train, train["y"])
    y_pred = pipeline.predict(test)
    print(classification_report(test["y"], y_pred=y_pred))
