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
from argopt import argopt
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from xgboost import XGBClassifier

from csv_detective.machine_learning.features import ItemSelector, CustomFeatures, ColumnInfoExtractor


# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler())


def get_files(data_path, ext="csv"):
    return glob.glob(data_path + "/*.{}".format(ext))


def extract_id(file_path):
    import os
    resource_id = os.path.basename(file_path)[:-4]
    return resource_id




if __name__ == '__main__':
    # try_detective(begin_from="DM1_2018_EHPAD")
    parser = argopt(__doc__).parse_args()
    tagged_file_path = parser.i
    csv_folder_path = parser.p
    output_path = parser.output or "."
    num_files = parser.num_files
    num_rows = parser.num_rows

    n_cores = int(parser.cores)

    pipeline = Pipeline([
        # Extract column info information from csv

        # Use FeatureUnion to combine the features from subject and body
        ('union', FeatureUnion(
            transformer_list=[

                # Pipeline for pulling custom features from the columns
                ('custom_features', Pipeline([
                    ('selector', ItemSelector(key='per_file_rows')),
                    ('customfeatures', CustomFeatures(n_jobs=n_cores)),
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
                    # ('count', CountVectorizer(ngram_range=(1, 3), analyzer="char_wb", binary=False, max_features=2000)),
                    ('hash', HashingVectorizer(n_features=2 ** 2, ngram_range=(1, 3), analyzer="char_wb")),

                ])),

            ],

            # weight components in FeatureUnion
            # transformer_weights={
            #     'column_custom': .499,
            #     'cell_bow': .499,
            #     'header_bow': 0.02,
            # },

        )),

        # Use a SVC classifier on the combined features
        # ('LR', LogisticRegression(multi_class="ovr", n_jobs=-1, solver="lbfgs")),
        # ('XG', XGBClassifier(n_jobs=5)),
        ('MLP', MLPClassifier(hidden_layer_sizes=(100, 100), activation="relu")),

    ])

    train, test = ColumnInfoExtractor(n_files=num_files, n_rows=num_rows, train_size=.7, n_jobs=n_cores).transform(
        annotations_file=tagged_file_path,
        csv_folder=csv_folder_path)

    pipeline.fit(train, train["y"])
    y_test = test["y"]
    y_pred = pipeline.predict(test)

    print(classification_report(y_test, y_pred=y_pred))

