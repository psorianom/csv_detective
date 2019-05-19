from sklearn.model_selection import StratifiedShuffleSplit

from machine_learning.features import ColumnInfoExtractor
import pandas as pd
import numpy as np


annotations_file = "/media/stuff/Pavel/Documents/Eclipse/workspace/csv_detective/csv_detective/machine_learning/data/columns_annotation.csv"
csv_folder = "/data/datagouv/csv_top/"

columns = ColumnInfoExtractor(n_files=10, n_rows=100, n_jobs=5, train_size=0.7, save_dataset=True)
train, test = columns.transform(annotations_file=annotations_file, csv_folder=csv_folder)
print()