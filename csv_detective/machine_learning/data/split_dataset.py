from sklearn.model_selection import StratifiedShuffleSplit

from machine_learning.features import ColumnInfoExtractor
import pandas as pd
import numpy as np


annotations_file = "/media/stuff/Pavel/Documents/Eclipse/workspace/csv_detective/csv_detective/machine_learning/data/columns_annotation.csv"
csv_folder = "/data/datagouv/csv_top/"

columns = ColumnInfoExtractor(n_files=10, n_rows=50, n_jobs=10, save_dataset=True)
columns.transform(annotations_file=annotations_file, csv_folder=csv_folder)


df = pd.read_csv("/media/stuff/Pavel/Documents/Eclipse/workspace/csv_detective/csv_detective/machine_learning/data/columns_annotation.csv")
df.human_detected = df.human_detected.fillna("O")
y = df.human_detected.values
X = np.zeros(len(y))
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
indices = list(sss.split(X, y))
train_indices, test_indices = indices[0][0], indices[0][1]

train_df = df.iloc[train_indices]
test_df = df.iloc[test_indices]
