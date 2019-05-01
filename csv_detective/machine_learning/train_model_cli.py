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

import pandas as pd
from argopt import argopt
from joblib import Parallel, delayed
from functools import partial
from tqdm import tqdm
import logging
import numpy as np

from csv_detective.machine_learning.training import train_routine

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def run_csv_detective(file_path):
    logger.info("Extracting features from file {}".format(file_path))
    try:
        feat_dict = train_routine(file_path)
        return feat_dict
    except Exception as e:
        logger.info(e)

    # if len(inspection_results) > 2 and len(inspection_results["columns"]):
    #     inspection_results["file"] = file_path
    #     logger.info(file_path, inspection_results)
    #     return inspection_results
    # else:
    #     logger.info("Analysis output of file {} was empty".format(tagged_file_path))


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


def get_csv_detective_analysis(list_files, begin_from=None, n_datasets=None, n_jobs=2):
    if n_datasets:
        list_files = list_files[:n_datasets]

    if begin_from:
        indx_begin = [i for i, path in enumerate(list_files) if begin_from in path]
        if indx_begin:
            list_files = list_files[indx_begin[0]:]

    run_csv_detective_p = partial(run_csv_detective)
    list_dict_result = Parallel(n_jobs=n_jobs)(
        delayed(run_csv_detective_p)(file_path) for file_path in tqdm(list_files))
    list_dict_result = [d for d in list_dict_result if d]
    return list_dict_result


def load_annotations_ids(tagged_file_path):
    df_annotation = pd.read_csv(tagged_file_path)
    y_true = df_annotation.human_detected.fillna("O").values
    csv_ids = df_annotation.id.unique()

    return y_true, csv_ids



def create_list_files(csv_folder, list_resources_ids):
    csv_paths = ["./{0}/{1}.csv".format(csv_folder, resource_id) for resource_id in list_resources_ids]
    return csv_paths

if __name__ == '__main__':
    # try_detective(begin_from="DM1_2018_EHPAD")
    parser = argopt(__doc__).parse_args()
    tagged_file_path = parser.i
    csv_folder_path = parser.p
    output_path = parser.output or "."
    n_cores = int(parser.cores)

    y_true, csv_ids = load_annotations_ids(tagged_file_path)
    y_true = y_true[:5]

    csv_path_list = create_list_files(csv_folder_path, csv_ids)

    if n_cores > 1:
        list_dict_result = get_csv_detective_analysis(csv_path_list, begin_from=None, n_datasets=5, n_jobs=n_cores)
    else:
        list_dict_result = get_csv_detective_analysis_single(csv_path_list, begin_from=None, n_datasets=5)

    assert len(list_dict_result) == len(y_true)

    pass