import os
from astropy.table import Table
import numpy as np

def get_abundance_cols(col_names):
    abund_col_names = [col for col in col_names if '_abund_' in col and '_e_' not in col]  # and 'fe_' not in col]
    return abund_col_names


def get_abundance_error_cols(col_names):
    abund_col_names = [col for col in col_names if '_abund_' in col and '_e_' in col]  # and 'fe_' not in col]
    return abund_col_names


def move_to_dir(path):
    if not(os.path.isdir(path)):
        os.mkdir(path)
    os.chdir(path)


def normalize_table(data, cols):
    data_out = Table(data)
    for col in cols:
        data_mean = np.nanmean(data[col])
        data_std = np.nanstd(data[col])
        data_out[col] = (data_out[col] - data_mean) / data_std
    return data_out
