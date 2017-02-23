import os, imp
import numpy as np
import pandas as pd

from astropy.table import Table
from common_functions import move_to_dir, get_abundance_cols, get_element_names
from line_class import *

import matplotlib
if os.environ.get('DISPLAY') is None:
    # enables figure saving on clusters with ssh connection without -X attribute
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

imp.load_source('helper', '../Carbon-Spectra/helper_functions.py')
from helper import get_spectra


# ----------------------------------------
# Constants
# ----------------------------------------
GALAH_BANDS = [[4718, 4903],  # all possible wavelengths in GALAH bands numbered from 1 to 4
               [5649, 5873],
               [6481, 6739],
               [7590, 7890]]

# ----------------------------------------
# Read all data sets
# ----------------------------------------
print 'Reading data sets'
galah_data_dir = '/home/klemen/GALAH_data/'
spectra_dir = '/media/storage/HERMES_REDUCED/dr5.1/'
cannon_data = Table.read(galah_data_dir+'sobject_iraf_cannon_1.2.fits')
# general_data = Table.read(galah_data_dir+'sobject_iraf_general_1.1.fits')
# param_data = Table.read(galah_data_dir+'sobject_iraf_param_1.1.fits')
line_list = pd.read_csv(galah_data_dir+'GALAH_Cannon_linelist.csv')

# search for elements data in both tables
line_list_elements = np.unique(line_list['Element'])
cannon_elements = get_element_names(get_abundance_cols(cannon_data.colnames))

# ----------------------------------------
# Collection of absorption lines classes
# ----------------------------------------
move_to_dir('Absorption_lines_data_2')
abs_line_collection = list([])
for n_line in range(len(line_list)):
    cur_abs_line = AbsorptionLine(element=line_list['Element'][n_line], wvl_center=line_list['line_centre'][n_line],
                                  wvl_step=0.06, wvl_width=2.)
    abs_line_collection.append(cur_abs_line)

# print abs_line_collection
# ----------------------------------------
# Collect data
# ----------------------------------------
for sob_id in cannon_data['sobject_id']:

print ' Working on '+str(sob_id)
	try:
		spectrum, wavelengths = get_spectra(str(sob_id), root=spectra_dir, bands=[1,2,3,4], read_sigma=False)
	except:
		print '  Problem reading spectra'
		for abs_line in abs_line_collection:
			abs_line.write_line(abs_line.nan_row)
		continue
	print '  Reading finished, resampling absorption lines spectra'
	for abs_line in abs_line_collection:
		abs_line.resample_and_add_line(spectrum, wavelengths)
