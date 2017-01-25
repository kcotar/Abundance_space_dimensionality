import os, imp
import numpy as np
import pandas as pd

from astropy.table import Table

from analyse_functions import *

imp.load_source('helper', '../tSNE_test/helper_functions.py')
from helper import get_spectra, move_to_dir, get_abundance_cols, get_element_names


# ----------------------------------------
# Functions
# ----------------------------------------


# ----------------------------------------
# Read all data sets
# ----------------------------------------
print 'Reading data sets'
galah_data_dir = '/home/klemen/GALAH_data/'
spectra_dir = '/media/storage/HERMES_REDUCED/dr5.1/'
cannon_data = Table.read(galah_data_dir+'sobject_iraf_cannon_1.2.fits')
# general_data = Table.read(galah_data_dir+'sobject_iraf_general_1.1.fits')
# param_data = Table.read(galah_data_dir+'sobject_iraf_param_1.1.fits')
linelist = pd.read_csv(galah_data_dir+'GALAH_Cannon_linelist.csv')

# search for elements data in both tables
linelist_elements = np.unique(linelist['Element'])
cannon_elements = get_element_names(get_abundance_cols(cannon_data.colnames))
# for test purposes
# linelist_elements = ['Si']
# cannon_elements = ['Si']

# ----------------------------------------
# Search for repeated observations in successive nights
# ----------------------------------------
galahid_all = cannon_data[np.logical_and(cannon_data['sobject_id']>140310000000000, cannon_data['sobject_id']<140320000000000)]['galah_id']
galahid_unique, galahid_counts = np.unique(galahid_all, return_counts=True)
# there might be additional observations of those objects outside this data range
# there are also another repeated observations () outside this date range

move_to_dir('Repeated_observations')
move_to_dir('Elements_sorted')
# move_to_dir('GalahId_sorted')
for galahid in galahid_unique[galahid_counts >= 3]:
    if galahid < 0:
        continue
    # move_to_dir(str(galahid))
    print 'Start procedure on ' + str(galahid)
    cannon_data_subset = cannon_data[cannon_data['galah_id'] == galahid]
    # retrieve all spectra and wavelength data
    print ' Retrieving spectral data'
    n_obs = len(cannon_data_subset)
    spetra_data, wvl_data = get_sobject_spectra(cannon_data_subset['sobject_id'], root=spectra_dir, bands=[1,2,3,4])
    # plot them according to the selected abundance
    for elem_col in get_abundance_cols(cannon_data.colnames):
        elem = get_element_names([elem_col])[0]
        move_to_dir(elem)
        if elem not in linelist_elements:
            print ' Element '+elem+' not in line list'
            continue
        # plot all spectra ranges for the same element and different observation of the same star
        linelist_subset = linelist[linelist['Element'] == elem]
        print ' Plotting ' + elem + ' data'
        plot_title = 'GALAH id: '+str(galahid)+'   Element: '+elem+'   Observations: '+', '.join([str(id) for id in cannon_data_subset['sobject_id']]),
        plot_path = str(galahid)+'_'+elem+'.png'
        plot_spectra_collection(spetra_data, wvl_data, cannon_data_subset, linelist_subset, elem_col,
                                path=plot_path, title=plot_title)
        os.chdir('..')
    # os.chdir('..')
