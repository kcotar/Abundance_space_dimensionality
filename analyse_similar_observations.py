import os, imp, path
import matplotlib
if os.environ.get('DISPLAY') is None:
    # enables figure saving on clusters with ssh conection
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

teff_range = np.arange(3000, 7700, 100)
logg_range = np.arange(0, 8, 0.5)
feh_range = np.arange(-3, 3.5, 0.5)
# abund_range = np.arange(-2., 2., 0.02)

print 'Processing started'
move_to_dir('Similar_observations')
for abund in get_abundance_cols(cannon_data.colnames)[1:]:
    abund_range = np.arange(np.round(np.min(cannon_data[abund]), 2), np.round(np.max(cannon_data[abund]), 2), 0.05)
    elem = get_element_names([abund])[0]
    linelist_subset = linelist[linelist['Element'] == elem]
    n_lines = len(linelist_subset)
    txt_out = open(elem+'_data.csv', 'w')
    # create csv header
    header_line = 'teff_min,teff_max,logg_min,logg_max,feh_min,feh_max,abund_min,abund_max,'
    header_line += ','.join(['line'+str(a) for a in np.arange(n_lines)+1])+'\n'
    txt_out.write()
    move_to_dir(elem)
    for i_abund in range(len(abund_range) - 1):
        for i_teff in range(len(teff_range)-1):
            for i_logg in range(len(logg_range)-1):
                for i_feh in range(len(feh_range)-1):
                    idx_use = np.logical_and(cannon_data['teff_cannon']>=teff_range[i_teff],
                                             cannon_data['teff_cannon']<teff_range[i_teff+1])
                    idx_use = np.logical_and(idx_use,
                                             np.logical_and(cannon_data['logg_cannon']>=logg_range[i_logg],
                                                            cannon_data['logg_cannon']<logg_range[i_logg+1]))
                    idx_use = np.logical_and(idx_use,
                                             np.logical_and(cannon_data['feh_cannon'] >= feh_range[i_feh],
                                                            cannon_data['feh_cannon'] < feh_range[i_feh+1]))
                    idx_use = np.logical_and(idx_use,
                                             np.logical_and(cannon_data[abund] >= abund_range[i_abund],
                                                            cannon_data[abund] < abund_range[i_abund+1]))
                    n_use = np.sum(idx_use)
                    if n_use >= 2:
                        print 'Working on teff: ' + str(teff_range[i_teff]) + ' logg: ' + str(logg_range[i_logg]) \
                              + ' feh: ' + str(feh_range[i_feh]) + ' attribute: ' + abund + ' abundance: ' \
                              + str(abund_range[i_abund])
                        print ' found: '+str(n_use)

                        cannon_data_subset = cannon_data[idx_use]
                        # retrieve all spectra and wavelength data
                        print ' Retrieving spectral data'
                        n_obs = len(cannon_data_subset)
                        spetra_data, wvl_data = get_sobject_spectra(cannon_data_subset['sobject_id'], root=spectra_dir,
                                                                    bands=[1, 2, 3, 4])
                        print ' Plotting data'
                        plot_title = ''
                        plot_path = 'teff_'+str(teff_range[i_teff])+'_logg_'+str(logg_range[i_logg])\
                                    +'_feh_'+str(feh_range[i_feh])+ '_abund_'+str(abund_range[i_abund])+'.png'
                        plot_spectra_collection(spetra_data, wvl_data, cannon_data_subset, linelist_subset, abund,
                                                path=plot_path, title=plot_title)
    os.chdir('..')