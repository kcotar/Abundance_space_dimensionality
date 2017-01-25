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

teff_range = np.arange(3800, 7200, 150)
logg_range = np.arange(0.0, 6.5, 0.5)
feh_range = np.arange(-1.5, 1.25, 0.25)
abund_range = np.arange(-2.5, 2.5, 0.05)

print 'Processing started'
move_to_dir('Similar_observations')
for abund in get_abundance_cols(cannon_data.colnames)[1:]:
    # select appropriate abundances value range subset to decrease processing time
    idx_min = np.nanargmin(np.abs(abund_range - np.percentile(cannon_data[abund], 2)))  # percentile used to remove extreme abundance values
    idx_max = np.nanargmin(np.abs(abund_range - np.percentile(cannon_data[abund], 98)))
    abund_range_use = abund_range[idx_min:idx_max+1]
    # determine element name and retrieve its list of observed absorption lines
    elem = get_element_names([abund])[0]
    linelist_subset = linelist[linelist['Element'] == elem]
    n_lines = len(linelist_subset)
    if n_lines == 0:
        print 'No absorption lines found for this element.'
        os.chdir('..')
        continue
    # create csv header
    txt_out_filename = elem+'_data.csv'
    txt_out = open(txt_out_filename, 'w')
    header_line = 'teff_min,teff_max,logg_min,logg_max,feh_min,feh_max,abund_min,abund_max,'
    header_line += ','.join(['line'+str(a) for a in np.arange(n_lines)+1])+'\n'
    txt_out.write(header_line)
    txt_out.close()
    move_to_dir(elem)
    # print number of possible parameter combinations
    print 'Number of combinations in parameter space: '+str(len(teff_range)*len(logg_range)*len(feh_range)*len(abund_range_use))
    for i_abund in range(len(abund_range_use) - 1):
        idx_abund = np.logical_and(cannon_data[abund] >= abund_range_use[i_abund],
                                   cannon_data[abund] < abund_range_use[i_abund+1])
        for i_teff in range(len(teff_range)-1):
            idx_teff = np.logical_and(cannon_data['teff_cannon'] >= teff_range[i_teff],
                                      cannon_data['teff_cannon'] < teff_range[i_teff+1])
            for i_logg in range(len(logg_range)-1):
                idx_logg = np.logical_and(cannon_data['logg_cannon'] >= logg_range[i_logg],
                                          cannon_data['logg_cannon'] < logg_range[i_logg+1])
                for i_feh in range(len(feh_range)-1):
                    idx_feh = np.logical_and(cannon_data['feh_cannon'] >= feh_range[i_feh],
                                             cannon_data['feh_cannon'] < feh_range[i_feh+1])
                    idx_use = np.logical_and(np.logical_and(idx_abund, idx_teff),
                                             np.logical_and(idx_logg, idx_feh))
                    n_use = np.sum(idx_use)
                    out_line = '{:04.0f},{:04.0f},{:03.1f},{:03.1f},{:04.2f},{:04.2f},{:04.2f},{:04.2f},'.\
                        format(teff_range[i_teff], teff_range[i_teff+1], logg_range[i_logg], logg_range[i_logg+1],
                               feh_range[i_feh], feh_range[i_feh+1], abund_range_use[i_abund], abund_range_use[i_abund+1])
                    if n_use >= 2:
                        print ('Working on teff: {:04.0f} logg: {:03.1f}  feh: {:04.2f}  attribute: '+abund+' abundance: {:04.2f} ').\
                            format(teff_range[i_teff], logg_range[i_logg], feh_range[i_feh], abund_range_use[i_abund])
                        print ' Found: '+str(n_use)

                        cannon_data_subset = cannon_data[idx_use]
                        # retrieve all spectra and wavelength data
                        print ' Retrieving spectral data'
                        n_obs = len(cannon_data_subset)
                        spetra_data, wvl_data = get_sobject_spectra(cannon_data_subset['sobject_id'],
                                                                    root=spectra_dir, bands=[1, 2, 3, 4])
                        print ' Plotting data'
                        plot_title = 'Observations: '+', '.join([str(so_id) for so_id in cannon_data_subset['sobject_id'][:8]])
                        plot_path = 'teff_'+str(teff_range[i_teff])+'_logg_'+str(logg_range[i_logg])\
                                    +'_feh_'+str(feh_range[i_feh])+ '_abund_'+str(abund_range_use[i_abund])+'.png'
                        plot_spectra_collection(spetra_data, wvl_data, cannon_data_subset, linelist_subset, abund,
                                                path=plot_path, title=plot_title)

                        print ' Reducing spectral data'
                        reduced = reduce_spectra_collection(spetra_data, wvl_data, linelist_subset)
                        out_line += ','.join(['{:05.3f}'.format(val) for val in reduced])
                    else:
                        out_line += ','.join([str(np.nan) for a in range(n_lines)])
                    txt_out = open('../'+txt_out_filename, 'a')  # ../ added as it is located in a parent directory
                    txt_out.write(out_line + '\n')
                    txt_out.close()
    os.chdir('..')
