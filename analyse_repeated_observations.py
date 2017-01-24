import os, imp, path
import matplotlib
if os.environ.get('DISPLAY') is None:
    # enables figure saving on clusters with ssh conection
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.table import Table

imp.load_source('helper', '../tSNE_test/helper_functions.py')
from helper import get_spectra, move_to_dir, get_abundance_cols, get_element_names

GALAH_BLUE  = [4718, 4903]
GALAH_GREEN = [5649, 5873]
GALAH_RED   = [6481, 6739]
GALAH_IR    = [7590, 7890]

# ----------------------------------------
# Read all data sets
# ----------------------------------------
def range_in_band(x_range):
    mean_range = np.mean(x_range)
    if mean_range > GALAH_BLUE[0] and mean_range < GALAH_BLUE[1]:
        return 1
    elif mean_range > GALAH_GREEN[0] and mean_range < GALAH_GREEN[1]:
        return 2
    elif mean_range > GALAH_RED[0] and mean_range < GALAH_RED[1]:
        return 3
    elif mean_range > GALAH_IR[0] and mean_range < GALAH_IR[1]:
        return 4
    else:
        return np.nan

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
for galahid in galahid_unique[galahid_counts >= 3]:
    if galahid < 0:
        continue
    move_to_dir(str(galahid))
    print 'Start procedure on ' + str(galahid)
    cannon_data_subset = cannon_data[cannon_data['galah_id'] == galahid]
    # retrieve all spectra and wavelength data
    print ' Retrieving spectral data'
    n_obs = len(cannon_data_subset)
    spetra_data = list()
    wvl_data = list()
    for sobj_id in cannon_data_subset['sobject_id']:
        spectrum, wavelengths = get_spectra(str(sobj_id),
                                            root=spectra_dir)
        spetra_data.append(spectrum)
        wvl_data.append(wavelengths)
    # plot them according to the selected abundance
    for elem_col in get_abundance_cols(cannon_data.colnames):
        elem = get_element_names([elem_col])[0]
        if elem not in linelist_elements:
            print ' Element '+elem+' not in line list'
            continue
        # plot all spectra ranges for the same element and different observation of the same star
        linelist_subset = linelist[linelist['Element'] == elem]
        linelist_subset_keys = linelist_subset.keys()
        total_plots = len(linelist_subset)  # might be smaller than space allocated for all plots
        x_plots = 5
        y_plots = np.ceil(1.*total_plots/x_plots)
        fig, ax = plt.subplots(np.int8(y_plots), np.int8(x_plots),
                               figsize=(3*x_plots, 3*y_plots))
        print ' Plotting '+elem+' data'
        for i_plot in np.arange(total_plots):
            p_x_pos = i_plot % x_plots
            p_y_pos = np.int8(np.floor(i_plot / x_plots))
            if total_plots > x_plots:
                plot_pos = (p_y_pos, p_x_pos)
            else:
                plot_pos = p_x_pos
            # add individual spectra to the plot
            x_range = [linelist_subset['segment_start'].get_values()[i_plot], linelist_subset['segment_end'].get_values()[i_plot]]
            for i_data in range(len(cannon_data_subset)):
                # determine which band should be plotted
                idx_band = range_in_band(x_range)-1
                ax[plot_pos].plot(wvl_data[i_data][idx_band],
                                spetra_data[i_data][idx_band],
                                label=str(cannon_data_subset[elem_col][i_data]))
            # add legend in the middle of the plot
            if p_x_pos==0 and p_y_pos==0:
                ax[plot_pos].legend(loc='center right', bbox_to_anchor=(-0.3, 0.5))
            ax[plot_pos].axvline(x=linelist_subset['line_centre'].get_values()[i_plot], color='black', linewidth=1)
            ax[plot_pos].axvline(x=linelist_subset['line_start'].get_values()[i_plot], color='black', linewidth=2)
            ax[plot_pos].axvline(x=linelist_subset['line_end'].get_values()[i_plot], color='black', linewidth=2)
            ax[plot_pos].set(xlim=x_range,
                           ylim=[0.3, 1.1])
        fig.suptitle('GALAH id: '+str(galahid)+'   Element: '+elem+'   Observations: '+', '.join([str(id) for id in cannon_data_subset['sobject_id']]),
                     y=1.)
        fig.tight_layout()
        fig.savefig(str(galahid)+'_'+elem+'.png', dpi=300, bbox_inches='tight')

    os.chdir('..')
