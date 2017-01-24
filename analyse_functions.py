import os, imp, path
import matplotlib
if os.environ.get('DISPLAY') is None:
    # enables figure saving on clusters with ssh connection
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

imp.load_source('helper', '../tSNE_test/helper_functions.py')
from helper import get_spectra

# ----------------------------------------
# Global variables
# ----------------------------------------
GALAH_BLUE  = [4718, 4903]
GALAH_GREEN = [5649, 5873]
GALAH_RED   = [6481, 6739]
GALAH_IR    = [7590, 7890]

# ----------------------------------------
# Functions
# ----------------------------------------
def range_in_band(x_range, bands_in_dataset=[1,2,3,4]):
    mean_range = np.mean(x_range)
    if mean_range > GALAH_BLUE[0] and mean_range < GALAH_BLUE[1]:
        use_band = np.where(np.array(bands_in_dataset) == 1)
    elif mean_range > GALAH_GREEN[0] and mean_range < GALAH_GREEN[1]:
        use_band = np.where(np.array(bands_in_dataset) == 2)
    elif mean_range > GALAH_RED[0] and mean_range < GALAH_RED[1]:
        use_band = np.where(np.array(bands_in_dataset) == 3)
    elif mean_range > GALAH_IR[0] and mean_range < GALAH_IR[1]:
        use_band = np.where(np.array(bands_in_dataset) == 4)
    else:
        return np.nan
    if np.size(use_band) != 1:
        # band was probably not read
        return np.nan
    else:
        return int(use_band[0])

def get_sobject_spectra(sobject_ids, root=None, bands=[1,2,3,4]):
    spectra_data = list()
    wvl_data = list()
    for sobj_id in sobject_ids:
        spectrum, wavelengths = get_spectra(str(sobj_id), root=root, bands=bands)
        spectra_data.append(spectrum)
        wvl_data.append(wavelengths)
    return (spectra_data, wvl_data)

def plot_spectra_collection(spetra_data, wvl_data, cannon_data_subset, linelist_subset, elem_col, path='spectra.png', title=None):
    total_plots = len(linelist_subset)  # might be smaller than space allocated for all plots
    x_plots = 5
    y_plots = np.ceil(1.*total_plots/x_plots)
    fig, ax = plt.subplots(np.int8(y_plots), np.int8(x_plots),
                           figsize=(3*x_plots, 3*y_plots))
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
            idx_band = range_in_band(x_range, bands_in_dataset=[1,2,3,4])
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
    fig.suptitle(title, y=1.)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)