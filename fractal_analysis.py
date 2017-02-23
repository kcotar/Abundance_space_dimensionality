import os, imp
import numpy as np

import matplotlib
if os.environ.get('DISPLAY') is None:
    # enables figure saving on clusters with ssh connection
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astropy.table import Table
from fractal_dimension import *

imp.load_source('helper', '../tSNE_test/helper_functions.py')
from helper import move_to_dir, get_abundance_cols, get_element_names

galah_data_dir = '/home/klemen/GALAH_data/'  # the same for gigli and local pc
# --------------------------------------------------------
# ---------------- Read Data -----------------------------
# --------------------------------------------------------
# read GALAH data
# galah_general = Table.read(galah_data_dir+'sobject_iraf_general_1.1.fits')
galah_cannon = Table.read(galah_data_dir+'sobject_iraf_cannon_1.2.fits')

use_cols = get_abundance_cols(galah_cannon.colnames)
use_cols_elem = get_element_names(use_cols)

galah_abund_data = galah_cannon[use_cols].to_pandas().values

idx_random = np.random.randint(0, len(galah_abund_data), 25000)
fract = Fractal(galah_abund_data[idx_random, :], verbose=True)
log_eps, log_c2 = fract.correlation_dimension(eps_steps=100)
# fit linear function to the result
idx_use = np.logical_and(log_eps >= -3., log_eps <= -0.5)
fit = np.polyfit(log_eps[idx_use], log_c2[idx_use], deg=1)
print 'Slope '+str(fit[0])

# Compute distance for all possible pairs of points
plt_xlim = (-4,2)
plt.plot(log_eps, log_c2)
plt.plot([plt_xlim[0], plt_xlim[1]], [plt_xlim[0]*fit[0]+fit[1], plt_xlim[1]*fit[0]+fit[1]])
plt.xlim(plt_xlim)
plt.ylim((-20,0))
plt.title('Correlation dimension with slope {:2.2f}.'.format(fit[0]))
plt.ylabel('log(C2(eps))')
plt.xlabel('log(eps)')
plt.tight_layout()
plt.savefig('corr_dim.png')
plt.close()
