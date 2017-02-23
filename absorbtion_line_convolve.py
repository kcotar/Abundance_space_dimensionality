import os, glob
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

move_to_dir('Absorption_lines_data')
for element_abs_file in glob.glob('*_center_*_step_*_width_*.txt'):
    print 'Working on '+element_abs_file
    cur_abs_line = AbsorptionLine(read=True, filename=element_abs_file)
    print ' Data convolution'
    n_out = 3
    n_step = 3
    convolved_data = cur_abs_line.convolve_data(n_out=n_out, n_step=n_step)
    print ' Saving data'
    out_file = element_abs_file[:-4]+'_convol-step_'+str(n_out)+'.txt'
    np.savetxt(out_file, convolved_data, delimiter=',')


