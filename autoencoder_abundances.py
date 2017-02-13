import os, imp
import numpy as np
from astropy.table import Table

import matplotlib
if os.environ.get('DISPLAY') is None:
    # enables figure saving on clusters with ssh connection
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

imp.load_source('helper', '../tSNE_test/helper_functions.py')
from helper import move_to_dir, get_abundance_cols, get_element_names
imp.load_source('data_normalization', '../Stellar_parameters_interpolator/data_normalization.py')
import data_normalization
imp.load_source('class_wrapper', '../Stellar_parameters_interpolator/classifier.py')
from class_wrapper import *
from pca_simple_functions import visualize_components


galah_data_dir = '/home/klemen/GALAH_data/'  # the same for gigli and local pc
# --------------------------------------------------------
# ---------------- Read Data -----------------------------
# --------------------------------------------------------
# read GALAH data
galah_general = Table.read(galah_data_dir+'sobject_iraf_general_1.1.fits')
galah_cannon = Table.read(galah_data_dir+'sobject_iraf_cannon_1.2.fits')

use_cols = get_abundance_cols(galah_cannon.colnames)

# regression of parameters using neural network
train_data = np.array(galah_cannon[use_cols].to_pandas())
class_obj = CLASSIFIER(train_data, train_data, inputs_norm='standardize', outputs_norm='standardize',
                       method='regression', algorithm='ANN', ann_layers=(21,11,5,11,21))
class_obj.train()
print 'Estimating values'
results_array = class_obj.get_label(train_data)

ann_weights = class_obj.model.coefs_
ann_biases = class_obj.model.intercepts_
ann_weights_layer1 = ann_weights[0]
# print i_n in range(len(ann_weights_layer1)):

move_to_dir('Autoencoder_test')

# plot results
for i_c in range(len(use_cols)):
    plt_range = (np.nanmin(train_data[:, i_c]), np.nanmax(train_data[:, i_c]))
    plt.plot([plt_range[0], plt_range[1]],[plt_range[0], plt_range[1]], c='black', lw=0.5)
    plt.scatter(train_data[:, i_c], results_array[:, i_c], lw=0, s=0.2, alpha=0.3)
    plt.title('Autoencoder of ' + use_cols[i_c])
    plt.xlabel('Original')
    plt.ylabel('Result')
    plt.xlim(plt_range)
    plt.ylim(plt_range)
    plt.tight_layout()
    plt.savefig(use_cols[i_c]+'.png', dpi=300)
    plt.close()

visualize_components(np.transpose(ann_weights_layer1), use_cols, prefix='', y_range=(-0.3, 0.3))
