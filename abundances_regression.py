import os, imp, itertools
import numpy as np

import matplotlib
if os.environ.get('DISPLAY') is None:
    # enables figure saving on clusters with ssh connection
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astropy.table import Table
from sklearn.metrics import mean_squared_error

imp.load_source('helper', '../tSNE_test/helper_functions.py')
from helper import move_to_dir, get_abundance_cols, get_element_names
imp.load_source('data_normalization', '../Stellar_parameters_interpolator/data_normalization.py')
imp.load_source('class_wrapper', '../Stellar_parameters_interpolator/classifier.py')
from class_wrapper import *
from pca_simple_functions import visualize_components

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


galah_data_dir = '/home/klemen/GALAH_data/'  # the same for gigli and local pc
# --------------------------------------------------------
# ---------------- Read Data -----------------------------
# --------------------------------------------------------
# read GALAH data
# galah_general = Table.read(galah_data_dir+'sobject_iraf_general_1.1.fits')
galah_cannon = Table.read(galah_data_dir+'sobject_iraf_cannon_1.2.fits')

use_cols = get_abundance_cols(galah_cannon.colnames)
use_cols_elem = get_element_names(use_cols)


# regression of parameters using neural network
train_data = np.array(galah_cannon[use_cols].to_pandas())

abund_to_determine = 2
move_to_dir('Abundances_regression_'+str(abund_to_determine))

n_runs = 5
n_hidden_layers = range(2, 14-abund_to_determine)

# generate all possible combinations for regression
abund_reg_combs = list(itertools.combinations(use_cols, abund_to_determine))
results_rmse = np.ndarray((len(n_hidden_layers), n_runs, len(abund_reg_combs), abund_to_determine))

for i_run in range(n_runs):
        for i_a_c in range(len(abund_reg_combs)):
            abund_target = np.array(abund_reg_combs[i_a_c])
            idx_cols_target = [i_c for i_c in range(len(use_cols)) if use_cols[i_c] in abund_target]
            idx_cols_train = [i_c for i_c in range(len(use_cols)) if i_c not in idx_cols_target]
            subdir = '_'.join(abund_target)
            print 'Estimating combination '+subdir
            move_to_dir(subdir)
            for i_hid in range(len(n_hidden_layers)):
                n_hidden = n_hidden_layers[i_hid]
                move_to_dir('Nodes_' + str(n_hidden))

                class_obj = CLASSIFIER(train_data[:, idx_cols_train], train_data[:, idx_cols_target], inputs_norm='standardize', outputs_norm='standardize',
                                    method='regression', algorithm='ANN', ann_layers=(n_hidden, n_hidden))
                class_obj.train()
                print 'Estimating values'
                results_array = class_obj.get_label(train_data[:, idx_cols_train])
                results_shape = np.shape(results_array)
                if len(results_shape) == 1:
                    results_array = results_array.reshape(results_shape[0], 1)

                ann_weights = class_obj.model.coefs_
                ann_biases = class_obj.model.intercepts_
                ann_weights_layer1 = ann_weights[0]
                # print i_n in range(len(ann_weights_layer1)):

                # compute RMSE values
                for i_c in range(len(abund_target)):
                    results_rmse[i_hid, i_run, i_a_c, i_c] = np.sqrt(mean_squared_error(train_data[:,idx_cols_target[i_c]],
                                                                                        results_array[:,i_c]))
                # plot results
                for i_c in range(len(abund_target)):
                    plt_range = (np.nanmin(train_data[:,idx_cols_target[i_c]]), np.nanmax(train_data[:,idx_cols_target[i_c]]))
                    plt.plot([plt_range[0], plt_range[1]],[plt_range[0], plt_range[1]], c='black', lw=0.5)
                    plt.scatter(train_data[:,idx_cols_target[i_c]], results_array[:,i_c], lw=0, s=0.2, alpha=0.2)
                    plt.title('Regression of ' + abund_target[i_c])
                    plt.xlabel('Original')
                    plt.ylabel('Result')
                    plt.xlim(plt_range)
                    plt.ylim(plt_range)
                    plt.tight_layout()
                    plt.savefig(abund_target[i_c]+'_'+str(i_run)+'.png', dpi=300)
                    plt.close()

                # visualize_components(np.transpose(ann_weights_layer1), use_cols, prefix='', y_range=(-0.3, 0.3))
                os.chdir('..')
            os.chdir('..')

# visualize RMSE values
for i_a_c in range(len(abund_reg_combs)):
    abund_target = np.array(abund_reg_combs[i_a_c])
    idx_cols_target = [i_c for i_c in range(len(use_cols)) if use_cols[i_c] in abund_target]
    move_to_dir('_'.join(abund_target))
    for i_c in range(len(abund_target)):
        for i_run in range(n_runs):
            plt.plot(n_hidden_layers, results_rmse[:, i_run, i_a_c, i_c], c='blue', alpha=0.3)
        plt.title('RMSE for ' + abund_target[i_c])
        plt.xlabel('Number of nodes in hidden layers')
        plt.ylabel('RMSE')
        plt.tight_layout()
        plt.savefig('rmse_'+abund_target[i_c]+'.png')
        plt.close()
    os.chdir('..')
