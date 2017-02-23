import os, imp
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
galah_general = Table.read(galah_data_dir+'sobject_iraf_general_1.1.fits')
galah_cannon = Table.read(galah_data_dir+'sobject_iraf_cannon_1.2.fits')

move_to_dir('Autoencoder_test_scipy_17_n_17')

use_cols = get_abundance_cols(galah_cannon.colnames)
use_cols_elem = get_element_names(use_cols)

# regression of parameters using neural network
train_data = np.array(galah_cannon[use_cols].to_pandas())

n_runs = 5
n_hidden_layers = range(2, 14)
results_rmse = np.ndarray((len(n_hidden_layers), len(use_cols), n_runs))

for i_run in range(n_runs):
    for i_hid in range(len(n_hidden_layers)):
        n_hidden = n_hidden_layers[i_hid]
        move_to_dir('{:02.0f}'.format(n_hidden))

        class_obj = CLASSIFIER(train_data, train_data, inputs_norm='standardize', outputs_norm='standardize',
                               method='regression', algorithm='ANN', ann_layers=(17,n_hidden,17))
        class_obj.train()
        print 'Estimating values'
        results_array = class_obj.get_label(train_data)

        ann_weights = class_obj.model.coefs_
        ann_biases = class_obj.model.intercepts_
        ann_weights_layer1 = ann_weights[0]
        # print i_n in range(len(ann_weights_layer1)):

        # model = Sequential()
        # model.add(Dense(n_hidden, input_dim=13, activation='sigmoid'))
        # # model.add(Dropout(0.5))

        # model.add(Dense(13, activation='sigmoid'))
        # model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        # model.fit(train_data, train_data, nb_epoch=30, batch_size=16384)
        # results_array = model.predict(train_data)

        # compute RMSE values
        for i_c in range(len(use_cols)):
            results_rmse[i_hid, i_c, i_run] = np.sqrt(mean_squared_error(train_data[:, i_c], results_array[:, i_c]))

        # plot results
        for i_c in range(len(use_cols)):
            plt_range = (np.nanmin(train_data[:, i_c]), np.nanmax(train_data[:, i_c]))
            plt.plot([plt_range[0], plt_range[1]],[plt_range[0], plt_range[1]], c='black', lw=0.5)
            plt.scatter(train_data[:, i_c], results_array[:, i_c], lw=0, s=0.2, alpha=0.2)
            plt.title('Autoencoder of ' + use_cols[i_c])
            plt.xlabel('Original')
            plt.ylabel('Result')
            plt.xlim(plt_range)
            plt.ylim(plt_range)
            plt.tight_layout()
            plt.savefig(use_cols[i_c]+'_'+str(i_run)+'.png', dpi=300)
            plt.close()

        # visualize_components(np.transpose(ann_weights_layer1), use_cols, prefix='', y_range=(-0.3, 0.3))

        os.chdir('..')

# visualize rmse values
fig, ax = plt.subplots(4, 4, figsize=(12, 12))
for i_c in range(len(use_cols)):
    x_sub = i_c % 4
    y_sub = np.int8(np.floor(i_c / 4))
    plot_pos = (y_sub, x_sub)
    ax[plot_pos].set_title('RMSE for ' + use_cols_elem[i_c])
    for i_run in range(n_runs):
        ax[plot_pos].plot(n_hidden_layers, results_rmse[:, i_c, i_run], c='blue', alpha=0.3)
    if y_sub == 3:
        ax[plot_pos].set_xlabel('Number of nodes in hidden layer')
    if x_sub == 0:
        ax[plot_pos].set_ylabel('RMSE')
fig.tight_layout()
fig.savefig('rmse_autoencoder_all.png', dpi=300)
plt.close(fig)
