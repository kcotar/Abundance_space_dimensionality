import os
import matplotlib
if os.environ.get('DISPLAY') is None:
    # enables figure saving on clusters with ssh conection
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np
import pandas as pd

from astropy.table import Table
from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis, FastICA, SparsePCA, KernelPCA, NMF
from sklearn.preprocessing import scale
from sklearn.cross_decomposition import CCA
from common_functions import *


def calculate_explained_variance(orig_data, transformed_data):
    covariance_orig = np.cov(np.transpose(orig_data))
    covariance_tran = np.cov(np.transpose(transformed_data))
    diag_orig = np.diag(covariance_orig)
    diag_tran = np.diag(covariance_tran)
    if len(diag_orig) > len(diag_tran):
        expl_var = diag_tran/np.sum(diag_orig) * 100.
    else:
        expl_var = diag_tran/np.sum(diag_tran) * 100.
    return expl_var


def add_curve(data, label=None, color=None, axes=None):
    y_data = np.hstack((0., np.cumsum(data)))
    x_data = np.arange(len(y_data))
    axes.plot(x_data, y_data, label=label, c=color)


def get_atom_numbers(labels, table=None):
    atom_num = []
    if table is None:
        print 'Periodic table not supplied'
    else:
        for label in labels:
            idx_elem = np.where(table['Symbol'] == label)
            if np.shape(idx_elem)[1] == 1:
                atom_num.append(table['Atomic_Number'][int(idx_elem[0])])
            else:
                atom_num.append(np.nan)
    return atom_num


def visualize_components(comp_data, labels, prefix=None):
    x_labels = [val.split('_')[0].capitalize() for val in labels]
    x_pos = np.arange(len(x_labels))
    bar_width = 0.5
    # add elements to the graph legend
    label_0 = mp.Patch(color='brown', label='Light odd-Z')
    label_1 = mp.Patch(color='green', label='Alpha elements')
    label_2 = mp.Patch(color='blue', label='Iron peak')
    label_3 = mp.Patch(color='red', label='Neutron capture')
    label_4 = mp.Patch(color='purple', label='Proton capture')
    # determine which elements should be coloured
    bar_colors = ['black'] * len(x_labels)
    for i_b in range(len(x_labels)):
        if x_labels[i_b] in light_oddz_el:
            bar_colors[i_b] = 'brown'
        if x_labels[i_b] in alpha_el:
            bar_colors[i_b] = 'green'
        if x_labels[i_b] in iron_peak_el:
            bar_colors[i_b] = 'blue'
        if x_labels[i_b] in neutron_el:
            bar_colors[i_b] = 'red'
        if x_labels[i_b] in proton_el:
            bar_colors[i_b] = 'purple'
    i_c = 1
    # reorder elements by its atomic numbers
    element_num = get_atom_numbers(x_labels, table=periodic_table)
    bar_order = np.argsort(element_num)
    # create plot for individual component
    for comp in comp_data:
        fig, ax = plt.subplots(1, 1)
        ax.bar(x_pos, comp[bar_order], width=bar_width, lw=0, color=np.array(bar_colors)[bar_order])
        ax.set(xticks=x_pos+bar_width/2., xticklabels=np.array(x_labels)[bar_order], ylabel=prefix.upper()+' weight',
               xlabel='Element', ylim=[-1, 1], xlim=[-1, len(x_labels)])
        ax.legend(handles=[label_0, label_1, label_2, label_3, label_4], loc='upper right', bbox_to_anchor=(1.42, 0.65))
        ax.grid()
        fig.savefig(prefix+'_component_weights_{:02}.png'.format(i_c), bbox_inches='tight')
        i_c += 1
    fig.clear()


# ----------------------------------------
# Read all data sets
# ----------------------------------------
print 'Reading data sets'
galah_data_dir = '/home/klemen/GALAH_data/'
cannon_data = Table.read(galah_data_dir+'sobject_iraf_cannon_1.2.fits')
# general_data = Table.read(galah_data_dir+'sobject_iraf_general_1.1.fits')
# param_data = Table.read(galah_data_dir+'sobject_iraf_param_1.1.fits')
periodic_table = pd.read_csv('periodic_table.csv')


# Z - atomic number, protons in nucleus aka its charge number
# A - mass number, protons and neutrons in nucleus aka its mass
# N - neutron number
# Light odd-Z elements
light_oddz_el = ['Na', 'Al', 'K', 'Sc']
# Aplha elements
# mainly produced by core-collapse supernovae (II, Ib, Ic)
alpha_el = ['C', 'O', 'Ne', 'Mg', 'Si', 'S', 'Ar', 'Ca']
# Iron peak elements
# mainly produced in Ia supernovae
iron_peak_el = ['Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni']
# Neutron capture elements
# elements with A ~> 65, above iron peak
neutron_el = ['Cu', 'Zn', 'Ga', 'Ge', 'As', 'Ba']  # includes s- and r- process elements
# Proton capture elements
# some elements between Selenium and Mercury ??
proton_el = []  # aka p- process elements

# ----------------------------------------
# Begin analysis
# ----------------------------------------
use_cols = get_abundance_cols(cannon_data.colnames)


# print 'Perform FEH analysis'
# feh_steps = np.arange(-2.5, 3, 0.25)
# idx_sorted = np.argsort(cannon_data['feh_cannon'])
# data_chunks = range(0, len(idx_sorted), 10000)
# n_steps = len(data_chunks)-1
# for id_step in range(n_steps):
#     print 'Step '+str(id_step)+' out of '+str(n_steps)
#     idx_use = idx_sorted[data_chunks[id_step]:data_chunks[id_step+1]]
#     # idx_use = np.logical_and(cannon_data['feh_cannon'] > feh_steps[id_step], cannon_data['feh_cannon'] <= feh_steps[id_step+1])
#     # if np.sum(idx_use) < 10:
#     #     continue
#     # print ' datapoints '+str(np.sum(idx_use))
#     use_dataset = np.array(cannon_data[use_cols][idx_use].to_pandas())
#     decomp = PCA(svd_solver='full', n_components=None, copy=True)
#     decomp.fit(np.array(cannon_data_norm[use_cols][idx_use].to_pandas()))
#     # decomp = FactorAnalysis(n_components=len(use_cols), copy=True, svd_method='lapack')
#     # decomp.fit(use_dataset)
#     color_perc = 1.0*id_step/n_steps
#     add_curve(calculate_explained_variance(use_dataset, decomp.transform(use_dataset)), color=(1-color_perc,0,color_perc)) #, label = '> {:.2f} & < {:.2f}'.format(feh_steps[id_step], feh_steps[id_step+1]))
# plt.xlabel('Number of components')
# plt.ylabel('Variance percentage')
# plt.grid()
# plt.xlim([0, len(use_cols)])
# plt.ylim([0,105])
# plt.legend(loc='center right', bbox_to_anchor=(1.3, 0.5))
# plt.tight_layout()
# plt.savefig('pca_feh_range.png', bbox_inches='tight')
# plt.close()

# abund_data = np.array(cannon_data_norm[use_cols].to_pandas())
abund_data = np.array(cannon_data[use_cols].to_pandas())
abund_data_norm = np.array(normalize_table(cannon_data, use_cols)[use_cols].to_pandas())
# OR for data standardization
# abund_data_norm = scale(abund_data)
abund_data_use = abund_data_norm

move_to_dir('Results_standardized')

fig, ax = plt.subplots(1, 1)

print 'Perform PCA analysis'
pca = PCA(svd_solver='full', n_components=len(use_cols), copy=True)
pca.fit(abund_data_use)
visualize_components(pca.components_, use_cols, prefix='pca')
add_curve(calculate_explained_variance(abund_data_use, pca.transform(abund_data_use)), label='PCA', axes=ax)

print 'Perform SDV analysis'
sdv = TruncatedSVD(n_components=len(use_cols)-1)
sdv.fit(abund_data_use)
add_curve(calculate_explained_variance(abund_data_use, sdv.transform(abund_data_use)), label='SDV', axes=ax)

print 'Perform Factorial analysis'
fa = FactorAnalysis(n_components=len(use_cols), copy=True, svd_method='lapack')
fa.fit(abund_data_use)
visualize_components(fa.components_, use_cols, prefix='fa')
add_curve(calculate_explained_variance(abund_data_use, fa.transform(abund_data_use)), label='Factor', axes=ax)

print 'Perform FastICA analysis'
fast_ica = FastICA(n_components=len(use_cols))
fast_ica.fit(abund_data_use)
add_curve(calculate_explained_variance(abund_data_use, fast_ica.transform(abund_data_use)), label='Fast ICA', axes=ax)

# print 'Perform NMF analysis'
# min_val = np.nanmin(abund_data)
# if min_val < 0:
#     min_val = np.abs(min_val)
# else:
#     min_val = 0
# nmf = NMF(n_components=len(use_cols), solver='cd')
# nmf.fit(abund_data+min_val)
# add_curve(calculate_explained_variance(abund_data+min_val, nmf.transform(abund_data+min_val)), label='NMF', axes=ax)


# !!! The following two should be used with reduce set of data or maybe lower number of jobs.
# !!! High usage of memory is expected.
#

# n_rand = 30000
# cpu_jobs = 2
# idx_random = np.random.randint(0, len(abund_data), n_rand)
# abund_data_use = abund_data[idx_random, :]
#
# # print 'Perform SparsePCA analysis'
# # sparse_pca = SparsePCA(n_components=len(use_cols), n_jobs=4)
# # sparse_pca.fit(abund_data_use)
# # add_curve(calculate_explained_variance(abund_data_use, sparse_pca.transform(abund_data_use)), label='SparsePCA', axes=ax)
#
# print 'Perform KernelPCA analysis'
# kernel_pca = KernelPCA(n_components=len(use_cols), n_jobs=cpu_jobs, copy_X=True, kernel='poly').fit(abund_data_use)
# add_curve(calculate_explained_variance(abund_data_use, kernel_pca.transform(abund_data_use)), label='KernelPCA - poly', axes=ax)
# kernel_pca = KernelPCA(n_components=len(use_cols), n_jobs=cpu_jobs, copy_X=True, kernel='linear').fit(abund_data_use)
# add_curve(calculate_explained_variance(abund_data_use, kernel_pca.transform(abund_data_use)), label='KernelPCA - lin', axes=ax)
# kernel_pca = KernelPCA(n_components=len(use_cols), n_jobs=cpu_jobs, copy_X=True, kernel='sigmoid').fit(abund_data_use)
# add_curve(calculate_explained_variance(abund_data_use, kernel_pca.transform(abund_data_use)), label='KernelPCA - sigmoid', axes=ax)
# kernel_pca = KernelPCA(n_components=len(use_cols), n_jobs=cpu_jobs, copy_X=True, kernel='cosine').fit(abund_data_use)
# add_curve(calculate_explained_variance(abund_data_use, kernel_pca.transform(abund_data_use)), label='KernelPCA - cosine', axes=ax)
# kernel_pca = KernelPCA(n_components=len(use_cols), n_jobs=cpu_jobs, copy_X=True, kernel='rbf').fit(abund_data_use)
# add_curve(calculate_explained_variance(abund_data_use, kernel_pca.transform(abund_data_use)), label='KernelPCA - rbf', axes=ax)

ax.set(xlabel='Number of components', ylabel='Variance percentage', xlim=[0, len(use_cols)], ylim=[0,105])
ax.legend(loc='center right', bbox_to_anchor=(1.3, 0.5))
ax.grid()
fig.savefig('component_analysis.png', bbox_inches='tight')
