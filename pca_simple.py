import numpy as np

from astropy.table import Table
from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis, FastICA, SparsePCA, KernelPCA, NMF
from sklearn.preprocessing import scale
from sklearn.cross_decomposition import CCA
from common_functions import *
from pca_simple_functions import *

# ----------------------------------------
# Read all data sets
# ----------------------------------------
print 'Reading data sets'
galah_data_dir = '/home/klemen/GALAH_data/'
cannon_data = Table.read(galah_data_dir+'sobject_iraf_cannon_1.2.fits')
# general_data = Table.read(galah_data_dir+'sobject_iraf_general_1.1.fits')
# param_data = Table.read(galah_data_dir+'sobject_iraf_param_1.1.fits')


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
