import os
import matplotlib
if os.environ.get('DISPLAY') is None:
    # enables figure saving on clusters with ssh conection
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mp

import numpy as np
import pandas as pd

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

periodic_table = pd.read_csv('periodic_table.csv')


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


def visualize_components(comp_data, labels, prefix=None, y_range=(-1,1)):
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
               xlabel='Element', ylim=y_range, xlim=(-1, len(x_labels)))
        ax.legend(handles=[label_0, label_1, label_2, label_3, label_4], loc='upper right', bbox_to_anchor=(1.42, 0.65))
        ax.grid()
        fig.savefig(prefix+'_component_weights_{:02}.png'.format(i_c), bbox_inches='tight')
        plt.close(fig)
        i_c += 1