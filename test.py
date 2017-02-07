from analyse_functions import *
import pandas as pd

galah_data_dir = '/home/klemen/GALAH_data/'
linelist = pd.read_csv(galah_data_dir+'GALAH_Cannon_linelist.csv')

plots_for_element = 'Ba'
data = pd.read_csv('Similar_observations/'+plots_for_element+'_data.csv')
lines_wvl = linelist[linelist['Element']==plots_for_element]['line_centre'].values
plot_abundace_grid(data, prefix=plots_for_element+'_data', lines_wvl=lines_wvl)
