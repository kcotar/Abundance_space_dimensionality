from analyse_functions import *
import pandas as pd

data = pd.read_csv('Similar_observations/K_data_test.csv')
plot_abundace_grid(data, prefix='K_data')
