import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from itertools import combinations

from scipy.stats import norm, f, chi2

import warnings
warnings.filterwarnings("ignore")

import inspect
import tqdm

# store builtin print
old_print = print
def new_print(*args, **kwargs):
    # if tqdm.tqdm.write raises error, use builtin print
    try:
        tqdm.tqdm.write(*args, **kwargs)
    except:
        old_print(*args, ** kwargs)
# globaly replace print with new_print
inspect.builtins.print = new_print

import time
import pickle

def cpc(truth, prediction):
    '''
    truth : vector of true values
    prediction : vector of predicted values
    '''
    return 2 * sum( np.minimum( truth , prediction ) ) / ( sum( truth ) + sum( prediction ))

def cpcd(truth, prediction, dist):
    true_hist, pred_hist = np.array([]), np.array([])
    for i in range(int(np.ceil(dist.max()))):
        bin_dist = np.ma.masked_inside(dist, i, i+1) > 0
        true_hist = np.append( true_hist, sum( truth[ bin_dist ]))
        pred_hist = np.append( pred_hist, sum( prediction[ bin_dist ]))

    return 2 * sum( np.minimum( true_hist , pred_hist ) ) / ( sum( true_hist ) + sum( pred_hist ))

def mae(truth, prediction):
    return sum( abs(truth - prediction) ) / len(truth)

def r_squared(truth, prediction):
    return 1 - sum( (truth - prediction)**2 ) / sum( (truth - np.mean(truth))**2 )
flow = pd.read_excel('../data/state/state_flows.xls', sheet_name='flow', index_col=0).fillna(0)

flow_error = pd.read_excel('../data/state/state_flows.xls', sheet_name='flow-moe', index_col=0).fillna(0)
# extract standard error by dividing by z-score of 95th percentile
flow_error /= norm.ppf(0.95)

population = pd.read_excel('../data/state/state_pop.xls', sheet_name='States', index_col=0).T

housing = pd.read_csv('../data/state/state_house_index.csv', index_col=0).iloc[-18:-3]
income = pd.read_excel('../data/state/state_income.xls', sheet_name='Income', index_col=0).iloc[-17:-2]

distance = pd.read_csv('../data/state/state_distance.csv', index_col=0)
area = pd.read_excel('../data/state/State_Area.xls', index_col=0).drop('area (sq. mi)',axis=1)

affordability = pd.DataFrame( housing.values /income.values , index=housing.index, columns=housing.columns)

states = flow.columns
years = housing.index

population.index = years
income.index = years

# N = 10

# df = pd.DataFrame()

# for t in range(len(years)):
#     for i in range(len(states)):
#         for j in range(len(states)):
#             if i != j:
#                 df = df.append(pd.Series([2005 + t , states[i] , states[j] , 
#                                             flow.iloc[i + len(states)*t,j] , flow_error.iloc[i + len(states)*t,j] , 
#                                             population[states[i]].iloc[t] , population[states[j]].iloc[t] , 
#                                             housing[states[i]].iloc[t] , housing[states[j]].iloc[t] , 
#                                             income[states[i]].iloc[t] , income[states[j]].iloc[t] ,
#                                             affordability[states[i]].iloc[t] , affordability[states[j]].iloc[t] , 
#                                             distance.iloc[i,j]]).T, ignore_index=True)
# df.columns = ['Year', 'State_i', 'State_j', 'M_ij_mean', 'M_ij_sd', 'P_i', 'P_j', 'H_i', 'H_j', 'I_i' , 'I_j' , 'AF_i' , 'AF_j', 'D_ij']

# df['A_i'] = area.values[LabelEncoder().fit_transform(df.State_i.values)]
# df['A_j'] = area.values[LabelEncoder().fit_transform(df.State_j.values)]
# df['rho_i'] = df.P_i / df.A_i
# df['rho_j'] = df.P_j / df.A_j

# df['SP_ij'] = np.ndarray(len(df))
# df['SH_ij'] = np.ndarray(len(df))
# df['SI_ij'] = np.ndarray(len(df))
# df['SA_ij'] = np.ndarray(len(df))
# df['SAF_ij'] = np.ndarray(len(df))
# df['Srho_ij'] = np.ndarray(len(df))

# for t in range(len(years)):
#     for i in df[df['Year'] == years[t]].index:

#         intervening_states = states[distance[df.iloc[i]['State_i']] < distance[df.iloc[i]['State_i']][df.iloc[i]['State_j']] ].drop(df.iloc[i]['State_i'])

#         df.iloc[i, df.columns.get_loc('SP_ij')] = sum(population.iloc[t][intervening_states])

#         df.iloc[i, df.columns.get_loc('SH_ij')] = sum(housing.iloc[t][intervening_states])

#         df.iloc[i, df.columns.get_loc('SI_ij')] = sum(income.iloc[t][intervening_states])

#         df.iloc[i, df.columns.get_loc('SA_ij')] = sum(area.loc[intervening_states].values)

#         df.iloc[i, df.columns.get_loc('SAF_ij')] = df.iloc[i, df.columns.get_loc('SH_ij')] / df.iloc[i, df.columns.get_loc('SI_ij')]  \
#                                                             if df.iloc[i, df.columns.get_loc('SI_ij')] else 0

#         df.iloc[i, df.columns.get_loc('Srho_ij')] = df.iloc[i, df.columns.get_loc('SP_ij')] /  df.iloc[i, df.columns.get_loc('SA_ij')] \
#                                                             if df.iloc[i, df.columns.get_loc('SA_ij')] else 0   


# # sample paths of flows
# np.random.seed(0)

# observations = np.random.normal(  np.repeat(df['M_ij_mean'], N)  ,  np.repeat(df['M_ij_sd'], N)).astype('int')
# observations[observations < 0] = 0

# df = pd.DataFrame(np.repeat(df.values, N, axis=0), columns=df.columns)
# df.insert(df.columns.get_loc('P_i'), 'M_ij', observations)

# df.insert(df.columns.get_loc('M_ij'), 'path_ind', np.tile(np.arange(N), len(years)*len(states)*(len(states)-1)))

# df.loc[:,'M_ij':] = df.loc[:,'M_ij':].astype(float)

# df.insert(df.columns.get_loc('path_ind'), 'State_pair' , df.State_i + ' - ' + df.State_j)
# df.to_csv('../data/state/state_flow_samples.csv')




# join with dataset with additional climate-related features
df = pd.concat([pd.read_csv('../data/state/state_flow_samples.csv', index_col='Year').drop('Unnamed: 0', axis=1), 
                pd.read_csv('../data/state/state_flow_samples_climate.csv', index_col=0).sort_values(['Year','State_pair'])[['ADC_i','ADC_j']]], axis=1)

test_years = list(years[-3:])

df_train = df.query("Year not in @test_years")
df_test = df.query("Year in @test_years")

# critical value for significance of feature weights
# critical_value = chi2.ppf(0.95, df=1)

# train_len_repeat = 20
# test_len_repeat = 5

# sample flows for training data
# np.random.seed(0)

# observations = np.random.normal(  np.repeat(df_train['M_ij_mean'], train_len_repeat)  ,  np.repeat(df_train['se(M_ij)'], train_len_repeat)).astype('int')
# observations[observations < 0] = 0
# df_train = pd.DataFrame(np.repeat(df_train.values, train_len_repeat, axis=0), columns=df_train.columns)
# df_train.insert(df_train.columns.get_loc('P_i'), 'M_ij', observations)

# observations = np.random.normal(  np.repeat(df_test['M_ij_mean'], test_len_repeat)  ,  np.repeat(df_test['se(M_ij)'], test_len_repeat)).astype('int')
# observations[observations < 0] = 0
# df_test = pd.DataFrame(np.repeat(df_test.values, test_len_repeat, axis=0), columns=df_test.columns)
# df_test.insert(df_test.columns.get_loc('P_i'), 'M_ij', observations)