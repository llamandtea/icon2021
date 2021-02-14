import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import cluster
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
import os


import bnlearn as bn


data_frame = pd.read_csv("..\DataSets\Ordinale3.csv")

#cnode = pomegranate.distributions.ConditionalProbabilityTable.from_samples(data_frame.iloc[:10, :])

#out_bayes_network = pomegranate.BayesianNetwork.discrete_extract_with_constraints(data_frame.iloc[:10, :])
df_short = data_frame.iloc[:4500, 1:]
#df_short["IsCanceled"] = df_short["IsCanceled"].apply(lambda x: "True" if x>0 else "False")
#df_short["IsRepeatedGuest"] = df_short["IsRepeatedGuest"].apply(lambda x: "True" if x>0 else "False")
#df_short = data_frame.iloc[:10, 2:4]

#print(df_short)



# Pre-processing of the input dataset
#dfhot, dfnum = bn.df2onehot(df_short)

# Structure learning
#DAG = bn.structure_learning.fit(dfnum)

DAG_filtered = bn.structure_learning.fit(df_short, methodtype='hc', bw_list_method='filter', black_list=['DistributionChannel', 'Meal', 'WasInWaitingList', 'TotalOfSpecialRequests',
 'BookingChanges', 'Minors', 'MarketSegment', 'Adults'])
#DAG_filtered = bn.structure_learning.fit(dfnum, methodtype='hc')
DAG_param_filtered = bn.parameter_learning.fit(DAG_filtered, df_short)
#bn.reverse.arc(DAG_filtered, 'ADR', 'Season')

Gf = bn.plot(DAG_filtered)


#model = bn.structure_learning.fit(df_short)
#G = bn.plot(model)



