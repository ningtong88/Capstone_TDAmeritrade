# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import wrds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import timeit
from arch.bootstrap import CircularBlockBootstrap, optimal_block_length
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from math import log
from scipy.stats import boxcox
from os import getenv, path
from pathlib import Path


# %%



# %%
d


# %%
def sim_returns(data, block_size = 20, total_sim = 10, random_seed = 1):
    ret_index = [x for x, y in enumerate(d.columns) if y[1] == 'ret']
    ret_sim = data.iloc[:,ret_index].to_numpy()
    rs = np.random.RandomState(random_seed)
    ret_sim_mod = CircularBlockBootstrap(block_size, ret_sim, random_state = rs)
    col = ret_sim.shape[1]
    sim = np.zeros((total_sim,len(data), col))
    count = 0
    for y in ret_sim_mod.bootstrap(total_sim):
        sim[count,:,:] = y[0][0]
        count +=1
    return sim


# %%
with open('low_var_port.pkl', 'rb') as f: d = pickle.load(f)
d = d.iloc[:,[1,5,9]]
d.iloc[0] = [1,1,1]
portfolio_weights = {'SPY':.5, 'TLT': .4, 'GLD': .1}
initial_money = 1000000
rebalance_days = 4500
threshold = .05


# %%
d


# %%
#Ning's rewrite
total_sim = 100
col = 2
bootstrap = sim_returns(d,block_size = 20, total_sim = total_sim, random_seed = 1)
arr_value = np.zeros((total_sim,len(d), col))
counter = 0
for x in bootstrap:
    x[0] = [1,1,1]
    portfolio_weights = {'SPY':.5, 'TLT': .4, 'GLD': .1}
    initial_money = 1000000
    rebalance_days = 1
    threshold = 10


    arr_data = x

    ls_tickers = []
    temp_tickers = list([y[0] for x, y in enumerate(arr_data_columns)])
    for i in temp_tickers:
        if i not in ls_tickers:
            ls_tickers.append(i)

    arr_port_det = np.zeros([d.shape[0], 9]) #Need to change back to data
    arr_port_sum = np.zeros([d.shape[0], 2]) #Need to change back to data
    arr_portfolio_weights = np.array([i for i in portfolio_weights.values()])

    #This loop goes through the data set "d", which should be a daily time series of asset returns and prices.
    #Returns should be broken into returns including dividends, returns without dividends, and dividend only returns. 
    cum_ret={}
    for k in range(len(ls_ret_index)):
        cum_ret[k]=arr_data[:,ls_ret_index[k]].cumprod()

    cum_ret_1= pd.DataFrame(data=cum_ret)
    cum_ret_1.columns = ls_tickers
    cum_ret_1.columns = [str(col) + '_cum_ret' for col in cum_ret_1.columns]
    num_rows, num_cols = arr_data.shape
    ls_cum_ret_index = [x for x, y in enumerate(cum_ret_1)]
    ls_cum_ret_index=[x+num_cols for x in ls_cum_ret_index]
    cum_ret_2 = np.array(cum_ret_1)

    #Since the cum_ret_2 keeps track of the cumulative returns and when we rebalance, the cumulative return
    #needs to be reset to 1, so create an array to keep track the cumulative returns right before rebalancing,
    #and this will be used to reset the cumulative returns.

    cum_ret_tracking = np.array(cum_ret_1.iloc[1])

    cum_ret_tracking[:]=1

    trading_day_counter=1
    count = 0
    initial_arr_asset_val = initial_money*arr_portfolio_weights

    for row in cum_ret_2:
        tradeflag = 0
        arr_rebal = np.zeros(len(ls_tickers))
        arr_latest_ret = row

        cur_asset_val = initial_arr_asset_val * arr_latest_ret/cum_ret_tracking
        total_value = np.sum(cur_asset_val)
        arr_actual_weights = cur_asset_val/total_value
        dev_weights = np.absolute((arr_actual_weights/arr_portfolio_weights)-1)

        if trading_day_counter % rebalance_days == 0 or np.amax(dev_weights) > threshold:

            tradeflag = 1
            cum_ret_tracking = arr_latest_ret

            arr_new_port_det = np.array([], dtype='f8')
          
            cur_asset_val = total_value*arr_portfolio_weights
      
            initial_arr_asset_val = cur_asset_val
            trading_day_counter=1
        else:
            trading_day_counter += 1    

        #Reaggregates all the data into the dataframe "portfolio_detail" and "portfolio_sum"
        #total_value

        arr_new_port_det = np.array([], dtype='f8')
        ls_new_port_det = []
        for i in range(len(ls_tickers)):

            ls_new_port_det = ls_new_port_det + [row[ls_ret_index[i]], arr_actual_weights[i], cur_asset_val[i]]

        arr_port_det[count] = ls_new_port_det

        #arr_new_port_sum = [total_value, total_asset_value, arr_cash[-1].item(), arr_cash[-1].item()/(total_value)]
        arr_new_port_sum = [total_value]

        arr_port_sum[count] = arr_new_port_sum
        tradeflag = 0
        count += 1
    arr_value[counter] = arr_port_sum
    counter +=1
    


# %%
arr_value


# %%
import random

b = list(range(4035))
a = [arr_value[i][:,0] for i in range(total_sim)]

rand = lambda: random.randint(0, 255)
fig = plt.figure(figsize=(10,7.5))
ax = fig.add_subplot(111)
for ydata in a:
    clr = '#%02X%02X%02X' % (rand(),rand(),rand())
    plot, = ax.plot(b, ydata, color=clr)


# %%

a = [arr_value[i][:,0] for i in range(total_sim)]
a


# %%
arr_port_det = np.zeros([d.shape[0], 9]) #Need to change back to data
arr_port_sum = np.zeros([d.shape[0], 2]) #Need to change back to data
arr_portfolio_weights = np.array([i for i in portfolio_weights.values()])

#This loop goes through the data set "d", which should be a daily time series of asset returns and prices.
#Returns should be broken into returns including dividends, returns without dividends, and dividend only returns. 
cum_ret={}
for k in range(len(ls_ret_index)):
    cum_ret[k]=d[:,ls_ret_index[k]].cumprod()

cum_ret_1= pd.DataFrame(data=cum_ret)
cum_ret_1.columns = ls_tickers
cum_ret_1.columns = [str(col) + '_cum_ret' for col in cum_ret_1.columns]
num_rows, num_cols = arr_data.shape
ls_cum_ret_index = [x for x, y in enumerate(cum_ret_1)]
ls_cum_ret_index=[x+num_cols for x in ls_cum_ret_index]
cum_ret_2 = np.array(cum_ret_1)

#Since the cum_ret_2 keeps track of the cumulative returns and when we rebalance, the cumulative return
#needs to be reset to 1, so create an array to keep track the cumulative returns right before rebalancing,
#and this will be used to reset the cumulative returns.

cum_ret_tracking = np.array(cum_ret_1.iloc[1])

cum_ret_tracking[:]=1

trading_day_counter=1
count = 0
initial_arr_asset_val = initial_money*arr_portfolio_weights

for row in cum_ret_2:
    tradeflag = 0
    arr_rebal = np.zeros(len(ls_tickers))
    arr_latest_ret = row

    cur_asset_val = initial_arr_asset_val * arr_latest_ret/cum_ret_tracking
    total_value = np.sum(cur_asset_val)
    arr_actual_weights = cur_asset_val/total_value
    dev_weights = np.absolute((arr_actual_weights/arr_portfolio_weights)-1)

    if trading_day_counter % rebalance_days == 0 or np.amax(dev_weights) > threshold:

        tradeflag = 1
        cum_ret_tracking = arr_latest_ret

        arr_new_port_det = np.array([], dtype='f8')
        print(trading_day_counter)
        cur_asset_val = total_value*arr_portfolio_weights
        print(cur_asset_val)
        initial_arr_asset_val = cur_asset_val
        trading_day_counter=1
    else:
        trading_day_counter += 1    

    #Reaggregates all the data into the dataframe "portfolio_detail" and "portfolio_sum"
    #total_value

    arr_new_port_det = np.array([], dtype='f8')
    ls_new_port_det = []
    for i in range(len(ls_tickers)):

        ls_new_port_det = ls_new_port_det + [row[ls_ret_index[i]], arr_actual_weights[i], cur_asset_val[i]]

    arr_port_det[count] = ls_new_port_det

    #arr_new_port_sum = [total_value, total_asset_value, arr_cash[-1].item(), arr_cash[-1].item()/(total_value)]
    arr_new_port_sum = [total_value]

    arr_port_sum[count] = arr_new_port_sum
    tradeflag = 0
    count += 1


# %%
df_portfolio_sum = pd.DataFrame(arr_port_sum, index = d.index)


# %%
df_portfolio_detail = pd.DataFrame(arr_port_det, index = d.index)


# %%
df_portfolio_detail.head(50)


# %%
cum_ret_1


