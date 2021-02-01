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


# %%
with open('low_var_port.pkl', 'rb') as f: d = pickle.load(f)

# %%
portfolio_weights = {'SPY':.5, 'TLT': .4, 'GLD': .1}

# %% [markdown]
# # Allocate shares

# %%
'''
Distributes shares among a discrete allocated portfolio based on initial price, money and portfolio weights of the stock
Inputs:
    1) data - Dataframe of assets with price multiindexed column names with the highest level being the tickers and the next level
    being prc | ret | retx | retd.
    2) initial_money - Initial money to be used to purchase assets
    3) portfolio_weights - Dictionary of {tickers : weight or asset} e.g. {'spy':0.5, 'tlt': 0.4, "gld": 0.1}
'''

def allocate_shares(data, portfolio_weights,initial_money = 10000000):
    latest_prices = {}
    date_start = data.index[0].date().isoformat()
    for k in portfolio_weights.keys():
        latest_prices[k] = data.loc[date_start][k]['prc']
        srs_latest_prices = pd.Series(latest_prices)
        
    da = DiscreteAllocation(portfolio_weights, srs_latest_prices, total_portfolio_value=initial_money)
    shares, cash = da.lp_portfolio()
    return shares, cash, srs_latest_prices, date_start
# %%
allocate_shares(d,portfolio_weights)
# %%
'''
Distributes shares among a continuous allocated portfolio based on initial price, money and portfolio weights of the stock
Inputs:
    1) data - Dataframe of assets with price multiindexed column names with the highest level being the tickers and the next level
    being prc | ret | retx | retd.
    2) initial_money - Initial money to be used to purchase assets
    3) portfolio_weights - Dictionary of {tickers : weight or asset} e.g. {'spy':0.5, 'tlt': 0.4, "gld": 0.1}
    
Outputs: 
    1) return shares
    2) latest_prices
    3) date_start
'''
#I think it might be enough to use this for allocating
def allocate_shares_continuous(data, portfolio_weights,initial_money = 10000000):
    latest_prices = {}
    shares = {}
    date_start = data.index[1].date().isoformat()
    for k in portfolio_weights.keys():
        latest_prices[k] = data.loc[date_start][k]['prc']
        shares[k] = initial_money*portfolio_weights[k]/latest_prices[k]
    return shares, latest_prices, date_start

# %%
d.index[1].date().isoformat()
# %%
allocate_shares_continuous(d,portfolio_weights)

# %%
#general class for allocate share
class allocate_shares_generator:
    def __init__(self,data,portfolio_weights,initial_money=10000000):
        self.data = data
        self.portfolio_weights=portfolio_weights
        self.initial_money=initial_money
        self.latest_prices = {}
    
    def allocate_shares(self):        
        date_start = self.data.index[0].date().isoformat()
        for k in self.portfolio_weights.keys():
            self.latest_prices[k] = self.data.loc[date_start][k]['prc']
            srs_latest_prices = pd.Series(self.latest_prices)        
        da = DiscreteAllocation(self.portfolio_weights, srs_latest_prices, total_portfolio_value=self.initial_money)
        shares, cash = da.lp_portfolio()
        return shares, cash, srs_latest_prices, date_start

#create sub class for continous
class allocate_shares_sub(allocate_shares_generator):
    def allocate_shares(self):
        shares = {}
        date_start = self.data.index[1].date().isoformat()
        for k in self.portfolio_weights.keys():
            self.latest_prices[k] = self.data.loc[date_start][k]['prc']
            shares[k] = self.initial_money*self.portfolio_weights[k]/self.latest_prices[k]
        return shares, self.latest_prices, date_start

# %%
allocate_shares_discrete = allocate_shares_generator(d,portfolio_weights).allocate_shares()
allocate_shares_discrete
# %%
allocate_shares_continuous = allocate_shares_sub(d,portfolio_weights).allocate_shares()
allocate_shares_continuous

# %% [markdown]
# # Set up dataframe to track trades

# %%
'''
Initializes dataframes used for discrete asset allocation rebalancing, along with assets on the first day. These will be used during rebalancing to figure out the number of assets
and corresponding columns to values. The dataframes are then used when converting arrays to dataframes to establish
columns and indicies.

Inputs:
    1) shares - dictionary of the quantity of each share.
    2) cash - Money leftover or in cash reserves
    3) latest_prices - The prices on the first day

Outputs:
    1) df_trades_initial
    2) df_cash_initial 
    3) df_portfolio_sum_initial
    4) df_portfolio_detail_initial
'''


def initialize_df(data, portfolio_weights,initial_money=1000000):
    #we might use allocate_shares_continuous
    shares, cash, srs_latest_prices, date_start = allocate_shares(data, portfolio_weights,initial_money)
    ls_trades = []
    ls_tickers = []
    ls_labels = ['shr chg', 'prc', 'csh chg', 'tot shr', 'tot bal', 'divflag', 'rebalflag']
    for k in srs_latest_prices.index:
        ls_trades = ls_trades + [shares[k], srs_latest_prices[k], -shares[k]*srs_latest_prices[k], shares[k], shares[k]*srs_latest_prices[k],0, 0]
        ls_tickers.append(k)
    ls_trades_iterables = [ls_tickers, ls_labels]
    trades_columns = pd.MultiIndex.from_product(ls_trades_iterables, names=['ticker', 'metric'])
    #DataFrame requires [[list]] to make sure they know it is row rather than column
    df_trades_initial = pd.DataFrame(np.array([ls_trades]),columns=trades_columns, index = [date_start])
    #Initialize df_cash dataframe
    df_cash_initial = pd.DataFrame(cash, columns = ['cash'], index = [date_start])
    
    #Initialize df_portfolio_detail dataframe
    ls_port_det_int_labels = ['price', 'return', 'weight', 'shares', 'value']
    ls_port_det_int_iterables = [ls_tickers, ls_port_det_int_labels]
    columns = pd.MultiIndex.from_product(ls_port_det_int_iterables, names=['ticker', 'metric'])
    df_portfolio_detail_initial = pd.DataFrame(columns=columns)

    #Initialize df_portfolio_sum dataframe
    ls_port_sum_labels = ['total_value', 'asset_value', 'cash', 'cash_pct']
    df_portfolio_sum_initial = pd.DataFrame(columns=ls_port_sum_labels)
    return df_trades_initial, df_cash_initial, df_portfolio_sum_initial, df_portfolio_detail_initial
# %%
initialize_df(d,portfolio_weights)
# %%
'''
Initializes dataframes used for continuous asset allocation rebalancing, along with assets on the first day. These will be used during rebalancing to figure out the number of assets
and corresponding columns to values. The dataframes are then used when converting arrays to dataframes to establish
columns and indicies.

Inputs:
    1) shares - dictionary of the quantity of each share.
    2) cash - Money leftover or in cash reserves
    3) latest_prices - The prices on the first day

Outputs:
    1) df_trades_initial
    2) df_cash_initial 
    3) df_portfolio_sum_initial
    4) df_portfolio_detail_initial
'''

def initialize_df_continuous(data, portfolio_weights,initial_money=1000000):
    ls_trades = []
    ls_tickers = []
    ls_labels = ['shr chg', 'prc', 'csh chg', 'tot shr', 'tot bal', 'divflag', 'rebalflag']
    shares, latest_prices, date_start = allocate_shares_continuous(data, portfolio_weights,initial_money)
    for k in latest_prices.keys():
        ls_trades = ls_trades + [shares[k], latest_prices[k], -shares[k]*latest_prices[k], shares[k], shares[k]*latest_prices[k],0, 0]
        ls_tickers.append(k)
    ls_trades_iterables = [ls_tickers, ls_labels]
    trades_columns = pd.MultiIndex.from_product(ls_trades_iterables, names=['ticker', 'metric'])
    df_trades_initial = pd.DataFrame(np.array([ls_trades]),columns=trades_columns, index = [date_start])
    
    #Initialize df_cash dataframe
    df_cash_initial = pd.DataFrame(np.array([0]), columns = ['cash'], index = [date_start])
    
    #Initialize df_portfolio_detail dataframe
    ls_port_det_int_labels = ['price', 'return', 'weight', 'shares', 'value']
    ls_port_det_int_iterables = [ls_tickers, ls_port_det_int_labels]
    columns = pd.MultiIndex.from_product(ls_port_det_int_iterables, names=['ticker', 'metric'])
    df_portfolio_detail_initial = pd.DataFrame(columns=columns)

    #Initialize df_portfolio_sum dataframe
    ls_port_sum_labels = ['total_value', 'asset_value', 'cash', 'cash_pct']
    df_portfolio_sum_initial = pd.DataFrame(columns=ls_port_sum_labels)
    return df_trades_initial, df_cash_initial, df_portfolio_sum_initial, df_portfolio_detail_initial

# %%
initialize_df_continuous(d,portfolio_weights)

# %%
class initialize_df_continuous_generator:
    def __init__(self, data, portfolio_weights, initial_money=1000000):
        self.ls_trades = []
        self.ls_tickers = []
        self.ls_labels = ['shr chg', 'prc', 'csh chg', 'tot shr', 'tot bal', 'divflag', 'rebalflag']
        self.data = data
        self.portfolio_weights = portfolio_weights
        self.initial_money = initial_money
        self.shares, self.latest_prices, self.date_start = allocate_shares_continuous(self.data,self.portfolio_weights,self.initial_money)
        self.df_cash_initial = pd.DataFrame(np.array([0]), columns = ['cash'], index = [self.date_start])
        for self.k in self.latest_prices.keys():
            self.ls_trades = self.ls_trades + [self.shares[self.k], self.latest_prices[self.k], -self.shares[self.k]*self.latest_prices[self.k], self.shares[self.k], self.shares[self.k]*self.latest_prices[self.k],0, 0]
            self.ls_tickers.append(self.k)
        ls_trades_iterables = [self.ls_tickers, self.ls_labels]
        trades_columns = pd.MultiIndex.from_product(ls_trades_iterables, names=['ticker', 'metric'])
        self.df_trades_initial = pd.DataFrame(np.array([self.ls_trades]),columns=trades_columns, index = [self.date_start])
        self.ls_port_det_int_labels = ['price', 'return', 'weight', 'shares', 'value']
        self.ls_port_det_int_iterables = [self.ls_tickers, self.ls_port_det_int_labels]
        columns = pd.MultiIndex.from_product(self.ls_port_det_int_iterables, names=['ticker', 'metric'])
        self.df_portfolio_detail_initial = pd.DataFrame(columns=columns)
        #Initialize df_portfolio_sum dataframe
        self.ls_port_sum_labels = ['total_value', 'asset_value', 'cash', 'cash_pct']
        self.df_portfolio_sum_initial = pd.DataFrame(columns=self.ls_port_sum_labels)

# %%
class initialize_df_dicrete_generator(initialize_df_continuous_generator):
    def __init__(self, data, portfolio_weights,initial_money=1000000):
        super().__init__(data,portfolio_weights)
        self.ls_trades = []
        self.ls_tickers = []
        self.shares, self.cash, self.srs_latest_prices, self.date_start = allocate_shares(self.data,self.portfolio_weights,self.initial_money)
        self.df_cash_initial = pd.DataFrame(self.cash, columns = ['cash'], index = [self.date_start])
        for self.k in self.srs_latest_prices.index:
            self.ls_trades = self.ls_trades + [self.shares[self.k], self.srs_latest_prices[self.k], -self.shares[self.k]*self.srs_latest_prices[self.k], self.shares[self.k], self.shares[self.k]*self.srs_latest_prices[self.k],0, 0]
            self.ls_tickers.append(self.k)
        self.ls_trades_iterables = [self.ls_tickers, self.ls_labels]~
        self.trades_columns = pd.MultiIndex.from_product(self.ls_trades_iterables, names=['ticker', 'metric'])
        #DataFrame requires [[list]] to make sure they know it is row rather than column
        self.df_trades_initial = pd.DataFrame(np.array([self.ls_trades]),columns=self.trades_columns, index = [self.date_start])
# %%
x = initialize_df_continuous_generator(d,portfolio_weights)
print (x.df_trades_initial, x.df_cash_initial, x.df_portfolio_sum_initial, x.df_portfolio_detail_initial)

# %%
y = initialize_df_dicrete_generator(d, portfolio_weights)
print (y.df_trades_initial, y.df_cash_initial, y.df_portfolio_sum_initial, y.df_portfolio_detail_initial)

# %% [markdown]
# # Functions to track dividend payments and rebalance using discrete assets

# %%
def calendar_threshold_rebalance_discrete(data, portfolio_weights, initial_money=1000000, rebalance_days = 63, threshold = 1):
    trading_day_counter = 1
    df_trades, df_cash, df_portfolio_sum, df_portfolio_detail = initialize_df(data, portfolio_weights,initial_money= 1000000)
    arr_trades = np.append([[0]], np.array(df_trades), axis = 1)
    date_start = data.index[0].date().isoformat()
    arr_data_columns = data.columns.values
    arr_trades_columns = df_trades.columns.values
    arr_data = np.array(data)
    arr_cash = np.array(df_cash)
    arr_port_det = np.zeros([data.shape[0], df_portfolio_detail.shape[1]])
    arr_port_sum = np.zeros([data.shape[0], df_portfolio_sum.shape[1]])
    arr_portfolio_weights = np.array([i for i in portfolio_weights.values()])
    ls_tickers = list(set([y[0] for x, y in enumerate(arr_data_columns)]))
    ls_prc_index = [x for x, y in enumerate(arr_data_columns) if y[1] == 'prc']
    ls_ret_index = [x for x, y in enumerate(arr_data_columns) if y[1] == 'ret']
    ls_retx_index = [x for x, y in enumerate(arr_data_columns) if y[1] == 'retx']
    ls_retd_index = [x for x, y in enumerate(arr_data_columns) if y[1] == 'retd']

    ls_trades_shr_index = [x+1 for x, y in enumerate(arr_trades_columns) if y[1] == 'tot shr']

    arr_yesterday_prices =  arr_data[0][ls_prc_index]
    cash = df_cash.iloc[0][0]
    #This loop goes through the data set "d", which should be a daily time series of asset returns and prices.
    #Returns should be broken into returns including dividends, returns without dividends, and dividend only returns. 
    arr_cur_shrs = np.array(arr_trades[-1][ls_trades_shr_index])
    count = 0
    for row in arr_data:
        arr_leftover_cash = 0
        tradeflag = 0
        arr_divflag = np.zeros(len(ls_tickers))
        arr_div_cash = np.zeros(len(ls_tickers))
        arr_rebal = np.zeros(len(ls_tickers))
        arr_new_shares = np.zeros(len(ls_tickers))
        arr_latest_prices = row[ls_prc_index]    
        #Checks to see if there are any dividend payouts, and if so, calculates the number of shares that are purchased as part of 
        #the reinvestment and the leftover cash if a full share cannot be purchased. Dividend payouts are reinvested into the asset
        #that payed the dividend
        
        if np.any(row[ls_retd_index]>0):
            arr_divflag = (row[ls_retd_index]>0).astype(int)
            add_cash = arr_cash[-1]/sum(arr_divflag)
            arr_add_cash = arr_divflag*add_cash
            arr_div_cash = (arr_yesterday_prices*arr_trades[-1][ls_trades_shr_index]*row[ls_retd_index])+arr_add_cash
            arr_new_shares = np.floor(arr_div_cash/arr_latest_prices)
            arr_leftover_cash = arr_div_cash-(arr_new_shares*arr_latest_prices)
            
        #Executes the actual trade, recording it in the df_trades dataframe and updates the df_cash dataframe. This is done before
        #rebalancing incase a) rebalancing happens on the same day in the case of calendar rebalancing, or b) the dividend payout
        #puts the asset over the threshold.
        
        if 1 in arr_divflag:

            arr_div_flag_add = np.array([count]) 
            for i in range(len(ls_tickers)):
                arr_div_flag_add = np.append(arr_div_flag_add, np.array([arr_new_shares[i], arr_latest_prices[i], -arr_new_shares[i] * arr_latest_prices[i], arr_trades[-1][ls_trades_shr_index[i]]+arr_new_shares[i], (arr_trades[-1][ls_trades_shr_index[i]]+arr_new_shares[i])*arr_latest_prices[i], arr_divflag[i], tradeflag]))
            arr_trades = np.append(arr_trades,[arr_div_flag_add], axis = 0)
            arr_cash = np.append(arr_cash, [np.array([np.sum(arr_leftover_cash)])], axis=0)
            arr_divflag = np.zeros(len(ls_tickers))
            arr_cur_shrs = np.array(arr_trades[-1][ls_trades_shr_index])
            
        total_value = np.matmul(np.array(arr_latest_prices),arr_cur_shrs)+arr_cash[-1]
        arr_asset_val = (arr_cur_shrs * arr_latest_prices)
        arr_actual_weights = arr_asset_val/total_value
        dev_weights = np.absolute((arr_actual_weights/arr_portfolio_weights)-1)
        
        '''Checks two things:
                1) Calendar Rebalancing: If rebalancing should occur on this day based on the trade counter chosen. This sells all the shares
                and reblances using the original weights and the total value available. The share amount before rebalancing is then
                subtracted to find the total change in shares for each asset. The df_trades and df_cash dataframes are then updated.

                2) Threshold Rebalancing: if rebalancing should occur due to an asset passing the threshold. The % deviation is calculated by: 

                                       |[Actual % of portfolio] / [target % of portfolio] - 1|

                If this absolute value is greater than the threshold, rebalancing occurs. This sells all the shares and reblances 
                using the original weights and the total value available. The share amount before rebalancing is then subtracted 
                to find the total change in shares for each asset. The df_trades and df_cash dataframes are then updated.'''

        
        if trading_day_counter % rebalance_days == 0 or np.amax(dev_weights) > threshold:
            
            tradeflag = 1
            
            total_value = np.matmul(np.array(arr_latest_prices),arr_cur_shrs)+arr_cash[-1]
            
            
            da = DiscreteAllocation(portfolio_weights, pd.Series(arr_latest_prices, index = ls_tickers), total_portfolio_value=total_value)
            shares, cash = da.lp_portfolio()
            arr_rebal = np.array([i for i in shares.values()])-arr_cur_shrs
            arr_trade_add = np.array([count])
            for i in range(len(ls_tickers)):
                arr_trade_add = np.append(arr_trade_add, np.array([arr_new_shares[i], arr_latest_prices[i], -arr_new_shares[i] * arr_latest_prices[i], arr_trades[-1][ls_trades_shr_index[i]]+arr_new_shares[i], (arr_trades[-1][ls_trades_shr_index[i]]+arr_new_shares[i])*arr_latest_prices[i], arr_divflag[i], tradeflag]))
            arr_trades = np.append(arr_trades,[arr_trade_add], axis = 0)
            arr_cash = np.append(arr_cash, [np.array(cash)], axis=0)
            trading_day_counter=1
            arr_cur_shrs = np.array(arr_trades[-1][ls_trades_shr_index])
            arr_new_port_det = np.array([], dtype='f8')
            arr_asset_val = (arr_cur_shrs * arr_latest_prices)
            
        else:
            trading_day_counter += 1
        
        arr_yesterday_prices = arr_latest_prices    
            
        #Reaggregates all the data into the dataframe "portfolio_detail" and "portfolio_sum"
        
        arr_new_port_det = np.array([], dtype='f8')
        arr_asset_val = (arr_cur_shrs * arr_latest_prices)
        total_asset_value = np.sum(arr_asset_val)
        arr_actual_weights = arr_asset_val/total_value
        ls_new_port_det = []
        for i in range(len(ls_tickers)):
            
            ls_new_port_det = ls_new_port_det + [arr_latest_prices[i], row[ls_ret_index[i]], arr_actual_weights[i], arr_cur_shrs[i], arr_asset_val[i]]

        arr_port_det[count] = ls_new_port_det
        
        arr_new_port_sum = [total_value, total_asset_value, arr_cash[-1].item(), arr_cash[-1].item()/(total_value)]
        
        arr_port_sum[count] = arr_new_port_sum
        count += 1
    return arr_port_det, arr_port_sum, arr_trades, arr_cash


# %%
arr_port_det_disc, arr_port_sum_disc, arr_trades_disc, arr_cash_disc = calendar_threshold_rebalance_discrete(d, portfolio_weights, initial_money=1000000, rebalance_days = 45000, threshold = 5)

# %% [markdown]
# # Functions to track dividend payments and rebalance using continuous assets
# 
# The advantage is time to run. This takes 1/3 of the time to run as the discrete version, which will come into play when we are running it over 1000 simulations.

# %%
def calendar_threshold_rebalance_continuous(data, portfolio_weights, initial_money=1000000, rebalance_days = 63, threshold = 1):
    trading_day_counter = 1
    df_trades, df_cash, df_portfolio_sum, df_portfolio_detail = initialize_df_continuous(data, portfolio_weights,initial_money= 1000000)
    arr_trades = np.append([[0]], np.array(df_trades), axis = 1)
    date_start = data.index[0].date().isoformat()
    arr_data_columns = data.columns.values
    arr_trades_columns = df_trades.columns.values
    arr_data = np.array(data)
    arr_cash = np.array(df_cash)
    arr_port_det = np.zeros([data.shape[0], df_portfolio_detail.shape[1]])
    arr_port_sum = np.zeros([data.shape[0], df_portfolio_sum.shape[1]])
    ls_tickers = list(set([y[0] for x, y in enumerate(arr_data_columns)]))
    ls_prc_index = [x for x, y in enumerate(arr_data_columns) if y[1] == 'prc']
    ls_ret_index = [x for x, y in enumerate(arr_data_columns) if y[1] == 'ret']
    ls_retx_index = [x for x, y in enumerate(arr_data_columns) if y[1] == 'retx']
    ls_retd_index = [x for x, y in enumerate(arr_data_columns) if y[1] == 'retd']
    arr_portfolio_weights = np.array([i for i in portfolio_weights.values()])
    ls_trades_shr_index = [x+1 for x, y in enumerate(arr_trades_columns) if y[1] == 'tot shr']

    arr_yesterday_prices =  arr_data[0][ls_prc_index]
    
    #This loop goes through the data set "d", which should be a daily time series of asset returns and prices.
    #Returns should be broken into returns including dividends, returns without dividends, and dividend only returns. 
    arr_cur_shrs = np.array(arr_trades[-1][ls_trades_shr_index])

    count = 0
    for row in arr_data:
        
        tradeflag = 0
        arr_divflag = np.zeros(len(ls_tickers))
        arr_div_cash = np.zeros(len(ls_tickers))
        arr_rebal = np.zeros(len(ls_tickers))
        arr_new_shares = np.zeros(len(ls_tickers))
        arr_leftover_cash = 0
        arr_latest_prices = row[ls_prc_index]
            
        #Checks to see if there are any dividend payouts, and if so, calculates the number of shares that are purchased as part of 
        #the reinvestment and the leftover cash if a full share cannot be purchased. Dividend payouts are reinvested into the asset
        #that payed the dividend
        
        if np.any(row[ls_retd_index]>0):
            arr_divflag = (row[ls_retd_index]>0).astype(int)
            add_cash = arr_cash[-1]/sum(arr_divflag)
            arr_add_cash = arr_divflag*add_cash
            arr_div_cash = (arr_yesterday_prices*arr_trades[-1][ls_trades_shr_index]*row[ls_retd_index])+arr_add_cash
            arr_new_shares = arr_div_cash/arr_latest_prices
            arr_leftover_cash = arr_div_cash-(arr_new_shares*arr_latest_prices)

        #Executes the actual trade, recording it in the df_trades dataframe and updates the df_cash dataframe. This is done before
        #rebalancing incase a) rebalancing happens on the same day in the case of calendar rebalancing, or b) the dividend payout
        #puts the asset over the threshold.
        
        if 1 in arr_divflag:
    
            arr_div_flag_add = np.array([count]) 
            for i in range(len(ls_tickers)):
                arr_div_flag_add = np.append(arr_div_flag_add, np.array([arr_new_shares[i], arr_latest_prices[i], -arr_new_shares[i] * arr_latest_prices[i], arr_trades[-1][ls_trades_shr_index[i]]+arr_new_shares[i], (arr_trades[-1][ls_trades_shr_index[i]]+arr_new_shares[i])*arr_latest_prices[i], arr_divflag[i], tradeflag]))
            arr_trades = np.append(arr_trades,[arr_div_flag_add], axis = 0)
            arr_cash = np.append(arr_cash, [np.array([np.sum(arr_leftover_cash)])], axis=0)
            arr_divflag = np.zeros(len(ls_tickers))
            arr_cur_shrs = np.array(arr_trades[-1][ls_trades_shr_index])
         
  
        
        total_value = np.matmul(np.array(arr_latest_prices),arr_cur_shrs)+arr_cash[-1]
        arr_asset_val = (arr_cur_shrs * arr_latest_prices)
        arr_actual_weights = arr_asset_val/total_value
        dev_weights = np.absolute((arr_actual_weights/arr_portfolio_weights)-1)
        
        '''Checks two things:
                1) Calendar Rebalancing: If rebalancing should occur on this day based on the trade counter chosen. This sells all the shares
                and reblances using the original weights and the total value available. The share amount before rebalancing is then
                subtracted to find the total change in shares for each asset. The df_trades and df_cash dataframes are then updated.

                2) Threshold Rebalancing: if rebalancing should occur due to an asset passing the threshold. The % deviation is calculated by: 

                                       |[Actual % of portfolio] / [target % of portfolio] - 1|

                If this absolute value is greater than the threshold, rebalancing occurs. This sells all the shares and reblances 
                using the original weights and the total value available. The share amount before rebalancing is then subtracted 
                to find the total change in shares for each asset. The df_trades and df_cash dataframes are then updated.'''

        
        if trading_day_counter % rebalance_days == 0 or np.amax(dev_weights) > threshold:
            
            tradeflag = 1
            
            total_value = np.matmul(np.array(arr_latest_prices),arr_cur_shrs)+arr_cash[-1]
            
           
            arr_shares = (total_value*arr_portfolio_weights)/arr_latest_prices
     
            arr_rebal = arr_shares-arr_cur_shrs
            arr_trade_add = np.array([count])
            for i in range(len(ls_tickers)):
                arr_trade_add = np.append(arr_trade_add, np.array([arr_new_shares[i], arr_latest_prices[i], -arr_new_shares[i] * arr_latest_prices[i], arr_trades[-1][ls_trades_shr_index[i]]+arr_new_shares[i], (arr_trades[-1][ls_trades_shr_index[i]]+arr_new_shares[i])*arr_latest_prices[i], arr_divflag[i], tradeflag]))
            arr_trades = np.append(arr_trades,[arr_trade_add], axis = 0)
            arr_cash = np.append(arr_cash, np.array([[0]]), axis=0)
            trading_day_counter=1
            arr_cur_shrs = np.array(arr_trades[-1][ls_trades_shr_index])
            arr_asset_val = (arr_cur_shrs * arr_latest_prices)
            arr_actual_weights = arr_asset_val/total_value
            
        else:
            trading_day_counter += 1
        
        arr_yesterday_prices = arr_latest_prices    
        total_asset_value = np.sum(arr_asset_val)   
        #Reaggregates all the data into the dataframe "portfolio_detail" and "portfolio_sum"
        arr_new_port_det = np.array([], dtype='f8')
        ls_new_port_det = []
        for i in range(len(ls_tickers)):
            
            ls_new_port_det = ls_new_port_det + [arr_latest_prices[i], row[ls_ret_index[i]], arr_actual_weights[i], arr_cur_shrs[i], arr_asset_val[i]]
        arr_port_det[count] = ls_new_port_det
        
        arr_new_port_sum = [total_value, total_asset_value, arr_cash[-1].item(), arr_cash[-1].item()/(total_value)]
        
        arr_port_sum[count] = arr_new_port_sum
        count += 1

    return arr_port_det, arr_port_sum, arr_trades, arr_cash


# %%
arr_port_det_cont, arr_port_sum_cont, arr_trades_cont, arr_cash_cont=calendar_threshold_rebalance_continuous(d, portfolio_weights, initial_money=1000000, rebalance_days = 45000, threshold = 5)

# %% [markdown]
# ## array_to_dataframe turns the outputs of the rebalancing functions to dataframes

# %%
def array_to_dataframe(data, portfolio_weights, arr_port_det, arr_port_sum, arr_trades, arr_cash):
    df_trades_initial, df_cash_initial, df_portfolio_sum_initial, df_portfolio_detail_initial = initialize_df(data, portfolio_weights,initial_money=10000000)
    df_trades_index = data.index[list(arr_trades[:,0].astype(int))]
    df_trades = pd.DataFrame(arr_trades[:,1:], index = df_trades_index, columns = df_trades_initial.columns)
    df_portfolio_sum = pd.DataFrame(arr_port_sum, index = data.index, columns = df_portfolio_sum_initial.columns)
    df_portfolio_detail = pd.DataFrame(arr_port_det, index = data.index, columns = df_portfolio_detail_initial.columns)
    return df_portfolio_sum, df_portfolio_detail, df_trades


# %%
df_portfolio_sum_disc_disc, df_portfolio_detail_disc, df_trades_disc = array_to_dataframe(d, portfolio_weights, arr_port_det_disc, arr_port_sum_disc, arr_trades_disc, arr_cash_disc)
df_portfolio_sum_disc_cont, df_portfolio_detail_cont, df_trades_cont = array_to_dataframe(d, portfolio_weights, arr_port_det_cont, arr_port_sum_cont, arr_trades_cont, arr_cash_cont)

# %% [markdown]
# # ***Testing against 'ret'***

# %%
port_cumprod = [d['SPY']['ret'].iloc[1:].cumprod()[-1], d['TLT']['ret'].iloc[1:].cumprod()[-1], d['GLD']['ret'].iloc[1:].cumprod()[-1]]
port_weights = [.5,.4,.1]


# %%
ret_ret = np.matmul(np.array(port_cumprod),np.array(port_weights))


# %%
discrete_function_return = (df_portfolio_sum_disc_disc.iloc[-1][0]/df_portfolio_sum_disc_disc.iloc[0][0])- 1


# %%
continuous_function_return = (df_portfolio_sum_disc_cont.iloc[-1][0]/df_portfolio_sum_disc_cont.iloc[0][0])- 1


# %%
pd.DataFrame(np.array([[ret_ret - 1,discrete_function_return, continuous_function_return]]), columns = ['ret_cumprod', 'Discrete_shares_function', 'Continuous_Shares_function'], index = ['Returns']  )

# %% [markdown]
# ## Trial Code

# %%
initial_money


# %%
plt.plot(arr_port_sum[:,0])


# %%
plt.plot(arr_port_sum[:,0])


# %%
plt.plot(arr_port_det[:,2])


# %%
d.index


# %%
df_trades_initial, df_cash_initial, df_portfolio_sum_initial, df_portfolio_detail_initial = initialize_df(data, portfolio_weights,initial_money=10000000)
df_trades_index = data.index[list(arr_trades[:,0].astype(int))]
df_trades1 = pd.DataFrame(arr_trades[:,1:], index = df_trades_index, columns = df_trades_initial.columns)


# %%
df_trades


# %%
trading_day_counter = 1
df_trades, df_cash, df_portfolio_sum, df_portfolio_detail = initialize_df(data, portfolio_weights,initial_money= 1000000)
arr_trades = np.append([[0]], np.array(df_trades), axis = 1)
date_start = data.index[0].date().isoformat()
arr_data_columns = data.columns.values
arr_trades_columns = df_trades.columns.values
arr_data = np.array(data)
arr_cash = np.array(df_cash)
arr_port_det = np.zeros([data.shape[0], df_portfolio_detail.shape[1]])
arr_port_sum = np.zeros([data.shape[0], df_portfolio_sum.shape[1]])
ls_tickers = list(set([y[0] for x, y in enumerate(arr_data_columns)]))
ls_prc_index = [x for x, y in enumerate(arr_data_columns) if y[1] == 'prc']
ls_ret_index = [x for x, y in enumerate(arr_data_columns) if y[1] == 'ret']
ls_retx_index = [x for x, y in enumerate(arr_data_columns) if y[1] == 'retx']
ls_retd_index = [x for x, y in enumerate(arr_data_columns) if y[1] == 'retd']

ls_trades_shr_index = [x+1 for x, y in enumerate(arr_trades_columns) if y[1] == 'tot shr']

arr_yesterday_prices =  arr_data[0][ls_prc_index]

#This loop goes through the data set "d", which should be a daily time series of asset returns and prices.
#Returns should be broken into returns including dividends, returns without dividends, and dividend only returns. 
arr_cur_shrs = np.array(arr_trades[-1][ls_trades_shr_index])
count = 0
for row in arr_data:
    tradeflag = 0
    arr_divflag = np.zeros(len(ls_tickers))
    arr_div_cash = np.zeros(len(ls_tickers))
    arr_rebal = np.zeros(len(ls_tickers))
    arr_new_shares = np.zeros(len(ls_tickers))
    leftover_cash = 0
    arr_latest_prices = row[ls_prc_index]    
    #Checks to see if there are any dividend payouts, and if so, calculates the number of shares that are purchased as part of 
    #the reinvestment and the leftover cash if a full share cannot be purchased. Dividend payouts are reinvested into the asset
    #that payed the dividend

    if np.any(row[ls_retd_index]>0):
        arr_divflag = (row[ls_retd_index]>0).astype(int)
        arr_div_cash = arr_yesterday_prices*arr_trades[-1][ls_trades_shr_index]*row[ls_retd_index]
        arr_new_shares = np.floor(arr_div_cash/arr_latest_prices)
        arr_leftover_cash = arr_div_cash-(arr_new_shares*arr_latest_prices)

    #Executes the actual trade, recording it in the df_trades dataframe and updates the df_cash dataframe. This is done before
    #rebalancing incase a) rebalancing happens on the same day in the case of calendar rebalancing, or b) the dividend payout
    #puts the asset over the threshold.

    if 1 in arr_divflag:

        arr_div_flag_add = np.array([count]) 
        for i in range(len(ls_tickers)):
            arr_div_flag_add = np.append(arr_div_flag_add, np.array([arr_new_shares[i], arr_latest_prices[i], -arr_new_shares[i] * arr_latest_prices[i], arr_trades[-1][ls_trades_shr_index[i]]+arr_new_shares[i], (arr_trades[-1][ls_trades_shr_index[i]]+arr_new_shares[i])*arr_latest_prices[i], arr_divflag[i], tradeflag]))
        arr_trades = np.append(arr_trades,[arr_div_flag_add], axis = 0)
        arr_cash = np.append(arr_cash, [np.array(arr_cash[-1]+np.sum(arr_leftover_cash))], axis=0)
        arr_divflag = np.zeros(len(ls_tickers))
        arr_cur_shrs = np.array(arr_trades[-1][ls_trades_shr_index])

    #Checks to see if rebalancing should occur on this day based on the trade counter chosen. This sells all the shares
    #and reblances using the original weights and the total value available. The share amount before rebalancing is then
    #subtracted to find the total change in shares for each asset. The df_trades and df_cash dataframes are then updated.

    if trading_day_counter % rebalance_days == 0:

        tradeflag = 1

        total_money = np.matmul(np.array(arr_latest_prices),arr_cur_shrs)+arr_cash[-1]


        da = DiscreteAllocation(portfolio_weights, pd.Series(arr_latest_prices, index = ls_tickers), total_portfolio_value=total_money)
        shares, cash = da.lp_portfolio()
        arr_rebal = np.array([i for i in shares.values()])-arr_cur_shrs
        arr_trade_add = np.array([count])
        for i in range(len(ls_tickers)):
            arr_trade_add = np.append(arr_trade_add, np.array([arr_new_shares[i], arr_latest_prices[i], -arr_new_shares[i] * arr_latest_prices[i], arr_trades[-1][ls_trades_shr_index[i]]+arr_new_shares[i], (arr_trades[-1][ls_trades_shr_index[i]]+arr_new_shares[i])*arr_latest_prices[i], arr_divflag[i], tradeflag]))
        arr_trades = np.append(arr_trades,[arr_trade_add], axis = 0)
        arr_cash = np.append(arr_cash, [np.array(arr_cash[-1]+np.sum(arr_leftover_cash))], axis=0)
        trading_day_counter=1
        arr_cur_shrs = np.array(arr_trades[-1][ls_trades_shr_index])


    else:
        trading_day_counter += 1

    arr_yesterday_prices = arr_latest_prices    

    #Reaggregates all the data into the dataframe "portfolio_detail" and "portfolio_sum"

    arr_new_port_det = np.array([], dtype='f8')
    arr_asset_val = (arr_cur_shrs * arr_latest_prices)
    total_value = np.sum(arr_asset_val)
    arr_actual_weights = arr_asset_val/total_value
    ls_new_port_det = []
    for i in range(len(ls_tickers)):

        ls_new_port_det = ls_new_port_det + [arr_latest_prices[i], row[ls_ret_index[i]], arr_actual_weights[i], arr_cur_shrs[i], arr_asset_val[i]]

    arr_port_det[count] = ls_new_port_det

    arr_new_port_sum = [total_value+arr_cash[-1].item(), total_value, arr_cash[-1].item(), arr_cash[-1].item()/(total_value+arr_cash[-1].item())]

    arr_port_sum[count] = arr_new_port_sum
    count += 1


# %%
arr_port_sum


# %%
arr_port_sum


# %%
arr_port_det, arr_port_sum, arr_trades, arr_cash = calendar_threshold_rebalance_continuous(d, portfolio_weights, rebalance_days=4300)


# %%
arr_port_sum


# %%
df_portfolio


# %%
start_time = timeit.default_timer()
timer = []
data = d
trading_day_counter = 1
df_trades, df_cash, df_portfolio_sum, df_portfolio_detail = initialize_df(data, portfolio_weights,initial_money= 1000000)
arr_trades = np.array(df_trades)
date_start = data.index[0].date().isoformat()
arr_data_columns = data.columns.values
arr_trades_columns = df_trades.columns.values
arr_data = np.array(data)
arr_cash = np.array(df_cash)
arr_port_det = np.array(df_portfolio_detail, dtype = 'f8')
arr_port_sum = np.array(df_portfolio_sum)
ls_tickers = list(set([y[0] for x, y in enumerate(arr_data_columns)]))
ls_prc_index = [x for x, y in enumerate(arr_data_columns) if y[1] == 'prc']
ls_ret_index = [x for x, y in enumerate(arr_data_columns) if y[1] == 'ret']
ls_retx_index = [x for x, y in enumerate(arr_data_columns) if y[1] == 'retx']
ls_retd_index = [x for x, y in enumerate(arr_data_columns) if y[1] == 'retd']

ls_trades_shr_index = [x for x, y in enumerate(arr_trades_columns) if y[1] == 'tot shr']

arr_yesterday_prices =  arr_data[0][ls_prc_index]

#This loop goes through the data set "d", which should be a daily time series of asset returns and prices.
#Returns should be broken into returns including dividends, returns without dividends, and dividend only returns. 
arr_cur_shrs = np.array(arr_trades[-1][ls_trades_shr_index])
elapsed1 = timeit.default_timer() - start_time
for row in arr_data:
    start_time6 = timeit.default_timer()
    tradeflag = 0
    arr_divflag = np.zeros(len(ls_tickers))
    arr_div_cash = np.zeros(len(ls_tickers))
    arr_rebal = np.zeros(len(ls_tickers))
    arr_new_shares = np.zeros(len(ls_tickers))
    leftover_cash = 0
    arr_latest_prices = row[ls_prc_index]
    elapsed6 = timeit.default_timer() - start_time2    
    #Checks to see if there are any dividend payouts, and if so, calculates the number of shares that are purchased as part of 
    #the reinvestment and the leftover cash if a full share cannot be purchased. Dividend payouts are reinvested into the asset
    #that payed the dividend
    
    if np.any(row[ls_retd_index]>0):
        start_time2 = timeit.default_timer()
        arr_divflag = (row[ls_retd_index]>0).astype(int)
        arr_div_cash = arr_yesterday_prices*arr_trades[-1][ls_trades_shr_index]*row[ls_retd_index]
        arr_new_shares = np.floor(arr_div_cash/arr_latest_prices)
        arr_leftover_cash = arr_div_cash-(arr_new_shares*arr_latest_prices)
        elapsed2 = timeit.default_timer() - start_time2
    #Executes the actual trade, recording it in the df_trades dataframe and updates the df_cash dataframe. This is done before
    #rebalancing incase a) rebalancing happens on the same day in the case of calendar rebalancing, or b) the dividend payout
    #puts the asset over the threshold.
    
    if 1 in arr_divflag:
        start_time3 = timeit.default_timer()
        arr_div_flag_add = np.array([]) 
        for i in range(len(ls_tickers)):
            arr_div_flag_add = np.append(arr_div_flag_add, np.array([arr_new_shares[i], arr_latest_prices[i], -arr_new_shares[i] * arr_latest_prices[i], arr_trades[-1][ls_trades_shr_index[i]]+arr_new_shares[i], (arr_trades[-1][ls_trades_shr_index[i]]+arr_new_shares[i])*arr_latest_prices[i], arr_divflag[i], tradeflag]))
        arr_trades = np.append(arr_trades,[arr_div_flag_add], axis = 0)
        arr_cash = np.append(arr_cash, [np.array(arr_cash[-1]+np.sum(arr_leftover_cash))], axis=0)
        arr_divflag = np.zeros(len(ls_tickers))
        arr_cur_shrs = np.array(arr_trades[-1][ls_trades_shr_index])
        elapsed3 = timeit.default_timer() - start_time3
    #Checks to see if rebalancing should occur on this day based on the trade counter chosen. This sells all the shares
    #and reblances using the original weights and the total value available. The share amount before rebalancing is then
    #subtracted to find the total change in shares for each asset. The df_trades and df_cash dataframes are then updated.
    
    if trading_day_counter % rebalance_days == 0:
        
        tradeflag = 1
        
        total_money = np.matmul(np.array(arr_latest_prices),arr_cur_shrs)+arr_cash[-1]
        
        start_time4 = timeit.default_timer()
        da = DiscreteAllocation(portfolio_weights, pd.Series(arr_latest_prices, index = ls_tickers), total_portfolio_value=total_money)
        shares, cash = da.lp_portfolio()
        elapsed4 = timeit.default_timer() - start_time4
        timer.append(elapsed4)
        arr_rebal = np.array([i for i in shares.values()])-arr_cur_shrs
        arr_trade_add = np.array([])
        for i in range(len(ls_tickers)):
            arr_trade_add = np.append(arr_trade_add, np.array([arr_new_shares[i], latest_prices[i], -arr_new_shares[i] * latest_prices[i], arr_trades[-1][ls_trades_shr_index[i]]+arr_new_shares[i], (arr_trades[-1][ls_trades_shr_index[i]]+arr_new_shares[i])*latest_prices[i], arr_divflag[i], tradeflag]))
        arr_trades = np.append(arr_trades,[arr_trade_add], axis = 0)
        arr_cash = np.append(arr_cash, [np.array(arr_cash[-1]+np.sum(arr_leftover_cash))], axis=0)
        trading_day_counter=1
        arr_cur_shrs = np.array(arr_trades[-1][ls_trades_shr_index])
        
        
    else:
        trading_day_counter += 1
    
    arr_yesterday_prices = arr_latest_prices    
        
    #Reaggregates all the data into the dataframe "portfolio_detail" and "portfolio_sum"
    
    arr_new_port_det = np.array([], dtype='f8')
    arr_asset_val = (arr_cur_shrs * arr_latest_prices)
    total_value = np.sum(arr_asset_val)
    arr_actual_weights = arr_asset_val/total_value
    
    for i in range(len(ls_tickers)):
        
        arr_new_port_det = np.append(arr_new_port_det, [arr_latest_prices[i], row[ls_ret_index[i]], arr_actual_weights[i], arr_cur_shrs[i], arr_asset_val[i]])
    start_time5 = timeit.default_timer()
    arr_port_det = np.append(arr_port_det, [arr_new_port_det], axis = 0)
    
    arr_new_port_sum = np.array([total_value+arr_cash[-1].item(), total_value, arr_cash[-1].item(), arr_cash[-1].item()/(total_value+arr_cash[-1].item())])
    
    arr_port_sum = np.append(arr_port_sum, [arr_new_port_sum], axis = 0)
    elapsed5 = timeit.default_timer() - start_time5
elapsed = timeit.default_timer() - start_time


# %%
start_time = timeit.default_timer()
timer = []
data = d
trading_day_counter = 1
df_trades, df_cash, df_portfolio_sum, df_portfolio_detail = initialize_df(data, portfolio_weights,initial_money= 1000000)
arr_trades = np.array(df_trades)
date_start = data.index[0].date().isoformat()
arr_data_columns = data.columns.values
arr_trades_columns = df_trades.columns.values
arr_data = np.array(data)
arr_cash = np.array(df_cash)
arr_port_det = np.zeros([data.shape[0], df_portfolio_detail.shape[1]])
arr_port_sum = np.zeros([data.shape[0], df_portfolio_sum.shape[1]])
ls_tickers = list(set([y[0] for x, y in enumerate(arr_data_columns)]))
ls_prc_index = [x for x, y in enumerate(arr_data_columns) if y[1] == 'prc']
ls_ret_index = [x for x, y in enumerate(arr_data_columns) if y[1] == 'ret']
ls_retx_index = [x for x, y in enumerate(arr_data_columns) if y[1] == 'retx']
ls_retd_index = [x for x, y in enumerate(arr_data_columns) if y[1] == 'retd']

ls_trades_shr_index = [x for x, y in enumerate(arr_trades_columns) if y[1] == 'tot shr']

arr_yesterday_prices =  arr_data[0][ls_prc_index]

#This loop goes through the data set "d", which should be a daily time series of asset returns and prices.
#Returns should be broken into returns including dividends, returns without dividends, and dividend only returns. 
arr_cur_shrs = np.array(arr_trades[-1][ls_trades_shr_index])
elapsed1 = timeit.default_timer() - start_time
count = 0
for row in arr_data:
    start_time6 = timeit.default_timer()
    tradeflag = 0
    arr_divflag = np.zeros(len(ls_tickers))
    arr_div_cash = np.zeros(len(ls_tickers))
    arr_rebal = np.zeros(len(ls_tickers))
    arr_new_shares = np.zeros(len(ls_tickers))
    leftover_cash = 0
    arr_latest_prices = row[ls_prc_index]
    elapsed6 = timeit.default_timer() - start_time2    
    #Checks to see if there are any dividend payouts, and if so, calculates the number of shares that are purchased as part of 
    #the reinvestment and the leftover cash if a full share cannot be purchased. Dividend payouts are reinvested into the asset
    #that payed the dividend
    
    if np.any(row[ls_retd_index]>0):
        start_time2 = timeit.default_timer()
        arr_divflag = (row[ls_retd_index]>0).astype(int)
        arr_div_cash = arr_yesterday_prices*arr_trades[-1][ls_trades_shr_index]*row[ls_retd_index]
        arr_new_shares = np.floor(arr_div_cash/arr_latest_prices)
        arr_leftover_cash = arr_div_cash-(arr_new_shares*arr_latest_prices)
        elapsed2 = timeit.default_timer() - start_time2
    #Executes the actual trade, recording it in the df_trades dataframe and updates the df_cash dataframe. This is done before
    #rebalancing incase a) rebalancing happens on the same day in the case of calendar rebalancing, or b) the dividend payout
    #puts the asset over the threshold.
    
    if 1 in arr_divflag:
        start_time3 = timeit.default_timer()
        arr_div_flag_add = np.array([]) 
        for i in range(len(ls_tickers)):
            arr_div_flag_add = np.append(arr_div_flag_add, np.array([arr_new_shares[i], arr_latest_prices[i], -arr_new_shares[i] * arr_latest_prices[i], arr_trades[-1][ls_trades_shr_index[i]]+arr_new_shares[i], (arr_trades[-1][ls_trades_shr_index[i]]+arr_new_shares[i])*arr_latest_prices[i], arr_divflag[i], tradeflag]))
        arr_trades = np.append(arr_trades,[arr_div_flag_add], axis = 0)
        arr_cash = np.append(arr_cash, [np.array(arr_cash[-1]+np.sum(arr_leftover_cash))], axis=0)
        arr_divflag = np.zeros(len(ls_tickers))
        arr_cur_shrs = np.array(arr_trades[-1][ls_trades_shr_index])
        elapsed3 = timeit.default_timer() - start_time3
    #Checks to see if rebalancing should occur on this day based on the trade counter chosen. This sells all the shares
    #and reblances using the original weights and the total value available. The share amount before rebalancing is then
    #subtracted to find the total change in shares for each asset. The df_trades and df_cash dataframes are then updated.
    
    if trading_day_counter % rebalance_days == 0:
        
        tradeflag = 1
        
        total_money = np.matmul(np.array(arr_latest_prices),arr_cur_shrs)+arr_cash[-1]
        
        start_time4 = timeit.default_timer()
        da = DiscreteAllocation(portfolio_weights, pd.Series(arr_latest_prices, index = ls_tickers), total_portfolio_value=total_money)
        shares, cash = da.lp_portfolio()
        elapsed4 = timeit.default_timer() - start_time4
        timer.append(elapsed4)
        arr_rebal = np.array([i for i in shares.values()])-arr_cur_shrs
        arr_trade_add = np.array([])
        for i in range(len(ls_tickers)):
            arr_trade_add = np.append(arr_trade_add, np.array([arr_new_shares[i], latest_prices[i], -arr_new_shares[i] * latest_prices[i], arr_trades[-1][ls_trades_shr_index[i]]+arr_new_shares[i], (arr_trades[-1][ls_trades_shr_index[i]]+arr_new_shares[i])*latest_prices[i], arr_divflag[i], tradeflag]))
        arr_trades = np.append(arr_trades,[arr_trade_add], axis = 0)
        arr_cash = np.append(arr_cash, [np.array(arr_cash[-1]+np.sum(arr_leftover_cash))], axis=0)
        trading_day_counter=1
        arr_cur_shrs = np.array(arr_trades[-1][ls_trades_shr_index])
        
        
    else:
        trading_day_counter += 1
    
    arr_yesterday_prices = arr_latest_prices    
        
    #Reaggregates all the data into the dataframe "portfolio_detail" and "portfolio_sum"
    
    arr_new_port_det = np.array([], dtype='f8')
    arr_asset_val = (arr_cur_shrs * arr_latest_prices)
    total_value = np.sum(arr_asset_val)
    arr_actual_weights = arr_asset_val/total_value
    ls_new_port_det = []
    for i in range(len(ls_tickers)):
        
        ls_new_port_det = ls_new_port_det + [arr_latest_prices[i], row[ls_ret_index[i]], arr_actual_weights[i], arr_cur_shrs[i], arr_asset_val[i]]
    start_time5 = timeit.default_timer()
    arr_port_det[count] = ls_new_port_det
    
    arr_new_port_sum = [total_value+arr_cash[-1].item(), total_value, arr_cash[-1].item(), arr_cash[-1].item()/(total_value+arr_cash[-1].item())]
    
    arr_port_sum[count] = arr_new_port_sum
    count += 1
    elapsed5 = timeit.default_timer() - start_time5
elapsed = timeit.default_timer() - start_time


# %%
start_time = timeit.default_timer()
trading_day_counter = 1
df_trades, df_cash, df_portfolio_sum, df_portfolio_detail = initialize_df_continuous(data, portfolio_weights,initial_money= 1000000)
arr_trades = np.array(df_trades)
date_start = data.index[0].date().isoformat()
arr_data_columns = data.columns.values
arr_trades_columns = df_trades.columns.values
arr_data = np.array(data)
arr_cash = np.array(df_cash)
arr_port_det = np.zeros([data.shape[0], df_portfolio_detail.shape[1]])
arr_port_sum = np.zeros([data.shape[0], df_portfolio_sum.shape[1]])
ls_tickers = list(set([y[0] for x, y in enumerate(arr_data_columns)]))
ls_prc_index = [x for x, y in enumerate(arr_data_columns) if y[1] == 'prc']
ls_ret_index = [x for x, y in enumerate(arr_data_columns) if y[1] == 'ret']
ls_retx_index = [x for x, y in enumerate(arr_data_columns) if y[1] == 'retx']
ls_retd_index = [x for x, y in enumerate(arr_data_columns) if y[1] == 'retd']
arr_portfolio_weights = np.array([i for i in portfolio_weights.values()])
ls_trades_shr_index = [x for x, y in enumerate(arr_trades_columns) if y[1] == 'tot shr']

arr_yesterday_prices =  arr_data[0][ls_prc_index]

#This loop goes through the data set "d", which should be a daily time series of asset returns and prices.
#Returns should be broken into returns including dividends, returns without dividends, and dividend only returns. 
arr_cur_shrs = np.array(arr_trades[-1][ls_trades_shr_index])
elapsed1 = timeit.default_timer() - start_time
count = 0
for row in arr_data:
    start_time6 = timeit.default_timer()
    tradeflag = 0
    arr_divflag = np.zeros(len(ls_tickers))
    arr_div_cash = np.zeros(len(ls_tickers))
    arr_rebal = np.zeros(len(ls_tickers))
    arr_new_shares = np.zeros(len(ls_tickers))
    leftover_cash = 0
    arr_latest_prices = row[ls_prc_index]
    elapsed6 = timeit.default_timer() - start_time2    
    #Checks to see if there are any dividend payouts, and if so, calculates the number of shares that are purchased as part of 
    #the reinvestment and the leftover cash if a full share cannot be purchased. Dividend payouts are reinvested into the asset
    #that payed the dividend
    
    if np.any(row[ls_retd_index]>0):
        start_time2 = timeit.default_timer()
        arr_divflag = (row[ls_retd_index]>0).astype(int)
        arr_div_cash = arr_yesterday_prices*arr_trades[-1][ls_trades_shr_index]*row[ls_retd_index]
        arr_new_shares = np.floor(arr_div_cash/arr_latest_prices)
        arr_leftover_cash = arr_div_cash-(arr_new_shares*arr_latest_prices)
        elapsed2 = timeit.default_timer() - start_time2
    #Executes the actual trade, recording it in the df_trades dataframe and updates the df_cash dataframe. This is done before
    #rebalancing incase a) rebalancing happens on the same day in the case of calendar rebalancing, or b) the dividend payout
    #puts the asset over the threshold.
    
    if 1 in arr_divflag:
        start_time3 = timeit.default_timer()
        arr_div_flag_add = np.array([]) 
        for i in range(len(ls_tickers)):
            arr_div_flag_add = np.append(arr_div_flag_add, np.array([arr_new_shares[i], arr_latest_prices[i], -arr_new_shares[i] * arr_latest_prices[i], arr_trades[-1][ls_trades_shr_index[i]]+arr_new_shares[i], (arr_trades[-1][ls_trades_shr_index[i]]+arr_new_shares[i])*arr_latest_prices[i], arr_divflag[i], tradeflag]))
        arr_trades = np.append(arr_trades,[arr_div_flag_add], axis = 0)
        arr_cash = np.append(arr_cash, [np.array(arr_cash[-1]+np.sum(arr_leftover_cash))], axis=0)
        arr_divflag = np.zeros(len(ls_tickers))
        arr_cur_shrs = np.array(arr_trades[-1][ls_trades_shr_index])
        elapsed3 = timeit.default_timer() - start_time3
    #Checks to see if rebalancing should occur on this day based on the trade counter chosen. This sells all the shares
    #and reblances using the original weights and the total value available. The share amount before rebalancing is then
    #subtracted to find the total change in shares for each asset. The df_trades and df_cash dataframes are then updated.
    
    if trading_day_counter % rebalance_days == 0:
        
        tradeflag = 1
        
        total_money = np.matmul(np.array(arr_latest_prices),arr_cur_shrs)+arr_cash[-1]
        
        start_time4 = timeit.default_timer()
        arr_shares = (total_money*arr_portfolio_weights)/arr_latest_prices
        elapsed4 = timeit.default_timer() - start_time4
        timer.append(elapsed4)
        arr_rebal = arr_shares-arr_cur_shrs
        arr_trade_add = np.array([])
        for i in range(len(ls_tickers)):
            arr_trade_add = np.append(arr_trade_add, np.array([arr_new_shares[i], arr_latest_prices[i], -arr_new_shares[i] * arr_latest_prices[i], arr_trades[-1][ls_trades_shr_index[i]]+arr_new_shares[i], (arr_trades[-1][ls_trades_shr_index[i]]+arr_new_shares[i])*arr_latest_prices[i], arr_divflag[i], tradeflag]))
        arr_trades = np.append(arr_trades,[arr_trade_add], axis = 0)
        arr_cash = np.append(arr_cash, [np.array(arr_cash[-1]+np.sum(arr_leftover_cash))], axis=0)
        trading_day_counter=1
        arr_cur_shrs = np.array(arr_trades[-1][ls_trades_shr_index])
        
        
    else:
        trading_day_counter += 1
    
    arr_yesterday_prices = arr_latest_prices    
        
    #Reaggregates all the data into the dataframe "portfolio_detail" and "portfolio_sum"
    
    arr_new_port_det = np.array([], dtype='f8')
    arr_asset_val = (arr_cur_shrs * arr_latest_prices)
    total_value = np.sum(arr_asset_val)
    arr_actual_weights = arr_asset_val/total_value
    ls_new_port_det = []
    for i in range(len(ls_tickers)):
        
        ls_new_port_det = ls_new_port_det + [arr_latest_prices[i], row[ls_ret_index[i]], arr_actual_weights[i], arr_cur_shrs[i], arr_asset_val[i]]
    start_time5 = timeit.default_timer()
    arr_port_det[count] = ls_new_port_det
    
    arr_new_port_sum = [total_value+arr_cash[-1].item(), total_value, arr_cash[-1].item(), arr_cash[-1].item()/(total_value+arr_cash[-1].item())]
    
    arr_port_sum[count] = arr_new_port_sum
    count += 1
    elapsed5 = timeit.default_timer() - start_time5
elapsed = timeit.default_timer() - start_time


# %%
arr_port_det


# %%
arr_port_sum[-1]


# %%
arr_port_sum


# %%
portfolio_sum


# %%

start_time = timeit.default_timer()
rebalance_days = 4100
trading_day_counter = 1
df_trades = df_trades_initial
portfolio_detail = df_portfolio_detail_initial
portfolio_sum = df_portfolio_sum_initial
df_cash = df_cash_initial
yesterday_prices =  {'SPY': d.loc[date_start]['SPY']['prc'], 'TLT': d.loc[date_start]['TLT']['prc'],           'GLD': d.loc[date_start]['GLD']['prc']}
portfolio_weights = {'SPY':0.5, 'TLT': 0.4, "GLD": 0.1}
#This loop goes through the data set "d", which should be a daily time series of asset returns and prices.
#Returns should be broken into returns including dividends, returns without dividends, and dividend only returns. 

for index, row in d.iterrows():
    tradeflag = 0
    s_divflag, t_divflag, g_divflag = (0,0,0)
    tlt_div_cash = 0
    spy_div_cash = 0
    gld_div_cash = 0
    rebal_spy = 0
    rebal_tlt = 0
    rebal_gld = 0
    leftover_cash = 0
    latest_prices = {'SPY': row['SPY']['prc'], 'TLT': row['TLT']['prc'],           'GLD': row['GLD']['prc']}
    new_spy_shares = 0
    new_gld_shares = 0
    new_tlt_shares = 0
    
    
    #Checks to see if there are any dividend payouts, and if so, calculates the number of shares that are purchased as part of 
    #the reinvestment and the leftover cash if a full share cannot be purchased. Dividend payouts are reinvested into the asset
    #that payed the dividend
    
    if row['SPY']['retd'] > 0:
        s_divflag=1
        spy_div_cash = yesterday_prices['SPY']*df_trades['SPY']['tot shr'].iloc[-1]*row['SPY']['retd']
        new_spy_shares = int(spy_div_cash/latest_prices['SPY'])
        leftover_cash = spy_div_cash-(new_spy_shares*latest_prices['SPY'])
        
    if row['TLT']['retd'] > 0:
        t_divflag = 1
        tlt_div_cash = yesterday_prices['TLT']*df_trades['TLT']['tot shr'].iloc[-1]*row['TLT']['retd']
        new_tlt_shares = int(tlt_div_cash/latest_prices['TLT'])
        leftover_cash = tlt_div_cash-(new_tlt_shares*latest_prices['TLT'])+leftover_cash
    
    if row['GLD']['retd'] > 0:
        g_divflag=1
        gld_div_cash = yesterday_prices['GLD']*df_trades['GLD']['tot shr'].iloc[-1]*row['GLD']['retd']
        new_gld_shares = int(gld_div_cash/latest_prices['GLD'])
        leftover_cash = gld_div_cash-(new_gld_shares*latest_prices['GLD'])+leftover_cash

    #Executes the actual trade, recording it in the df_trades dataframe and updates the df_cash dataframe. This is done before
    #rebalancing incase a) rebalancing happens on the same day in the case of calendar rebalancing, or b) the dividend payout
    #puts the asset over the threshold.
    
    if 1 in [s_divflag, t_divflag, g_divflag]:
        div_trade_add = pd.DataFrame([[new_spy_shares, latest_prices['SPY'], -new_spy_shares * latest_prices['SPY'],df_trades['SPY']['tot shr'].iloc[-1]+new_spy_shares, (df_trades['SPY']['tot shr'].iloc[-1]+new_spy_shares)*latest_prices['SPY'], s_divflag, tradeflag,
                        new_tlt_shares, latest_prices['TLT'], -new_tlt_shares * latest_prices['TLT'],df_trades['TLT']['tot shr'].iloc[-1]+new_tlt_shares, (df_trades['TLT']['tot shr'].iloc[-1]+new_tlt_shares)*latest_prices['TLT'], t_divflag, tradeflag,
                        new_gld_shares, latest_prices['GLD'], -new_gld_shares * latest_prices['GLD'],df_trades['GLD']['tot shr'].iloc[-1]+new_gld_shares, (df_trades['GLD']['tot shr'].iloc[-1]+new_gld_shares)*latest_prices['GLD'], g_divflag, tradeflag]], index = [index], columns = df_trades.columns) 
        df_trades = df_trades.append(div_trade_add)
        df_cash = df_cash.append(pd.DataFrame(df_cash['cash'].iloc[-1]+leftover_cash,columns = ['cash'], 
                         index = [row.name]))
        divflag=0
    
    #Checks to see if rebalancing should occur on this day based on the trade counter chosen. This sells all the shares
    #and reblances using the original weights and the total value available. The share amount before rebalancing is then
    #subtracted to find the total change in shares for each asset. The df_trades and df_cash dataframes are then updated.
    
    if trading_day_counter % rebalance_days == 0:
        tradeflag = 1
        latest_prices = pd.Series(latest_prices)
        cur_shrs = np.array([df_trades['SPY']['tot shr'].iloc[-1], df_trades['TLT']['tot shr'].iloc[-1],df_trades['GLD']['tot shr'].iloc[-1]])
        total_money = np.matmul(np.array(latest_prices),cur_shrs)+df_cash['cash'].iloc[-1]
        
        
        da = DiscreteAllocation(portfolio_weights, latest_prices, total_portfolio_value=total_money)
        shares, cash = da.greedy_portfolio()
        rebal_spy = shares['SPY']-df_trades['SPY']['tot shr'].iloc[-1]
        rebal_tlt = shares['TLT']-df_trades['TLT']['tot shr'].iloc[-1]
        rebal_gld = shares['GLD']-df_trades['GLD']['tot shr'].iloc[-1]
        trade_add = pd.DataFrame([[rebal_spy, latest_prices['SPY'], -rebal_spy*latest_prices['SPY'],shares['SPY'], shares['SPY']*latest_prices['SPY'], divflag, tradeflag,
                        rebal_tlt, latest_prices['TLT'], -rebal_tlt*latest_prices['TLT'], shares['TLT'], shares['TLT']*latest_prices['TLT'], divflag, tradeflag,
                        rebal_gld, latest_prices['GLD'], -rebal_gld*latest_prices['GLD'], shares['GLD'], shares['GLD']*latest_prices['GLD'], divflag, tradeflag]], 
                        index = [index], columns = df_trades.columns)
        df_trades = df_trades.append(trade_add)
        df_cash = df_cash.append(pd.DataFrame(cash,columns = ['cash'],index = [row.name]))
        trading_day_counter=1
    else:
        trading_day_counter += 1
    yesterday_prices = latest_prices    
        
    #Reaggregates all the data into the dataframe "portfolio_detail" and "portfolio_sum"
    
    share_dic = {'SPY':df_trades['SPY']['tot shr'].iloc[-1],
                 'TLT':df_trades['TLT']['tot shr'].iloc[-1],
                 'GLD':df_trades['GLD']['tot shr'].iloc[-1]}
    
    tic_vals = {k: share_dic[k]*latest_prices[k] for k in share_dic}
    total_value = sum(share_dic[k]*latest_prices[k] for k in share_dic)
    actual_weights = {k:tic_vals[k]/total_value for k in tic_vals}
    
    new_port_det = np.array([[latest_prices['SPY'], row["SPY"]['ret'], actual_weights["SPY"], share_dic['SPY'], tic_vals['SPY'],
                    latest_prices['TLT'], row["TLT"]['ret'], actual_weights["TLT"], share_dic['TLT'], tic_vals['TLT'],
                    latest_prices['GLD'], row["GLD"]['ret'], actual_weights["GLD"], share_dic['GLD'], tic_vals['GLD']]])
    portfolio_detail = portfolio_detail.append(pd.DataFrame(new_port_det,columns = portfolio_detail.columns))
    
    new_port_sum = np.array([[total_value+df_cash.iloc[-1][0], total_value, df_cash.iloc[-1][0], df_cash.iloc[-1][0]/(total_value+df_cash.iloc[-1][0])]])
    
    portfolio_sum = portfolio_sum.append(pd.DataFrame(new_port_sum, columns = portfolio_sum.columns))

portfolio_detail.index = d.index
portfolio_sum.index = d.index
elapsed = timeit.default_timer() - start_time


# %%
threshold = .1
df_trades = df_trades_initial
df_cash = df_cash_initial

portfolio_detail = df_portfolio_detail_initial
portfolio_sum = df_portfolio_sum_initial
yesterday_prices =  {'SPY': d.loc[date_start]['SPY']['prc'], 'TLT': d.loc[date_start]['TLT']['prc'],           'GLD': d.loc[date_start]['GLD']['prc']}

#This loop goes through the data set "d", which should be a daily time series of asset returns and prices.
#Returns should be broken into returns including dividends, returns without dividends, and dividend only returns. 

for index, row in d.iterrows():
    tradeflag = 0
    s_divflag, t_divflag, g_divflag = (0,0,0)
    tlt_div_cash = 0
    spy_div_cash = 0
    gld_div_cash = 0
    rebal_spy = 0
    rebal_tlt = 0
    rebal_gld = 0
    leftover_cash = 0
    
    latest_prices = {'SPY': row['SPY']['prc'], 'TLT': row['TLT']['prc'],           'GLD': row['GLD']['prc']}
    new_spy_shares = 0
    new_gld_shares = 0
    new_tlt_shares = 0
    
        #Checks to see if there are any dividend payouts, and if so, calculates the number of shares that are purchased as part of 
    #the reinvestment and the leftover cash if a full share cannot be purchased. Dividend payouts are reinvested into the asset
    #that payed the dividend
    
    if row['SPY']['retd'] > 0:
        s_divflag=1
        spy_div_cash = yesterday_prices['SPY']*df_trades['SPY']['tot shr'].iloc[-1]*row['SPY']['retd']
        new_spy_shares = int(spy_div_cash/latest_prices['SPY'])
        leftover_cash = spy_div_cash-(new_spy_shares*latest_prices['SPY'])
        
    if row['TLT']['retd'] > 0:
        t_divflag = 1
        tlt_div_cash = yesterday_prices['TLT']*df_trades['TLT']['tot shr'].iloc[-1]*row['TLT']['retd']
        new_tlt_shares = int(tlt_div_cash/latest_prices['TLT'])
        leftover_cash = tlt_div_cash-(new_tlt_shares*latest_prices['TLT'])+leftover_cash
    
    if row['GLD']['retd'] > 0:
        g_divflag=1
        gld_div_cash = yesterday_prices['GLD']*df_trades['GLD']['tot shr'].iloc[-1]*row['GLD']['retd']
        new_gld_shares = int(gld_div_cash/latest_prices['GLD'])
        leftover_cash = gld_div_cash-(new_gld_shares*latest_prices['GLD'])+leftover_cash

    #Executes the actual trade, recording it in the df_trades dataframe and updates the df_cash dataframe. This is done before
    #rebalancing incase a) rebalancing happens on the same day in the case of calendar rebalancing, or b) the dividend payout
    #puts the asset over the threshold.
    
    if 1 in [s_divflag, t_divflag, g_divflag]:
        div_trade_add = pd.DataFrame([[new_spy_shares, latest_prices['SPY'], -new_spy_shares * latest_prices['SPY'],df_trades['SPY']['tot shr'].iloc[-1]+new_spy_shares, (df_trades['SPY']['tot shr'].iloc[-1]+new_spy_shares)*latest_prices['SPY'], s_divflag, tradeflag,
                        new_tlt_shares, latest_prices['TLT'], -new_tlt_shares * latest_prices['TLT'],df_trades['TLT']['tot shr'].iloc[-1]+new_tlt_shares, (df_trades['TLT']['tot shr'].iloc[-1]+new_tlt_shares)*latest_prices['TLT'], t_divflag, tradeflag,
                        new_gld_shares, latest_prices['GLD'], -new_gld_shares * latest_prices['GLD'],df_trades['GLD']['tot shr'].iloc[-1]+new_gld_shares, (df_trades['GLD']['tot shr'].iloc[-1]+new_gld_shares)*latest_prices['GLD'], g_divflag, tradeflag]], index = [index], columns = df_trades.columns) 
        df_trades = df_trades.append(div_trade_add)
        df_cash = df_cash.append(pd.DataFrame(df_cash['cash'].iloc[-1]+leftover_cash,columns = ['cash'], 
                         index = [row.name]))
        divflag=0
    
    
    tic_vals = {k: share_dic[k]*latest_prices[k] for k in share_dic}
    total_value = sum(share_dic[k]*latest_prices[k] for k in share_dic)
    actual_weights = {k:tic_vals[k]/total_value for k in tic_vals}
    dev_weights = {k:abs((actual_weights[k]/portfolio_weights[k])-1) for k in actual_weights}
    
    #Checks to see if rebalancing should occur due to an asset passing the threshold. The % deviation is calculated by: 
    #
    #                       |[Actual % of portfolio] / [target % of portfolio] - 1|
    #
    #If this absolute value is greater than the threshold, rebalancing occurs. This sells all the shares and reblances 
    #using the original weights and the total value available. The share amount before rebalancing is then subtracted 
    #to find the total change in shares for each asset. The df_trades and df_cash dataframes are then updated.
    
    if max(dev_weights[k] for k in actual_weights) > threshold:
        tradeflag = 1
        latest_prices = pd.Series(latest_prices)
        cur_shrs = np.array([df_trades['SPY']['tot shr'].iloc[-1], df_trades['TLT']['tot shr'].iloc[-1],df_trades['GLD']['tot shr'].iloc[-1]])
        total_money = np.matmul(np.array(latest_prices),cur_shrs)+df_cash['cash'].iloc[-1]
        
        
        da = DiscreteAllocation(portfolio_weights, latest_prices, total_portfolio_value=total_money)
        shares, cash = da.greedy_portfolio()
        rebal_spy = shares['SPY']-df_trades['SPY']['tot shr'].iloc[-1]
        rebal_tlt = shares['TLT']-df_trades['TLT']['tot shr'].iloc[-1]
        rebal_gld = shares['GLD']-df_trades['GLD']['tot shr'].iloc[-1]
        trade_add = pd.DataFrame([[rebal_spy, latest_prices['SPY'], -rebal_spy*latest_prices['SPY'],shares['SPY'], shares['SPY']*latest_prices['SPY'], divflag, tradeflag,
                        rebal_tlt, latest_prices['TLT'], -rebal_tlt*latest_prices['TLT'], shares['TLT'], shares['TLT']*latest_prices['TLT'], divflag, tradeflag,
                        rebal_gld, latest_prices['GLD'], -rebal_gld*latest_prices['GLD'], shares['GLD'], shares['GLD']*latest_prices['GLD'], divflag, tradeflag]], 
                        index = [index], columns = df_trades.columns)
        df_trades = df_trades.append(trade_add)
        df_cash = df_cash.append(pd.DataFrame(cash,columns = ['cash'],index = [row.name]))
        trading_day_counter=1
    else:
        trading_day_counter += 1
        yesterday_prices = latest_prices 
    
    #Reaggregates all the data into the dataframe "portfolio_detail" and "portfolio_sum"
    
    share_dic = {'SPY':df_trades['SPY']['tot shr'].iloc[-1],
                 'TLT':df_trades['TLT']['tot shr'].iloc[-1],
                 'GLD':df_trades['GLD']['tot shr'].iloc[-1]}
    
    tic_vals = {k: share_dic[k]*latest_prices[k] for k in share_dic}
    total_value = sum(share_dic[k]*latest_prices[k] for k in share_dic)
    actual_weights = {k:tic_vals[k]/total_value for k in tic_vals}
    
    new_port_det = np.array([[latest_prices['SPY'], row["SPY"]['ret'], actual_weights["SPY"], share_dic['SPY'], tic_vals['SPY'],
                    latest_prices['TLT'], row["TLT"]['ret'], actual_weights["TLT"], share_dic['TLT'], tic_vals['TLT'],
                    latest_prices['GLD'], row["GLD"]['ret'], actual_weights["GLD"], share_dic['GLD'], tic_vals['GLD']]])
    portfolio_detail = portfolio_detail.append(pd.DataFrame(new_port_det,columns = portfolio_detail.columns))
    
    new_port_sum = np.array([[total_value+df_cash.iloc[-1][0], total_value, df_cash.iloc[-1][0], df_cash.iloc[-1][0]/(total_value+df_cash.iloc[-1][0])]])
    
    portfolio_sum = portfolio_sum.append(pd.DataFrame(new_port_sum, columns = portfolio_sum.columns))

portfolio_detail.index = d.index
portfolio_sum.index = d.index


# %%
portfolio_detail


# %%
portfolio_sum


# %%
df_trades

# %% [markdown]
# ## Analysis of Returns
# 
# This is done for Threshold Rebalancing
# 
# ### Compaison of underlying assets

# %%
portfolio_cumret = portfolio_sum.total_value/money
spy_cum_ret = d.SPY.ret.cumprod()
gld_cum_ret = d.GLD.ret.cumprod()
tlt_cum_ret = d.TLT.ret.cumprod()


# %%
spy_cum_ret[-1]+gld_cum_ret[-1]+tlt_cum_ret[-1]


# %%
gld_cum_ret[-1]


# %%
spy_ret = np.array(d.SPY.ret[1:]-1)*.5
gld_ret = np.array(d.GLD.ret[1:]-1)*.1
tlt_ret = np.array(d.TLT.ret[1:]-1)*.4
port_ret = spy_ret+gld_ret+tlt_ret
(port_ret+1).cumprod()


# %%
spy_ret = np.array(d.SPY.ret-1)

(spy_ret[1:3]+1).cumprod()*.5


# %%
((spy_ret[1:]*.5)+1).cumprod()+((gld_ret[1:]*.5)+1).cumprod()


# %%
cum_ret = pd.DataFrame(np.array(port_ret).T, columns = ['portfolio'], index = portfolio_detail.index)


# %%
spy_cum_ret = d.SPY.ret[1:].cumprod()*.5
gld_cum_ret = d.GLD.ret[1:].cumprod()*.1
tlt_cum_ret = d.TLT.ret[1:].cumprod()*.4

print(spy_cum_ret+gld_cum_ret+tlt_cum_ret)


# %%
spy_cum_ret = ((d.SPY.ret[1:]-1)*.5+1).cumprod()
gld_cum_ret = ((d.GLD.ret[1:]-1)*.1+1).cumprod()
tlt_cum_ret = ((d.TLT.ret[1:]-1)*.4+1).cumprod()

print(spy_cum_ret,gld_cum_ret,tlt_cum_ret)


# %%
portfolio_weights = np.array([.5,.4,.1])
StockReturns = pd.DataFrame(np.array([d.SPY.ret[1:]-1, d.GLD.ret[1:]-1, d.TLT.ret[1:]-1]).T)
port_ret = StockReturns.mul(portfolio_weights, axis=1).sum(axis=1)


# %%
((d.SPY.ret[1:]).cumprod()*.5)


# %%
np.array([d.SPY.ret[1:]-1, d.GLD.ret[1:]-1, d.TLT.ret[1:]-1]).


# %%
port = np.array([d.SPY.ret,d.TLT.ret, d.GLD.ret]).T


# %%
port

# %% [markdown]
# ### Comparison between an untouched portfolio and a rebalanced portfolio

# %%
portfolio_weights = np.array([.5, .4, .1])


# %%
port_ret = np.matmul(port, portfolio_weights)


# %%
port_ret.T.shape


# %%
df_port_ret = pd.DataFrame([port_ret, cum_ret['portfolio']]).T


# %%
df_port_ret.columns = ['untouched_port', 'rebal_port']


# %%
df_port_ret['untouched_port'] = df_port_ret['untouched_port'].cumprod()


# %%
df_port_ret.plot(figsize = (10,10))


# %%
df_port_ret


# %%
portfolio_sum


# %%
df_trades


# %%
df_trades[df_trades.TLT.rebalflag == 1]


# %%
portfolio_detail


# %%
15666.33*3.7


# %%
portfolio_sum.total_value[-1]+15666.33*3.72/2


