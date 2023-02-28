"""
Bank data
Initial data
"""
__date__ = "2023-02-27"
__author__ = "NickTerziyski"
# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd
import seaborn as sns
# %% --------------------------------------------------------------------------
# Set Random State
# -----------------------------------------------------------------------------
rng = np.random.RandomState(123)
# %% --------------------------------------------------------------------------
# Transactions monthly load
# -----------------------------------------------------------------------------
trans_filepath = r'Data\transactions_tm1_e.csv'
trans_df_original = pd.read_csv(trans_filepath)
trans_df_original['date'] = pd.to_datetime(trans_df_original['date'])
trans_df = trans_df_original.groupby(by = ['customer_id', 'date']).sum()
trans_df.drop(['account_id', 'deposit', 'withdrawal'], axis = 'columns', inplace=True)
# %% --------------------------------------------------------------------------
# Other data load
# -----------------------------------------------------------------------------
cust_filepath = r'Data\customers_tm1_e.csv'
cust_df = pd.read_csv(cust_filepath, index_col='customer_id')
file_path3 = r'Data\FEDFUNDS.csv'
fed_data = pd.read_csv(file_path3)
# %% --------------------------------------------------------------------------
# Join trans cust
# -----------------------------------------------------------------------------
merged_df = trans_df.join(cust_df) 
# %% --------------------------------------------------------------------------
# To datetime
# -----------------------------------------------------------------------------
merged_df['dob'] = pd.to_datetime(merged_df['dob'])
merged_df['creation_date'] = pd.to_datetime(merged_df['creation_date'])
merged_df['state'] = merged_df['state'].astype(str)
# %% --------------------------------------------------------------------------
#  create age, account maturity columns
# -----------------------------------------------------------------------------
merged_df['age'] = ((merged_df.index.get_level_values('date') - merged_df['dob']).dt.days / 365.25).round(2)
merged_df['account_maturity'] = ((merged_df.index.get_level_values('date') - merged_df['creation_date']).dt.days / 365.25).round(2)
# %% --------------------------------------------------------------------------
# # concurrent bank balance
# # -----------------------------------------------------------------------------
merged_df['conc_bal'] = merged_df.groupby('customer_id')['amount'].cumsum()
merged_df['current_balance'] = merged_df['start_balance'] + merged_df['conc_bal']
# %% --------------------------------------------------------------------------
# Fix states
# -----------------------------------------------------------------------------
merged_df.replace(to_replace='NY', value = 'New York', inplace=True)
merged_df.replace('TX', value='Texas', inplace=True)
merged_df.replace('CALIFORNIA', value='California', inplace=True)
merged_df.drop(merged_df.loc[merged_df['state']=='-999'].index, inplace=True)
# %% --------------------------------------------------------------------------
# Fix balance
# -----------------------------------------------------------------------------
merged_df = merged_df.drop(merged_df[merged_df.current_balance > 200000].index)
merged_df = merged_df.drop(merged_df[merged_df.current_balance < 0].index)
merged_df = merged_df.drop(columns=['conc_bal'])

# %% --------------------------------------------------------------------------
# add pct change balance column
# -----------------------------------------------------------------------------
merged_df['balance pct change'] = merged_df['amount']/merged_df['current_balance']*100
# %% --------------------------------------------------------------------------
# Fed data
# -----------------------------------------------------------------------------
start_date = '2006-12-31'
end_date = '2020-05-31'
mask = (fed_data['DATE'] >= start_date) & (fed_data['DATE'] <= end_date)
new_fed = fed_data.loc[mask]
new_fed = new_fed.rename(columns={'DATE': 'date'})
new_fed['date'] = pd.to_datetime(new_fed['date'])
new_fed['date'] = new_fed['date'] + MonthEnd(0)
new_fed = new_fed.set_index('date')
# %% --------------------------------------------------------------------------
# Join fedfunds
# -----------------------------------------------------------------------------
merged_df = merged_df.join(new_fed)

# %% --------------------------------------------------------------------------
# add growth quarterly data
# -----------------------------------------------------------------------------
gdp_fpath = r'Data\GDP.csv'
gdp = pd.read_csv(gdp_fpath)
gdp = gdp.rename(columns={'DATE': 'date'})
gdp['date'] = pd.to_datetime(gdp['date'])


# %% --------------------------------------------------------------------------
# load gdp
# -----------------------------------------------------------------------------
gdp = gdp.set_index('date')

# %% --------------------------------------------------------------------------
# forward fill
# -----------------------------------------------------------------------------
gdp = gdp.resample('M').ffill()

# %% --------------------------------------------------------------------------
# merge growth data
# -----------------------------------------------------------------------------
merged_df = merged_df.join(gdp)
# %% --------------------------------------------------------------------------
# add growth quarterly data
# -----------------------------------------------------------------------------
inf_fpath = r'Data\INFLATION.csv'
inf = pd.read_csv(inf_fpath)
inf = inf.rename(columns={'DATE': 'date'})
inf['date'] = pd.to_datetime(inf['date'])

# %% --------------------------------------------------------------------------
# load inflation
# -----------------------------------------------------------------------------
inf['date'] = inf['date'] + MonthEnd(0)
inf = inf.set_index('date')

# %% --------------------------------------------------------------------------
# merge growth data
# -----------------------------------------------------------------------------
merged_df = merged_df.join(inf)
merged_df = merged_df.rename(columns={'CORESTICKM159SFRBATL': 'Inflation'})

# %% --------------------------------------------------------------------------
# plotting inflation with current balance for a sample customer
# -----------------------------------------------------------------------------

customer_107 = merged_df.query('customer_id == 107')

sns.regplot(x='current_balance', y='FEDFUNDS', data=customer_107)

# Add axis labels
plt.xlabel('Account Balance')
plt.ylabel('Interest Rate')

# Display the plot
plt.show()

# %% --------------------------------------------------------------------------
# {1:Enter description for cell}
# -----------------------------------------------------------------------------
plt.scatter(x='Inflation', y='FEDFUNDS', data=merged_df)

# %% --------------------------------------------------------------------------
# correlation heatmap
# -----------------------------------------------------------------------------
sns.heatmap(merged_df.corr())