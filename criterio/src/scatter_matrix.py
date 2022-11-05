import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import sys
import pathlib
FILE_PATH = str(pathlib.Path(__file__).parent.resolve())
PROJECT_ROOT = re.search(r'[\w.-_\\ ()\[\]]+criterio', FILE_PATH).group()
sys.path.append(PROJECT_ROOT) 
import criterio.market as mm


# Def stocks tickets

ETFS = ['QQQ','SPY','DIA','XLF','EWZ','IWM','EEM','XLE','ARKK']

COMPANIES = ['DIS', 'KO', 'NKE', 'WMT', 'CAT', 'DE', 'JNJ', 'JPM', 'MCD', 'AUY',
           'YPF', 'AMD', 'TSLA', 'MSFT', 'AAPL', 'AMZN', 'FB', 'NFLX', 'EBAY',
           'BA', 'MELI', 'BIIB', 'BRFS', 'GS', 'TEN', 'XOM', 'ERJ', 'COST',
           'UNH', 'GOLD', 'VIST', 'AXP', 'GOOGL', 'FCX', 'BABA', 'HMY', 'C',
           'NVDA', 'INTC', 'WFC', 'PBR', 'BIOX', 'X', 'QCOM', 'BBD',
           'BIDU', 'PYPL', 'VALE', 'ETSY', 'AVGO', 'LMT', 'ITUB', 'SAP', 'BAC',
           'DESP', 'FSLR', 'GE', 'HMC', 'VZ']

TICKETS = ETFS + COMPANIES


TICKETS_SELECTED = ['QQQ','EEM','XLE','ARKK']

MARKET = mm.get_market_data()

scatter_matrix = MARKET[TICKETS].corr()

# print(scatter_matrix)

np_scatter_matrix = np.array(scatter_matrix)

# np.set_printoptions(threshold = 4)
# print(repr(np_scatter_matrix))
# print(np.array_repr(np_scatter_matrix))

for l in np_scatter_matrix:
    print(repr(l))

# pd.plotting.scatter_matrix(MARKET[TICKETS_SELECTED], diagonal='kde',alpha=0.1)
# plt.show()
