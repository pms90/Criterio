import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
import re
import sys
import pathlib


FILE_PATH = str(pathlib.Path(__file__).parent.resolve())
PROJECT_ROOT = re.search(r'[\w.-_\\ ()\[\]]+criterio', FILE_PATH).group()
sys.path.append(PROJECT_ROOT) 

import criterio.market as mm




# -----------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------

# Def stocks tickets

ETFS = ['QQQ','SPY','DIA','XLF','EWZ','IWM','EEM','XLE','ARKK','ARKQ']

COMPANIES = ['DIS', 'KO', 'NKE', 'WMT', 'CAT', 'DE', 'JNJ', 'JPM', 'MCD', 'AUY',
           'YPF', 'AMD', 'TSLA', 'MSFT', 'AAPL', 'AMZN', 'FB', 'NFLX', 'EBAY',
           'BA', 'MELI', 'BIIB', 'BRFS', 'GS', 'TEN', 'XOM', 'ERJ', 'COST',
           'UNH', 'GOLD', 'VIST', 'AXP', 'GOOGL', 'FCX', 'BABA', 'HMY', 'C',
           'NVDA', 'INTC', 'WFC', 'PBR', 'BIOX', 'X', 'QCOM', 'BBD',
           'BIDU', 'PYPL', 'VALE', 'ETSY', 'AVGO', 'LMT', 'ITUB', 'SAP', 'BAC',
           'DESP', 'FSLR', 'GE', 'HMC', 'VZ']

TICKETS = ETFS + COMPANIES


# -----------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------

# path = mm.PROJECT_ROOT+'\\data\\zones\\'+'buy_zone'+'.pkl'
# buy_zone = mm.load_object(path)

# path = mm.PROJECT_ROOT+'\\data\\zones\\'+'good_buys_zones'+'.pkl'
# gbz = mm.load_object(path)

# print('len(Zonas):', len(gbz))

# for z in gbz:
#     # plt.plot(z, color='black', alpha=0.05)
#     plt.plot(z[0], z[-1], '.', color='black', alpha=0.2)
# plt.plot(buy_zone[0], buy_zone[-1], '.', alpha=0.8)
# plt.show()


# -----------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------

# path = mm.PROJECT_ROOT+'\\data\\zones\\'+'buy_zone'+'.pkl'
# buy_zone = mm.load_object(path)

# path = mm.PROJECT_ROOT+'\\data\\zones\\'+'sell_zone'+'.pkl'
# sell_zone = mm.load_object(path)

# print(buy_zone.shape, sell_zone.shape)
# print(repr(buy_zone))
# print(repr(sell_zone))

# plt.plot(buy_zone)
# plt.plot(sell_zone)
# plt.show()



# # -----------------------------------------------------------------------------
# #  Graficar zonas de compra y venta sobre un ticket
# # -----------------------------------------------------------------------------

from sklearn.metrics import mean_squared_error

# rmsl = []
# for zone in gbzones:
#     mse = mean_squared_error(zone, sell_zone, squared=True)
#     rmsl.append(mse)
# rmsl.sort()
# for i, r in enumerate(rmsl):
#     print(i, r)




ticket = 'YPF'
stock_path = str(PROJECT_ROOT+'\\data\\tickets\\'+ticket+'.pkl')
stock = mm.load_object(stock_path)

t = stock.dates
p = stock.prices
b_i = []
b_tz = []
b_pz = []
s_i = []
s_tz = []
s_pz = []

lw = 25
rw = 5

plots=0

for i in range(len(p))[lw+rw+1:]:
    zone = p[i-lw-rw:i+1]
    zone = zone/zone[-1]
    buy_zone = mm.buy_zone[60-lw:60+rw+1]
    buy_zone = buy_zone/buy_zone[-1]
    sell_zone = mm.sell_zone[60-lw:60+rw+1]
    sell_zone = sell_zone/sell_zone[-1]

    b_mse_1 = mean_squared_error(zone[:-2*rw], buy_zone[:-2*rw], squared=True)
    b_mse_2 = mean_squared_error(zone[-2*rw:], buy_zone[-2*rw:], squared=True)
    s_mse_1 = mean_squared_error(zone[:-2*rw], sell_zone[:-2*rw], squared=True)
    s_mse_2 = mean_squared_error(zone[-2*rw:], sell_zone[-2*rw:], squared=True)

    if b_mse_1<0.001 and b_mse_2<0.0005:
        b_i.append(i)
        b_tz.append(t[i])
        b_pz.append(p[i])
        # plt.plot(zone)
        # plt.plot(buy_zone)
        # plt.show()
        # plots += 1
        # if plots > 10: break
        
    if s_mse_1<0.001 and s_mse_2<0.0005:
        s_i.append(i)
        s_tz.append(t[i])
        s_pz.append(p[i])
        # plt.plot(zone)
        # plt.plot(sell_zone)
        # plt.show()
        # plots += 1
        # if plots > 10: break


plt.plot(t, p, linewidth=0.6)
plt.plot(t, stock.m1, color='gray', alpha=0.1)
plt.plot(t, stock.m3, color='gray', alpha=0.1)
plt.plot(t, stock.m4, color='gray', alpha=0.1)
plt.plot(b_tz,b_pz, '*', color='orange', alpha=0.5)
plt.plot(s_tz,s_pz, '*', color='red', alpha=0.5)

for e in b_i:
    plt.plot(t[e-lw-rw:e+1], buy_zone*p[e], linewidth=0.75, color='orange')

plt.show()

# plt.plot(mm.buy_zone[60-lw:60+rw])
# plt.show()


# # -----------------------------------------------------------------------------
# #  Graficar zonas de compra y venta sobre un ticket
# # -----------------------------------------------------------------------------

# from sklearn.metrics import mean_squared_error

# # rmsl = []
# # for zone in gbzones:
# #     mse = mean_squared_error(zone, sell_zone, squared=True)
# #     rmsl.append(mse)
# # rmsl.sort()
# # for i, r in enumerate(rmsl):
# #     print(i, r)




# ticket = 'DESP'
# stock_path = str(THIS_FILE_PATH+'\\data\\'+ticket+'.pkl')
# stock = mm.load_object(stock_path)

# t = stock.dates
# p = stock.prices
# b_tz = []
# b_pz = []
# s_tz = []
# s_pz = []

# rhw = 15
# for i in range(len(p))[hw+rhw:]:
#     zone = p[i-hw-rhw:i]/p[i-rhw]
#     b_mse = mean_squared_error(zone, buy_zone[:-(hw-rhw)], squared=True)
#     s_mse = mean_squared_error(zone, sell_zone[:-(hw-rhw)], squared=True)
#     if b_mse<0.006:
#         print(b_mse)
#         b_tz.append(t[i])
#         b_pz.append(p[i])
#     if s_mse<0.008:
#         s_tz.append(t[i])
#         s_pz.append(p[i])

# plt.plot(t,p)
# plt.plot(t,stock.m1)
# plt.plot(t,stock.m3)
# plt.plot(t,stock.m4)
# plt.plot(b_tz,b_pz, '*', color='orange')
# plt.plot(s_tz,s_pz, '*', color='red')
# plt.show()

# fig2 = plt.figure()
# plt.plot(buy_zone)
# plt.show()
# fig3 = plt.figure()
# plt.plot(sell_zone)
# plt.show()











# # fig1 = plt.figure()
# # plt.plot(good_buys(t, p)[2])
# # plt.show()

# # fig2 = plt.figure()
# # plt.plot(t, p)
# # plt.plot(good_buys(t, p)[0], good_buys(t, p)[1], '*', color='orange')
# # plt.show()


































# 