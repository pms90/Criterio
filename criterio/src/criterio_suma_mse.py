import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
import pathlib
THIS_FILE_PATH = str(pathlib.Path(__file__).parent.resolve())

# import os
# WORK_DIR = os.getcwd()
# THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# print('WORK_DIR', WORK_DIR)
# print('THIS_FILE_DIR', THIS_FILE_DIR)

# import site
# site.addsitedir(THIS_FILE_DIR)

import criterio1.module1 as mm



# -----------------------------------------------------------------------------
# mse: mean squared error
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

# print('Definiendo MARKET (objeto)')
# MARKET = mm.get_market_data()

# xle = mm.Stock(MARKET['XLE'])
# eem = mm.Stock(MARKET['EEM'])
# spy = mm.Stock(MARKET['SPY'])

# xle_path = str(mm.file_path()+'\\data\\XLE.pkl')
# xle = mm.load_object(xle_path)
# eem_path = str(mm.file_path()+'\\data\\EEM.pkl')
# eem = mm.load_object(eem_path)
# spy_path = str(mm.file_path()+'\\data\\SPY.pkl')
# spy = mm.load_object(spy_path)





# # Save Market
# for ticket in TICKETS:
#     print('Creating '+ticket+' Stock object.')
#     stock = mm.Stock(MARKET[ticket][:], normalize=-1)
#     print('Saving '+ticket+'.pkl')
#     out_path = mm.file_path()+'\\data\\'+ticket+'.pkl'
#     mm.save_object(stock, out_path)





# for ticket in TICKETS:
#     stock_path = str(mm.file_path()+'\\data\\'+ticket+'.pkl')
#     stock = mm.load_object(stock_path)  
#     p, d = apply_criterio(stock, [-1.3804947612096556, 0.005198044180955985])
#     print(ticket.ljust(5), round(p,1), round(100*d/stock.len))





# -----------------------------------------------------------------------------
# Zonas de compra
# -----------------------------------------------------------------------------


def good_buys(t, p, hw=30):
    gbd = []
    gbp = []
    gpgp = np.zeros(2*hw) 
    gpgp_count = 0
    for i in range(len(p))[hw:-hw]: # muy optimizable
        m = min(p[i-hw:i+hw]) 
        if p[i]==m:
            gbd.append(t[i])
            gbp.append(p[i])

            gp = p[i-hw:i+hw]/p[i]
            gpgp = gpgp+np.array(gp)
            gpgp_count += 1
            # plt.plot(gp, linewidth=0.5, alpha=0.8)

    return gbd, gbp, (gpgp/gpgp_count)

hw = 60

gbm = np.zeros(2*hw)
gbm_count = 0
gbzones = []
for ticket in TICKETS:
    if ticket=='DESP':
        print('Stock separado para testing')
        continue
    stock_path = str(THIS_FILE_PATH+'\\data\\'+ticket+'.pkl')
    stock = mm.load_object(stock_path)  
    t = stock.dates
    p = stock.prices
    good_buys_zone = good_buys(t, p, hw)[2]
    gbm = gbm + good_buys_zone
    gbm_count += 1
    gbzones.append(good_buys_zone)
    # plt.plot(good_buys_zone, alpha=0.7)
buy_zone = gbm/gbm_count







# -----------------------------------------------------------------------------
#  Zonas de venta
# -----------------------------------------------------------------------------

def good_sells(t, p, hw=30):
    gbd = []
    gbp = []
    gpgp = np.zeros(2*hw) 
    gpgp_count = 0
    for i in range(len(p))[hw:-hw]: # muy optimizable
        m = max(p[i-hw:i+hw]) 
        if p[i]==m:
            gbd.append(t[i])
            gbp.append(p[i])

            gp = p[i-hw:i+hw]/p[i]
            gpgp = gpgp+np.array(gp)
            gpgp_count += 1
            # plt.plot(gp, linewidth=0.5, alpha=0.8)

    return gbd, gbp, (gpgp/gpgp_count)



gsm = np.zeros(2*hw)
gsm_count = 0
gszones = []
for ticket in TICKETS:
    if ticket=='DESP':
        print('Stock separado para testing')
        continue
    stock_path = str(THIS_FILE_PATH+'\\data\\'+ticket+'.pkl')
    stock = mm.load_object(stock_path)  
    t = stock.dates
    p = stock.prices
    good_buys_zone = good_sells(t, p, hw)[2]
    gsm = gsm + good_buys_zone
    gsm_count += 1
    gszones.append(good_buys_zone)
    # plt.plot(good_buys_zone, alpha=0.7)
sell_zone = gsm/gsm_count













# -----------------------------------------------------------------------------
#  Graficar zonas de compra y venta sobre un ticket
# -----------------------------------------------------------------------------

from sklearn.metrics import mean_squared_error

# rmsl = []
# for zone in gbzones:
#     mse = mean_squared_error(zone, sell_zone, squared=True)
#     rmsl.append(mse)
# rmsl.sort()
# for i, r in enumerate(rmsl):
#     print(i, r)




ticket = 'DESP'
stock_path = str(THIS_FILE_PATH+'\\data\\'+ticket+'.pkl')
stock = mm.load_object(stock_path)

t = stock.dates
p = stock.prices
b_tz = []
b_pz = []
s_tz = []
s_pz = []

rhw = 15
for i in range(len(p))[hw+rhw:]:
    zone = p[i-hw-rhw:i]/p[i-rhw]
    b_mse = mean_squared_error(zone, buy_zone[:-(hw-rhw)], squared=True)
    s_mse = mean_squared_error(zone, sell_zone[:-(hw-rhw)], squared=True)
    if b_mse<0.006:
        print(b_mse)
        b_tz.append(t[i])
        b_pz.append(p[i])
    if s_mse<0.008:
        s_tz.append(t[i])
        s_pz.append(p[i])

plt.plot(t,p)
plt.plot(t,stock.m1)
plt.plot(t,stock.m3)
plt.plot(t,stock.m4)
plt.plot(b_tz,b_pz, '*', color='orange')
plt.plot(s_tz,s_pz, '*', color='red')
plt.show()

fig2 = plt.figure()
plt.plot(buy_zone)
plt.show()
fig3 = plt.figure()
plt.plot(sell_zone)
plt.show()











# fig1 = plt.figure()
# plt.plot(good_buys(t, p)[2])
# plt.show()

# fig2 = plt.figure()
# plt.plot(t, p)
# plt.plot(good_buys(t, p)[0], good_buys(t, p)[1], '*', color='orange')
# plt.show()


































# 