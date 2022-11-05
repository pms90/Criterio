import matplotlib.pyplot as plt
import numpy as np
import re
import sys
import pathlib
FILE_PATH = str(pathlib.Path(__file__).parent.resolve())
PROJECT_ROOT = re.search(r'[\w.-_\\ ()\[\]]+criterio', FILE_PATH).group()
sys.path.append(PROJECT_ROOT) 
import criterio.market as mm

# -----------------------------------------------------------------------------
# Def stocks tickets
# -----------------------------------------------------------------------------

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
# Zonas de compra
# -----------------------------------------------------------------------------


def good_buys(t, p, hw=30):
    gbd = []
    gbp = []
    gpgp = np.zeros(2*hw) 
    gpgp_count = 0
    gbzones_ticket = []
    for i in range(len(p))[hw:-hw]: # muy optimizable
        m = min(p[i-hw:i+hw]) 
        if p[i]==m:
            gbd.append(t[i])
            gbp.append(p[i])
            gp = p[i-hw:i+hw]/p[i] # normalizar
            gp = np.array(gp)
            gpgp = gpgp+gp
            gbzones_ticket.append(gp)
            gpgp_count += 1
            # plt.plot(gp, linewidth=0.5, alpha=0.8)
    return gbd, gbp, (gpgp/gpgp_count), gbzones_ticket

def good_sells(t, p, hw=30):
    gbd = []
    gbp = []
    gpgp = np.zeros(2*hw) 
    gpgp_count = 0
    gbzones_ticket = []
    for i in range(len(p))[hw:-hw]: # muy optimizable
        m = max(p[i-hw:i+hw]) 
        if p[i]==m:
            gbd.append(t[i])
            gbp.append(p[i])
            gp = p[i-hw:i+hw]/p[i] # normalizar
            gp = np.array(gp)
            gpgp = gpgp+gp
            gbzones_ticket.append(gp)
            gpgp_count += 1
            # plt.plot(gp, linewidth=0.5, alpha=0.8)
    return gbd, gbp, (gpgp/gpgp_count), gbzones_ticket


hw = 60

gbm = np.zeros(2*hw)
gbm_count = 0
gbzones = []

gsm = np.zeros(2*hw)
gsm_count = 0
gszones = []

for ticket in TICKETS:
    # if ticket=='DESP':
    #     print('Stock separado para testing')
    #     continue
    stock_path = str(mm.PROJECT_ROOT+'\\data\\tickets\\'+ticket+'.pkl')
    stock = mm.load_object(stock_path)  

    t = stock.dates
    p = stock.prices

    good_buys_zone = good_buys(t, p, hw)
    gbm = gbm + good_buys_zone[2]
    gbm_count += 1
    gbzones+=good_buys_zone[3]
    # plt.plot(good_buys_zone, alpha=0.7)

    good_sells_zone = good_sells(t, p, hw)
    gsm = gsm + good_sells_zone[2]
    gsm_count += 1
    gszones+=good_sells_zone[3]
    # plt.plot(good_sells_zone, alpha=0.7)

buy_zone = gbm/gbm_count
sell_zone = gsm/gsm_count


out_path = mm.PROJECT_ROOT+'\\data\\zones\\'+'good_buys_zones'+'.pkl'
mm.save_object(gbzones, out_path)
out_path = mm.PROJECT_ROOT+'\\data\\zones\\'+'good_sells_zones'+'.pkl'
mm.save_object(gszones, out_path)

out_path = mm.PROJECT_ROOT+'\\data\\zones\\'+'buy_zone'+'.pkl'
mm.save_object(buy_zone, out_path)
out_path = mm.PROJECT_ROOT+'\\data\\zones\\'+'sell_zone'+'.pkl'
mm.save_object(sell_zone, out_path)
































# 