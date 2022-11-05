import numpy as np
from sklearn.metrics import mean_squared_error
import criterio.market as mm


# -----------------------------------------------------------------------------
# Def stocks tickets
# -----------------------------------------------------------------------------

ETFS = ['QQQ','SPY','DIA','XLF','EWZ','IWM','EEM','XLE','ARKK','ARKQ']

COMPANIES = ['DIS', 'KO', 'NKE', 'WMT', 'CAT', 'DE', 'JNJ', 'JPM', 'MCD', 
        'YPF', 'AMD', 'TSLA', 'MSFT', 'AAPL', 'AMZN', 'META', 'NFLX', 'EBAY',
        'BA', 'MELI', 'BIIB', 'BRFS', 'GS', 'TEN', 'XOM', 'ERJ', 'COST',
        'UNH', 'GOLD', 'VIST', 'AXP', 'GOOGL', 'FCX', 'BABA', 'HMY', 'C',
        'NVDA', 'INTC', 'WFC', 'PBR', 'BIOX', 'X', 'QCOM', 'BBD',
        'BIDU', 'PYPL', 'VALE', 'ETSY', 'AVGO', 'LMT', 'ITUB', 'SAP', 'BAC',
        'DESP', 'FSLR', 'GE', 'HMC', 'VZ', 'AUY'
        ]

TICKERS = ETFS + COMPANIES

# -----------------------------------------------------------------------------
# Def funcion criterio de compra y venta
# -----------------------------------------------------------------------------

buy_zone = np.array([
       1.32624786, 1.32497076, 1.32332715, 1.32071488, 1.31687751,
       1.31146969, 1.307898  , 1.30822904, 1.30884173, 1.30830541,
       1.30978576, 1.31040061, 1.30526184, 1.30381994, 1.29914089,
       1.29641471, 1.29805674, 1.29366932, 1.29218908, 1.28719439,
       1.28344188, 1.27741757, 1.27445503, 1.27524037, 1.27466184,
       1.26941835, 1.26800794, 1.26454856, 1.26069213, 1.25649106,
       1.25462772, 1.2500794 , 1.24947635, 1.24453914, 1.23885306,
       1.23975982, 1.23349445, 1.23229684, 1.22510238, 1.22158075,
       1.21750228, 1.21187834, 1.20577293, 1.19420454, 1.18935061,
       1.18449768, 1.17385226, 1.17062394, 1.16356666, 1.15417985,
       1.14405225, 1.13856854, 1.13175906, 1.11710945, 1.11171844,
       1.10114461, 1.0875303 , 1.07227678, 1.05463926, 1.03376644,
       1.        , 1.03522336, 1.05610466, 1.07143855, 1.08182574,
       1.09173012, 1.10034997, 1.10761141, 1.11315194, 1.11702432,
       1.12342896, 1.12991306, 1.13475147, 1.14124813, 1.14797704,
       1.15336039, 1.15473407, 1.16159455, 1.1688763 , 1.17456186,
       1.1768546 , 1.17857695, 1.18338797, 1.18557424, 1.19167812,
       1.19813592, 1.20337301, 1.20654717, 1.21071744, 1.21446664,
       1.21833573, 1.2202379 , 1.22547348, 1.22947882, 1.23109634,
       1.23255936, 1.23464453, 1.23559816, 1.23654824, 1.23763124,
       1.24137014, 1.24631346, 1.25126426, 1.25726629, 1.26286905,
       1.26566153, 1.26770304, 1.27023297, 1.2716439 , 1.26964588,
       1.27307267, 1.27620658, 1.27996703, 1.28243847, 1.28155994,
       1.28372719, 1.28439142, 1.28384753, 1.28401769, 1.28445279])
sell_zone = np.array([
       0.83198896, 0.83316768, 0.83135007, 0.83285428, 0.83276027,
       0.83265292, 0.83215954, 0.8325939 , 0.83283858, 0.83436776,
       0.83624423, 0.8372518 , 0.83734055, 0.83786709, 0.83860133,
       0.8392713 , 0.84075975, 0.84253522, 0.84355975, 0.84655289,
       0.84986587, 0.85068896, 0.85102653, 0.85217338, 0.85298077,
       0.8540731 , 0.85527185, 0.85754018, 0.8600439 , 0.86214543,
       0.86253566, 0.8644852 , 0.86805356, 0.87219232, 0.87444016,
       0.87606897, 0.87793055, 0.8807116 , 0.88262999, 0.88541144,
       0.88925626, 0.89251453, 0.89555601, 0.89808809, 0.90220211,
       0.90484036, 0.90775952, 0.91080144, 0.91381293, 0.91677246,
       0.91998213, 0.92351811, 0.9278974 , 0.93156728, 0.93634424,
       0.94267675, 0.94947313, 0.9573037 , 0.96519265, 0.97760401,
       1.        , 0.97929221, 0.96607828, 0.9548706 , 0.94816077,
       0.94067455, 0.93561773, 0.93125331, 0.92714558, 0.92254629,
       0.91923462, 0.91482524, 0.91116368, 0.9057018 , 0.90282115,
       0.89934672, 0.89687547, 0.89333345, 0.89096573, 0.88986101,
       0.88734518, 0.88593212, 0.88189691, 0.88084011, 0.87863708,
       0.87708178, 0.87470166, 0.87238588, 0.87017915, 0.8688823 ,
       0.86802354, 0.86703213, 0.86536794, 0.86492794, 0.86388884,
       0.86376239, 0.86311841, 0.86111674, 0.85996399, 0.85867844,
       0.85815643, 0.85613868, 0.85505008, 0.85402476, 0.85367661,
       0.85173392, 0.8503297 , 0.84939804, 0.8466617 , 0.84679695,
       0.84631661, 0.846497  , 0.84497267, 0.84350171, 0.84360589,
       0.84398592, 0.84428448, 0.84548376, 0.84568525, 0.84504418])

def criterio(stock, parameters: list[float], index=-1) -> str:
    
    global buy_zone, sell_zone

    p = stock.prices
    ma1 = stock.m1
    # ma3 = stock.m3
    ma4 = stock.m4
    vr = stock.vr
    rf_desv = stock.rf_desv
    mvr = stock.mvr    
    sigma = mvr[index]*rf_desv[index]
    decision = 'Hold'

    PC1 = parameters[0]
    PC2 = parameters[1]
    PV1 = parameters[2]
    PV2 = parameters[3]    

    lw = 25
    rw = 5
    mm = len(buy_zone)//2 #60

    zone = p[index-lw-rw:index+1]
    zone = zone/zone[-1]
    buy_zone_i = buy_zone[mm-lw:mm+rw+1]
    buy_zone_i = buy_zone_i/buy_zone_i[-1]
    sell_zone_i = sell_zone[mm-lw:mm+rw+1]
    sell_zone_i = sell_zone_i/sell_zone_i[-1]

    b_mse_1 = mean_squared_error(
        zone[:-2*rw], 
        buy_zone_i[:-2*rw], 
        squared=True
        )
    b_mse_2 = mean_squared_error(
        zone[-2*rw:], 
        buy_zone_i[-2*rw:], 
        squared=True
        )
    s_mse_1 = mean_squared_error(
        zone[:-2*rw], 
        sell_zone_i[:-2*rw], 
        squared=True
        )
    s_mse_2 = mean_squared_error(
        zone[-2*rw:], 
        sell_zone_i[-2*rw:], 
        squared=True
        )

    ccZ = b_mse_1<0.004 and b_mse_2<0.002
    cvZ = s_mse_1<0.004 and s_mse_2<0.002

    cc1 = p[index-1] < ma1[index-1]
    cc2 = p[index] > ma1[index]
    cc3 = p[index] < vr[index] + sigma * PC1
    cc4 = p[index] < ma4[index] * PC2
    if cc3 and cc4 and ccZ:
    # if ccZ:
        decision = 'Buy'

    cv1 = p[index-1] > ma1[index-1]
    cv2 = p[index] < ma1[index]
    cv3 = p[index] > vr[index] + sigma * PV1
    cv4 = p[index] > ma4[index] * PV2
    if cv3 and cv4 and cvZ:
    # if cvZ:
        decision = 'Sell'   

    return decision






# -----------------------------------------------------------------------------
# Def funcion de costo para algoritmo Gradient Descent
# -----------------------------------------------------------------------------



def cost(parameters, print_=0):
    ol = []
    total_d = 0
    for ticker in TICKERS:        
        stock_path = str(mm.PROJECT_ROOT+'\\data\\tickers\\'+ticker+'.pkl')
        stock = mm.load_object(stock_path)  
        ap, d, p = mm.apply_criterio(criterio, stock, parameters)
        if print_==2:
            print(ticker.ljust(5), round(p,1), round(100*d/stock.len))
        ol.append(ap)
        total_d += d
    out = np.mean(ol)

    if print_ in [1,2]:
        print()
        print(round(out,2),'%', total_d/10000)
        print()

    print(parameters, out)

    return -out