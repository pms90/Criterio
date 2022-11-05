import numpy as np
import pandas as pd
import statistics as st
import os
import pathlib
import yfinance as yf
import pickle

# -----------------------------------------------------------------------------
# Constantes
# -----------------------------------------------------------------------------

PARENT = str(pathlib.Path(__file__).parent.resolve())
PROJECT_ROOT = os.path.split(PARENT)[0]

# -----------------------------------------------------------------------------
# Clases
# -----------------------------------------------------------------------------

class Wallet:

    '''
    Wallet (Billetera) simula la billetera de un inversor que compra y vende
    acciones en la bolsa. Tiene la información de cuánto es el capital en la 
    billetera y el registro de las operaciónes realizadas (fecha y flujo de
    capital). 

    El método report() devuelve el resultado neto de todas las operaciones 
    registradas a travez de 3 valores: ganancia anualizada, total de días 
    en lo que se estuvo en posecion de acciones, y la ganancia (o perdida) 
    total como porcentaje del capital inicial.

    Los impuestos que se pagan en cada operación se definen como constantes.
    '''

    INITIAL_CAPITAL = 1_000_000    
    BROKER_FEE = 0.005
    MARKET_TAX = 0.0025
    HALF_SPREAD = 0.01
    FEES = BROKER_FEE + MARKET_TAX + HALF_SPREAD    # [%] 

    def __init__(self) -> None:
        self.liquid = self.INITIAL_CAPITAL
        self.stocks = 0
        self.state = 'to buy'
        self.buy_dates = []
        self.buy_prices= []
        self.sell_dates = []
        self.sell_prices = []        
        self.capital_history = [self.INITIAL_CAPITAL]
    
    def buy(self, date, price) -> None:
        taxed_price = price * (1/(1-self.FEES))
        stocks_bought = self.liquid // taxed_price
        self.stocks += stocks_bought
        self.liquid = self.liquid % taxed_price
        self.buy_dates.append(date)
        self.buy_prices.append(taxed_price)
        self.capital_history.append(self.capital(taxed_price))
        self.state = 'to sell'

    def sell(self, date, price) -> None:
        taxed_price = price * (1-self.FEES)
        self.liquid += taxed_price*self.stocks
        self.stocks = 0
        self.sell_dates.append(date)
        self.sell_prices.append(taxed_price)
        self.capital_history.append(self.capital(taxed_price))
        self.state = 'to buy'        
    
    def capital(self, price):
        return (self.liquid + self.stocks * price)

    def __str__(self) -> str:
        return f'''
        Wallet 
        Liquid={self.liquid}
        Stocks:{self.stocks}        
        '''

    def __repr__(self) -> str:
        return f'''
        Wallet
        Capital_history: {self.capital_history}
        '''
    def report(self) -> tuple[float, int, float]:
        import numpy as np
        exposed_days = 0
        anual_profit = 0
        profit = 0
        om = 0

        if self.buy_dates!=[]:            
            for i in range(len(self.buy_dates)):
                start_date = self.buy_dates[i].to_numpy()
                end_date = self.sell_dates[i].to_numpy()
                d = npday2int(end_date-start_date)          
                #^ Dias entre compra y venta 
                exposed_days += d 

            profit = (self.capital_history[-1]/self.INITIAL_CAPITAL) -1
            #^ Variacion % entre capital inicial y final
            anual_profit = (1+profit)**(365/exposed_days) - 1 

            # mm = []
            # for i in range(len(self.buy_dates)):
            #     start_date = self.buy_dates[i].to_numpy()
            #     end_date = self.sell_dates[i].to_numpy()
            #     d = npday2int(end_date-start_date)     
            #     sell_price = self.sell_prices[i]
            #     buy_price = self.buy_prices[i]
            #     m = (sell_price-buy_price)/d
            #     mm.append(m)         
            # om = np.mean(mm)     


        return anual_profit, exposed_days, profit







class Stock:

    '''
    Stock representa a una accion del mercado. Guarda el valor de su 
    cotizacíon en el tiempo así como otros indicadors asociados (ej: moving 
    average)

    El método plot() permite graficar estos valores con Matplotlib.
    '''

    M1_WINDOW = 20           # market days (no days)
    M3_WINDOW = 200
    M4_WINDOW = 400
    VR_WINDOW = 84           
    VR_DESV_WINDOW = 84     

    def __init__(self, ticker,  stock_data, normalize=-1):  
        self.ticker = ticker
        self.__stock_data = stock_data    
        not_null_data = stock_data[stock_data.notnull()]
        self.dates = not_null_data.index
        self.prices = not_null_data.values
        self.len = len(self.prices)
        self.normalize_sate = normalize

        if normalize!=None:
            divider = self.prices[normalize]
            for i in range(self.len):
                self.prices[i] = self.prices[i]/divider

        self.m1 = mo(self.prices, self.M1_WINDOW)
        self.m3 = mo(self.prices, self.M3_WINDOW)
        self.m4 = mo(self.prices, self.M4_WINDOW)

        self.mvr = mo(self.prices, self.VR_WINDOW)
        refs = references(self.prices, self.mvr, self.VR_WINDOW)
        self.rf, self.rf_desv, self.vr, self.vr_up, self.vr_down = refs

    def __str__(self) -> str:
        return self.ticker

    def __repr__(self) -> str:        
        return f'''
        Ticker: {self.ticker}
        Normalización: {self.normalize_sate}
        Fecha primera cotización: {self.dates[0]} 
        Fecha ultima cotización: {self.dates[-1]} 
        '''

    def normalize(self, divider_index=-1) -> None:
        divider = self.prices[divider_index]
        for i in range(self.len):
            self.prices[i] = self.prices[i]/divider
            self.m1[i] = self.m1[i]/divider
            self.m3[i] = self.m3[i]/divider
            self.m4[i] = self.m4[i]/divider

    def plot(self) -> None:
        import matplotlib.pyplot as plt
        plt.plot(self.dates, self.prices, linewidth=0.75)
        plt.plot(self.dates, self.m1, color='red', alpha=0.5, linewidth=0.75)
        plt.plot(self.dates, self.m4, color='green', alpha=0.5, linewidth=0.75)
        plt.plot(self.dates, self.vr_up, '--k', alpha=0.4, linewidth=0.5)
        plt.plot(self.dates, self.vr_down, '--k', alpha=0.4, linewidth=0.5)
        plt.show()

    def __add__(self, other):
        data1 = self.__stock_data
        data2 = other.__stock_data
        data12 = pd.concat([data1,data2], axis=1)
        data12 = data12[data12.notnull()]
        last1 = data12.iloc[-1,0]
        last2 = data12.iloc[-1,1]
        for i in range(len(data12)):
            data12.iloc[i,0] = data12.iloc[i,0]/last1
            data12.iloc[i,1] = data12.iloc[i,1]/last2
        new = [0]*len(data12)
        for i in range(len(data12)):
            new[i]=(data12.iloc[i,0]+data12.iloc[i,1])/2
        data12['c'] = new
        compose = data12['c']
        return Stock(compose)

                

# -----------------------------------------------------------------------------
# Funciones
# -----------------------------------------------------------------------------

def apply_criterio(criterio: callable, stock, 
                    parameters: list[float], report='no', plot=False,
                    save_plot=False) -> tuple[float, int, float]:

    w = Wallet()

    marg = 3*84  

    for i in range(stock.len)[marg:]:

        decision = criterio(stock, parameters, index=i)

        if decision=='Hold':
            # # Condicion de vender si pasan mas de DMAX dias
            # DMAX = 90
            # if w.state == 'to sell':
            #     start_date = w.buy_dates[-1].to_numpy()
            #     end_date = stock.dates[i].to_numpy()
            #     d = mm.npday2int(end_date-start_date) 
            #     if d>DMAX:
            #         w.sell(stock.dates[i], stock.prices[i])
            continue

        if decision=='Buy':
            if w.state == 'to buy':
                w.buy(stock.dates[i], stock.prices[i])

        if decision=='Sell':
            if w.state == 'to sell':
                w.sell(stock.dates[i], stock.prices[i])

    if w.state == 'to sell':
        w.sell(stock.dates[-1], stock.prices[-1])         
    
    if report=='simple':
        print("{:4.1f}".format(w.report()[2]*100))
    if report=='full':
        print(stock.ticker.ljust(4), end='  ')
        print("{:4.1f}".format(w.report()[0]*100)+'%',
              str(w.report()[1])+'d',
              "{:3.1f}".format(w.report()[2]*100)+'%')
      

    if plot==True or save_plot==True:

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12,6))

        plt.plot(stock.dates, stock.prices)
        # plt.plot(stock.dates, stock.m1, color='red', alpha=0.4)
        # plt.plot(stock.dates, stock.m4, color='green', alpha=0.4)
        # plt.plot(stock.dates, stock.vr_up, '--k', alpha=0.3)
        # plt.plot(stock.dates, stock.vr_down, '--k', alpha=0.3)       
        plt.plot(w.buy_dates, w.buy_prices,'*', color='orange', label='Compra')   
        plt.plot(w.sell_dates, w.sell_prices, '*', color='red', label='Venta')       

        plt.legend()
        title = stock.ticker.rjust(4)
        title += ' | Ganancia anualizada: '
        title += "{:4.1f}".format(w.report()[0]*100).rjust(5)+'%'
        title += ' | Días de exposicion: '+str(w.report()[1]).rjust(6)
        title += ' | Ganancia total: '
        title += "{:3.1f}".format(w.report()[2]*100).rjust(5)+'%'
        plt.title(title)

        plt.ylabel('Cotización')


        for d1,d2,p1,p2 in zip(w.buy_dates,w.sell_dates,
                                w.buy_prices,w.sell_prices):
            plt.plot([d1,d2], [p1,p2], color='orange')

        if save_plot == True:
            name = stock.ticker
            save_path = PROJECT_ROOT+'\\plots\\'+name+'.png'
            plt.savefig(save_path, dpi=100)

        if plot==True:
            plt.show()        



    return w.report()



def download_market_data(target_tickers: list[str], time_period: str, 
    save_in_xlsx: bool, save_as_Stock_pkl: bool) -> None:    

    data = yf.download(
        tickers=target_tickers,
        period=time_period, 
        interval='1d'
        )

    data_close = data['Close']

    if save_in_xlsx==True:
        out_path = PROJECT_ROOT+r"\data\MARKET.xlsx"
        dfs = {'Market Data': data_close} 
        # diccionario de dataframes, uno por Hoja del exel
        writer = pd.ExcelWriter(out_path, engine='xlsxwriter')
        for sheetname, df in dfs.items():  
            df.to_excel(writer, sheet_name=sheetname, index=True) 
            worksheet = writer.sheets[sheetname]  # pull worksheet object
            for idx, col in enumerate(df):  # loop through all columns
                series = df[col]
                max_len = max((
                    series.astype(str).map(len).max(),  # len of largest item
                    len(str(series.name))  # len of column name/header
                    )) + 1                 # adding a little extra space
                worksheet.set_column(idx, idx, max_len)  # set column width
        writer.save()

    if save_as_Stock_pkl==True:
        # Save Market tickers as stocks objects in .pkl
        for ticker in target_tickers:
            print('Creating '+ticker.ljust(4)+' Stock object', end=' ')
            stock = Stock(ticker, data_close[ticker][:], normalize=-1)
            out_path = PROJECT_ROOT+'\\data\\tickers\\'+ticker+'.pkl'
            save_object(stock, out_path)
            print('.......... OK')


def update_data(TICKERS: list[str]) -> None:
    download_market_data(
        target_tickers = TICKERS,
        time_period = '288mo',
        save_in_xlsx = True,
        save_as_Stock_pkl = True
    )




# Crear DataFrame MARKET
def get_market_df():
    # import pandas as pd
    io = PROJECT_ROOT+r"\data\MARKET.xlsx"
    out = pd.read_excel(io, index_col=0)
    return out



# Moving average y plot moving average
def mo(data,interval):
#   import numpy as np
  L = len(data)
  w = interval # ma window
  cus = np.cumsum(data)/w
  mo = np.zeros_like(data)
  mo[0:w+1] = data[0:w+1]
  for i in range(L):
    if i<w: continue
    mo[i] = cus[i]-cus[i-w]
  return mo

# Ajustar polinomio de grado n
def pfit(x_data, y_data, x_fit=[], n=1):
    # import numpy as np
    if x_fit==[]:
        L = len(y_data)
        # x_fit = np.arange(0,L,1)
        x_fit = x_data.copy()
    coef = np.polyfit(x_data, y_data, n)
    fit_fn = np.poly1d(coef) 
    fit_values = fit_fn(x_fit)
    return fit_values, fit_fn

# Valores de referencia
def references(p, ma, ma_window):
    # import numpy as np
    # import statistics as st
    L = len(p)
    rf = p.copy()*0
    rf_desv = p.copy()*0
    vr = ma.copy()
    vr_up = ma.copy()
    vr_down = ma.copy()
    for i in range(L):
        if i <= ma_window+10:
            continue
        relative_vars = (p[ma_window:i] - ma[ma_window:i])/ma[ma_window:i]
        rf[i] = np.mean(relative_vars) 
        rf_desv[i] = st.stdev(relative_vars) 
        vr[i] = ma[i]*( 1+ rf[i] + 0 * rf_desv[i] ) 
        vr_up[i] = ma[i]*( 1+ rf[i] + 1 * rf_desv[i] ) 
        vr_down[i] = ma[i]*( 1+ rf[i] - 1 * rf_desv[i] ) 
    return rf, rf_desv, vr, vr_up, vr_down

def delta_np_time(start,end,unit='D'):
    # import numpy as np
    '''units = ['Y','M','D','h','m','s','ns']
    '''  
    diff = end - start
    diff = diff.astype('timedelta64['+unit+']')
    diff = diff / np.timedelta64(1, unit)
    return int(diff)

def npday2int(delta):
    # import numpy as np
    days = delta / np.timedelta64(1,'D')
    return int(days)

def save_object(obj, full_file_path):
    # import pickle
    if not full_file_path.endswith('.pkl'):
        full_file_path += '.pkl'
    with open(full_file_path, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_object(file_path):
    # import pickle
    with open(file_path, 'rb') as inp:
        return pickle.load(inp)

def load(stock_name):
    path = PROJECT_ROOT+'\\data\\tickers\\'+stock_name+'.pkl'
    stock = load_object(path)
    return stock


def file_path():
    # import pathlib
    return str(pathlib.Path(__file__).parent.resolve())






