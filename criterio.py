from scipy import optimize
import criterio.market as mm
import definitions

'''

Se muestra a continuación, en forma de tests independientes, el uso del código. 

Se han definido como ejemplo, en "definitions.py", un criterio de compra y 
venta, una funcióon de costo, y los tickers del mercado que se usarán como 
datos.

A partir del caso particular de estas definiciones, se simula el resultado de 
haber aplicado ese criterio en el mercado durante los ultimos 20 años,  
se visualizan las compras y ventas que habrían tenido lugar, y se muestra el 
resultado neto. 

Luego se ajustan los paramaetros que determinan el criterio mediante el 
algoritmo Gradient Descent para obtener los que minimizen la función de costo 
definida.

'''

PROJECT_ROOT = mm.PROJECT_ROOT      # path, type:str

# -----------------------------------------------------------------------------
# TICKERS: iniciales que identifican en el mercado a cada accion (ej: 'KO' es
# el ticker de Coca-Cola Company).
# -----------------------------------------------------------------------------
 
ETFS = definitions.ETFS
COMPANIES = definitions.COMPANIES
TICKERS = ETFS + COMPANIES          # type: list[str]

# -----------------------------------------------------------------------------
# TEST: Aplicar el criterio definido en "definitions.py" a una acción en
# particular, en este caso Walmart cuyo ticker es 'YPF', y graficar.
# -----------------------------------------------------------------------------

test_apply_one = 1

if test_apply_one==1:
    mm.apply_criterio(
        criterio = definitions.criterio,
        stock = mm.load('YPF'),
        parameters =  [-1.03743099, 0.60614302, 0.92400256, 1.23213628],
        report = 'full',
        plot = True,
        save_plot = False
    )


# -----------------------------------------------------------------------------
# TEST: Aplicar el criterio definido en "definitions.py" a todos los tickers
# -----------------------------------------------------------------------------

test_apply_all = 0

if test_apply_all==1:
    for ticker in TICKERS:
        mm.apply_criterio(
            criterio = definitions.criterio,
            stock = mm.load(ticker),
            parameters =  [-1.03743099, 0.60614302, 0.92400256, 1.23213628],
            report = 'full',
            plot = False,
            save_plot = True     
        )


# -----------------------------------------------------------------------------
# TEST: Optimizar parametros del criterio definido en "definitions.py" mediante
# la aplicacionon de el algoritmo "Gradient Descent".
# -----------------------------------------------------------------------------

gd = 0

if gd==1:

    print('Aplicando Gradiend Descent...')

    init_parameters = [-0.9, 0.85, 0.9, 1.15]

    m_scy  = optimize.fmin_cg(f = definitions.cost,
                              x0 = init_parameters,
                              epsilon = 0.1, 
                              fprime = None) #epsilon = 1.0e-03

    print(f'Parámetros ideales para el criterio definido: {list(m_scy)}')



# -----------------------------------------------------------------------------
# TEST: Actulaizar datos de mercado, crear objetos "Stock" para cada accion
# y guardar los objetos como .pkl. 
# -----------------------------------------------------------------------------

test_update = 0

if test_update==1:
    mm.update_data(TICKERS)












