from scipy import optimize
import criterio.market as mm
import definitions

'''

El objetivo del proyecto es sistematizar la búsqueda de un criterio óptimo de
compra-venta de acciones en el mercado mediante backtesting usando datos
históricos de cotizaciones diarias del mercado de los últimos 20 años.

El usuario del código puede definir cualquier criterio de compra-venta que
esté determinado por cualquier número de parámetros, con la única condición
de que tomen valores reales.

El código permite simular el resultado que se hubiese obtenido al aplicar el
criterio definido de compra-venta durante los úlimos 20 años y graficarlo.

Mediante una aplicación del algoritmo Gadient Descent el código pretende
permitir encontrar los parámetros óptimos que puede tener el criterio 
propueto.

Aquí se muestra un posible uso del código, presentado en forma de snippets o
"TEST" independientes.

Se han definido como ejemplo, en "definitions.py", un criterio de compra y 
venta, una funcióon de costo, y los tickers del mercado que se usarán como 
datos.

A partir del caso particular de estas definiciones, se simula el resultado de 
haber aplicado ese criterio en el mercado durante los ultimos 20 años,  
se visualizan las compras y ventas que habrían tenido lugar, y se muestra el 
resultado neto. 

Luego se ajustan los paramaetros que determinan el criterio mediante el 
algoritmo Gradient Descent para obtener los que minimizan la función de costo 
definida (maximicen la ganancia).

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










