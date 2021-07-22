#METRONOMO DE BEETHOVEN ###

# Importamos librerias
from scipy.special import factorial2
from numpy import sin, deg2rad
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Definimos nuestra funcion de theta
def f_ang(theta):   
    """
    Ecuacion (2) del paper en funcion del angulo

    Parameters
    ----------
    theta : Float
        Angulo θ en grados.

    Returns
    -------
    Float
        Uno mas la sumatoria de la funcion evaluada desde n=1 hasta n=150.

    """
    serie = np.array([
        (factorial2(2*n-1)/factorial2(2*n)*(sin(deg2rad(theta)/2))**(2*n))**2 
        for n in range(1,151)])
    
    return 1 + np.sum(serie)

# Definimos la funcion omega de varias variables
def omega(g, theta, M, m, R, mu, l, L, r):  
    """
    Ecuacion (4) del paper en funcion de las variables influyentes en omega

    Parameters
    ----------
    g : float
        Aceleracion de la gravedad.
    theta : float
        Angulo θ.
    M : float
        Masa mayor, ubicada en la parte inferior del metronomo.
    m : float
        Masa menor, correspondiente al trapecio deslizante del metronomo.
    R : float
        Distancia desde la mitad de la varilla hasta M.
    mu : float
        Valor de μ, correspondiente a la masa de la varilla.
    l : float
        Distancia desde el eje hasta el extremo superior de la varilla.
    L : float
        Distancia desde el eje hasta el extremo inferior de la varilla..
    r : float
        Distancia desde la mitad de la varilla hasta m.

    Returns
    -------
    Float
        Raíz de la funcion omegaa.

    """
    Mp = M/m   # M'
    mup = mu/m  # μ'
    b_2 = (-1)/(Mp*R**2 +mup*(L**2+l**2-l*L)/3)  # Ec. (6)
    a_0 = (g/(f_ang(theta)**2))*(Mp*R-mup*(l-L)/2)/(Mp*R**2 + mup*(L**2+l**2-l*L)/3)  # Ec. (5)
    omegaa=((a_0+b_2*(g*r/(f_ang(theta)**2)))/(1-b_2*r**2))**1/2 # Ec. (4)
    return (omegaa)

rs = np.linspace(40, 208, 10, endpoint=True)  # Espacio de r en el que se evaluará Ω

fig = plt.figure(figsize=(15, 7))  # Tamaño que tomará la figura
Graf = plt.axes((0.15, 0.55, 0.7, 0.3))  # Ejes para la gráfica

# Valores referenciales para la funcion omega
omegas = omega(9800, 52.5, 31, 7, 36.4, 3.59, 138, 62, rs)  
# Ploteamos la grafica de la funcion en el espacio rs con los valores de omegas
GrafP, = Graf.plot(rs, omegas, '-ok', color='orange', linestyle='dashed', 
                   linewidth=2, markersize=8)

plt.xlabel('r [mm]', size=15, weight='bold')  # Nombre del eje x
plt.ylabel('Ω', size=15, weight='bold')  # Nombre del eje y
plt.title('Metrónomo de Bethoven', weight='bold', size=25)  # Titulo de la grafica
plt.grid()  # Cuadricula
plt.show   # Muestra la grafica

# Establecemos los ejes para los sliders
sl_g = plt.axes([0.1, 0.40, 0.35, 0.05])  # Eje del slider de la gravedad
sl_theta = plt.axes([0.1, 0.30, 0.35, 0.05])  # Eje del slider del angulo 
sl_M = plt.axes([0.1, 0.20, 0.35, 0.05])  # Eje del slider de la masa mayor
sl_m = plt.axes([0.1, 0.10, 0.35, 0.05])  # Eje del slider de la masa menor
sl_R = plt.axes([0.55, 0.20, 0.35, 0.05])  # Eje del slider de la distancia R
sl_μ = plt.axes([0.55, 0.10, 0.35, 0.05])  # Eje del slider de la masa de la varilla
sl_l = plt.axes([0.55, 0.40, 0.35, 0.05])  # Eje del slider de la distancia l
sl_L = plt.axes([0.55, 0.30, 0.35, 0.05])  # Eje del slider de la distancia L

# Creamos los Sliders
## Slider gravedad
g= Slider(ax=sl_g,                  # Eje en el que se encontrara el Slider
          label=('g [mm/s²]'),      # Nombre del Slider
          valmin=40,                # Valor minimo que puede tomar
          valmax=9807,              # Valor maximo que puede tomar
          valinit=9800,             # Valor inicial
          valstep=20,               # Amplitud entre cada dato a seleccionar
          color='mediumturquoise')  # Color del Slider

##Slider theta
theta= Slider(ax=sl_theta,
              label=('θ [°]'),
              valmin=40,
              valmax=60,
              valinit=52.5,
              valstep=1,
              color='mediumturquoise')

## Slider Masa
M = Slider(ax=sl_M,
           label=('M [g]'),
           valmin=15,
           valmax=50,
           valinit=31,
           valstep=1,
           color='mediumturquoise')

## Slider masa menor
m = Slider(ax=sl_m,
           label=('m [g]'),
           valmin=1,
           valmax=14,
           valinit=7,
           valstep=0.1,
           color='mediumturquoise')

## Slider R mayor
R = Slider(ax=sl_R,
           label=('R [mm]'),
           valmin=35,
           valmax=100,
           valinit=36.4,
           valstep=1,
           color='mediumturquoise')

## Slider Mu
mu = Slider(ax=sl_μ,
            label=('μ [g]'),
            valmin=2,
            valmax=6,
            valinit=3.59,
            valstep=0.1,
            color='mediumturquoise')

## Slider l
l = Slider(ax=sl_l,
           label=('l [mm]'),
           valmin=120,
           valmax=210,
           valinit=138,
           valstep=1,
           color='mediumturquoise')

## Slider L
L = Slider(ax=sl_L,
           label=('L [mm]'),
           valmin=35,
           valmax=70,
           valinit=36.4,
           valstep=1,
           color='mediumturquoise')


# Creamos una funcion que servira para que los valores de las variables de los
# sliders varien segun el usuario los manipule
def ACTUALIZAR(valor):
    GrafP.set_ydata(omega(g.val, theta.val, M.val, m.val, R.val, mu.val, 
                          l.val, L.val, rs))
    fig.canvas.draw_idle()
    
g.on_changed(ACTUALIZAR)       
theta.on_changed(ACTUALIZAR)
M.on_changed(ACTUALIZAR)
m.on_changed(ACTUALIZAR)
R.on_changed(ACTUALIZAR)
mu.on_changed(ACTUALIZAR)
l.on_changed(ACTUALIZAR)
L.on_changed(ACTUALIZAR)

#  Creamos el Boton 
sl_Boton = plt.axes([0.865, 0.47, 0.08, 0.05])  # Esyablecemos el eje del Boton
Reiniciar_Grafico = Button(ax=sl_Boton,         # Eje a tomar
                           label="Reiniciar",   # Palabra indicativa del Boton
                           color='palegreen',   # Color del Boton 
                           hovercolor='red')    # Color al posicionar el cursor 
                                                # en el Boton

# Definimos una funcion para la accion del boton 
def reiniciar(evento):
    g.reset()        
    theta.reset()
    M.reset()
    m.reset()
    R.reset()
    mu.reset()
    l.reset()
    L.reset()
# Reinicia cada valor a su valinit en cada Slider al hacer un click sobre el 
Reiniciar_Grafico.on_clicked(reiniciar) 