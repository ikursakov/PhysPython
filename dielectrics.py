# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Взаимодействие между сферическим полостями внутри диэлектрика между обкладками плоского конденсатора

# Рассмотрим две сферические полости с радиусами $a$, находящиеся внутри диэлектрика с проницаемостью $\varepsilon$ между обкладками плоского конденсатора с разностью потенциалов $U$ на расстоянии $l$ друг от друга
#
# ![Полости в конденсаторе](pics/de_condenser.png)
#

# Дипольные моменты пузырьков противоположны тем, что получились бы при заполнении их данным диэлектриком:
#
#  $$ \begin{array}{c}
#  \mathbf{p}_п = \frac{4}{3}\pi a^3 \mathbf{P} = \frac{(\varepsilon -1)a^3\mathbf{E}}{3} \\
#   \mathbf{E} = (Ud,0) \\
#   p_{пx} = \frac{(\varepsilon -1)a^3Ud}{3}
#   \end{array}
#  $$

# Так как напряженность поля, созданного диполем есть
# $$ \begin{array}{c}
# \mathbf{E}_d = 3(\mathbf{pr})\frac{\mathbf{r}}{r^5} - \frac{\mathbf{p}}{r^3} = E_{dx}\vec{i} + E_{dy}\vec{j} \\
# E_{dx} = \frac{(\varepsilon -1)a^3x^2Ud}{\sqrt{x^2+y^2}^5} +\frac{(\varepsilon -1)a^3Ud}{3\sqrt{x^2+y^2}^3} \\
# E_{dy} = \frac{(\varepsilon -1)a^3xyUd}{\sqrt{x^2+y^2}^5} \\ 
# E_d = \sqrt{E_{dx}^2 + E_{dy}^2}
# \end{array} $$

# Визуализируем полученное поле. Положим напряжение между пластинами конденсатора $U = 1000 В$, расстояние между пластинами $d = 4 мм$, радиус пузырька $a = 0.01 мм$, проницаемость диэлектрика $\varepsilon = 6$. Пусть пузырек располагается в начале координат, а левая обкладка конденсатора соответсвует $x_l = -1 мм$

import numpy
import math
from matplotlib import pyplot
# отображение картинок в блокноте
# # %matplotlib inline
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

eps_0 = 8.85e-12
N = 200                     # Число узлов сетки в каждом направлении
x_start, x_end = -1.0, 3.0            # границы по x
y_start, y_end = -5.0, 5.0            # границы по y
x = numpy.linspace(x_start, x_end, N)    # одномерный массив x
y = numpy.linspace(y_start, y_end, N)    # одномерный массив y
X, Y = numpy.meshgrid(x, y)              # создает сетку

eps = 6
U = 1000
d = 4e-3
a = 1e-5


def 


E_dx = (eps - 1)*a**3*U*d *(X**2/(X**2+Y**2)**5 + 1/3 * (X**2+Y**2)**-3)
E_dy = (eps - 1)*a**3*U*d *(X*Y/(X**2+Y**2)**5 )

size = 5
pyplot.figure(figsize=(size, (y_end-y_start)/(x_end-x_start)*size))
pyplot.grid(True)
pyplot.xlabel('x', fontsize=16)
pyplot.ylabel('y', fontsize=16)
xmax,ymax = x_start-1, x_end+1
pyplot.xlim(xmax,ymax)
ymin,ymax  = y_start-1, y_end+1
pyplot.ylim(ymin,ymax)
pyplot.streamplot(X, Y, E_dx, E_dy, density=2, linewidth=1, arrowsize=1,\
                  arrowstyle='->')
pyplot.scatter(0, 0, color='#CD2305', s=80, marker='o')
pyplot.axvline(x_start, (y_start - ymin) / (ymax -ymin),
               (y_end - ymin)/ (ymax -ymin),
color='#CD2305', linewidth=2);


