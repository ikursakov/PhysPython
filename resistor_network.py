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

# # Сопротивление цепи проводников

# Перевод и незначительные изменения ноутбука [Resistor Network](https://nbviewer.jupyter.org/urls/www.numfys.net/media/notebooks/resistor_network.ipynb) By Magnus A. Gjennestad, Vegard Hagen, Aksel Kvaal, Morten Vassvik, Trygve B. Wiig, Peter Berg and Magnus Dahle, [NumPhys Project](https://www.numfys.net)

# ## Постановка задачи
#
# В этом примере мы рассмотрим цупь, состоящую из $N$ повторяющихся сегментов. Каждый сегмент цепи сотоит из двух резисторов с сопротивлениями $R$ и $12R$ соответственно. Сегменты подключены к батарее с напряжением $V_0$, как показано на рисунке.
#

# ![circuit1](pics/4ec_image_1.png)

# Наша цель – определить силу тока, протекающего через цепь. На следующем рисунке приведены обозначения, которыми мы будем пользоваться для решения задачи. В этих обозначениях, нам нужно определить силу тока $I_{1,1}$

# ![circuit2](pics/4ec_image_2.png)

# ## Частный случай: $N=1$

# При решении подобных задач полезно бывает рассмотреть частные случаи, если таковые имеются, перед тем как решать задачу в общей постановке. Рассмотрим два таких частных – предельных случая.
#
# Сначала обратимся к случаю, когда $N=1$. Тогда цепь будет выглядеть следующим образом:

# ![circuit2](pics/4ec_image_3.png)

# Для решения задачи в такой упрощенной постановке обозначим $R_{eff}$ эффективное сопротивление всей цепи, подключенной к батарее. С таким обозначением цепь будет выглядеть так:

# ![circuit2](pics/4ec_image_4.png)

# где $R_{eff} = R+12R=13R$. Тогда, по закону Ома:
#
# $$
# I_{1,1} = \frac{V_0}{13R} = \frac{1}{13} \frac{\textrm{V}}{\Omega} \approx 0.0769 \textrm{A}
# $$

# ## Частный случай: $N\to\infty$

# В качестве другого частного случая рассмотрим другую крайность – когда число сегментов $N$ стремится к бесконечности. Этот случай не такой тривиальный. Очевидно при $N\to\infty$ число путей, по которым может течь ток, возростает. Следовательно, можно ожидать, что сопротивление будет падать при приближении к пределу. Подумайте, как бы вы решали эту задачу, перед тем как продолжить чтение.

# Снова введем величину $R_{eff}$, и обозначим ей сопротивление всей цепи c бесконечным числом сегментов (как на предыдущей картинке). Теперь возьмем эту цепь и добавим к ней еще один сегмент, тогда получится результат, изображенный на следующей диаграмме.

# ![circuit2](pics/4ec_image_5.png)

# Поскольку число сегментов $N$ бесконечно большое, добавление к бескончности еще одного сегмента не должно изменить величины эффективного сопротивления $R_{eff}$ всей цепи. Другими словами, в получившейся цепи резиисторы $12R$ подключены параллельно $R_{eff}$, но сопротивление всей цепи $R_{total}$ по-прежнему равно $R_{eff}$. Исходя из этого, можно записать соотношение для $R_{eff}$:
#
# $$
# R_{total} = R_{eff} = R + \frac{1}{\frac{1}{12R} + \frac{1}{R_{eff}}}
# $$
#
#

# Решая это квадратное уравнение для $R_{eff}$, и отбросив отрицательный корень, получим
#
# $$
# R_{eff} = 4R = 4\Omega
# $$

# Таким образом по закону Ома мы находим
# $$
# I_{1,1} = \frac{V_0}{4R} = \frac{1}{4} \frac{\textrm{V}}{\Omega} \approx 0.25 \textrm{A}
# $$

# ## Составляем систему уравнений

# Теперь перейдем к более общему случаю, когда $1<N<\infty$. Для получения решения мы составим систему из $N$ уравнений с $N$ неизвестными, которую запишем в матричном виде. Затем мы решим полученную систему с помощью Python. Неизвестными будут $N$ напряжения $V_i$, $i=1,\ldots,N;$.

# To obtain the $N$ equations and $N$ unknowns, we first apply Ohm's law to all resistors in the circuit. This yields
#
# Чтобы получить $N$ уравнений для $N$ неизвестных, запишем закон Ома для всех сопроивлений в цепи. Получ
# им
# $$
# I_{i,1} = \frac{V_{i-1} - V_i}{R}, \quad i=1,\ldots,N;
# $$
#
# для $N$ сопртивлений величины $R$ и
#
# $$
# I_{i,2} = \frac{V_i}{12R}, \quad i=1,\ldots,N;
# $$
#
# для $N$ сопртивлений величины $12R$.

# Следующим шагом будет устранение токов $I_{i,1}$ и $I_{i,2}$ для $i=1,\ldots,N;$ в приведенных выше уравнениях. Для этого обратимся к закону сохранения заряда и заметим, что сумма всех токов, входящих в узел на принципиальной схеме, равна сумме всех токов, вытекающих из него. Это утверждение равнозначно тому, что в узле не создаются и не разрушаются никакие заряды, и часто упоминается как первое правило Кирхгофа.

# Для узлов $V_i$, где $i=1,\ldots,N-1;$ получим
#
# $$
# I_{i,1} = I_{i,2} + I_{i+1,1}
# $$
#
# А для последнего узла $V_N$:
#
# $$
# I_{N,1} = I_{N,2}
# $$

# Подставив ранее полученные выражения для $I_{i,1}$ и $I_{i,2}$ в последние соотношения, запишем следующие соотношения: 
#
# - для первого узла цепи $i=1$:
#
# $$
# \frac{25}{12R} V_1 - \frac{1}{R}V_2 = V_0
# $$
#
# - для узлов $V_i$ с $i=2 до N-1:$
#
# $$
# -\frac{1}{R}V_{i-1} + \frac{25}{12R}V_i - \frac{1}{R} V_{i+1} = 0
# $$
#
# - и для последнего узла:
#
# $$
# -\frac{1}{R}V_{N-1} + \frac{13}{12R}V_N = 0
# $$
#
# Если подсчитать количество получившихся уравнений, то можно увидеть, что три последних выражения содержат в сумме $N$ уравнений. Ровно столько нам и нужно, чтобы определить все $N$ напряжений $V_i$ для $i=2,\ldots,N-1.$
#

# Более того, полученную систему уравнений можно записать в матричном виде $\mathcal{A}\boldsymbol{V}=\boldsymbol{b}$
#
# $$
# \begin{bmatrix} 25/12R & -1/R  & 0    & \dots & 0 \\ 
#                 -1/R  & 25/12R & -1/R & \dots & 0 \\
#                 \vdots& \ddots &\ddots& \ddots& \vdots \\
#                 0     & \dots &-1/R  & 25/12R & -1/R  \\
#                 0     & \dots & 0    & -1/R   & 25/12R
# \end{bmatrix} \cdot 
# \begin{bmatrix} V_1   \\ V_2 \\ \vdots \\ V_{N-1} \\ V_N \end{bmatrix} =
# \begin{bmatrix} V_0/R \\ 0   \\ \vdots \\ 0       \\ 0   \end{bmatrix} 
# $$

# Теперь поиск неизвестных напряжений равнозначен решению матричного уравнения $\mathcal{A}\boldsymbol{V}=\boldsymbol{b}$ для неизвестного вектора напряжения $\boldsymbol{V}$. Матрица $\mathcal{A}$ и вектор $\boldsymbol{b}$ задаются сопротивлениями внутри цепи и напряжением $V_0$. Впоследствии мы можем вычислить суммарный ток по закону Ома,
#
# $$
# I_{1,1} = \frac{V_0 - V_1}{R}
# $$
#
# Чтобы решить систему, используя Python, нужно начать с задания $V_0$, $R$ и $N$.

R  = 1.0 # Сопротивление [Ohm]
V0 = 1.0 # Приложенное напряжение [V]
N  = 10  # Количество сегментов [безразмерный]

# Затем необходимо задать матрицу $\mathcal{A}$ и вектор $\boldsymbol{b}$ матричного уравнения. Начнем с инициализации их обоих нулями, а также определим две переменные, $a=25/12R$ и $c=-1/R$ для упрощения их заполнения.

# +
# Для ускорения расчетов и удобства вычислений
# воспользуемся массивами numpy
import numpy as np 

A = np.zeros((N,N)) # Матрица размерности NxN
b = np.zeros(N)     # Ветор из N элементов
a = 25.0/(12*R)     # Скаляр (constant)
c = -1.0/R          # Скаляр (constant)
# -

# Далее, составим вектор $\boldsymbol{b}$ и матрицу $\mathcal{A}$ ряд за рядом.

# +
# В векторе правых частей b все элементы, кроме первого, равны нулю
b[0] = V0/R

# Заполняем первый ряд
A[0,0] = a
A[0,1] = c 

# Заполняем последний ряд
A[N-1,N-1] = a
A[N-1,N-2] = c

# Заполняем оставшиеся ряды
# При использовании Python 2.7 (или старше) целесообразнее применять
#      'xrange()' вместо 'range()', особенно для больших значений 'N'.
for row in range(1,N-1):
    A[row,row-1] = c
    A[row,row  ] = a
    A[row,row+1] = c

# Чтобы посмотреть, венро ли инициализоывны A и b, их можно распечатать:
# print(A,b)
# -

# Теперь можно решить систему уравнений с помощью встроенного решателя *Numerical Python Linear Algebra Solver*

# +
Voltages = np.linalg.solve(A,b)
print("Напряжения = ", Voltages)

I11 = (V0 - Voltages[0])/R
print("\nI_11     = ", I11)
# -

# Можно заметить, что когда $N$ становится большим (на самом деле уже при $N\geq15$), $I_{1,1}$ стремится к пределу, который мы нашли аналитически $N\to\infty$. То есть $I_{1,1}\to 1/4$.
#
# Обратите внимание, несмотря на то, что встроенный решатель, который мы используем в этом примере, призван быть эфффективным, это решатель для задач широкого класса, который <strong>не</strong> использует тот факт, что $\mathcal{A}$ является разреженной матрицей для этой проблемы.
#
# Если $\mathcal{A}$ станет действительно большой, то этот решатель в конечном итоге станет очень медленным, так как он выполняет большое количество ненужных итераций с нулевыми элементами. В качестве альтернативы можно использовать встроенную функциональность пакета *Scientific Python Sparse Linear Algebra*. Однако, этот модуль требует, чтобы матрица $\mathcal{A}$ хранилась в определенным (разреженном) формате.

# +
from scipy import sparse
import scipy.sparse.linalg as ssl
from scipy.sparse.linalg import spsolve

# Поскольку у нашей матрицы ненулевые элементы
# расположены на главной диагонали, а также 
# НАД главной диагональю (sup) и ПОД ней (sub),
# это нужно описать при помощи синтаксиса Python.

# Создадим разреженную матрицу

sup_diag = np.ones(N)*c
sub_diag = np.ones(N)*c
the_diag = np.ones(N)*a
           
# Опступы 
# от главной диагонали:   -1     0        1
all_diags  =       [sub_diag, the_diag, sup_diag]
offsets    =        np.array([-1, 0, 1])


csc="csc" # Компьютеоный формат, в котором будет храниться матрица

# Определяем разреженную матрицу
A_sparse = sparse.spdiags(all_diags, offsets, N,N, format=csc)


# print(A_sparse.todense()) # печатает разреженую матрицу в обычном (NxN) формате.


Voltages = spsolve(A_sparse,b)
print("Voltages = ", Voltages)

I11 = (V0 - Voltages[0])/R
print("\nI_11     = ", I11)

# -

# Теперь попробуйте сравнить время работы разреженного решателя со стандартным по мере увеличения $N$ до $10, 100, 1000, \ldots$.
