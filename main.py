from scipy.special import comb
from scipy.optimize import fsolve
from scipy.integrate import quad

import numpy as np

import matplotlib.pyplot as plt

import sympy as sp

from mpl_toolkits.mplot3d import Axes3D

import math

#################################### 13 ####################################
def e13_1() -> float:
    numerator = math.factorial(7) + comb(11, 32, exact=True)
    denominator = math.sqrt(1 / (1 + 0.2435))
    result = 2**-3 + (numerator / denominator)
    return result


def e13_2() -> float:
    term1 = (1 / 0.3532) ** (1/3)
    term2 = (12 * math.e ** (1 / 4.8)) / math.sqrt(7 ** -3)
    result = term1 - term2
    return result


def e13_3(dataset_file: str) -> float:
    feet_to_meters = 0.3048

    braking_distances = []
    with open(dataset_file, "r") as file:
        for line in file :
            braking_distances.extend(map(float, line.split()))

    braking_distances_meters = [distance * feet_to_meters for distance in braking_distances]
    average_braking_distance = sum(braking_distances_meters) / len(braking_distances_meters)
    return average_braking_distance


#################################### 14 ####################################
def e14_1():
    nums = [i for i in range(1, 100+1)]
    return [math.sin(x) for x in nums]


def e14_2():
    def sgn(x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0
    
    x_values = np.linspace(-2, 2, 400)

    # Применяем функцию sgn ко всем значениям массива
    y_values = np.array([sgn(x) for x in x_values])

    # Строим график
    plt.plot(x_values, y_values, label="sgn(x)")
    plt.title("Graph of the Sign Function")
    plt.xlabel("x")
    plt.ylabel("sgn(x)")
    plt.grid(True)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.legend()

    # Показываем график
    plt.show()


def e14_3(): 
    def int_frac_parts(x):
        int_part = int(x)  # Целая часть
        frac_part = x - int_part  # Дробная часть
        return int_part, frac_part
        
        # Генерация значений x от -3 до 3
    x_values = np.linspace(-3, 3, 400)

    # Применение функции к каждому x
    int_parts = np.array([int_frac_parts(x)[0] for x in x_values])
    frac_parts = np.array([int_frac_parts(x)[1] for x in x_values])

    # Построение графиков
    plt.figure(figsize=(10, 6))

    # График целой части
    plt.plot(x_values, int_parts, label="Integer part", color="blue")

    # График дробной части
    plt.plot(x_values, frac_parts, label="Fractional part", color="orange")

    # Настройки графика
    plt.title("Integer and Fractional Parts of x")
    plt.xlabel("x")
    plt.ylabel("Values")
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.grid(True)
    plt.legend()

    # Показ графика
    plt.show()


def e14_4(): 
    def xsinx(x):
        return np.sin(x) * x
    
    # Генерируем массив x от -20 до 20 с 401 точкой
    x_values = np.linspace(-20, 20, 401)

    # Вычисляем значения функции f(x)
    y_values = xsinx(x_values)

    # Строим график
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label=r"$f(x) = \sin(x) \cdot x$", color="blue")
    plt.title("Graph of f(x) = sin(x) * x")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.axhline(0, color='black',linewidth=0.5, ls='--')
    plt.axvline(0, color='black',linewidth=0.5, ls='--')
    plt.grid(True)
    plt.legend()

    # Показываем график
    plt.show()


#################################### 15 ####################################
def e15_1():
    # Определяем функцию
    def f(x):
        return x**3 - 6*x + 9

    # Производная для нахождения экстремумов
    def f_prime(x):
        return 3*x**2 - 6

    # Вторая производная для нахождения точек перегиба
    def f_double_prime(x):
        return 6*x

    # Аналитическое решение уравнения f(x) = 0
    roots = fsolve(f, [-3, 0, 3])  # начальные приближения

    # Аналитические решения для нахождения экстремумов (f'(x) = 0)
    extrema = fsolve(f_prime, [-3, 3])

    # Аналитические решения для нахождения точек перегиба (f''(x) = 0)
    inflection_points = fsolve(f_double_prime, [0])

    # Генерация данных для построения графика
    x = np.linspace(-4, 4, 400)
    y = f(x)

    # Построение графика
    plt.plot(x, y, label="f(x) = x^3 - 6x + 9")
    plt.axhline(0, color='black',linewidth=1)
    plt.axvline(0, color='black',linewidth=1)

    # Нанесение корней на график
    for root in roots:
        plt.plot(root, f(root), 'ro')
        plt.text(root, f(root), f'Root ({root:.2f}, {f(root):.2f})', color='red')

    # Нанесение экстремумов на график
    for ext in extrema:
        plt.plot(ext, f(ext), 'bo')
        plt.text(ext, f(ext), f'Extremum ({ext:.2f}, {f(ext):.2f})', color='blue')

    # Нанесение точек перегиба на график
    for inflection in inflection_points:
        plt.plot(inflection, f(inflection), 'go')
        plt.text(inflection, f(inflection), f'Inflection ({inflection:.2f}, {f(inflection):.2f})', color='green')

    # Настройки графика
    plt.title('График функции f(x) = x^3 - 6x + 9 с нанесением ключевых точек')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Вывод корней, экстремумов и точек перегиба
    print(f"Корни: {roots}")
    print(f"Экстремумы: {extrema}")
    print(f"Точки перегиба: {inflection_points}")


def e15_2():
    # Определяем функцию
    def f(x):
        return 1.3*x**4 - 3*x**3 + 2*x - 0.2*x**4 + 3

    # Производная функции для нахождения экстремумов
    def f_prime(x):
        return 4*(1.3 - 0.2)*x**3 - 9*x**2 + 2

    # Определяем асимптоту для высоких значений x
    def asymptote(x):
        return (1.3 - 0.2)*x**4

    # Аналитическое решение уравнения f(x) = 0 для поиска корней
    roots = fsolve(f, [-5, 0, 5])

    # Решение уравнения f'(x) = 0 для поиска локальных экстремумов
    extrema = fsolve(f_prime, [-5, 0, 5])

    # Генерация данных для построения графика
    x = np.linspace(-5, 5, 500)
    y = f(x)
    asymp = asymptote(x)

    # Построение графика функции
    plt.plot(x, y, label="f(x) = 1.1x^4 - 3x^3 + 2x + 3")
    plt.axhline(0, color='black',linewidth=1)
    plt.axvline(0, color='black',linewidth=1)

    # Добавление асимптоты пунктирной линией
    plt.plot(x, asymp, 'k--', label="Асимптота")

    # Нанесение корней на график
    for root in roots:
        plt.plot(root, f(root), 'ro')
        plt.text(root, f(root), f'Root ({root:.2f}, {f(root):.2f})', color='red')

    # Нанесение экстремумов на график
    for ext in extrema:
        plt.plot(ext, f(ext), 'bo')
        plt.text(ext, f(ext), f'Extremum ({ext:.2f}, {f(ext):.2f})', color='blue')

    # Настройка графика
    plt.title('График функции с нанесением корней, экстремумов и асимптоты')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Вывод корней и экстремумов
    print(f"Корни: {roots}")
    print(f"Экстремумы: {extrema}")
    

def e15_3():
    # Определяем функцию
    def f(x):
        return x**4 - 2*x**3 - 8*x**2 + 18*x - 9

    # Первая производная для поиска локальных экстремумов
    def f_prime(x):
        return 4*x**3 - 6*x**2 - 16*x + 18

    # Вторая производная для поиска точек перегиба
    def f_double_prime(x):
        return 12*x**2 - 12*x - 16

    # Находим корни уравнения f(x) = 0
    roots = fsolve(f, [-3.5, 0, 3.5])

    # Находим локальные экстремумы, решая f'(x) = 0
    extrema = fsolve(f_prime, [-3.5, 0, 3.5])

    # Находим точки перегиба, решая f''(x) = 0
    inflection_points = fsolve(f_double_prime, [-3.5, 0, 3.5])

    # Генерация данных для построения графика
    x = np.linspace(-3.5, 3.5, 500)
    y = f(x)

    # Построение графика
    plt.plot(x, y, label="f(x) = x^4 - 2x^3 - 8x^2 + 18x - 9")
    plt.axhline(0, color='black',linewidth=1)
    plt.axvline(0, color='black',linewidth=1)

    # Нанесение корней на график
    for root in roots:
        plt.plot(root, f(root), 'ro')
        plt.text(root, f(root), f'Root ({root:.2f}, {f(root):.2f})', color='red')

    # Нанесение локальных экстремумов на график
    for ext in extrema:
        plt.plot(ext, f(ext), 'bo')
        plt.text(ext, f(ext), f'Extremum ({ext:.2f}, {f(ext):.2f})', color='blue')

    # Нанесение точек перегиба на график
    for inflection in inflection_points:
        plt.plot(inflection, f(inflection), 'go')
        plt.text(inflection, f(inflection), f'Inflection ({inflection:.2f}, {f(inflection):.2f})', color='green')

    # Настройка графика
    plt.title('График функции с корнями, экстремумами и точками перегиба')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Вывод корней, экстремумов и точек перегиба
    print(f"Корни: {roots}")
    print(f"Экстремумы: {extrema}")
    print(f"Точки перегиба: {inflection_points}")


#################################### 16 ####################################
def e16_1(): 
    # Определяем переменную и функцию для точного вычисления
    x = sp.Symbol('x')
    f_exact = sp.Abs(x - 2)**(1/3)

    # Вычисление точного значения интеграла с помощью sympy
    exact_integral = sp.integrate(f_exact, (x, 0, 1))
    print(f"Точное значение интеграла: {exact_integral.evalf()}")

    # Определяем функцию для приближенного численного вычисления
    def f_approx(x):
        return np.abs(x - 2)**(1/3)

    # Численное вычисление интеграла с помощью scipy
    approx_integral, error = quad(f_approx, 0, 1)
    print(f"Приближенное значение интеграла: {approx_integral}")
    print(f"Погрешность численного метода: {error}")

    # Вычисление абсолютной ошибки
    absolute_error = abs(exact_integral.evalf() - approx_integral)
    print(f"Абсолютная ошибка: {absolute_error}")                                                                                     


def e16_2():
    # Определение переменной и функции для точного вычисления
    x = sp.Symbol('x')
    f_exact = x**2 * sp.cos(sp.cos(x))

    # Вычисление точного значения интеграла с помощью sympy
    exact_integral = sp.integrate(f_exact, (x, 0, sp.pi/2))
    print(f"Точное значение интеграла: {exact_integral.evalf()}")

    # Определение функции для приближённого численного вычисления
    def f_approx(x):
        return x**2 * np.cos(np.cos(x))

    # Численное вычисление интеграла с помощью scipy
    approx_integral, error = quad(f_approx, 0, np.pi/2)
    print(f"Приближённое значение интеграла: {approx_integral}")
    print(f"Погрешность численного метода: {error}")

    # Вычисление абсолютной ошибки
    absolute_error = abs(exact_integral.evalf() - approx_integral)
    print(f"Абсолютная ошибка: {absolute_error}")


def e16_3():
    # a) Интеграл ∫ cos(cos(x)) dx от 0 до +∞
    def f_a(x):
        return np.cos(np.cos(x))

    # b) Интеграл ∫ x * e^(-x^2) dx от 0 до 4
    def f_b(x):
        return x * np.exp(-x**2)

    # c) Интеграл ∫ (ln(ln(x))) / x^2 dx от 1 до +∞
    def f_c(x):
        return np.log(np.log(x)) / x**2

    # d) Интеграл ∫ (sin(sin(x))) / x dx от 0 до π
    def f_d(x):
        if x == 0:
            return 0
        return np.sin(np.sin(x)) / x

    # e) Интеграл ∫ dx / (x^2 - x^3) от 0 до 4
    def f_e(x):
        if x == 0 or x == 1:
            return 0
        return 1 / (x**2 - x**3)

    # Численное вычисление каждого интеграла
    # Параметр limit регулирует число делений при неограниченном интегрировании

    integral_a, error_a = quad(f_a, 0, np.inf, limit=100)
    integral_b, error_b = quad(f_b, 0, 4)
    integral_c, error_c = quad(f_c, 1, np.inf, limit=100)
    integral_d, error_d = quad(f_d, 0, np.pi, limit=100)
    integral_e, error_e = quad(f_e, 0, 4)

    # Вывод результатов
    print(f"Интеграл a: {integral_a}, Погрешность: {error_a}")
    print(f"Интеграл b: {integral_b}, Погрешность: {error_b}")
    print(f"Интеграл c: {integral_c}, Погрешность: {error_c}")
    print(f"Интеграл d: {integral_d}, Погрешность: {error_d}")
    print(f"Интеграл e: {integral_e}, Погрешность: {error_e}")


def e16_4():
    # Подзадача (a)
    x = sp.Symbol('x')
    parabola = 4 - x**2

    # Найдем точки пересечения с осью абсцисс
    roots = sp.solve(parabola, x)
    print(f"Корни уравнения: {roots}")

    # Интегрируем от -2 до 2
    area_a = sp.integrate(parabola, (x, roots[0], roots[1]))
    print(f"Площадь под параболой: {area_a.evalf()}")

    # Подзадача (b)
    def func_b(x):
        return 1 / np.sqrt(x)

    # Интегрируем от 0 до 1
    area_b, _ = quad(func_b, 0, 1)
    print(f"Площадь под функцией: {area_b}")    


#################################### 17 ####################################
def e17_1():
    # Определяем функцию f(x, y)
    def f(x, y):
        return x**3 - 3600*x - 50*y**2

    # Создаем сетку значений x и y в диапазоне [-100, 100]
    x = np.linspace(-100, 100, 400)
    y = np.linspace(-100, 100, 400)
    x, y = np.meshgrid(x, y)

    # Вычисляем значения функции f(x, y)
    z = f(x, y)

    # Создаем 3D-график
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Построение поверхности
    ax.plot_surface(x, y, z, cmap='viridis')

    # Настройки для удобного просмотра
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(x, y)')
    ax.view_init(elev=30, azim=60)  # Угол обзора (можно изменить для разных ракурсов)

    # Показ графика
    plt.show()


def e17_2():
    # Определяем функцию f(x, y)
    def f(x, y):
        return y * np.exp(-x**2)

    # Создаем сетку значений x и y в диапазоне [-5, 5]
    x = np.linspace(-5, 5, 400)
    y = np.linspace(-5, 5, 400)
    x, y = np.meshgrid(x, y)

    # Вычисляем значения функции f(x, y)
    z = f(x, y)

    # Создаем 3D-график
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Построение поверхности
    ax.plot_surface(x, y, z, cmap='plasma')

    # Настройки для удобного просмотра
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(x, y)')
    ax.view_init(elev=30, azim=60)  # Угол обзора

    # Показ графика
    plt.show()


def e17_3():
    # Определяем переменные
    x = sp.symbols('x')

    # Функция a)
    f_a = 3 * sp.exp(-2*x) + (x**(1/3) / (sp.ln(x) * sp.sqrt(x)))

    # Первая, вторая и третья производные функции a)
    f_a_prime = sp.diff(f_a, x)
    f_a_double_prime = sp.diff(f_a_prime, x)
    f_a_triple_prime = sp.diff(f_a_double_prime, x)

    # Функция b)
    f_b = (7*x**4 + sp.cos(1 - x)) / (sp.asin(sp.asin(x)))

    # Первая, вторая и третья производные функции b)
    f_b_prime = sp.diff(f_b, x)
    f_b_double_prime = sp.diff(f_b_prime, x)
    f_b_triple_prime = sp.diff(f_b_double_prime, x)

    # Выводим результаты
    print("Первая производная функции a:", f_a_prime)
    print("Вторая производная функции a:", f_a_double_prime)
    print("Третья производная функции a:", f_a_triple_prime)

    print("\nПервая производная функции b:", f_b_prime)
    print("Вторая производная функции b:", f_b_double_prime)
    print("Третья производная функции b:", f_b_triple_prime)


#################################### 20 ####################################
def e20_1(): 
    # Исходный вектор b
    b = [1, 3, 6, 8, 11, np.nan, 10, 9, 7, 5, 2, 2, 2, 0, 0]

    # Переставим вектор в обратном порядке
    b_reversed = b[::-1]

    # Выводим результат
    print("Исходный вектор:", b)
    print("Переставленный вектор:", b_reversed)
    
    
def e20_2():
    # Определяем вектор d
    d = [4, -3, 6, 8, 11, 0, 5, 9, 17, 5, 3, 2, -1, 0, 4, 12]

    # Вычисляем сумму элементов на нечетных местах
    # В Python индексация начинается с 0, поэтому выбираем элементы с нечетными индексами
    sum_odd_positions = sum(d[i] for i in range(0, len(d), 2))

    # Выводим результат
    print("Сумма элементов на нечетных позициях:", sum_odd_positions)
    
    
def e20_3():
    l = [1, 2, 3, 4, 5, 1, 2, 1]
    l = [
        "one" if n == 1 else "two" if n == 2 else n
        for n in l
    ]
    print(l)


def e20_4():
    # Определяем функцию f(x) по системе
    def f(x):
        if x <= 0:
            return np.cos(np.cos(x))
        elif 0 < x <= 4:
            return 1 + np.sqrt(x)
        else:
            return 4

    # Определяем функцию g(x)
    def g(x):
        return -x if x >= 1 else x

    # Векторизуем функции для работы с массивами
    f_vec = np.vectorize(f)
    g_vec = np.vectorize(g)

    # Определяем диапазон x от -6 до 6
    x_values = np.linspace(-6, 6, 500)

    # Вычисляем значения функций
    f_values = f_vec(x_values)
    g_values = g_vec(x_values)
    f_g_values = f_vec(g_values)

    # Построение графиков
    plt.plot(x_values, f_values, label='f(x)', color='blue')
    plt.plot(x_values, g_values, label='g(x)', color='green')
    plt.plot(x_values, f_g_values, label='f(g(x))', color='red')

    # Настройки графиков
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Графики функций f(x), g(x) и f(g(x))')
    plt.legend()
    plt.grid(True)

    # Показать график
    plt.show()


#################################### 21 ####################################
def e21_1():
    # Зададим параметры для метода Эйлера
    x_start = 0     # начальное значение x
    x_end = 13      # конечное значение x
    y0 = 3          # начальное условие y(0)
    h = 0.01        # шаг метода Эйлера

    # Дифференциальное уравнение y' = -sin(sin(x))
    def f(x, y):
        return -np.sin(np.sin(x))

    # Метод Эйлера
    def euler_method(x_start, x_end, y0, h):
        n = int((x_end - x_start) / h)  # количество шагов
        x_values = np.linspace(x_start, x_end, n)
        y_values = np.zeros(n)
        y_values[0] = y0
        
        for i in range(1, n):
            y_values[i] = y_values[i-1] + h * f(x_values[i-1], y_values[i-1])
        
        return x_values, y_values

    # Точное решение y(x) = cos(cos(x))
    def exact_solution(x):
        return np.cos(np.cos(x))

    # Получим результаты численного метода Эйлера
    x_vals, y_vals_approx = euler_method(x_start, x_end, y0, h)

    # Точное решение на тех же x
    y_vals_exact = exact_solution(x_vals)

    # Построение графика
    plt.plot(x_vals, y_vals_approx, label="Приближённое решение (метод Эйлера)", color='blue')
    plt.plot(x_vals, y_vals_exact, label="Точное решение y(x) = cos(cos(x))", color='red', linestyle='dashed')
    plt.title('Сравнение приближённого решения и точного решения')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()


def e21_2():
    # Параметры метода Эйлера
    x_start = 0      # начальное значение x
    x_end = 5        # конечное значение x
    y0 = 1           # начальное условие y(0)
    h = 0.01         # шаг метода Эйлера

    # Дифференциальное уравнение y' = y + xy^3
    def f(x, y):
        return y + x * y**3

    # Метод Эйлера
    def euler_method(x_start, x_end, y0, h):
        n = int((x_end - x_start) / h)  # количество шагов
        x_values = np.linspace(x_start, x_end, n)
        y_values = np.zeros(n)
        y_values[0] = y0
        
        for i in range(1, n):
            y_values[i] = y_values[i-1] + h * f(x_values[i-1], y_values[i-1])
        
        return x_values, y_values

    # Получим результаты численного метода Эйлера
    x_vals, y_vals_approx = euler_method(x_start, x_end, y0, h)

    # Построение графика
    plt.plot(x_vals, y_vals_approx, label="Приближённое решение (метод Эйлера)", color='blue')
    plt.title('Численное решение уравнения Бернулли методом Эйлера')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def e21_3():
    # Определяем функцию y(x)
    def y_exact(x):
        return 1 / np.sqrt(0.5 * x * np.exp(-2*x) - x + 0.5)

    # Задаем шаг dx и диапазон x
    dx = 0.01
    x_values = np.arange(0, 0.63 + dx, dx)

    # Вычисляем значения функции y(x) для точного решения
    y_values_exact = y_exact(x_values)

    # Строим график
    plt.plot(x_values, y_values_exact, label='Точное решение', color='blue')

    # Настройки графика
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('График функции y(x)')
    plt.legend()
    plt.grid(True)

    # Показать график
    plt.show()


if __name__ == '__main__':
    # 13
    # print(f"{e13_1():.6f}")
    # print(f"{e13_2():.3f}")
    # print(f"Average braking distance: {e13_3("cars.txt"):.2f} meters")
    
    # 14 
    # print(e14_1())
    # e14_2()
    # e14_3()
    # e14_4()

    # 15
    # e15_1()
    # e15_2()
    # e15_3()   
    
    # 16
    # e16_1()
    # e16_2() 
    # e16_3()  
    # e16_4()
    
    # 17
    # e17_1()
    # e17_2()
    # e17_3()
    
    #20
    # e20_1()
    # e20_2()
    # e20_3()
    # e20_4()
    
    #21
    # e21_1()
    # e21_2()
    # e21_3()
    
    
    
    
    
        