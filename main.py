from scipy.special import comb
from scipy.optimize import fsolve

import numpy as np
import matplotlib.pyplot as plt

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


#################################### 13 ####################################
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
    
    #15
    # e15_1()
    # e15_2()
    e15_3()
        