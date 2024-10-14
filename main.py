from scipy.special import comb

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
        