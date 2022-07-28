import numpy as np
import random

# Обучающая выборка (идеальное изображение цифр от О до 9)
num0 = list('111101101101111')
num1 = list('001001001001001')
num2 = list('111001111100111')
num3 = list('111001111001111')
num4 = list('101101111001001')
num5 = list('111100111001111')
num6 = list('111100111101111')
num7 = list('111001001001001')
num8 = list('111101111101111')
num9 = list('111101111001111')

# Список всех цифр от О до 9 в едином массиве
nums = [num0, num1, num2, num3, num4, num5, num6, num7, num8, num9]

tema = 7
n_sensor = 15

# Веса
weights = []
for i in range(15):
    weights.append(0)


def perceptron(Sensor):
    b = 7  # Порог функции активации
    s = 0  # Начальное значение суммы
    for i in range(n_sensor):
        s += int(Sensor[i])
        if s >= b:
            return True
        else:
            return False


# Уменьшение значений весов
# Если сеть ошиблась и выдала Да при входной цифре, отличной от пятерки
def decrease(number):
    for i in range(n_sensor):
        if int(number[i]) == 1:
            weights[i] -= 1


# Увеличение значений весов
# Если сеть не ошиблась и выдала Да при поданной на вход цифре 5
def increase(nurnЬer):
    for i in range(n_sensor):
        if int(nurnЬer[i]) == 1:  # Возбужденный ли вход
            weights[i] += 1


# Тренировка сети
n = 1  # количество уроков
for i in range(n):
    j = random.randint(0, 9)
    r = perceptron(nums[j])
    if j != tema:
        if r:
            decrease(nums[j])
    else:
        if not r:
            increase(nums[tema])
    print(j, weights)
