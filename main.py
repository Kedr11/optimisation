import math
import numpy as np

from scipy.optimize import minimize
from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# ФИЗИЧЕСКИЕ КОНСТАНТЫ

Pi = math.pi

P = 4 * 10 ** 6  # давление
Vmin = 0.4  # минимальный объем
pl = 7800  # плотность
omax = 9.5 * 10 ** 6  # допустимое напряжение


# ФИЗИЧЕСКАЯ МОДЕЛЬ
iterations = 100


def mass(R, L, TS, TH):
    return pl * (
            4 / 3 * Pi * ((R + TH) ** 3 - R ** 3)
            + Pi * L * ((R + TS) ** 2 - R ** 2)
    )


def volume_s(R, L):
    return 4 / 3 * Pi * R ** 3 + Pi * R ** 2 * L


def stress(P, R, TH, TS):
    return max(
        P * R / (2 * TH),
        P * R / TS
    )


# ЦЕЛЕВАЯ ФУНКЦИЯ СО ШТРАФАМИ

def objective(x):
    R, L, TS, TH = x

    m = mass(R, L, TS, TH)
    V = volume_s(R, L)
    s = stress(P, R, TH, TS)

    penalty = 0.0

    if V < Vmin:
        penalty += 1000

    if s > omax:
        penalty += 1000

    return m + penalty

# ГРАНИЦЫ ПАРАМЕТРОВ

bounds = [
    (0.2, 1.0),  # R
    (0.2, 1.8),  # L
    (0.01, 0.15),  # TS
    (0.01, 0.1)  # TH
]


# СЛУЧАЙНЫЕ НАЧАЛЬНЫЕ ТОЧКИ

def random_samples(n):
    return np.array([
        [np.random.uniform(low, high) for (low, high) in bounds]
        for _ in range(n)
    ])


# EXPECTED IMPROVEMENT

def expected_improvement(x, gp, y_min):
    x = np.array(x).reshape(1, -1)

    mu, sigma = gp.predict(x, return_std=True)

    if sigma == 0:
        return 0.0

    improvement = y_min - mu
    Z = improvement / sigma

    return improvement * norm.cdf(Z) + sigma * norm.pdf(Z)


# НАЧАЛЬНЫЕ ДАННЫЕ

X = random_samples(10)
y = np.array([objective(x) for x in X])

# GAUSSIAN PROCESS

kernel = Matern(nu=2.5)

gp = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True
)

# ОСНОВНОЙ ЦИКЛ BO


for i in range(iterations):
    # обучаем GP
    gp.fit(X, y)

    # лучший найденный результат
    y_min = np.min(y)

    # функция для minimize (minimize умеет только минимизировать)
    def neg_ei(x):
        return -expected_improvement(x, gp, y_min)


    # случайная стартовая точка
    x0 = random_samples(1)[0]

    # оптимизация acquisition function
    res = minimize(
        neg_ei,
        x0=x0,
        bounds=bounds
    )

    # новая точка
    x_new = res.x
    y_new = objective(x_new)

    # добавляем данные
    X = np.vstack([X, x_new])
    y = np.append(y, y_new)

    print(f"Итерация {i + 1}: масса = {y_new:.2f}")

# =========================
# ЛУЧШЕЕ РЕШЕНИЕ
# =========================

best_index = np.argmin(y)
best_x = X[best_index]
best_value = y[best_index]

print("\nЛучшее решение:")
print(f"R  = {best_x[0]}")
print(f"L  = {best_x[1]}")
print(f"TS = {best_x[2]}")
print(f"TH = {best_x[3]}")
print(f"Минимальная масса = {best_value}")
