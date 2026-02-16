import math
import numpy as np

from scipy.optimize import minimize
from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# ===============================
# ФИЗИЧЕСКИЕ КОНСТАНТЫ
# ===============================

Pi = math.pi

P = 4e6              # давление
Vmin = 0.4           # минимальный объем
pl = 7800            # плотность
omax = 9.5e6         # допустимое напряжение

iterations = 100

# ===============================
# ФИЗИЧЕСКАЯ МОДЕЛЬ
# ===============================

def mass(R, L, TS, TH):
    return pl * (
        4 / 3 * Pi * ((R + TH) ** 3 - R ** 3)
        + Pi * L * ((R + TS) ** 2 - R ** 2)
    )

def volume_s(R, L):
    return 4 / 3 * Pi * R ** 3 + Pi * R ** 2 * L

def stress(R, TS, TH):
    return max(
        P * R / (2 * TH),
        P * R / TS
    )

# Ограничение в виде g(x) >= 0
def stress_constraint(x):
    R, L, TS, TH = x
    return omax - stress(R, TS, TH)

# ===============================
# ГРАНИЦЫ
# ===============================

bounds = [
    (0.2, 1.0),
    (0.2, 1.8),
    (0.01, 0.15),
    (0.01, 0.1)
]

# ===============================
# СЛУЧАЙНЫЕ НАЧАЛЬНЫЕ ТОЧКИ
# ===============================

def random_samples(n):
    return np.array([
        [np.random.uniform(low, high) for (low, high) in bounds]
        for _ in range(n)
    ])

# ===============================
# EXPECTED IMPROVEMENT
# ===============================

def expected_improvement(x, gp, y_min):
    x = np.array(x).reshape(1, -1)
    mu, sigma = gp.predict(x, return_std=True)

    sigma = max(sigma[0], 1e-9)
    mu = mu[0]

    improvement = y_min - mu
    Z = improvement / sigma

    return improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

# ===============================
# PROBABILITY OF FEASIBILITY
# ===============================

def probability_feasible(x, gp_constraint):
    x = np.array(x).reshape(1, -1)
    mu, sigma = gp_constraint.predict(x, return_std=True)

    sigma = max(sigma[0], 1e-9)
    mu = mu[0]

    # Вероятность того, что g(x) >= 0
    return norm.cdf(mu / sigma)

# ===============================
# НАЧАЛЬНЫЕ ДАННЫЕ
# ===============================

X = random_samples(10)

y_mass = np.array([
    mass(*x) for x in X
])

y_constraint = np.array([
    stress_constraint(x) for x in X
])

# ===============================
# GAUSSIAN PROCESS
# ===============================

kernel = Matern(nu=2.5)

gp_mass = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True
)

gp_constraint = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True
)

# ===============================
# ОСНОВНОЙ ЦИКЛ BO
# ===============================

for i in range(iterations):

    # обучаем модели
    gp_mass.fit(X, y_mass)
    gp_constraint.fit(X, y_constraint)

    y_min = np.min(y_mass)

    # acquisition = EI * P_feasible
    def neg_acquisition(x):

        R, L, TS, TH = x

        # фильтр по объему
        if volume_s(R, L) < Vmin:
            return 0.0

        ei = expected_improvement(x, gp_mass, y_min)
        p_feas = probability_feasible(x, gp_constraint)

        return -(ei * p_feas)

    # случайная стартовая точка
    x0 = random_samples(1)[0]

    res = minimize(
        neg_acquisition,
        x0=x0,
        bounds=bounds
    )

    x_new = res.x

    # вычисляем истинные значения
    y_new_mass = mass(*x_new)
    y_new_constraint = stress_constraint(x_new)

    # добавляем в данные
    X = np.vstack([X, x_new])
    y_mass = np.append(y_mass, y_new_mass)
    y_constraint = np.append(y_constraint, y_new_constraint)

    print(f"Итерация {i+1}: масса = {y_new_mass:.2f}")

# ===============================
# ВЫБОР ЛУЧШЕГО ДОПУСТИМОГО
# ===============================

feasible_mask = np.array([
    (volume_s(x[0], x[1]) >= Vmin) and
    (stress_constraint(x) >= 0)
    for x in X
])

if np.any(feasible_mask):
    feasible_masses = y_mass[feasible_mask]
    best_index = np.where(feasible_mask)[0][np.argmin(feasible_masses)]
else:
    best_index = np.argmin(y_mass)

best_x = X[best_index]
best_value = y_mass[best_index]

print("\nЛучшее решение:")
print(f"R  = {best_x[0]}")
print(f"L  = {best_x[1]}")
print(f"TS = {best_x[2]}")
print(f"TH = {best_x[3]}")
print(f"Минимальная масса = {best_value}")