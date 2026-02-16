import numpy as np
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import time

# Инициализация генератора
rng = np.random.default_rng(int(time.time_ns()))

# Константы
P = 4e6             # Внутреннее давление (Па)
V_min = 0.4         # Минимальный внутренний объем (м^3)
rho = 7800          # Плотность материала (кг/м^3)
sigma_max = 9.5e6   # Максимально допустимое напряжение (Па)

# Ограничения для переменных: [R, L, T_s, T_h]
bounds = np.array([
    [0.2, 1.0],    # R (м)
    [0.2, 1.8],    # L (м)
    [0.01, 0.15],  # T_s (м)
    [0.01, 0.10]   # T_h (м)
], dtype=float)

# Геометрия и физика
def vessel_volume(R, L):
    # Подсчет внутреннего объема сосуда
    return (4 / 3) * np.pi * R**3 + np.pi * R**2 * L

def stress(R, T_s, T_h):
    # Подсчет максимального напряжения
    hemi_stress = P * R / (2 * T_h)   # Полусфера
    cylinder_stress = P * R / T_s     # Цилиндр
    return max(hemi_stress, cylinder_stress)

def material_volume(R, L, T_s, T_h):
    # Подсчет объема материала сосуда
    sphere_volume = (4 / 3) * np.pi * ((R + T_h)**3 - R**3)
    cylinder_volume = np.pi * L * ((R + T_s)**2 - R**2)
    return sphere_volume + cylinder_volume

def true_mass(x):
    # Подсчет массы сосуда без штрафов
    R, L, T_s, T_h = x
    return rho * material_volume(R, L, T_s, T_h)

# Ограничения
def constraint_violations(x):
    # Подсчет величины нарушения ограничений
    R, L, T_s, T_h = x
    V = vessel_volume(R, L)
    sigma = stress(R, T_s, T_h)

    v_violation = max(0, V_min - V)          # Объема не хватает
    s_violation = max(0, sigma - sigma_max)  # Напряжение превышено
    return v_violation, s_violation

def is_feasible(x, v_tol=1e-6, s_tol=1e4):
    # Проверка, является ли решение допустимым
    v_viol, s_viol = constraint_violations(x)
    return (v_viol <= v_tol) and (s_viol <= s_tol)

def is_within_bounds(x):
    # Проверка, находится ли x в пределах допустимых границ
    return np.all(x >= bounds[:, 0]) and np.all(x <= bounds[:, 1])

# Целевая функция со штрафами
def penalized_objective(x, penalty_weight=1e6):
    # Целевая функция со штрафами за нарушение ограничений
    if not is_within_bounds(x):
        return np.inf

    R, L, T_s, T_h = x
    if (R <= 0) or (L <= 0) or (T_s <= 0) or (T_h <= 0):
        return np.inf

    m = true_mass(x)
    if not np.isfinite(m) or m <= 0:
        return np.inf

    v_violation, s_violation = constraint_violations(x)
    epsilon = 1e-12

    # Нормализация нарушений ограничений
    v_norm = v_violation / (V_min + epsilon)
    s_norm = s_violation / (sigma_max + epsilon)

    # Логарифмический мягкий штраф
    v_penalty = np.log1p(v_norm) if v_violation > 0 else 0.0
    s_penalty = np.log1p(s_norm) if s_violation > 0 else 0.0

    return m + penalty_weight * (v_penalty + s_penalty)


def expected_improvement(X, gp, f_min, xi=0.05):
    # Вычисление функции ожидаемого улучшения (EI)
    mu, sigma = gp.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-9)

    improvement = f_min - mu - xi
    Z = improvement / sigma

    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma <= 1e-9] = 0.0
    return ei

# Сэмплирование
def sample_points_lhs(n, seed):
    # Генерация точек
    sampler = qmc.LatinHypercube(d=bounds.shape[0], seed=seed, scramble=True)
    u = sampler.random(n)
    return bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])

def sample_points_random(n):
    # Случайная генерация точек в пределах границ
    return rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, bounds.shape[0]))


# Байесовская оптимизация
def optimize(
    n_initial_points=50,
    n_iterations=100,
    n_candidates=5000,
    patience=25
):
    # Начальные точки
    X = sample_points_lhs(n_initial_points, seed=1)
    y = np.array([penalized_objective(x) for x in X])

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5)

    best_feasible_mass = np.inf
    best_feasible_x = None
    no_improvement = 0

    for iteration in range(n_iterations):
        # Адаптивный уровень шума GP
        alpha = 1e-4 + 1e-5 * iteration

        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            alpha=alpha,
            n_restarts_optimizer=5,
            random_state=1
        )
        gp.fit(X, y)

        # Выбор кандидатов
        if iteration < 5:
            candidates = sample_points_lhs(n_candidates, seed=iteration)
        else:
            candidates = sample_points_random(n_candidates)

        f_min = np.min(y)
        ei = expected_improvement(candidates, gp, f_min)
        next_point = candidates[np.argmax(ei)]
        next_value = penalized_objective(next_point)

        X = np.vstack([X, next_point])
        y = np.append(y, next_value)

        # Мониторинг лучших допустимых решений
        if is_feasible(next_point):
            m = true_mass(next_point)
            if m < best_feasible_mass:
                best_feasible_mass = m
                best_feasible_x = next_point
                no_improvement = 0
            else:
                no_improvement += 1
        elif best_feasible_x is not None:
            no_improvement += 1

        # Ранняя остановка: нет улучшений 'patience' итераций подряд
        if best_feasible_x is not None and no_improvement >= patience:
            break

    # Если допустимое решение не найдено — берем лучшее по штрафу
    if best_feasible_x is None:
        best_feasible_x = X[np.argmin(y)]

    return best_feasible_x
# Вывод результата
def print_solution(x):
    R, L, T_s, T_h = x
    m = true_mass(x)
    V = vessel_volume(R, L)
    sigma = stress(R, T_s, T_h)
    v_viol, s_viol = constraint_violations(x)

    print("\n" + "=" * 50)
    print("ЛУЧШЕЕ НАЙДЕННОЕ РЕШЕНИЕ")
    print("=" * 50)
    print(f"R   = {R:.6f} м")
    print(f"L   = {L:.6f} м")
    print(f"T_s = {T_s:.6f} м")
    print(f"T_h = {T_h:.6f} м")
    print("-" * 50)
    print(f"Масса:        {m:.2f} кг")
    print(f"Объем:        {V:.6f} м³ (мин {V_min:.3f})")
    print(f"Напряжение:   {sigma / 1e6:.3f} МПа")
    print(f"V violation:  {v_viol:.8e} м³")
    print(f"S violation:  {s_viol / 1e3:.3f} кПа")
    print(f"Допустимо:    {'ДА' if is_feasible(x) else 'НЕТ'}")
    print("=" * 50)

# Запуск
if __name__ == "__main__":
    best_solution = optimize()
    print_solution(best_solution)