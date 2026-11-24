from fractions import Fraction
from itertools import combinations, product

import numpy as np
from scipy.optimize import linprog


A1_CENTER = np.array(
    [
        [0.95, 1.00],
        [1.05, 1.00],
        [1.10, 1.00],
    ]
)

A1_REGRESSION_D = np.array(
    [
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ]
)

A1_TOMOGRAPHY_D = np.ones((3, 2))

A2_CENTER = np.array(
    [
        [1.10, 0.90, 1.10],
        [1.40, 1.00, 0.80],
        [0.80, 1.40, 1.20],
    ]
)

A2_TOMOGRAPHY_D = np.ones((3, 3))


def build_tomography_matrix(center, D, x):
    # Строим для задачи томографии матрицу A', для которой
    # A'x = 0 и |A' - A_c| <= delta D.
    residual = center @ x
    capacity = D @ np.abs(x)
    row_factors = np.zeros(center.shape[0])

    for row in range(center.shape[0]):
        if capacity[row] <= 1e-12:
            if abs(residual[row]) > 1e-10:
                raise ValueError("Для найденного x нельзя построить матрицу A'")
        else:
            row_factors[row] = -residual[row] / capacity[row]

    E = row_factors[:, None] * D * np.sign(x)
    return center + E


def solve_regression(center):
    # Первый столбец содержит измеренные t_i,
    # второй столбец единиц является точным.
    center = np.asarray(center, dtype=float)

    if center.ndim != 2 or center.shape[1] != 2:
        raise ValueError("Для регрессии нужна матрица размера m x 2")
    if not np.allclose(center[:, 1], 1.0):
        raise ValueError("Второй столбец должен состоять из единиц")

    values = center[:, 0]
    minimum = np.min(values)
    maximum = np.max(values)

    delta = (maximum - minimum) / 2.0
    new_column_value = minimum + delta

    A_prime = center.copy()
    A_prime[:, 0] = new_column_value

    x = np.array([1.0, -new_column_value])

    return delta, x, A_prime


def solve_tomography(center):
    # Все элементы матрицы имеют одинаковый радиус delta.
    center = np.asarray(center, dtype=float)

    if center.ndim != 2:
        raise ValueError("Матрица центра должна быть двумерной")

    row_count, column_count = center.shape
    best_delta = np.inf
    best_x = None

    # Из-за однородности условия полагаем ||x||_1 = 1.
    # При фиксированных знаках x = signs * u, где u >= 0.
    for signs in product((-1.0, 1.0), repeat=column_count):
        signs = np.array(signs)
        signed_center = center * signs

        # Переменные задачи: u_1, ..., u_n, delta.
        objective = np.zeros(column_count + 1)
        objective[-1] = 1.0

        inequalities = np.vstack(
            (
                np.column_stack((signed_center, -np.ones(row_count))),
                np.column_stack((-signed_center, -np.ones(row_count))),
            )
        )

        normalization = np.zeros((1, column_count + 1))
        normalization[0, :column_count] = 1.0

        result = linprog(
            c=objective,
            A_ub=inequalities,
            b_ub=np.zeros(2 * row_count),
            A_eq=normalization,
            b_eq=np.array([1.0]),
            bounds=[(0.0, None)] * (column_count + 1),
            method="highs",
        )

        if result.success and result.fun < best_delta:
            best_delta = result.fun
            best_x = signs * result.x[:column_count]

    if best_x is None:
        raise ValueError("Не удалось найти критический радиус")

    D = np.ones_like(center)
    A_prime = build_tomography_matrix(center, D, best_x)

    return best_delta, best_x, A_prime


def maximal_minors(matrix):
    # Для матрицы m x n вычисляем все миноры порядка n.
    column_count = matrix.shape[1]
    values = []

    for rows in combinations(range(matrix.shape[0]), column_count):
        values.append(np.linalg.det(matrix[list(rows), :]))

    return np.array(values)


def print_result(name, center, D, delta, x, A_prime):
    exact_delta = Fraction(delta).limit_denominator(1_000_000)

    print(f"\n{name}")
    print(f"Критический радиус: delta_* = {delta:.10f} = {exact_delta}")
    print(f"Диапазон особенности: [{delta:.10f}, +inf)")
    print("Вектор x_*:")
    print(x)
    print("Матрица A':")
    print(A_prime)

    # E требуется только для проверки принадлежности A' интервалу.
    E = A_prime - center
    residual = np.linalg.norm(A_prime @ x, ord=np.inf)
    interval_error = np.max(np.abs(E) - delta * D)
    rank = np.linalg.matrix_rank(A_prime, tol=1e-8)

    print(f"rank(A') = {rank}")
    print(f"max |A'x_*| = {residual:.3e}")
    print(f"max (|A' - A_c| - delta_* D) = {interval_error:.3e}")

    if A_prime.shape[0] == A_prime.shape[1]:
        print(f"det(A') = {np.linalg.det(A_prime):.3e}")
    else:
        print("Максимальные миноры A':")
        print(maximal_minors(A_prime))


def main():
    np.set_printoptions(precision=10, suppress=True)

    delta, x, A_prime = solve_regression(A1_CENTER)
    print_result(
        "Линейная регрессия: изменяется первый столбец A1",
        A1_CENTER,
        A1_REGRESSION_D,
        delta,
        x,
        A_prime,
    )

    delta, x, A_prime = solve_tomography(A1_CENTER)
    print_result(
        "Малоракурсная томография: изменяются все элементы A1",
        A1_CENTER,
        A1_TOMOGRAPHY_D,
        delta,
        x,
        A_prime,
    )

    delta, x, A_prime = solve_tomography(A2_CENTER)
    print_result(
        "Малоракурсная томография: изменяются все элементы A2",
        A2_CENTER,
        A2_TOMOGRAPHY_D,
        delta,
        x,
        A_prime,
    )


if __name__ == "__main__":
    main()
