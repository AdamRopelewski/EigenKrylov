import numpy as np
import time
import random


def generate_random_symmetric_matrix(n, a=100, b=1000):
    """
    Generuje losową macierz symetryczną o rozmiarze n x n.

    :param n: Rozmiar macierzy.
    :type n: int
    :param a: Minimalna wartość losowa, domyślnie 100.
    :type a: int, opcjonalnie
    :param b: Maksymalna wartość losowa, domyślnie 1000.
    :type b: int, opcjonalnie
    :return: Losowa macierz symetryczna.
    :rtype: list
    """
    A = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            A[i][j] = random.randint(a, b)
    for i in range(n):
        for j in range(i):
            A[i][j] = A[j][i]
    return A


def upperTriangleMatrixSolver(u, b: list) -> list:
    """
    Rozwiązuje układ równań dla macierzy trójkątnej górnej.

    :param u: Macierz trójkątna górna.
    :type u: list of lists
    :param b: Wektor wyników.
    :type b: list
    :return: Wektor rozwiązań układu równań.
    :rtype: list
    """
    x = [0] * len(b)
    n = len(b)
    x[n - 1] = b[n - 1] / u[n - 1][n - 1]
    for i in range(n - 2, -1, -1):
        temp = 0
        for k in range(i + 1, n):
            temp += u[i][k] * x[k]
        x[i] = (b[i] - temp) / u[i][i]
    return x


def matrixSolver(A):
    """
    Rozwiązuje układ równań liniowych przy użyciu eliminacji Gaussa.

    :param A: Macierz współczynników układu równań.
    :type A: list
    :return: Zmodyfikowana macierz współczynników po wykonaniu eliminacji Gaussa.
    :rtype: list
    """
    row_count = len(A)
    column_count = len(A[0])
    for i in range(0, row_count):  # wybór wiersza odejmowanego n - liczba wierszy
        for j in range(i + 1, row_count):  # wybór wiersza w którym odejmuję
            scalingFactor = float(A[j][i] / A[i][i])  # wyliczenie mnożnika
            for k in range(0, column_count):
                A[j][k] -= A[i][k] * scalingFactor
    return A


def extractVectorFromMatrix(matrix):
    """
    Wyodrębnia ostatnią kolumnę z macierzy i zwraca ją jako wektor.

    :param matrix: Macierz wejściowa.
    :type matrix: list of lists
    :return: Ostatnia kolumna jako wektor oraz zmodyfikowana macierz.
    :rtype: tuple
    """
    matrix = matrix.transpose()
    y = matrix[-1]
    matrix = matrix[:-1]
    matrix = matrix.transpose()
    return y, matrix


def f(x, A):
    """
    Zwraca wartość wielomianu w punkcie x.

    :param x: Punkt, w którym obliczana jest wartość wielomianu.
    :type x: float
    :param A: Współczynniki wielomianu.
    :type A: list
    :return: Wartość wielomianu.
    :rtype: float
    """
    n = len(A)
    result = 1
    for i in range(n):
        result = result * x + A[i]
    return result


def find_root_interval(a, b, A):
    """
    Znajduje przedział dla metody siecznych.

    :param a: Początek przedziału.
    :type a: float
    :param b: Koniec przedziału.
    :type b: float
    :param A: Współczynniki wielomianu.
    :type A: list
    :return: Znaleziony przedział.
    :rtype: tuple
    """
    for i in range(a, b * 10):
        i /= 10
        if f(i, A) * f(i + 1, A) < 0:
            return i, i + 1
    return None, None


def secant(x1, x2, epsilon, A):
    """
    Metoda siecznych. Zwraca przybliżoną wartość pierwiastka.

    :param x1: Początkowa wartość x1.
    :type x1: float
    :param x2: Początkowa wartość x2.
    :type x2: float
    :param epsilon: Dokładność obliczeń.
    :type epsilon: float
    :param A: Współczynniki wielomianu.
    :type A: list
    :return: Przybliżona wartość pierwiastka.
    :rtype: float
    """
    f_x1 = f(x1, A)
    f_x2 = f(x2, A)
    while abs(x2 - x1) > epsilon:
        if (f_x2 - f_x1) == 0:
            break
        x3 = x2 - f_x2 * (x2 - x1) / (f_x2 - f_x1)
        f_x3 = f(x3, A)
        x1, x2 = x2, x3
        f_x1, f_x2 = f_x2, f_x3
    precission = len(str(int(1 / epsilon))) - 1
    return round(x2, precission)


def horner(A, x0):
    """
    Dzieli wielomian przez dwumian (x - x0)
    i zwraca współczynniki wielomianu wynikowego.

    :param A: Współczynniki wielomianu.
    :type A: list
    :param x0: Wartość, przez którą dzielony jest wielomian.
    :type x0: float
    :return: Współczynniki wielomianu wynikowego.
    :rtype: list
    """
    B = [1]
    B.extend(A)
    A = B
    n = len(A)
    coefficients = [0.0] * (n - 1)
    coefficients[0] = A[0]
    for i in range(1, n - 1):
        coefficients[i] = coefficients[i - 1] * x0 + A[i]
    return coefficients[1:]


def matrixNorm(A):
    """
    Zwraca normę (maksium) macierzy.

    :param A: Macierz wejściowa.
    :type A: list of lists
    :return: Norma macierzy.
    :rtype: float
    """
    n = len(A)
    max_sum = 0
    for i in range(n):
        sum = 0
        for j in range(n):
            sum += abs(A[i][j])
        if sum > max_sum:
            max_sum = sum
    return max_sum


def calculateEigenValues(characteristic_polynomial, epsilon, x1, x2):
    """
    Oblicza wartości własne macierzy na podstawie wielomianu charakterystycznego.

    :param characteristic_polynomial: Współczynniki wielomianu charakterystycznego.
    :type characteristic_polynomial: list
    :param epsilon: Dokładność obliczeń.
    :type epsilon: float
    :param x1: Początkowa wartość x1 dla metody siecznych.
    :type x1: float
    :param x2: Początkowa wartość x2 dla metody siecznych.
    :type x2: float
    :return: Lista wartości własnych.
    :rtype: list
    """
    lambdas = []
    for i in range(len(characteristic_polynomial)):
        lambdas.append(secant(x1, x2, epsilon, characteristic_polynomial))
        characteristic_polynomial = horner(characteristic_polynomial, lambdas[i])
    return lambdas


def checkEigenValues(A, eigen_values: list, epsilon: float):
    """
    Sprawdza poprawność obliczonych wartości własnych.

    :param A: Macierz wejściowa.
    :type A: list of lists
    :param eigen_values: Lista wartości własnych do sprawdzenia.
    :type eigen_values: list
    :param epsilon: Dokładność porównania.
    :type epsilon: float
    """
    control_eigen_values = np.linalg.eig(A)[0]
    control_eigen_values.sort()
    print("Uzyskane wartości własne:")
    print(eigen_values)
    print("Sprawdzenie:")
    print(control_eigen_values)

    score = 0
    n = len(eigen_values)
    for i in range(n):
        if abs(eigen_values[i] - control_eigen_values[i]) < epsilon * 100:
            score += 1
    print("Dokładność powównania: epsilon * 100")
    print(f"Wynik: {score}/{n}")


def krylow(A):
    """
    Oblicza wielomian charakterystyczny macierzy kwadratowej przy użyciu metody Krylowa.

    :param A: Macierz wejściowa.
    :type A: list of lists
    :return: Współczynniki wielomianu charakterystycznego.
    :rtype: list
    """
    n = len(A)
    b = [0.0] * n
    b[0] = 1
    matrix1 = [[0.0] * n for _ in range(n)]
    matrix1[0] = b
    for i in range(1, n):
        matrix1[i] = np.dot(A, matrix1[i - 1])
    matrix1.reverse()
    rightSide = -np.dot(A, matrix1[0])
    matrix_to_solve = matrix1
    matrix_to_solve.append(rightSide)
    extended_matrix_to_solve = np.array(matrix_to_solve).transpose()
    extended_triangle_matrix = matrixSolver(extended_matrix_to_solve)
    y, triangle_Matrix = extractVectorFromMatrix(extended_triangle_matrix)
    characteristic_polynomial = upperTriangleMatrixSolver(triangle_Matrix, y)
    return characteristic_polynomial


def vectorNorm(v):
    """
    Oblicza normę wektora.

    :param v: Wektor wejściowy.
    :type v: list
    :return: Norma wektora.
    :rtype: float
    """
    return sum([x * x for x in v]) ** 0.5


def eigenVector(A, eigen_value):
    """
    Oblicza wektor własny dla danej wartości własnej macierzy.

    :param A: Macierz wejściowa.
    :type A: list of lists
    :param eigen_value: Wartość własna.
    :type eigen_value: float
    :return: Wektor własny.
    :rtype: list
    """
    n = len(A)
    I = np.identity(n)
    B = np.array(A) - eigen_value * I
    y = [0] * n
    y[-1] = 1
    B = B.transpose()
    B = matrixSolver(B.tolist())
    x = upperTriangleMatrixSolver(B, y)
    vector_norm = vectorNorm(x)
    x = [round(i / vector_norm, 6) for i in x]
    return x


def Ropelewski_Adam_(A, epsilon=1e-6):
    """
    Oblicza wartości własne oraz wektory własne macierzy kwadratowej, symetrycznej.

    :param A: Macierz kwadratowa, symetryczna.
    :type A: list of lists
    :param epsilon: Dokładność obliczeń.
    :type epsilon: float
    :return: Wartości własne i wektory własne macierzy.
    :rtype: tuple
    """
    characteristic_polynomial = krylow(A)
    matrix_norm_value = matrixNorm(A)
    print(f"Norma macierzy: {matrix_norm_value}")
    x1, x2 = find_root_interval(
        -matrix_norm_value, matrix_norm_value, characteristic_polynomial
    )
    if x1 is None or x2 is None:
        raise ValueError("Nie znaleziono przedziału dla metody siecznych.")
    eigenValues = calculateEigenValues(characteristic_polynomial, epsilon, x1, x2)
    eigenValues.sort()
    checkEigenValues(A, eigenValues, epsilon)
    list_of_eigen_vectors = []
    for eigenValue in eigenValues:
        list_of_eigen_vectors.append(eigenVector(A, eigenValue))
    return eigenValues, list_of_eigen_vectors


if __name__ == "__main__":
    A = [[2, 1, 2], [1, 2, 1], [2, 1, 1]]
    n = 6
    A = generate_random_symmetric_matrix(n, 100, 1000)
    epsilon = 1e-8
    current_time = time.time()
    eigenValues, list_of_eigen_vectors = Ropelewski_Adam_(A, epsilon)
    print(
        f"Czas wykonania (wartości własne): {time.time() - current_time:.4f} sekundy.\n"
    )
    print("Uzyskane wektory własne:")
    for row in list_of_eigen_vectors:
        print(row)
    print("\nSprawdzenie:")
    for row in np.linalg.eig(A)[1].transpose():
        print(row)
