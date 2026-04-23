
import itertools


def f_or(n):
    return {
        x: (1.0 if any(xi == 1 for xi in x) else -1.0)
        for x in itertools.product([-1, 1], repeat=n)
    }

def f_or01(n: int) -> dict:
    return {
        x: int(any(x))
        for x in itertools.product([0, 1], repeat=n)
    }


def f_deutsch_jozsa(m: int):
    """
    x in {-1,1}^N, N = 2^m
    f(x) = +1 for constant
    f(x) = -1 for balanced

    Returns only promised inputs.
    """
    N = 2 ** m
    f_values = {}

    # constant inputs
    f_values[tuple([1] * N)] = 1.0
    f_values[tuple([-1] * N)] = 1.0

    # balanced inputs
    for plus_positions in itertools.combinations(range(N), N // 2):
        x = [-1] * N
        for i in plus_positions:
            x[i] = 1
        f_values[tuple(x)] = -1.0

    return f_values