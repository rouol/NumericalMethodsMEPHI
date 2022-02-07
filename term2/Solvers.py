import numpy as np
from numba import jit
from typing import Callable


def NumericalDerivative(f: Callable[[float], float], h: float = 0.01) -> Callable[[float], float]:
    return lambda x: (f(x + h) - f(x - h)) / (2 * h)


def CompositeNumericalDerivative(f: Callable[[float], float], h: float = 0.01, order: int = 1) -> Callable[[float], float]:
    if order == 0:
        return f
    else:
        df = f
        for _ in range(order):
            df = NumericalDerivative(df, h)
        return df
