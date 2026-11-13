import math
import random
from typing import Tuple


class ExponentialFuzzyVariable:

    def __init__(self, lam: float, xi_p: float = None, normalized: bool = True, name: str = ""):
        if lam <= 0:
            raise ValueError("Параметр лямбда (lam) должен быть строго больше 0.")

        self.lam = float(lam)
        self.xi_p = float(xi_p) if xi_p is not None else random.uniform(0.05, 0.99)
        self.normalized = normalized
        self.name = name or f"FV(λ={self.lam:.3f})"

    def membership(self, x: float) -> float:
        if x < 0:
            return 0.0
        if self.normalized:
            return math.exp(-self.lam * x)
        else:
            return self.lam * math.exp(-self.lam * x)

    @property
    def x_star(self) -> float:
        return -(1.0 / self.lam) * math.log(self.xi_p)

    @property
    def mean_x(self) -> float:
        return 1.0 / self.lam

    def get_alpha_cut(self, alpha: float) -> Tuple[float, float]:
        if not (0 < alpha <= 1.0):
            raise ValueError("Уровень отсечения alpha должен быть в пределах (0; 1].")

        A = 0.0
        if self.normalized:
            B = -math.log(alpha) / self.lam
        else:
            if alpha >= self.lam:
                return (0.0, 0.0)
            B = -(1.0 / self.lam) * math.log(alpha / self.lam)

        return (round(A, 5), round(B, 5))

    def add(self, other: 'ExponentialFuzzyVariable', alpha: float = 0.7) -> Tuple[
        'ExponentialFuzzyVariable', Tuple[float, float]]:
        a1, b1 = self.get_alpha_cut(alpha)
        a2, b2 = other.get_alpha_cut(alpha)
        res_interval = (round(a1 + a2, 5), round(b1 + b2, 5))

        new_lam = (self.lam * other.lam) / (self.lam + other.lam)
        res_fv = ExponentialFuzzyVariable(new_lam, normalized=self.normalized, name=f"({self.name}+{other.name})")
        return res_fv, res_interval

    def subtract(self, other: 'ExponentialFuzzyVariable', alpha: float = 0.7) -> Tuple[
        'ExponentialFuzzyVariable', Tuple[float, float]]:
        a1, b1 = self.get_alpha_cut(alpha)
        a2, b2 = other.get_alpha_cut(alpha)
        res_A = max(0.0, round(a1 - b2, 5))
        res_B = max(1e-4, round(abs(b1 - b2), 5))
        res_interval = (res_A, res_B)

        new_lam = -math.log(alpha) / res_B if res_B > 0 else 1.0
        res_fv = ExponentialFuzzyVariable(max(0.01, new_lam), normalized=self.normalized,
                                          name=f"({self.name}-{other.name})")
        return res_fv, res_interval

    def multiply(self, other: 'ExponentialFuzzyVariable', alpha: float = 0.7) -> Tuple[
        'ExponentialFuzzyVariable', Tuple[float, float]]:
        a1, b1 = self.get_alpha_cut(alpha)
        a2, b2 = other.get_alpha_cut(alpha)
        res_A = round(a1 * a2, 5)
        res_B = max(1e-4, round(b1 * b2, 5))
        res_interval = (res_A, res_B)

        new_lam = -math.log(alpha) / res_B
        res_fv = ExponentialFuzzyVariable(max(0.01, new_lam), normalized=self.normalized,
                                          name=f"({self.name}*{other.name})")
        return res_fv, res_interval

    def divide(self, other: 'ExponentialFuzzyVariable', alpha: float = 0.7) -> Tuple[
        'ExponentialFuzzyVariable', Tuple[float, float]]:
        a1, b1 = self.get_alpha_cut(alpha)
        a2, b2 = other.get_alpha_cut(alpha)
        if b2 == 0:
            raise ZeroDivisionError("Деление на вырожденный нулевой интервал невозможно.")

        res_A = 0.0
        res_B = max(1e-4, round(b1 / b2, 5))
        res_interval = (res_A, res_B)

        new_lam = -math.log(alpha) / res_B
        res_fv = ExponentialFuzzyVariable(max(0.01, new_lam), normalized=self.normalized,
                                          name=f"({self.name}/{other.name})")
        return res_fv, res_interval

    def compare(self, other: 'ExponentialFuzzyVariable', alpha: float = 0.7) -> str:
        _, b1 = self.get_alpha_cut(alpha)
        _, b2 = other.get_alpha_cut(alpha)
        if math.isclose(b1, b2, rel_tol=1e-4):
            return f"{self.name} == {other.name} (Отрезки равны: [0; {b1:.4f}])"
        elif b1 > b2:
            return f"{self.name} > {other.name} (Отрезок [0; {b1:.4f}] шире [0; {b2:.4f}])"
        else:
            return f"{self.name} < {other.name} (Отрезок [0; {b1:.4f}] уже [0; {b2:.4f}])"

    def __repr__(self):
        return f"ExponentialFuzzyVariable(name='{self.name}', lam={self.lam:.4f}, x*={self.x_star:.4f})"