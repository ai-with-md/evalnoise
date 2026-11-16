"""Small numeric helpers: the t distribution, without scipy."""

from __future__ import annotations

import math


def _beta_fraction(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    result = d
    for m in range(1, 500):
        m2 = 2 * m
        for numerator in (
            m * (b - m) * x / ((a + m2 - 1.0) * (a + m2)),
            -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0)),
        ):
            d = 1.0 + numerator * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + numerator / c
            c = c if abs(c) > tiny else tiny
            result *= d * c
        if abs(d * c - 1.0) < 1e-14:
            break
    return result


def incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _beta_fraction(a, b, x) / a
    return 1.0 - math.exp(log_front) * _beta_fraction(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    """Chance that a t variable with ``df`` degrees of freedom is at most ``t``.

    >>> round(t_cdf(2.228, 10), 3)
    0.975
    """
    tail = 0.5 * incomplete_beta(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


def t_quantile(p: float, df: float) -> float:
    """The value a t variable with ``df`` degrees of freedom stays below with chance ``p``.

    >>> round(t_quantile(0.975, 1), 2), round(t_quantile(0.975, 10), 3), round(t_quantile(0.975, 1e9), 3)
    (12.71, 2.228, 1.96)
    """
    if not 0 < p < 1:
        raise ValueError("p must be between 0 and 1")
    if p < 0.5:
        return -t_quantile(1 - p, df)
    low, high = 0.0, 1.0
    while t_cdf(high, df) < p:
        high *= 2
    for _ in range(200):
        mid = (low + high) / 2
        if t_cdf(mid, df) < p:
            low = mid
        else:
            high = mid
        if high - low < 1e-10:
            break
    return (low + high) / 2
