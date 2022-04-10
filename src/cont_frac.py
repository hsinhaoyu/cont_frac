import math
import numpy as np
from typing import NamedTuple, Iterator, Tuple
from functools import reduce
from itertools import tee


# Rational(a, b) = a/b
class Rational(NamedTuple('Rational', [('a', int), ('b', int)])):
    def __repr__(self):
        return f'{self.a}/{self.b}'


# Convert a rational number to a continued fraction


def qr(a: int, b: int) -> Tuple[int, int]:
    """a = b * q + r, return (q, r)"""
    q = math.floor(a / b)  # the quotient
    r = a - b * q  # the reminder
    return (q, r)


def r2cf_(rn: Rational) -> Iterator[Tuple[int, int]]:
    a, b = rn
    while True:
        q, r = qr(a, b)
        yield b, q
        if r == 0:
            break
        a, b = b, r


def r2cf(rn: Rational) -> Iterator[int]:
    def second(x: tuple):
        return x[1]

    return map(second, r2cf_(rn))


# Calculate the convergents of a continued fraction


def cf_convergents0(cf: Iterator[int]) -> Iterator[Rational]:
    """For a continued fraction cf, return an iterator of rational numbers to approximate it"""
    p1, p0 = 1, 0
    q1, q0 = 0, 1

    for a in cf:
        p = a * p1 + p0
        q = a * q1 + q0
        yield Rational(p, q)

        p0, p1 = p1, p
        q0, q1 = q1, q


def cf2r0(cf: Iterator[int]) -> Rational:
    """Given a finite-term continued fraction, return its value as a rational number.
    This function will get into an infinite loop if the iterator doesn't stop.
    """
    return list(cf_convergents0(cf))[-1]


def h(a):
    return np.array([[a, 1], [1, 0]])


def cf_convergents1_(cf: Iterator[int]) -> Iterator:
    res = np.array([[1, 0], [0, 1]])
    for a in cf:
        res = np.matmul(res, h(a))
        yield res


def cf_convergents1(cf: Iterator[int]) -> Iterator[Rational]:
    mLst = cf_convergents1_(cf)
    for m in mLst:
        yield Rational(m[0, 0], m[1, 0])


def cf2r1(cf: Iterator[int]) -> Rational:
    """Given a finite-term continued fraction, return its value as a rational number.
    This function will get into an infinite loop if the iterator doesn't stop.
    """
    return list(cf_convergents1(cf))[-1]


# Calculate the quotient and reminder of a 2-by-2 matrix

flip_m = np.array([[0, 1], [1, 0]])


def qr_matrix(m):
    m2 = m.copy()

    if m2[1][0] != 0 and m2[1][1] != 0:
        d0 = math.floor(m[0][0] / m[1][0])
        d1 = math.floor(m[0][1] / m[1][1])

        # the quotient is between d0 and d1 (inclusive)
        (d0, d1) = sorted([d0, d1])
        if d0 == d1:
            # if d1 and d2 are the same, the coefficient is determined
            # calculate the remain, and flip the matrix
            r = m2[0] - m2[1] * d0
            m2[0] = r
            m2 = np.matmul(flip_m, m2)
            return d1, m2
        elif d1 == d0 + 1:
            # if d1 is d0 + 1, there is a situation where coefficient can be determined
            r = m2[0] - m2[1] * d1
            if r[1] < 0:
                # this means d1 doesn't work, try d0
                r = m2[0] - m2[1] * d0
                if d0 < 0:
                    return None
                else:
                    m2[0] = r
                    m2 = np.matmul(flip_m, m2)
                    return d0, m2
            else:
                # cannot rule out d1
                return None
        else:
            # the range is too big, so the coefficient cannot be determined
            return None
    else:
        # coefficient cannot be determined for unbounded function
        return None


def euclid_matrix_(m):
    while True:
        res = qr_matrix(m)
        if res:
            q, r = res
            yield q, r
            m = r
        else:
            break


def cf_transform_(cf: Iterator[int], m0=np.identity(2, int)) -> Iterator:
    m = m0
    for a in cf:
        m = np.matmul(m, h(a))
        q = -1
        for (q, r) in euclid_matrix_(m):
            yield q, r, m
            m = r
        if q == -1:
            # for this coefficient a, the convergent cannot be turned into a continued fraction
            yield (None, None, m)

    # we will only reach this point if the series is finite
    for s in r2cf(Rational(m[0][0], m[1][0])):
        yield s, None, m


def cf_transform(cf: Iterator[int], m0=np.identity(2, int)) -> Iterator:
    for res in cf_transform_(cf, m0):
        if res:
            (q, r, m) = res
            if q is not None:
                # cf_transform_ can return None to indicate that it needs more coefficients
                # to continue. It can be ignored
                yield q


def cf_e():
    yield 2
    k = 0
    while True:
        # a finite generator comprehension
        for i in (j for j in [1, 2 * k + 2, 1]):
            yield i
        k = k + 1
