import math
import numpy as np
from typing import NamedTuple, Iterator, Tuple
from functools import reduce
from itertools import tee


# Rational(a, b) = a/b
class Rational(NamedTuple('Rational', [('a', int), ('b', int)])):

    def __repr__(self):
        return f'{self.a}/{self.b}'


def euclid(a: int, b: int) -> Tuple[int, int]:
    """a = b * q + r"""
    q = math.floor(a / b)
    r = a - b * q
    return (q, r)


def r2cf_(rn: Rational) -> Iterator[Tuple[int, int]]:
    a, b = rn
    r = 1000
    while r != 0:
        q, r = euclid(a, b)
        yield b, q
        a = b
        b = r


def r2cf(rn: Rational) -> Iterator[int]:

    def second(x: tuple):
        return x[1]

    return map(second, r2cf_(rn))


def cf_convergent0(cf: Iterator[int]) -> Iterator[Rational]:
    p1, p0 = 1, 0
    q1, q0 = 0, 1

    for a in cf:
        p = a * p1 + p0
        q = a * q1 + q0
        yield Rational(p, q)

        p0, p1 = p1, p
        q0, q1 = q1, q


def h(a):
    return np.array([[a, 1], [1, 0]])


def cf_convergent1_(cdf: Iterator[int]) -> Iterator:
    res = np.array([[1, 0], [0, 1]])
    for a in cdf:
        res = np.matmul(res, h(a))
        yield res


def cf_convergent1(cdf: Iterator[int]) -> Iterator[Rational]:
    mLst = cf_convergent1_(cdf)
    for m in mLst:
        yield Rational(m[0, 0], m[1, 0])


def r2cf_tab(rn: Rational):

    def row(st: str, x: tuple):
        b, q = x
        return st + f"{b : > 5}  {q : < 5}\n"

    str0 = f"{rn.a : > 5}\n"
    return reduce(row, r2cf_(rn), str0) + f"{0 : > 5}\n"


def cf_convergent1_tab_pp(row1, row2, row3):

    def f_(item):
        if item is None:
            return f"{'': >4}"
        else:
            return f"{item: >4}"

    print(reduce(lambda a, b: a + b, map(f_, row1), ""))
    print(reduce(lambda a, b: a + b, map(f_, row2), ""))
    print(reduce(lambda a, b: a + b, map(f_, row3), ""))


def cf_convergent1_tab(cf: Iterator[int]):
    row1, row2, row3 = [], [], []
    row2 = [0, 1]
    row3 = [1, 0]

    (cf1, cf2) = tee(cf)
    for (mat, a) in zip(cf_convergent1_(cf1), cf2):
        row1.append(a)
        row2.append(mat[0][0])
        row3.append(mat[1][0])

    row1.reverse()
    row2.reverse()
    row3.reverse()

    row1 = [None] + row1
    cf_convergent1_tab_pp(row1, row2, row3)
