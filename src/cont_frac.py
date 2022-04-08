import math
import numpy as np
from typing import NamedTuple, Iterator, Tuple
from functools import reduce
from itertools import tee


# Rational(a, b) = a/b
class Rational(NamedTuple('Rational', [('a', int), ('b', int)])):
    def __repr__(self):
        return f'{self.a}/{self.b}'


def qr(a: int, b: int) -> Tuple[int, int]:
    """a = b * q + r"""
    q = math.floor(a / b)  # the quotion
    r = a - b * q  # the reminder
    return (q, r)


def euclid_(rn: Rational) -> Iterator[Tuple[int, int]]:
    a, b = rn
    while True:
        q, r = qr(a, b)
        yield b, q
        if r == 0:
            break
        a, b = b, r


def euclid(rn: Rational) -> Iterator[int]:
    def second(x: tuple):
        return x[1]

    return map(second, euclid_(rn))


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


def cf_convergent1_(cf: Iterator[int]) -> Iterator:
    res = np.array([[1, 0], [0, 1]])
    for a in cf:
        res = np.matmul(res, h(a))
        yield res


def cf_convergent1(cf: Iterator[int]) -> Iterator[Rational]:
    mLst = cf_convergent1_(cf)
    for m in mLst:
        yield Rational(m[0, 0], m[1, 0])


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
                if r[1] < 0:
                    # d0 also does'nt work
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


def cf_convergent2_(cf: Iterator[int], m0=np.identity(2, int)) -> Iterator:
    m = m0
    for a in cf:
        m = np.matmul(m, h(a))
        q = -1
        for (q, r) in euclid_matrix_(m):
            yield q, r, m
            m = r
        if q == -1:
            # for this coefficient a, the convergent cannot be turned into cont. frac. form
            yield (None, None, m)

    # we will only reach this point if the series is finite
    for s in euclid(Rational(m[0][0], m[0][1])):
        yield s, None, m


def cf_convergent2(cf: Iterator[int], m0=np.identity(2, int)) -> Iterator:
    for res in cf_convergent2_(cf, m0):
        if res:
            (q, r, m) = res
            yield q


def euclid_tab(rn: Rational):
    def row(st: str, x: tuple):
        b, q = x
        return st + f"{b : > 5}  {q : < 5}\n"

    str0 = f"{rn.a : > 5}\n"
    return reduce(row, euclid_(rn), str0) + f"{0 : > 5}\n"


def euclid_matrix_tab(m):
    def get_row(res):
        q, r = res
        return (r[0][0], r[0][1], r[1][0], r[1][1], q)

    def row(st, res):
        (n1, n2, _, _, q) = res
        s = ""
        s = s + f"{n1 : > 4}"
        s = s + f"{n2 : > 4}"
        if q:
            s = s + f"  {q}\n"
        else:
            s = s + "\n"
        return st + s

    st0 = f"{m[0][0] : > 4}{m[0][1] : > 4}\n"
    res = euclid_matrix_(m)
    res = list(map(get_row, res))
    st = reduce(row, res, st0)
    st = st + f"{res[-1][2] : > 4}{res[-1][3] : >4}\n"
    return st


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


def cf_convergent2_tab(cf: Iterator[int], m0=np.identity(2, int)):
    (cf1, cf2) = tee(cf)
    for (a, (q, r, m)) in zip(cf1, cf_convergent2_(cf2, m0)):
        if q is None:
            pass
