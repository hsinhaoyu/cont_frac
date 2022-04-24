import math
import numpy as np
from typing import NamedTuple, Iterator, Tuple, Optional
from functools import reduce
from itertools import tee, islice


class Rational(NamedTuple('Rational', [('a', int), ('b', int)])):
    """Rational(a, b) = a/b"""
    def __repr__(self):
        return f'{self.a}/{self.b}'


# Convert a rational number to a continued fraction


def qr(a: int, b: int) -> Tuple[int, int]:
    """a = b * q + r, return (q, r)"""
    q = math.floor(a / b)  # the quotient
    r = a - b * q  # the remainder
    return (q, r)


def r2cf_(rn: Rational) -> Iterator[Tuple[int, int]]:
    """The Euclidean algorithm for representing a rational number as a continuous fraction.
    Return an iterator of quotients and remainders"""
    a, b = rn
    while True:
        q, r = qr(a, b)
        yield b, q
        if r == 0:
            break
        a, b = b, r


def r2cf(rn: Rational) -> Iterator[int]:
    """Represent a rational number as a continued fraction.
    Return an iterator of integers"""
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


# Calculate the convergents using matrix multiplication


def h(a: int) -> np.ndarray:
    return np.array([[a, 1], [1, 0]])


def cf_convergents1_(cf: Iterator[int]) -> Iterator[np.ndarray]:
    """Given a continuous fraction, return an iterator of 2x2 matrices representing convergents"""
    res = np.array([[1, 0], [0, 1]])
    for a in cf:
        res = np.matmul(res, h(a))
        yield res


def cf_convergents1(cf: Iterator[int]) -> Iterator[Rational]:
    """Given a continuous fraction, return an iterator of rational numbers representing convergents"""
    mLst = cf_convergents1_(cf)
    for m in mLst:
        yield Rational(m[0, 0], m[1, 0])


def cf2r1(cf: Iterator[int]) -> Rational:
    """Given a finite-term continued fraction, return its value as a rational number.
    This function will get into an infinite loop if the iterator doesn't stop.
    """
    return list(cf_convergents1(cf))[-1]


# Simple transformation of continued fraction

flip_m = np.array([[0, 1], [1, 0]])
identity_m = np.array([[1, 0], [0, 1]])


def flip_remain(m: np.ndarray, q: int):
    assert q >= 0
    r = m[0] - m[1] * q
    m[0] = r
    return np.matmul(flip_m, m)


def qr_matrix(m: np.ndarray) -> Tuple[int, np.ndarray]:
    """Calculate the quotient and the remainder of a 2x2 matrix"""

    assert not (m[1][0] == 0 and m[1][1] == 0)
    # this means that the series has already ended. Nothing further needs to be done
    # The caller should not call qr_matrix in this case

    m2 = m.copy()

    if m2[1][0] == 0 or m2[1][1] == 0:
        # If the function is unbounded, the quotient cannot be determined
        return (None, identity_m)
    elif m2[1][1] < 0:
        # This means that the denominator can be made 0 (i.e., a singularity)
        return (None, identity_m)
    else:
        # If the function is bounded...
        v0: float = m[0][0] / m[1][0]
        v1: float = m[0][1] / m[1][1]
        v0, v1 = sorted([v0, v1])
        d0: int = math.floor(v0)
        d1: int = math.floor(v1)
        if d0 == d1:
            # If d1 and d2 are the same, the quotient is determined
            # Calculate the remain, and flip the matrix
            m2 = flip_remain(m2, d1)
            return d1, m2
        elif d1 == d0 + 1:
            if d1 == v1:
                # This means that d1 is the upper-bound
                # it's only reached at 0 or infinity
                # So d0 is the quotient
                m2 = flip_remain(m2, d0)
                return d0, m2
            else:
                # The bounds are not tight enough to determine the quotient
                return (None, identity_m)
        else:
            # The bounds are not tight enough to determine the quotient
            return (None, identity_m)


def euclid_matrix_(m: np.ndarray) -> Iterator[Tuple[int, np.ndarray]]:
    """The Euclidean algorithm for the function express by matrix m.
    Returns an iterator of the quotient and the remainder"""
    while True:
        if m[0][0] == 0 and [0][1] == 0:
            # if there is no remain, stop
            break
        else:
            (q, r) = qr_matrix(m)
            if q is not None:
                yield q, r
                m = r
            else:
                # if the quotient cannot be determined, stop
                break


def cf_transform_(
    cf: Iterator[int], m0: np.ndarray = np.identity(2, int)
) -> Iterator[Tuple[Optional[int], Optional[np.ndarray], np.ndarray, int,
                    bool]]:
    """Transform the input continued fraction by matrix m
    returns another continued fraction"""
    m = m0
    for a in cf:
        m = np.matmul(m, h(a))
        new_a = True
        for (q, r) in euclid_matrix_(m):
            yield q, r, m, a, new_a
            new_a = False
            m = r
        if new_a:
            # Nothing was yielded. That means for this convergent cannot be turned into a continued fraction
            yield (None, None, m, a, new_a)

    # we will only reach this point if the series is finite
    if m[1][0] != 0:
        for s in r2cf(Rational(m[0][0], m[1][0])):
            yield s, None, m, a, False


def cf_transform(
    cf: Iterator[int], m0: np.ndarray = np.identity(2, int)) -> Iterator[int]:
    for res in cf_transform_(cf, m0):
        (q, r, m, a, new_a) = res
        if q is not None:
            yield q
            # q can be None, indicating that more coefficients are needed
            # to continue. It can be ignored


# Tensor representations of bihomographic functions


def tFrom2x4(m: np.ndarray) -> np.ndarray:
    ((a, b, c, d), (e, f, g, h)) = m.tolist()
    return np.array([[[b, d], [a, c]], [[f, h], [e, g]]])


def tTo2x4(m: np.ndarray) -> np.ndarray:
    (((b, d), (a, c)), ((f, h), (e, g))) = m.tolist()
    return np.array([[a, b, c, d], [e, f, g, h]])


def tensor_ref(t: np.ndarray, label: str):
    assert t.shape == (2, 2, 2)
    assert label in [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'xy', 'x', 'y', '1'
    ]
    lookup = {}
    lookup['a'] = t[0, 1, 0]
    lookup['b'] = t[0, 0, 0]
    lookup['c'] = t[0, 1, 1]
    lookup['d'] = t[0, 0, 1]
    lookup['e'] = t[1, 1, 0]
    lookup['f'] = t[1, 0, 0]
    lookup['g'] = t[1, 1, 1]
    lookup['h'] = t[1, 0, 1]
    lookup['xy'] = lookup['a'], lookup['e']
    lookup['x'] = lookup['b'], lookup['f']
    lookup['y'] = lookup['c'], lookup['g']
    lookup['1'] = lookup['d'], lookup['h']
    return lookup[label]


tForAddition = np.array([[[1, 0], [0, 1]], [[0, 1], [0, 0]]])
tForSubtraction = np.array([[[1, 0], [0, -1]], [[0, 1], [0, 0]]])
tForMultiplication = np.array([[[0, 0], [1, 0]], [[0, 1], [0, 0]]])
tForDivision = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]])


def apply_a(t: np.ndarray, a: int):
    ha = h(a)
    return np.einsum('dyx,xz->dyz', t, ha)


def h_rotated(b: int) -> np.ndarray:
    return np.array([[0, 1], [1, b]])


def apply_b(t: np.ndarray, b: int):
    hb = h_rotated(b)
    return np.einsum('zy,dyx->dzx', hb, t)


def arithmetic_convergents_(a: Iterator[int],
                            b: Iterator[int],
                            t0=tForAddition) -> Iterator[np.ndarray]:
    res = t0.copy()
    while True:
        an = next(a, None)
        bn = next(b, None)

        if an is None and bn is None:
            break

        if an is not None:
            res = apply_a(res, an)
            yield 'a', an, res
        if bn is not None:
            res = apply_b(res, bn)
            yield 'b', bn, res


def arithmetic_convergents(a: Iterator[int],
                           b: Iterator[int],
                           t0=tForAddition) -> Iterator[Rational]:
    c = arithmetic_convergents_(a, b, t0)
    next(c)  # skip the initial tensor
    for _, _, res in c:
        r = tensor_ref(res, 'xy')
        yield Rational(*r)


def tensor_term_ratios(t: np.ndarray) -> list:
    """ return
        [[a/e, floor(a/e), 'xy'],
         [b/f, floor(b/f), 'x'],
         [c/g, floor(c/g), 'y'],
         [d/h, floor(d/h), '1']]"""
    def f(label):
        numerator, denominator = tensor_ref(t, label)
        return numerator / denominator, math.floor(numerator /
                                                   denominator), label

    return [f('xy'), f('x'), f('y'), f('1')]


def qr_tensor(t: np.ndarray):
    if (t[1] <= 0).any():
        # if any number in the denominator is smaller or equal to 0, the function is unbounded
        pass
    else:
        r = tensor_term_ratios(t)
        r_sorted = sorted(r, key=lambda terms: terms[1], reverse=True)

        if r_sorted[0, 0] == r_sorted[0, 1]:
            # the max can only be reached at 0 or infinity, so the quotient needed -1
            r_sorted[0, 1] = r_sorted[0, 1] - 1

        if r_sorted[0, 1] == r_sorted[-1, 1]:
            #  If the floor of the first and the last are the same, the numbers are all the same
            # in this case, the value is the quotient
            pass
        else:
            # the quotient cannot be determined, but suggest which direction to go
            # if one of them is different, go to the dir that is different
            # if all three are different, what to do? compare magnitude?
            pass


# Examples of continuous fractions


def cf_e() -> Iterator[int]:
    """e as a continuous fraction"""
    yield 2
    k = 0
    while True:
        # a finite generator comprehension
        for i in (j for j in [1, 2 * k + 2, 1]):
            yield i
        k = k + 1


def cf_sqrt2():
    """A generator representing sqrt(2) as a continued fraction"""
    yield 1
    while True:
        yield 2
