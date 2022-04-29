import math
import numpy as np
from typing import NamedTuple, Iterator, Tuple, Optional, Callable, Union
from functools import reduce
from itertools import tee, islice


class Rational(NamedTuple('Rational', [('a', int), ('b', int)])):
    """Rational(a, b) = a/b"""
    def __repr__(self):
        return f'{self.a}/{self.b}'


# Convert a rational number to a continued fraction


def qr(a: int, b: int) -> Tuple[int, int]:
    """a = b * q + r, return (q, r)
    a is the numberator, and b is the denominator of a rational number"""
    q = math.floor(a / b)  # the quotient
    r = a - b * q  # the remainder
    return (q, r)


def r2cf_(rn: Rational) -> Iterator[Tuple[int, int]]:
    """The Euclidean algorithm for representing a rational number as a continued fraction.
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
    """Given a continued fraction, return an iterator of 2x2 matrices representing convergents"""
    res = np.array([[1, 0], [0, 1]])
    for a in cf:
        res = np.matmul(res, h(a))
        yield res


def cf_convergents1(cf: Iterator[int]) -> Iterator[Rational]:
    """Given a continued fraction, return an iterator of rational numbers representing convergents"""
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


def qr_matrix(m: np.ndarray) -> Tuple[Optional[int], np.ndarray]:
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
    cf: Iterator[int],
    m0: np.ndarray = np.identity(2, int),
    finite_term=True
) -> Iterator[Tuple[Optional[int], Optional[np.ndarray], np.ndarray, int,
                    bool]]:
    """Transform the input continued fraction, using the initial matrix m0.
    Returns a tuple of quotients, and internal states
    """
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

    # We will only reach this point if the series is finite
    # If cf has finite term, but it represents the beginning of a longer series, set finite_term to False
    if finite_term and m[1][0] != 0:
        for s in r2cf(Rational(m[0][0], m[1][0])):
            yield s, None, m, a, False


def cf_transform(cf: Iterator[int],
                 m0: np.ndarray = np.identity(2, int),
                 finite_term=True) -> Iterator[Union[int, np.ndarray]]:
    """Transform the input continued fraction, using the initial matrix m0.
    Returns another continued fraction"""
    for res in cf_transform_(cf, m0, finite_term):
        (q, r, m, a, new_a) = res
        if q is not None:
            yield q
        else:
            pass


            # q can be None, indicating that more terms are needed
            # to continue. It can be ignored
def cf_transform_func(cf, m0):
    outputs = []
    out_m = None
    for res in cf_transform_(cf, m0, finite_term=False):
        (q, r, m, a, new_a) = res
        if q is not None:
            outputs = outputs + [q]
            out_m = r
        else:
            out_m = m
    return outputs, out_m


# Tensor representations of bihomographic functions


def tFrom2x4(m: np.ndarray) -> np.ndarray:
    ((a, b, c, d), (e, f, g, h)) = m.tolist()
    return np.array([[[b, d], [a, c]], [[f, h], [e, g]]])


def tTo2x4(m: np.ndarray) -> np.ndarray:
    (((b, d), (a, c)), ((f, h), (e, g))) = m.tolist()
    return np.array([[a, b, c, d], [e, f, g, h]])


def tensor_ref(t: np.ndarray, label: str) -> Union[int, Tuple[int, int]]:
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


def apply_a(t: np.ndarray, a: int) -> np.ndarray:
    ha = h(a)
    return np.einsum('dyx,xz->dyz', t, ha)


def h_rotated(b: int) -> np.ndarray:
    return np.array([[0, 1], [1, b]])


def apply_b(t: np.ndarray, b: int) -> np.ndarray:
    hb = h_rotated(b)
    return np.einsum('zy,dyx->dzx', hb, t)


def apply_ab(t: np.ndarray, term: int, label: str) -> np.ndarray:
    assert label in ['a', 'b']
    if label == 'a':
        return apply_a(t, term)
    else:
        return apply_b(t, term)


tForAddition = np.array([[[1, 0], [0, 1]], [[0, 1], [0, 0]]])
tForSubtraction = np.array([[[1, 0], [0, -1]], [[0, 1], [0, 0]]])
tForMultiplication = np.array([[[0, 0], [1, 0]], [[0, 1], [0, 0]]])
tForDivision = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]])


def arithmetic_convergents_(a: Iterator[int],
                            b: Iterator[int],
                            t0=tForAddition) -> Iterator[np.ndarray]:
    """Given two continued fractions (a and b) and an arithmetic operation (specified by t0), return an iterator of tensors and addional information.
       New terms of the two continued fractions are applied alternately"""
    res = t0.copy()
    while True:
        an = next(a, None)
        bn = next(b, None)

        if an is None and bn is None:
            break

        if an is not None:
            res = apply_ab(res, an, 'a')
            yield 'a', an, res
        if bn is not None:
            res = apply_ab(res, bn, 'b')
            yield 'b', bn, res


def arithmetic_convergents(a: Iterator[int],
                           b: Iterator[int],
                           t0=tForAddition) -> Iterator[Rational]:
    """Given two continued fractions (a and b) and an arithmetic operation (specified by t0), return an interator of rational numbers"""
    c = arithmetic_convergents_(a, b, t0)
    for _, _, res in c:
        r = tensor_ref(res, 'xy')
        yield Rational(*r)


def qr_tensor(t: np.ndarray) -> Tuple[Optional[int], np.ndarray]:
    t1 = t.copy()
    if np.all(t1[1] > 0):
        # if the denominator matrix doesn't have any 0 or negative number
        r = t_ratios(t)
        if r[0][0] == r[0][1] == r[1][0] == r[1][1]:
            # if the integer parts are all the same, we've got a quotient
            q = r[0][0]
            r = np.array([t1[1], t1[0] - q * t1[1]])
            return (q, r)
        else:
            # the range is too big to determine the quotient
            return (None, np.array([identity_m, identity_m]))
    else:
        # the denominator can be zero. The dihomographic function is unbounded
        return (None, np.array([identity_m, identity_m]))


def t_ratios(t: np.ndarray) -> list:
    def r(label):
        numerator, denominator = tensor_ref(t, label)
        return [
            numerator / denominator,
            math.floor(numerator / denominator), label
        ]

    zz = [r('xy'), r('x'), r('y'), r('1')]
    # sort by the floating point ratio
    zz_sorted = sorted(zz, key=lambda item: item[0])

    zz_max = zz_sorted[-1]
    if zz_max[0] == zz_max[1]:
        # In this situation, the upper-bound zz_max[0] will never be reached
        # so we return the value - 1
        zz_max[1] = zz_max[1] - 1

    dict = {}
    for item in zz_sorted:
        dict[item[2]] = item[1]

    return [[dict['x'], dict['1']], [dict['xy'], dict['y']]]


def score(t):
    def r(label):
        numerator, denominator = tensor_ref(t, label)
        return math.floor(numerator / denominator)

    # number of zero in the denominator
    # a negative value means that the denominator can be made zero
    n_zero = np.count_nonzero(t[1] <= 0)
    if n_zero == 3 or n_zero == 4:
        # 0 can be removed in 2 moves
        return -2
    elif n_zero == 2:
        if tensor_ref(t, 'f') == 0 and tensor_ref(t, 'h') == 0:
            if r('xy') == r('y'):
                # 0 0
                # 2 2
                # after one step, in a good position
                return -0.5
            else:
                # 0 0
                # 2 3
                # both zeros can be removed in one step
                return -1.0
        elif tensor_ref(t, 'h') == 0 and tensor_ref(t, 'g') == 0:
            if r('xy') == r('x'):
                # 2 0
                # 2 0
                # in good position in one step
                return -0.5
            else:
                # 2 0
                # 3 0
                # both zeroes can be remoevd in one step
                return -1
        else:
            return -2
    elif n_zero == 1:
        if tensor_ref(t, 'e') == 0:
            # 3 8
            # 0 1
            # takes 2 moves to remove 0
            return -2.0
        elif tensor_ref(t, 'f') == 0:
            if r('xy') == r('y'):
                # 0 x
                # 2 2
                # in good position in one step
                return -0.5
            else:
                # 0 2
                # 3 4
                return -1.0
        elif tensor_ref(t, 'g') == 0:
            if (r('xy') == r('x')):
                # 2 3
                # 2 0
                return -0.5
            else:
                # 2 3
                # 3 0
                return -1.0
        else:
            if (r('xy') == r('x')) or (r('xy') == r('y')):
                # 2 0
                # 2 3
                return -0.5
            else:
                return -1.0
    else:  # no zereos in the denominator
        r = t_ratios(t)
        if r[0][0] == r[0][1] == r[1][0] == r[1][1]:
            # the 4 ratios are all the same. This is the best situation
            return 4.0
        elif (r[0][0] == r[1][0]) or (r[1][0] == r[1][1]):
            # 3 1       1 2
            # 3 2 or    3 3
            return 1.0
        else:
            return 0.0


def ABQueue(a: Iterator[int],
            b: Iterator[int]) -> Callable[np.ndarray, Tuple[int, str]]:
    current_a = None
    current_b = None
    last_tie = 'b'

    def ABQueue_(t: np.ndarray) -> Tuple[int, str]:
        nonlocal current_a
        nonlocal current_b
        nonlocal last_tie

        def dequeue(label: str) -> Tuple[int, str]:
            nonlocal current_a
            nonlocal current_b
            nonlocal last_tie
            assert label in ['a', 'b', 'alt']
            if label == 'a':
                next_term = current_a
                current_a = None
                return next_term, 'a'
            elif label == 'b':
                next_term = current_b
                current_b = None
                return next_term, 'b'
            elif label == 'alt':
                if last_tie == 'a':
                    next_term = current_b
                    current_b = None
                    last_tie = 'b'
                    return next_term, 'b'
                else:
                    next_term = current_a
                    current_a = None
                    last_tie = 'a'
                    return next_term, 'a'

        if current_a is None:
            current_a = next(a, None)
        if current_b is None:
            current_b = next(b, None)

        if current_a is None and current_b is None:
            # both a and b are empty
            return None, None
        elif current_a is None:
            # if a is empty, return b
            return dequeue('b')
        elif current_b is None:
            # if b is empty, return a
            return dequeue('a')
        else:
            #print(current_a, current_b)
            t_a = apply_ab(t, current_a, 'a')
            t_b = apply_ab(t, current_b, 'b')
            s_a, s_b = score(t_a), score(t_b)
            #print(t_a)
            #print(s_a, s_b)
            if s_a == s_b:
                return dequeue('alt')
            if s_a < s_b:
                return dequeue('b')
            else:
                return dequeue('a')

    return ABQueue_


def euclid_tensor_(t: np.ndarray) -> Iterator[Tuple[int, np.ndarray]]:
    while True:
        if np.all(t[0] == 0):
            # if the numerator is all zero, stop
            break
        else:
            (q, r) = qr_tensor(t)
            if q is not None:
                yield q, r
                t = r
            else:
                break


def cf_arithmetic_(cf_a: Iterator[int],
                   cf_b: Iterator[int],
                   t0: np.ndarray,
                   finite_term=True) -> Iterator:
    """Given two continued fraction, cf_a and cf_b, and an initial tensor specifying the operation, return the result as a continued fraction.
    Returns an iterator of quotients and other information.
    """
    t = t0
    next_ab = ABQueue(cf_a, cf_b)

    while True:
        term, label = next_ab(t)
        if term is None and label is None:
            # cf_a and cf_b are exhausted
            break
        else:
            t = apply_ab(t, term, label)
            new_term = True
            for (q, r) in euclid_tensor_(t):
                yield q, r, t, term, label, new_term
                t = r
                new_term = False
            if new_term:
                # Nothing was yielded. This means that an Euclidean step was not performed
                yield None, None, t, term, label, new_term

    # we will only reach this point if cf_a and cf_b have finite terms
    # If cf has finite term, but it represents the beginning of a longer series, set finite_term to False
    if finite_term and tensor_ref(t, 'e') != 0:
        for s in r2cf(Rational(*tensor_ref(t, 'xy'))):
            yield s, None, t, None, None, False
    else:
        # if the 'e' term is 0, that means the quotient is 0.
        # there is no need to return it
        pass


def cf_arithmetic(cf_a: Iterator[int],
                  cf_b: Iterator[int],
                  t0: np.ndarray,
                  finite_term=True) -> Iterator[int]:
    """Given two continued fraction, cf_a and cf_b, and an initial tensor specifying the operation, return the result as a continued fraction.
    """
    for res in cf_arithmetic_(cf_a, cf_b, t0, finite_term=finite_term):
        (q, r, t, term, label, new_term) = res
        if q is not None:
            yield q


def cf_arithmetic_func(cf_a, cf_b, t0):
    outputs = []
    out_t = None
    for res in cf_arithmetic_(cf_a, cf_b, t0, finite_term=False):
        q, r, t, term, label, new_term = res
        if q is not None:
            outputs = outputs + [q]
            out_t = r
        else:
            out_t = t
    return outputs, out_t


# Examples of continued fractions


def cf_e() -> Iterator[int]:
    """e as a continued fraction"""
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


def cf_sqrt6():
    """sqrt(6) = [2, 2, 4, 2, 4, 2...]"""
    yield 2
    yield 2
    while True:
        yield 4
        yield 2


def cf_coth1():
    """(e^2+1)/(e^2-1) = [1, 3, 5, 7...]"""
    s = 1
    while True:
        yield s
        s = s + 2


def cf_pi():
    return iter(
        [3, 7, 15, 1, 292, 1, 1, 1, 2, 1, 3, 1, 14, 2, 1, 1, 2, 2, 2, 2])
