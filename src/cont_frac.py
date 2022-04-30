import math
import numpy as np
from typing import NamedTuple, Iterator, Tuple, Optional, Callable, Union, List


class Rational(NamedTuple('Rational', [('a', int), ('b', int)])):
    """Rational(a, b) = a/b"""

    def __repr__(self):
        return f'{self.a}/{self.b}'


# Convert a rational number to a continued fraction


def qr(a: int, b: int) -> Tuple[int, int]:
    """
    Find the quotient and remainder of a rational number.

    a = b * q + r, return (q, r).
    :param a: The numerator of the rational number
    :param b: The denominator of the rational number
    :return: (quotient, remainder)
    """
    q = math.floor(a / b)  # the quotient
    r = a - b * q  # the remainder
    return (q, r)


def r2cf_(rn: Rational) -> Iterator[Tuple[int, int]]:
    """
    Turn a rational number into a continued fraction.

    The Euclidean algoirthm.
    :param rn: The rational number
    :return: An iterator of the old denominator and the quotient
    """
    a, b = rn
    while True:
        q, r = qr(a, b)
        yield b, q
        if r == 0:
            break
        a, b = b, r


def r2cf(rn: Rational) -> Iterator[int]:
    """
    Turn a rational number to a continued fraction.

    :param rn: The rational number
    :return: An iterator of integers
    """

    def second(x: tuple):
        return x[1]

    return map(second, r2cf_(rn))


# Calculate the convergents of a continued fraction


def cf_convergents0(cf: Iterator[int]) -> Iterator[Rational]:
    """
    Calculate the convergents of a continued fraction.

    :param cf: A continued fraction
    :return: An iterator of rational numbers
    """
    p1, p0 = 1, 0
    q1, q0 = 0, 1

    for a in cf:
        p = a * p1 + p0
        q = a * q1 + q0
        yield Rational(p, q)

        p0, p1 = p1, p
        q0, q1 = q1, q


def cf2r0(cf: Iterator[int]) -> Rational:
    """
    The value of a finite-length continued fraction as a rational number.

    This function will get into an infinite loop if the iterator doesn't stop.
    """
    return list(cf_convergents0(cf))[-1]


# Calculate the convergents using matrix multiplication


def h(a: int) -> np.ndarray:
    '''Homographic matrix for one term of continued fraction'''
    return np.array([[a, 1], [1, 0]])


def cf_convergents1_(cf: Iterator[int]) -> Iterator[np.ndarray]:
    """
    The convergents (as matrices) of a continued fraction.

    :param cf: The continued fraction
    :return: An iterator of 2x2 matrices representing the convergents
    """
    res = np.array([[1, 0], [0, 1]])
    for a in cf:
        res = np.matmul(res, h(a))
        yield res


def cf_convergents1(cf: Iterator[int]) -> Iterator[Rational]:
    """
    The convergents (as rational numbers ) of a continued fraction

    :param cf: The continued fraction
    :return: An iterator of rational numbers
    """
    mLst = cf_convergents1_(cf)
    for m in mLst:
        yield Rational(m[0, 0], m[1, 0])


def cf2r1(cf: Iterator[int]) -> Rational:
    """
    Turn a continued fraction into a rational number.

    The continued fraction must have finite length.
    If not, this function will get into an infinite loop.

    :param cf: The continued fraction
    :return: A rational number
    """
    return list(cf_convergents1(cf))[-1]


# Simple transformation of continued fraction

flip_m = np.array([[0, 1], [1, 0]])
identity_m = np.array([[1, 0], [0, 1]])


def flip_remain(m: np.ndarray, q: int) -> np.ndarray:
    """
    Caclulate the remainder, and then flip

    :param m: A 2x2 matrix
    :param q: The quotient
    :return: A 2x2 matrix
    """
    assert q >= 0
    r = m[0] - m[1] * q
    m[0] = r
    return np.matmul(flip_m, m)


def qr_matrix(m: np.ndarray) -> Tuple[Optional[int], np.ndarray]:
    """
    Calculate the quotient and the remainder of a 2x2 matrix

    :param m: A 2x2 matrix
    :return: The quotient (None if it cannot be found)
             The remainder (2x2 matrix; idetity if quotient cannot be found)
    """

    assert not (m[1][0] == 0 and m[1][1] == 0)
    # This means that the series has already ended.
    # Nothing further needs to be done
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
    """
    Symbolic Euclidean algorithm for a homographic function.

    :param m: The 2x2 matrix representing the function.
    :return: An iterator of the quotient and the remainder.
    """
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
    """
    Transform a continued fraction.

    Exposes the internal states of the algorithm for visualization.
    A step can do: 1. Update the homographic matrix
                   2. Update, and an Euclidean step
                   3. Only an Euclidean step

    :param cf: The continued fraction.
    :param m0: The initial 2x2 matrix representing the transformatiom.
    :param finite-term: True, if cf is a continued fraction with finite terms.
                        False, if cf represents a truncated continued fraction.
    :return: q: The quotient. None if the Euclidean step cannot be performed.
             r: The remainder. None if the Euclidean step cannot be performed.
             m: The 2x2 matrix before the Euclidean step
             a: The term of the continued fraction that was used in this update
             new_a: Was a new a term used?
                    False if this step only did Euclidean
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
            # Nothing was yielded.
            # This convergent cannot be turned into a continued fraction
            yield (None, None, m, a, new_a)

    # We will only reach this point if the series is finite
    if finite_term and m[1][0] != 0:
        for s in r2cf(Rational(m[0][0], m[1][0])):
            yield s, None, m, a, False


def cf_transform(cf: Iterator[int],
                 m0: np.ndarray = np.identity(2, int),
                 finite_term=True) -> Iterator[int]:
    """
    Transform the input continued fraction into a new continued fraction.

    :param cf: The continued fraction.
    :param m0: The 2x2 matrix representing the transformation.
    :param finite_term: Is cf a finite-term fraction?
                        Set to false if cf is a finite truncation of
                        an infinite continued fractuon
    :return: A new continued fraction
    """
    for res in cf_transform_(cf, m0, finite_term):
        (q, r, m, a, new_a) = res
        if q is not None:
            yield q
        else:
            # q can be None, indicating that more terms are needed
            # to continue. It can be ignored
            pass


def cf_transform_func(cf: Iterator[int],
                      m0: np.ndarray) -> Tuple[List[int], np.ndarray]:
    """
    Transform a continued continuation with finite number of terms.

    Returns a list (representing the transformed fraction) and the
    last homographic function so that it can be used on more terms.

    If cf is infinite, this function won't terminate.

    :param cf: The input continuned fraction.
    :param m0: A 2x2 matrix representing the transformation
    :return: A list representing the transformed fraction, and
             a 2x2 matrix representing the homographic function
    """
    outputs: List[int] = []
    out_m: Optional[np.ndarray] = None
    for res in cf_transform_(cf, m0, finite_term=False):
        (q, r, m, a, new_a) = res
        if q is not None:
            outputs = outputs + [q]
            out_m = r
        else:
            out_m = m
    assert isinstance(out_m, np.ndarray)
    return outputs, out_m


# Tensor representations of bihomographic functions


def tFrom2x4(m: np.ndarray) -> np.ndarray:
    """
    Translate a bihomographic function from algebraic to tensor form.

    :param m: A 2x4 matrix representing the function
    :return: A 2x2x2 tensor
    """
    ((a, b, c, d), (e, f, g, h)) = m.tolist()
    return np.array([[[b, d], [a, c]], [[f, h], [e, g]]])


def tTo2x4(m: np.ndarray) -> np.ndarray:
    """
    Translate a bihomographic function from tensor to algebraic form.

    :param m: A 2x2x2 tensor
    :return: A 2x4 matrix
    """

    (((b, d), (a, c)), ((f, h), (e, g))) = m.tolist()
    return np.array([[a, b, c, d], [e, f, g, h]])


def tensor_ref(t: np.ndarray, label: str) -> Union[int, Tuple[int, int]]:
    """
    Easy accessing elements of a tensor

    :param t: A 2x2x2 tensor
    :param label: a-h refers to a number in the tensor
                  xy, x, y, 1 refers to numerator/denominator pairs
    :return: A number, or a pair of numbers
    """
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
    '''Move tensor t to the left by taking a new term of a'''
    ha = h(a)
    return np.einsum('dyx,xz->dyz', t, ha)


def h_rotated(b: int) -> np.ndarray:
    '''A rotated version of the 2x2 homgraphic matrix'''
    return np.array([[0, 1], [1, b]])


def apply_b(t: np.ndarray, b: int) -> np.ndarray:
    '''Move tensor t downwards by taking a new term of b'''
    hb = h_rotated(b)
    return np.einsum('zy,dyx->dzx', hb, t)


def apply_ab(t: np.ndarray, term: int, label: str) -> np.ndarray:
    """
    Apply a new term to the bihomographic tensor

    :param t: The tensor
    :param term: The new term to be applied to the tensor
    :param label: Is the term from continued fraction a or b?
    :return: A new 2x2x2 tensor
    """
    assert label in ['a', 'b']
    if label == 'a':
        return apply_a(t, term)
    else:
        return apply_b(t, term)


# Initial tensors for basic arithemtic operations

tForAddition = np.array([[[1, 0], [0, 1]], [[0, 1], [0, 0]]])
tForSubtraction = np.array([[[1, 0], [0, -1]], [[0, 1], [0, 0]]])
tForMultiplication = np.array([[[0, 0], [1, 0]], [[0, 1], [0, 0]]])
tForDivision = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]])
# Continued fraction arithmethic - convergents


def arithmetic_convergents_(
        a: Iterator[int],
        b: Iterator[int],
        t0=tForAddition) -> Iterator[Tuple[str, int, np.ndarray]]:
    """
    The convergents of doing arithemtic on two continued fractions.

    This version exposes the internal steps for visualization and debugging.
    New terms of the two continued fractions are applied alternately

    :param a: The first continued fraction.
    :param b: The second continued fraction.
    :param t0: The 2x2x2 tensor representing the operation.
    :return: The source (a or b?), the term, and the result (a 2x2x2 tensor)
    """
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
    """
    The convergents of doing arithmetic on two continued fraction.

    This version returns an iteartor of rational numbers.
    :param a: The first continued fraction.
    :param b: The second continued fraction.
    :param t0: The 2x2x2 tensor representing the operation.
    :return: An iterator of rational numbers
    """
    c = arithmetic_convergents_(a, b, t0)
    for _, _, res in c:
        r = tensor_ref(res, 'xy')
        assert isinstance(r, tuple)
        yield Rational(r[0], r[1])


# Continued fraction arithmetic - Euclidean algorithmn


def qr_tensor(t: np.ndarray) -> Tuple[Optional[int], np.ndarray]:
    """
    Find the quotient and remainder of a 2x2x2 tensor

    :param t: The tensor
    :return: The quotient (None if cannot be determined)
             The remainder (2x2x2 tensor; identity if quotient is None
    """

    t1 = t.copy()
    if np.all(t1[1] > 0):
        # if the denominator matrix doesn't have any 0 or negative number
        r = t_ratios(t)
        if r[0][0] == r[0][1] == r[1][0] == r[1][1]:
            # if the integer parts are all the same, we've got a quotient
            q = r[0][0]
            rem = np.array([t1[1], t1[0] - q * t1[1]])
            return (q, rem)
        else:
            # the range is too big to determine the quotient
            return (None, np.array([identity_m, identity_m]))
    else:
        # the denominator can be zero. The dihomographic function is unbounded
        return (None, np.array([identity_m, identity_m]))


def t_ratios(t: np.ndarray) -> list:
    """
    Calculate the floor ratios of the numerator and the denominator terms.

    Make sure that the denominator doesn't have 0 or negative numbers

    :param t: the 2x2x2 tensor
    :return: A 2x2 matrix (as list of lists)
    """

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


# Continued fraction arithmetic - scoring


def score(t: np.ndarray) -> float:
    """
    A scoring function for the 2x2x2 tensor.

    A higher number means that the tensor is
    getting closer for a quotient to be determined
    """

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
        r_ = t_ratios(t)
        if r_[0][0] == r_[0][1] == r_[1][0] == r_[1][1]:
            # the 4 ratios are all the same. This is the best situation
            return 4.0
        elif (r_[0][0] == r_[1][0]) or (r_[1][0] == r_[1][1]):
            # 3 1       1 2
            # 3 2 or    3 3
            return 1.0
        else:
            return 0.0


def ABQueue(
    a: Iterator[int], b: Iterator[int]
) -> Callable[[np.ndarray], Tuple[Optional[int], Optional[str]]]:
    """
    Return next term either from the first or the second continued fraction.

    The source is selected by the scoring function.

    :param a: The first continued fraction.
    :param b: The second continued fraction.
    :return: A closure. Give 2x2x2 tensor, return a new term, and the source.
    """
    current_a = None
    current_b = None
    last_tie = 'b'

    def ABQueue_(t: np.ndarray) -> Tuple[Optional[int], Optional[str]]:
        nonlocal current_a
        nonlocal current_b
        nonlocal last_tie

        def dequeue(label: str) -> Tuple[Optional[int], Optional[str]]:
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
            else:
                assert label == 'alt'
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
            t_a = apply_ab(t, current_a, 'a')
            t_b = apply_ab(t, current_b, 'b')
            s_a, s_b = score(t_a), score(t_b)
            if s_a == s_b:
                return dequeue('alt')
            if s_a < s_b:
                return dequeue('b')
            else:
                return dequeue('a')

    return ABQueue_


# Continued fraction arithmetic - full version with Euclidean steps


def euclid_tensor_(t: np.ndarray) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Symbolic Euclidean step for tensor t

    :param t: The 2x2x2 tensor
    :return: An iterator of quotient (an int) and remainder (a tensor)
    """
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
    """
    Perform arithmetic operation on two continued fraction.

    :param cf_a: The first continued fraction.
    :param cf_b: The second continued fraction.
    :param t0: The 2x2x2 tensor representing the opertion.
    :param finite_term: Is this a finite-length continued fraction,
                        or a truncation of an infinite one?
    :return: q - quotient (None if no Euclidean step performed)
             r - remainder (2x2x2 tensor)
             t - the tensor before the Euclidean step
             term - the term used in the update
             label - where does the term come from? cf_a or cf_b?
             new_term - was a new term (either from cf_a or cf_b) used?
                        It's false if only an Euclidean step was performed.
    """
    t = t0
    next_ab = ABQueue(cf_a, cf_b)

    while True:
        term, label = next_ab(t)
        if term is None and label is None:
            # cf_a and cf_b are exhausted
            break
        else:
            assert isinstance(term, int)
            assert isinstance(label, str)
            t = apply_ab(t, term, label)
            new_term = True
            for (q, r) in euclid_tensor_(t):
                yield q, r, t, term, label, new_term
                t = r
                new_term = False
            if new_term:
                # Nothing was yielded.
                # This means that an Euclidean step was not performed
                yield None, None, t, term, label, new_term

    # we will only reach this point if cf_a and cf_b have finite terms
    if finite_term and tensor_ref(t, 'e') != 0:
        rxy = tensor_ref(t, 'xy')
        assert isinstance(rxy, tuple)
        for s in r2cf(Rational(rxy[0], rxy[1])):
            yield s, None, t, None, None, False
    else:
        # if the 'e' term is 0, that means the quotient is 0.
        # there is no need to return it
        pass


def cf_arithmetic(cf_a: Iterator[int],
                  cf_b: Iterator[int],
                  t0: np.ndarray,
                  finite_term=True) -> Iterator[int]:
    """
    Perform arithmetic operation on two continued fractions.

    :param cf_a: The first continued fraction.
    :param cf_b: The second continued fraction.
    :param t0: The operation, represented by a 2x2x2 tensor.
    :param finite_term: Is this a continued fraction with finite terms,
                        or a truncated one?
    :return: An iterator of the new continued fraction
    """
    for res in cf_arithmetic_(cf_a, cf_b, t0, finite_term=finite_term):
        (q, r, t, term, label, new_term) = res
        if q is not None:
            yield q


def cf_arithmetic_func(cf_a: Iterator[int], cf_b: Iterator[int],
                       t0: np.ndarray) -> Tuple[List[int], np.ndarray]:
    """
    Perform arithmetic operations on two finite-length continued fraction.

    Return a new fraction, and a bihomographic functiom, which can be
    applied to more terms.

    :param cf_a: The first finite-length continued fraction.
    :param cf_b: The second finite-length continued fraction.
    :param t0: A 2x2x2 tensor representing the operation.
    :return: A list, representing terms of the new continued fraction.
             A 2x2x2 tensor representing the final bihomographic function.
    """
    outputs: List[int] = []
    out_t: Optional[np.ndarray] = None
    for res in cf_arithmetic_(cf_a, cf_b, t0, finite_term=False):
        q, r, t, term, label, new_term = res
        if q is not None:
            outputs = outputs + [q]
            out_t = r
        else:
            out_t = t

    assert isinstance(out_t, np.ndarray)
    return outputs, out_t


# Examples of continued fractions


def cf_e() -> Iterator[int]:
    '''e as a continued fraction'''
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
