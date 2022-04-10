import numpy as np
import pytest
from cont_frac import *


class TestR2cf(object):

    def test_basic(self):
        res = r2cf(Rational(254, 100))
        assert list(res) == [2, 1, 1, 5, 1, 3]


class TestQRMatrix(object):

    def qr(self, m):
        res = qr_matrix(np.array(m))
        if res is None:
            return None
        else:
            (q, r) = res
            return q, r.tolist()

    def test_basic1(self):
        # (4x+2)/(3x+2) is between 4/2 and 2/2 (1 to 1.33)
        m = [[4, 2], [3, 2]]
        (q, r) = self.qr(m)
        assert q == 1 and r == [[3, 2], [1, 0]]

    def test_basic2(self):
        # (70x + 29)/ (12x + 5) is between 29/5 and 35/6 (5.8 to 5.88)
        m = [[70, 29], [12, 5]]
        (q, r) = self.qr(m)
        assert q == 5 and r == [[12, 5], [10, 4]]

    def test_basic3(self):
        # (12x + 5) / (10 x + 4) is between 6/5 and 5/4 (1.2 to 1.25)
        m = [[12, 5], [10, 4]]
        (q, r) = self.qr(m)
        assert q == 1 and r == [[10, 4], [2, 1]]

    def test_negative1(self):
        # (10x + 4) / (2x + 1) is bounded between 4 and 5
        # the quotient is 4, because it is 5 only at infinity
        m = [[10, 4], [2, 1]]
        (q, r) = self.qr(m)
        assert q == 4 and r == [[2, 1], [2, 0]]

    def test_negative2(self):
        # (8x + 3) / (2x + 1) is bounded between 3 and 4
        # it is only 4 if x is infinity, so the coefficient has to be 3
        m = [[8, 3], [2, 1]]
        (q, r) = self.qr(m)
        assert q == 3 and r == [[2, 1], [2, 0]]

    def test_divergent1(self):
        # 4x + 2 is unbounded
        m = [[4, 2], [0, 1]]
        res = self.qr(m)
        assert res is None

    def test_divergent2(self):
        # (4x + 2) / 3 is unbounded
        m = [[4, 2], [3, 0]]
        res = self.qr(m)
        assert res is None

    def test_zero_coeff(self):
        # (1x + 2) / (2x + 3) is bounded between 1/2 and 2/3 (0.5 to 0.666)
        m = [[1, 2], [2, 3]]
        (q, r) = self.qr(m)
        assert q == 0 and r == [[2, 3], [1, 2]]


# Apply this test to two functions, because they should be equivalent
@pytest.mark.parametrize('conv_func', [cf_convergents0, cf_convergents1])
def test_convergents_basci1(conv_func):
    l = list(conv_func(iter([2, 1, 1, 5, 1, 3])))
    assert l[0] == Rational(2, 1)
    assert l[1] == Rational(3, 1)
    assert l[2] == Rational(5, 2)
    assert l[3] == Rational(28, 11)
    assert l[4] == Rational(33, 13)
    assert l[5] == Rational(127, 50)
    assert len(l) == 6


# Apply this test to two functions, because they should be equivalent
@pytest.mark.parametrize('conv_func', [cf_convergents0, cf_convergents1])
def test_convergents_basic2(conv_func):
    l = list(cf_convergents0(iter([1, 2, 3, 4, 5])))
    assert l[0] == Rational(1, 1)
    assert l[1] == Rational(3, 2)
    assert l[2] == Rational(10, 7)
    assert l[3] == Rational(43, 30)
    assert l[4] == Rational(225, 157)
    assert len(l) == 5


# cf2r0 and cf2r1 should be equivalent
@pytest.mark.parametrize('cf2r_func', [cf2r0, cf2r1])
def test_cf2r(cf2r_func):
    assert cf2r_func([2, 1, 1, 5, 1, 3]) == Rational(127, 50)
    assert cf2r_func([1, 2, 3, 4, 5]) == Rational(225, 157)
