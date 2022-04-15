from cont_frac import *
from functools import reduce


# A class to facilitate tabular displays
class Chart(object):

    def __init__(self,
                 m=np.identity(2, int),
                 display_top=True,
                 display_right=True,
                 field_width=4):
        self.display_top = display_top
        self.display_right = display_right
        self.top = [None]
        self.right = [None]
        self.board = [[m[0][0], m[0][1]], [m[1][0], m[1][1]]]
        self.field_width = field_width

    def pp_item(self, item, right=False):
        if item is None:
            return " " * self.field_width
        else:
            if right:
                return f" {item : < {self.field_width}}"
            else:
                return f"{item : > {self.field_width}}"

    def pp_row(self, row):
        return reduce(lambda s, item: s + self.pp_item(item), row, "")

    def last_column(self):
        return (self.board[-2][0], self.board[-1][0])

    def push_top(self, i):
        self.top = [i] + self.top

    def push_right(self, i):
        self.right.append(i)

    def push_column(self, m, a):
        self.push_top(a)
        assert self.last_column() == (m[0][1], m[1][1])
        for i in range(len(self.board)):
            self.board[i] = [None] + self.board[i]
        self.board[-2][0] = m[0][0]
        self.board[-1][0] = m[1][0]

    def push_row(self, m, q):
        assert self.board[-1][0] == m[0][0]
        assert self.board[-1][1] == m[0][1]
        new_row = [None] * len(self.board[-1])
        new_row[0] = m[1][0]
        new_row[1] = m[1][1]
        self.board.append(new_row)
        self.right.append(q)

    def __repr__(self):
        s = ""
        if self.display_top:
            s = s + self.pp_row([None] + self.top) + "\n"

        for (i, row) in enumerate(self.board):
            s = s + self.pp_row(row)
            try:
                if self.display_right:
                    r = self.pp_item(self.right[i], right=True)
                else:
                    r = ""
                s = s + r + "\n"
            except IndexError:
                s = s + "\n"
        s = s[:-1]  # remove the last "\n"
        return s


def r2cf_tab(rn: Rational):

    def row(st: str, x: tuple):
        b, q = x
        return st + f"{b : > 5}  {q : < 5}\n"

    str0 = f"{rn.a : > 5}\n"
    print(reduce(row, r2cf_(rn), str0) + f"{0 : > 5}\n")


def cf_convergents1_tab(cf: Iterator[int]):
    chart = Chart(display_right=False)
    (cf1, cf2) = tee(cf)
    for (mat, a) in zip(cf_convergents1_(cf1), cf2):
        chart.push_column(mat, a)
    print(chart)


def euclid_matrix_tab(m):
    chart = Chart(m=m, display_top=True)
    for (q, r) in euclid_matrix_(m):
        chart.push_row(r, q)
    print(chart)


def cf_transform_tab(cf: Iterator[int],
                     m0=np.identity(2, int),
                     n=None,
                     field_width=4):
    chart = Chart(m=m0, field_width=field_width)
    if n:
        cf = islice(cf, n)

    (cf1, cf2) = tee(cf)

    res = cf_transform_(cf2, m0)
    # res may be longer than cf1, res might not be empty after this loop
    for (a, (q, r, m)) in zip(cf1, res):
        chart.push_column(m, a)
        if q is None:
            # this means that no euclid step was performed
            # do nothing
            pass
        else:
            chart.push_row(r, q)

    for item in res:
        # at this point, the quotients are quotients for rational numbers rather than matrices
        # so r should be None
        (q, r, m) = item
        assert r is None
        chart.push_right(q)
        pass

    print(chart)


# Utilities functions for LaTeX displays
def latex_cf(lst: list):
    if len(lst) == 1:
        return str(lst[0])
    else:
        x = str(lst[0]) + "+"
        x = x + r"\frac{1}{" + latex_cf(lst[1:]) + "}"
        return x


def latex_rational(r: Rational):
    return r"\frac{" + str(r.a) + "}{" + str(r.b) + "}"


def show_cf_expansion(r: Rational):
    print(r"\[")
    print(r"\frac{", r.a, "}{", r.b, "}=")
    nc = list(r2cf(r))
    print(latex_cf(nc))
    print(r"\]")


def show_rational_series(itr: Iterator[int]):
    rLst = list(cf_convergents0(itr))
    s = ""
    for r in rLst:
        s = s + "$" + latex_rational(r) + "$" + ","

    print(s[:-1])


# Pretty printing utilities


def pp_qr(qr: Tuple[int, np.ndarray]) -> None:
    """Pretty print a tuple of a quotient and a remainder matrix"""
    q, r = qr
    print(f"{q:>2} {r[0][0]:2} {r[0][1]:2}")
    print(f"   {r[1][0]:2} {r[1][1]:2}")


def pp_inf_cf(cf: list) -> None:
    """Pretty print a list representing the first couple terms of a longer continued fraction"""
    res = "["
    res = res + reduce(lambda s, n: s + str(n) + ",", cf, "")
    res = res[:-1] + "...]"
    return res
