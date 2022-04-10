from cont_frac import *
from functools import reduce


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
    return reduce(row, r2cf_(rn), str0) + f"{0 : > 5}\n"


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


def cf_convergent2_tab(cf: Iterator[int],
                       m0=np.identity(2, int),
                       field_width=4):
    chart = Chart(m=m0, field_width=field_width)
    (cf1, cf2) = tee(cf)
    for (a, (q, r, m)) in zip(cf1, cf_convergent2_(cf2, m0)):
        chart.push_column(m, a)
        if q is None:
            pass
        else:
            chart.push_row(r, q)
            if r is None:
                char.push_right(q)
    print(chart)
