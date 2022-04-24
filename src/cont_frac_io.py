from cont_frac import *
from functools import reduce
import csv


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

    def to_array(self):
        main_content = self.board.copy()
        top_content = self.top.copy()
        right_content = self.right.copy()
        m = len(main_content)
        n = len(main_content[0])

        delta = n - len(top_content)
        top_content = [None] * delta + top_content + [None]

        delta = m - len(right_content)
        right_content = right_content + [None] * delta

        zz = [top_content]
        for i in range(m):
            zz = zz + [main_content[i] + [right_content[i]]]
        return zz

    def export_csv(self, filename):
        array = self.to_array()
        with open(filename, mode='w') as out_file:
            writer = csv.writer(out_file)
            writer.writerows(array)


class Chart3D(object):
    def __init__(self, m=tForAddition):
        self.output = [None]
        self.boards = m.tolist()
        self.a = []
        self.b = [[]]
        self.current_tensor = m.copy()
        self.include_a = True
        self.include_b = True
        self.include_out = True

    def move_left(self, t, a):
        assert tensor_ref(t, 'y') == tensor_ref(self.current_tensor, 'xy')
        assert tensor_ref(t, '1') == tensor_ref(self.current_tensor, 'x')
        self.current_tensor = t

        # add a new column for all boards
        for i in range(len(self.boards)):
            for j in range(len(self.boards[0])):
                self.boards[i][j] = [None] + self.boards[i][j]

        self.boards[-2][-2][0] = tensor_ref(t, 'b')
        self.boards[-2][-1][0] = tensor_ref(t, 'a')
        self.boards[-1][-2][0] = tensor_ref(t, 'f')
        self.boards[-1][-1][0] = tensor_ref(t, 'e')
        self.a = [a] + self.a

    def move_down(self, t, b):
        assert tensor_ref(t, 'x') == tensor_ref(self.current_tensor, 'xy')
        assert tensor_ref(t, '1') == tensor_ref(self.current_tensor, 'y')
        self.current_tensor = t
        new_row_numerator = [None] * len(self.boards[0][0])
        new_row_denominator = [None] * len(self.boards[1][0])
        new_row_numerator[0] = tensor_ref(t, 'a')
        new_row_numerator[1] = tensor_ref(t, 'c')
        new_row_denominator[0] = tensor_ref(t, 'e')
        new_row_denominator[1] = tensor_ref(t, 'g')
        self.boards[-2] = self.boards[-2] + [new_row_numerator]
        self.boards[-1] = self.boards[-1] + [new_row_denominator]
        self.b[-1] = self.b[-1] + [b]

        # all boards under have to be expanded
        for i in range(0, len(self.boards) - 2):
            self.boards[i] = self.boards[i] + [[None] * len(new_row_numerator)]

    def move_under(self, t, output):
        assert tensor_ref(self.current_tensor, 'f') == tensor_ref(t, 'b')
        assert tensor_ref(self.current_tensor, 'h') == tensor_ref(t, 'd')
        assert tensor_ref(self.current_tensor, 'e') == tensor_ref(t, 'a')
        assert tensor_ref(self.current_tensor, 'g') == tensor_ref(t, 'c')
        self.current_tensor = t

        n_rows = len(self.boards[0])
        n_cols = len(self.boards[0][0])

        def new_row():
            return [None] * n_cols

        new_boards = [new_row() for i in range(n_rows)]

        new_boards[-2][0] = tensor_ref(t, 'f')
        new_boards[-2][1] = tensor_ref(t, 'h')
        new_boards[-1][0] = tensor_ref(t, 'e')
        new_boards[-1][1] = tensor_ref(t, 'g')
        self.boards = self.boards + [new_boards]

        new_b = [None] * len(self.b[-1])
        self.b = self.b + [new_b]

        self.output = self.output + [output]

    def board_to_array(self, board, b, out):
        new_content = []
        for i, row in enumerate(board):
            new_row = row.copy()

            if self.include_b:
                if i == 0:
                    new_row = new_row + [None]
                elif b is None:
                    new_row = new_row + [None]
                elif i <= len(b):
                    new_row = new_row + [b[i - 1]]
                else:
                    new_row = new_row + [None]

            if self.include_out:
                if i == len(board) - 1:
                    new_row = new_row + [out]
                else:
                    new_row = new_row + [None]

            new_content = new_content + [new_row]
        return new_content

    def to_array(self):
        content = []
        row = []
        n_rows = len(self.boards[0])
        n_cols = len(self.boards[0][0])

        if self.include_a:
            row = row + [None]
            if self.include_b:
                row = row + [None]
            if self.include_out:
                row = row + [None]
            row = self.a + row
            row = [None] * (n_cols - len(self.a) - 1) + row
            content = content + [row]

        for i in range(len(self.boards)):
            board = self.boards[i]
            b = self.b[i] if i < len(self.b) else None
            out = self.output[i] if i < len(self.output) else None
            content = content + self.board_to_array(board, b, out)

        return content

    @staticmethod
    def pp_item(item, field_width):
        if item is None:
            return " " * field_width
        else:
            return f"{item : > {field_width}}"

    @staticmethod
    def pp_row(row, field_width):
        return reduce(lambda s, item: s + Chart3D.pp_item(item, field_width),
                      row, "") + "\n"

    def __repr__(self):
        content = self.to_array()

        content_nonone = [[c for c in row if c is not None] for row in content]
        content_nonone = [r for r in content_nonone if r != []]
        mx = max(map(max, content_nonone))
        field_width = len(str(mx)) + 1

        s = reduce(lambda s, r: s + Chart3D.pp_row(r, field_width), content,
                   "")
        return s

    def export_csv(self, filename):
        content = self.to_array()
        with open(filename, mode='w') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(content)


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
    return chart


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

    res = cf_transform_(cf, m0)
    # res may be longer than cf1, res might not be empty after this loop
    for (q, r, m, a, new_a) in res:
        if new_a:
            chart.push_column(m, a)
        if q is None:
            # this means that no euclid step was performed
            # do nothing
            pass
        else:
            if r is not None:
                chart.push_row(r, q)
            else:
                # r is None, meaning that the quotients are for rational numbers rathen than matrices
                chart.push_right(q)
    return chart


def tabs3d(a, b, t0=tForAddition):
    c = Chart3D(t0)
    for direction, coefficient, t in arithmetic_convergents_(a, b, t0):
        if direction == 'a':
            c.move_left(t, coefficient)
        else:
            c.move_down(t, coefficient)
    return c


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
