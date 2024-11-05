import math
import sys

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, c):
        return Point(self.x * c, self.y * c)

    def len(self):
        return math.hypot(self.x, self.y)

def cross_prod(a, b):
    return a.x * b.y - a.y * b.x

def intersect(a1, a2, b1, b2):
    cp1 = cross_prod(a2 - a1, b1 - a1)
    cp2 = cross_prod(a2 - a1, b2 - a1)
    if cp1 < -1e-9 and cp2 < -1e-9:
        return False
    if cp1 > 1e-9 and cp2 > 1e-9:
        return False
    cp1 = cross_prod(b2 - b1, a1 - b1)
    cp2 = cross_prod(b2 - b1, a2 - b1)
    if cp1 < -1e-9 and cp2 < -1e-9:
        return False
    if cp1 > 1e-9 and cp2 > 1e-9:
        return False
    return True

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    while idx < len(data):
        a1 = Point(float(data[idx]), float(data[idx + 1]))
        a2 = Point(float(data[idx + 2]), float(data[idx + 3]))
        ah = float(data[idx + 4])
        b1 = Point(float(data[idx + 5]), float(data[idx + 6]))
        b2 = Point(float(data[idx + 7]), float(data[idx + 8]))
        bh = float(data[idx + 9])
        idx += 10

        a_side_len = (a2 - a1).len()
        b_side_len = (b2 - b1).len()
        a_diag_len = math.sqrt(a_side_len ** 2 / 2 + ah ** 2)
        b_diag_len = math.sqrt(b_side_len ** 2 / 2 + bh ** 2)
        a_alt_len = math.sqrt(a_side_len ** 2 / 4 + ah ** 2)
        b_alt_len = math.sqrt(b_side_len ** 2 / 4 + bh ** 2)
        ret = 1e9

        for ai in range(4):
            ap = Point(a1.y - a2.y, a2.x - a1.x)
            amid = (a1 + a2) * 0.5 + ap * 0.5
            for bi in range(4):
                bp = Point(b1.y - b2.y, b2.x - b1.x)
                bmid = (b1 + b2) * 0.5 + bp * 0.5

                for a_diag in range(2):
                    at = a1 if a_diag else (a1 + a2) * 0.5 + ap * (a_alt_len / a_side_len)
                    alen = a_diag_len if a_diag else 0.0
                    for b_diag in range(2):
                        bt = b1 if b_diag else (b1 + b2) * 0.5 + bp * (b_alt_len / b_side_len)
                        blen = b_diag_len if b_diag else 0.0

                        if not a_diag and (cross_prod(bmid - a1, a2 - a1) < 0 or not intersect(a1, a2, at, bt)):
                            continue
                        if not b_diag and (cross_prod(amid - b1, b2 - b1) < 0 or not intersect(b1, b2, at, bt)):
                            continue
                        ret = min(ret, alen + blen + (bt - at).len())

                b1 = b2 + bp
                b1, b2 = b2, b1
            a1 = a2 + ap
            a1, a2 = a2, a1

        print(f"{ret:.9f}")

if __name__ == "__main__":
    main()

