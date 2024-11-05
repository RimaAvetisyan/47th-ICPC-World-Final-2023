import sys
from itertools import chain
from math import isclose


def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0

    while idx < len(data):
        N1 = int(data[idx])
        idx += 1
        D1 = list(map(int, data[idx:idx + N1]))
        idx += N1

        N2 = int(data[idx])
        idx += 1
        D2 = list(map(int, data[idx:idx + N2]))
        idx += N2

        D1.sort()
        D2.sort()
        D1.append(int(2e9))
        D2.append(int(2e9))

        # Ensure D1 beats D2.
        prob = 0
        i1 = i2 = j2 = 0
        for i1 in range(N1):
            while D2[i2] < D1[i1]:
                i2 += 1
            while D2[j2] <= D1[i1]:
                j2 += 1
            prob += 2 * i2 + (j2 - i2)

        if prob < N1 * N2:
            N1, N2 = N2, N1
            D1, D2 = D2, D1

        # Generate candidate positions
        poss = list(set(chain.from_iterable((x - 1, x, x + 1) for x in D1 + D2 if x > 1)))
        poss.sort()
        poss = [x for x in poss if x <= 1.5e9]

        v = []
        i1 = i2 = j1 = j2 = 0
        for p in poss:
            while D1[i1] < p:
                i1 += 1
            while D2[i2] < p:
                i2 += 1
            while D1[j1] <= p:
                j1 += 1
            while D2[j2] <= p:
                j2 += 1
            v.append(((2 * i1 + (j1 - i1)) / (2 * N1), (2 * i2 + (j2 - i2)) / (2 * N2)))

        for rep in range(2):
            hull = []
            for i in range(len(v)):
                while len(hull) >= 2:
                    x1, y1 = v[hull[-2]]
                    x2, y2 = v[hull[-1]]
                    x3, y3 = v[i]
                    if (x3 - x1) * (y2 - y1) < (x2 - x1) * (y3 - y1):
                        break
                    hull.pop()
                hull.append(i)

            ret = 1.0
            for i in range(len(hull) - 1):
                x1, y1 = v[hull[i]]
                x2, y2 = v[hull[i + 1]]
                if x1 >= 0.5 or x2 < 0.5:
                    continue
                ret = y1 + (y2 - y1) / (x2 - x1) * (0.5 - x1)

            if rep == 0:
                print(f"{ret:.9f}", end=" ")
            else:
                print(f"{1 - ret:.9f}")

            # Swap and reverse for the next round
            v = [(1 - y, 1 - x) for x, y in v]
            v.reverse()
            poss.reverse()


if __name__ == "__main__":
    main()

