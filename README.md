# 47th ICPC World Final 2023

## Problem I : WaterWorld
<img align="center" src="./assets/images/A.png"/>
<img align="center" src="./assets/images/B.png"/>

## Solution

```py
while True:
    try:
        n, m = map(int, input().split())
        total = 0
        for _ in range(n * m):
            a = int(input())
            total += a
        average = total / (n * m)
        print(f"{average:.9f}")
    except EOFError:
        break
```
## Problem B : Schedule
<img align="center" src="./assets/images/C.png"/>
<img align="center" src="./assets/images/D.png"/>

### Solution

```py
from itertools import permutations

while True:
    try:
        N, W = map(int, input().split())
        c = 4
        while c <= W:
            sch = []
            cur = [1] * (c // 2) + [2] * (c - c // 2)
            
            # Use set to eliminate duplicates from permutations
            for perm in set(permutations(cur)):
                sch.append(list(perm))
            
            if len(sch) >= N:
                break
            c += 1

        if c > W:
            print("infinity")
            continue

        print(c)
        for i in range(W):
            for j in range(N):
                print(sch[j][i % c], end="")
            print()
    
    except EOFError:
        break
```
## Problem H : Jet Lag

<img align="center" src="./assets/images/E.png"/>
<img align="center" src="./assets/images/F.png"/>

### Solution

```py
while True:
    try:
        N = int(input())
        B = [0] * (N + 1)
        E = [0] * (N + 1)
        
        for i in range(1, N + 1):
            B[i], E[i] = map(int, input().split())
        
        S = []
        T = []
        
        i = N
        j = i
        
        while i > 0:
            if j == 0:
                print("impossible")
                break

            t = B[j] - (E[i] - B[j] + 1) // 2
            
            if E[j - 1] > t:
                j -= 1
                continue

            if E[j - 1] >= t - 1:
                S.append(E[j - 1])
                T.append(B[j] - (1 if E[j - 1] == t - 1 and E[i] - B[j] == 1 else 0))
            else:
                S.append(t)
                T.append(B[j])
                S.append(E[j - 1])
                T.append((E[j - 1] + t) // 2)
            
            i = j - 1
            j = i
        
        else:
            print(len(S))
            for k in range(len(S) - 1, -1, -1):
                print(S[k], T[k])
        
    except EOFError:
        break

```
## Problem A : Riddle of the Sphinx
<img align="center" src="./assets/images/G.png"/>
<img align="center" src="./assets/images/H.png"/>

### Solution

```py
print("1 0 0")
a = int(input())
print("0 1 0")
b = int(input())
print("0 0 1")
c = int(input())
print("1 1 1")
d = int(input())
print("1 2 3")
e = int(input())

# Evaluate conditions and print results
if a + b + c == d:
    print(a, b, c)
elif a + 2 * b + 3 * c == e:
    print(a, b, c)
elif (d - b - c) + 2 * b + 3 * c == e:
    print(d - b - c, b, c)
elif a + 2 * (d - c - a) + 3 * c == e:
    print(a, d - c - a, c)
else:
    print(a, b, d - a - b)
```
## Problem G : Turning Red
<img align="center" src="./assets/images/I.png"/>
<img align="center" src="./assets/images/J.png"/>

### Solution

```py
from sys import stdin
from functools import lru_cache

while True:
    try:
        # Input L and B
        L, B = map(int, input().split())
        
        lb = [[] for _ in range(L)]
        bl = [[] for _ in range(B)]
        ls = [0] * L
        
        # Read characters and convert them to integers based on "RGB" index
        for i in range(L):
            ch = input().strip()
            ls[i] = "RGB".index(ch)
        
        # Reading connections between L and B
        for i in range(B):
            K = int(input())
            bl[i] = [int(input()) - 1 for _ in range(K)]
            for x in bl[i]:
                lb[x].append(i)
        
        ret = 0
        push = [-1] * B
        
        # Check initial condition for each `lb[j]`
        for j in range(L):
            if not lb[j] and ls[j] != 0:
                print("impossible")
                break
        else:
            # Recursive function to determine the minimum transformations
            @lru_cache(None)
            def rec(i, p, cookie):
                if push[i] >= cookie:
                    return 0 if push[i] == cookie + p else int(1e9)
                push[i] = cookie + p
                result = p
                
                for j in bl[i]:
                    if len(lb[j]) == 2:
                        k = lb[j][0] ^ lb[j][1] ^ i
                        result += rec(k, (12 - ls[j] - p) % 3, cookie)
                        if result >= 1e9:
                            return int(1e9)
                    else:
                        if (ls[j] + p) % 3 != 0:
                            return int(1e9)
                return result
            
            # Calculate the minimum number of transformations
            for i in range(B):
                if push[i] == -1 and bl[i]:
                    best = int(1e9)
                    for p in range(3):
                        best = min(best, rec(i, p, p * 3))
                    if best == int(1e9):
                        print("impossible")
                        break
                    ret += best
            else:
                print(ret)
    
    except EOFError:
        break

```
## Problem F : Tilting Tiles
<img align="center" src="./assets/images/K.png"/>
<img align="center" src="./assets/images/L.png"/>

### Solution

```py
import math

def gcd(a, b):
    return math.gcd(a, b) if b != 0 else abs(a)

def tilt(dir, g):
    X, Y = len(g[0]), len(g)
    if dir & 1:
        X, Y = Y, X

    def get(x, y):
        if dir == 0:
            return g[y][x]
        elif dir == 1:
            return g[x][y]
        elif dir == 2:
            return g[y][X - 1 - x]
        else:
            return g[X - 1 - x][y]

    for y in range(Y):
        x2 = 0
        for x in range(X):
            if get(x, y):
                g[y][x2 if dir in [0, 2] else x] = get(x, y)
                x2 += 1
        for k in range(x2, X):
            if dir in [0, 2]:
                g[y][k] = 0
            else:
                g[k][y] = 0

def match(s, t):
    r, m = 0, 0
    while r <= len(s):
        if r == len(s):
            return 0, 0
        if s[r:] == t[:len(s) - r] and s[:r] == t[len(s) - r:]:
            break
        r += 1

    for m in range(r if r else 1, len(s)):
        if len(s) % m == 0 and s[:len(s) - m] == s[m:]:
            break

    return r, m

def main():
    while True:
        try:
            Y, X = map(int, input().split())
            g = [[0] * X for _ in range(Y)]
            g2 = [[0] * X for _ in range(Y)]

            for y in range(Y):
                line = input().strip()
                for x, ch in enumerate(line):
                    g[y][x] = ord(ch) - ord('a') + 1 if ch != '.' else 0

            for y in range(Y):
                line = input().strip()
                for x, ch in enumerate(line):
                    g2[y][x] = ord(ch) - ord('a') + 1 if ch != '.' else 0

            for sd in range(4):
                for dd in range(1, 4, 2):
                    tg = [row[:] for row in g]
                    d = sd
                    for i in range(7):
                        if tg == g2:
                            print("yes")
                            break
                        if i >= 2:
                            ng = [row[:] for row in tg]
                            for y in range(Y):
                                for x in range(X):
                                    if bool(g2[y][x]) != bool(ng[y][x]):
                                        break
                                    if ng[y][x]:
                                        ng[y][x] = y * X + x + 1
                            else:
                                residues, mods = [], []
                                for y in range(Y):
                                    for x in range(X):
                                        if ng[y][x]:
                                            s, t = "", ""
                                            ptr = ng[y][x]
                                            while ptr:
                                                x2, y2 = (ptr - 1) % X, (ptr - 1) // X
                                                ptr = 0
                                                s += chr(tg[y2][x2] + ord('a') - 1)
                                                t += chr(g2[y2][x2] + ord('a') - 1)
                                                ptr = ng[y2][x2]
                                            
                                            residue, mod = match(s, t)
                                            if mod == 0:
                                                break
                                            if mod == 1:
                                                continue
                                            
                                            for i in range(len(mods)):
                                                g = gcd(mod, mods[i])
                                                if residues[i] % g != residue % g:
                                                    break
                                            else:
                                                residues.append(residue)
                                                mods.append(mod)
                                else:
                                    print("yes")
                                    break
                        tilt(d, tg)
                        d = (d + dd) % 4
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                print("no")

        except EOFError:
            break

if __name__ == "__main__":
    main()

```
## Problem D : Carl's Vacation
<img align="center" src="./assets/images/M.png"/>
<img align="center" src="./assets/images/N.png"/>

### Solution :

```py
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

```
## Problem J : Bridging The Gap
<img align="center" src="./assets/images/O.png"/>

### Solution :

```py
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    
    while idx < len(data):
        N = int(data[idx])
        C = int(data[idx + 1])
        idx += 2
        T = list(map(int, data[idx:idx + N]))
        idx += N

        T.sort()

        # Initialize tot and cc arrays
        tot = [1e18] * (N + 1)
        cc = [1e18] * N
        tot[0] = cc[0] = 0

        # Compute prefix sums in tot array
        for i in range(N):
            tot[i + 1] = tot[i] + T[i]

        # Compute minimum costs for each possible group size
        for i in range(1, min(C, N)):
            for j in range(i, len(cc)):
                cc[j] = min(cc[j], cc[j - i] + T[i] + tot[i + 1])

        # Initialize mnc, mxc, and dyn arrays
        mnc = [-(i // C) - (i == N) for i in range(N + 1)]
        mxc = [(N - i + C - 1) // C - 1 for i in range(N + 1)]
        dyn = [[1e18] * (mxc[i] - mnc[i] + 1) for i in range(N + 1)]
        dyn[0][0] = 0

        # Dynamic programming to compute minimum sum for partitions
        for i in range(N):
            for ci in range(len(dyn[i])):
                if ci and dyn[i][ci] - dyn[i][0] >= cc[ci]:
                    continue
                c = mnc[i] + ci
                for j in range(min(N, i + C), i, -1):
                    extra = j - i - 1
                    c2 = c + extra
                    if c2 > mxc[j]:
                        break
                    dyn[j][c2 - mnc[j]] = min(dyn[j][c2 - mnc[j]], dyn[i][ci] + T[N - 1 - i] + tot[extra + 1])

        # Compute the final result
        ret = 1e18
        for c in range(mnc[N], 0):
            ret = min(ret, dyn[N][c - mnc[N]] + cc[-1 - c])

        print(int(ret))

if __name__ == "__main__":
    main()

```

## Problem K : Alea Lacta Est
<img align="center" src="./assets/images/P.png"/>
<img align="center" src="./assets/images/R.png"/>

### Solution :

```py
import sys
import heapq
from functools import lru_cache
from collections import defaultdict

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0

    while idx < len(data):
        N = int(data[idx])
        W = int(data[idx + 1])
        idx += 2

        D = data[idx:idx + N]
        idx += N

        dw = defaultdict(list)

        # Populate `dw` with all possible configurations
        def doit(d, b, s):
            if d == N:
                sorted_s = ''.join(sorted(s))
                dw[sorted_s].append(b)
                return
            for i, ch in enumerate(D[d]):
                doit(d + 1, b + ((i + 1) << (3 * d)), s + ch)

        doit(0, 0, "")

        curn = [0] * (1 << (3 * N))
        seen = [False] * (1 << (3 * N))
        cure = [0.0] * (1 << (3 * N))
        beste = [1e9] * (1 << (3 * N))
        q = []

        # Adjust expected values for configurations
        def sete(d, pw, b, e):
            if d == N:
                if pw == 1:
                    return
                curn[b] += 1
                cure[b] += e
                be = (pw + cure[b]) / curn[b]
                beste[b] = be
                heapq.heappush(q, (-be, b))
                return
            sete(d + 1, pw, b, e)
            sete(d + 1, pw * len(D[d]), b & ~(7 << (3 * d)), e)

        # Update values for configurations that reach a solved configuration
        def brec(d, b, e):
            if d == N:
                if not seen[b]:
                    sete(0, 1, b, e)
                seen[b] = True
                return
            if b & (7 << (3 * d)):
                brec(d + 1, b, e)
            else:
                for i in range(len(D[d])):
                    brec(d + 1, b + ((i + 1) << (3 * d)), e)

        # Process all words
        for _ in range(W):
            w = data[idx]
            idx += 1
            sorted_w = ''.join(sorted(w))
            for b in dw[sorted_w]:
                brec(0, b, 0.0)

        # Check if queue is empty and process the configurations
        if not q:
            print("impossible")
            continue

        while q:
            e, b = heapq.heappop(q)
            e = -e
            if seen[b]:
                continue
            seen[b] = True

            # Solve for configuration b
            brec(0, b, e)

        # Output result with high precision
        print(f"{beste[0]:.9f}")

if __name__ == "__main__":
    main()

```
## Problem E : A Recurring Problem
<img align="center" src="./assets/images/R1.png"/>
<img align="center" src="./assets/images/R2.png"/>

### Solution :

```py
import sys
from collections import defaultdict
from functools import lru_cache

# Memoization for count1 function
memo1 = [None] * 51

def count1(n):
    if n == 0:
        return 1
    if memo1[n] is not None:
        return memo1[n]
    
    ret = 0
    for a in range(1, n + 1):
        for c in range(1, (n // a) + 1):
            ret += count1(n - a * c)
    memo1[n] = ret
    return ret

# Memoization and global variables for the count function
memo = {}
curc = []
cura = []
saved = []

def count(seq, prev, save):
    empty = {}
    base = {0: 1}

    if seq[0] == 0:
        if all(x == 0 for x in seq):
            if save:
                curs = cura[:]
                while len(curs) < len(curc) + 30:
                    x = sum(curs[-len(curc) + i] * curc[i] for i in range(len(curc)))
                    curs.append(x)
                curs = curs[len(curc):]
                saved.append((curs, curc[:], cura[:]))
            return base
        return empty

    if any(x <= 0 for x in seq):
        return empty

    # Check memoization
    key = (tuple(seq), tuple(prev))
    if key in memo and not save:
        return memo[key]

    ret = defaultdict(int)
    if save:
        ret.clear()
    else:
        memo[key] = ret

    prev = [0] + prev
    for c in range(1, seq[0] + 1):
        for a in range(1, (seq[0] // c) + 1):
            prev[0] = a
            for i in range(len(seq)):
                seq[i] -= prev[i] * c
            tmp = prev.pop()

            if save:
                curc.insert(0, c)
                cura.insert(0, a)
            result = count(seq, prev, save)
            for v, n in result.items():
                ret[v + tmp * c] += n
            if save:
                curc.pop(0)
                cura.pop(0)

            prev.append(tmp)
            for i in range(len(seq)):
                seq[i] += prev[i] * c

    if not save:
        memo[key] = ret
    return ret

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0

    while idx < len(data):
        N = int(data[idx])
        idx += 1

        memo.clear()
        cura.clear()
        curc.clear()
        saved.clear()

        seq = []
        n = 1
        while True:
            if count1(n) < N:
                N -= count1(n)
                n += 1
            else:
                seq.append(n)
                break

        while len(seq) < 30 and seq[-1] < 1e16:
            m = count(seq, seq, False)
            tot = 0
            for v, n in m.items():
                if n < N:
                    N -= n
                else:
                    seq.append(v)
                    if n <= 20:
                        break
                    else:
                        break

        count(seq, seq, True)
        saved.sort()
        sv, cv, av = saved[N - 1]

        print(len(cv))
        print(" ".join(map(str, cv)))
        print(" ".join(map(str, av)))
        print(" ".join(map(str, sv[:10])))

if __name__ == "__main__":
    main()

```

## Problem C : Three Kinds of Dice
<img align="center" src="./assets/images/P1.png"/>
<img align="center" src="./assets/images/P2.png"/>

### Solution :
```py
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


```
