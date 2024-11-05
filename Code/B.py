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