def lu_decomposition(A):
    n = len(A)

    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for k in range(i, n):
            sum_upper = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = A[i][k] - sum_upper

        for k in range(i, n):
            if i == k:
                L[i][i] = 1.0
            else:
                sum_lower = sum(L[k][j] * U[j][i] for j in range(i))
                L[k][i] = (A[k][i] - sum_lower) / U[i][i]

    return L, U

A = [
    [2, -1, -2],
    [-4, 6, 3],
    [-4, -2, 8]
]

L, U = lu_decomposition(A)

print("L:")
for row in L:
    print(row)

print("\nU:")
for row in U:
    print(row)
