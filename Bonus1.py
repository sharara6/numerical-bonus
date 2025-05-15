def thomas_algorithm(a, b, c, d):
    n = len(d)
    # Forward sweep
    for i in range(1, n):
        w = a[i] / b[i - 1]
        b[i] = b[i] - w * c[i - 1]
        d[i] = d[i] - w * d[i - 1]

    # Back substitution
    x = [0] * n
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x

a = [0, -1, -1, -1]  
b = [2, 2, 2, 2]     
c = [-1, -1, -1, 0] 
d = [1, 0, 0, 1] 

x = thomas_algorithm(a, b, c, d)
print("Solution using Thomas Algorithm:", x)

import numpy as np

A = np.array([[2, -1, 0, 0],
              [-1, 2, -1, 0],
              [0, -1, 2, -1],
              [0, 0, -1, 2]])
D = np.array([1, 0, 0, 1])

x_builtin = np.linalg.solve(A, D)
print("Solution using NumPy:", x_builtin)
