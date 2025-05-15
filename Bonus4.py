import numpy as np
import matplotlib.pyplot as plt

def g(x):
    return np.exp(-x) - x

def dg(x):
    return -np.exp(-x) - 1

TRUE_ROOT = 0.56714329

def record_error(x):
    return abs((TRUE_ROOT - x) / TRUE_ROOT) * 100

def bisection(a, b, tol=1e-6, maxit=20):
    history = []
    for _ in range(maxit):
        mid = (a + b) / 2.0
        history.append(record_error(mid))
        if abs(b - a) < tol:
            break
        if g(a) * g(mid) < 0:
            b = mid
        else:
            a = mid
    return history

def false_position(a, b, tol=1e-6, maxit=20):
    history = []
    c = a
    for _ in range(maxit):
        fa, fb = g(a), g(b)
        c = (a * fb - b * fa) / (fb - fa)
        history.append(record_error(c))
        if abs(g(c)) < tol:
            break
        if fa * g(c) < 0:
            b = c
        else:
            a = c
    return history

def secant(x_prev, x_curr, tol=1e-6, maxit=20):
    history = []
    for _ in range(maxit):
        f0, f1 = g(x_prev), g(x_curr)
        x_next = x_curr - f1 * (x_curr - x_prev) / (f1 - f0)
        history.append(record_error(x_next))
        if abs(x_next - x_curr) < tol:
            break
        x_prev, x_curr = x_curr, x_next
    return history

def newton(x0, tol=1e-6, maxit=20):
    history = []
    x = x0
    for _ in range(maxit):
        x_new = x - g(x) / dg(x)
        history.append(record_error(x_new))
        if abs(x_new - x) < tol:
            break
        x = x_new
    return history

results = {
    "Bisection":    bisection(0.0, 1.0),
    "False-Pos.":   false_position(0.0, 1.0),
    "Secant":       secant(0.0, 1.0),
    "Newton–Raphson": newton(0.5)
}

fig, ax = plt.subplots()
ax.set_yscale('log')
ax.set_xlabel('Iteration #')
ax.set_ylabel('True % Relative Error')
ax.set_title('Root-Finding Convergence Comparison')
ax.grid(True, linestyle='--', linewidth=0.5)

for method, errs in results.items():
    ax.plot(range(1, len(errs)+1), errs, label=method)

ax.legend(loc='upper right')
plt.tight_layout()
plt.show()
