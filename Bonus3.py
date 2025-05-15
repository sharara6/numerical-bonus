import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([0.25, 0.75, 1.25, 1.75, 2.25])
y_data = np.array([0.28, 0.57, 0.68, 0.74, 0.79])

def model(a, x):
    a0, a1 = a
    return a0 * (1 - np.exp(-a1 * x))

def jacobian(a, x):
    a0, a1 = a
    e = np.exp(-a1 * x)
    J0 = 1 - e
    J1 = a0 * x * e
    return np.vstack([J0, J1]).T

def gauss_newton(x, y, a_init, tol_ss=1e-4, tol_eps=0.01, max_iter=50, damping=1e-6):
    a = np.array(a_init, dtype=float)
    history = {
        'a': [a.copy()],
        'ss': [],
        'eps': []
    }
    for k in range(max_iter):
        r = model(a, x) - y
        ss = np.sum(r ** 2)
        history['ss'].append(ss)
        J = jacobian(a, x)
        JTJ = J.T @ J
        try:
            delta = np.linalg.solve(JTJ + damping * np.eye(JTJ.shape[0]), J.T @ r)
        except np.linalg.LinAlgError:
            print(f"Iteration {k}: Singular matrix encountered. Aborting this run.")
            break
        history['eps'].append(delta.copy())
        a -= delta
        history['a'].append(a.copy())
        if ss < tol_ss or np.max(np.abs(delta)) < tol_eps:
            break
    return a, history

inits = [(1, 1), (0.5, 0.5), (-1, 1), (1, -1)]
results = []

for init in inits:
    try:
        a_fit, hist = gauss_newton(x_data, y_data, init)
        results.append((init, a_fit, hist))
        print(f"init={init} → fit a0={a_fit[0]:.4f}, a1={a_fit[1]:.4f},"
              f" iter={len(hist['ss'])}, final SS={hist['ss'][-1]:.2e}")
    except Exception as e:
        print(f"init={init} → failed with error: {e}")

plt.figure(figsize=(6, 4))
for init, _, H in results:
    plt.plot(H['ss'], label=f"init={init}")
plt.yscale('log')
plt.xlabel("Iteration")
plt.ylabel("Sum of squares")
plt.legend()
plt.title("Convergence of SS")
plt.tight_layout()
plt.show()

if results:
    _, _, H0 = results[0]
    eps = np.array(H0['eps'])
    if eps.size > 0:
        plt.figure(figsize=(6, 4))
        plt.plot(eps[:, 0], marker='o', label='ε₀')
        plt.plot(eps[:, 1], marker='s', label='ε₁')
        plt.xlabel("Iteration")
        plt.ylabel("Parameter increment")
        plt.legend()
        plt.title(f"Parameter updates (init={results[0][0]})")
        plt.tight_layout()
        plt.show()

x_plot = np.linspace(0, 2.5, 200)
plt.figure(figsize=(6, 4))
plt.scatter(x_data, y_data, color='k', label='data')

for init, _, H in results:
    for a in H['a']:
        plt.plot(x_plot, model(a, x_plot), alpha=0.2)
    plt.plot(x_plot, model(H['a'][-1], x_plot),
             label=f"final (init={init})", linewidth=2)

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Fit evolution and final curves")
plt.tight_layout()
plt.show()
