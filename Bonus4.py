import numpy as np
import matplotlib.pyplot as plt

def target_function(x):
    return np.exp(-x) - x

def derivative(x):
    return -np.exp(-x) - 1

ACTUAL_ROOT = 0.56714329

def percent_error(value):
    return abs((ACTUAL_ROOT - value) / ACTUAL_ROOT) * 100

def bisection_generator(interval_start, interval_end, tolerance=1e-6, iterations=20):
    a, b = interval_start, interval_end
    for i in range(iterations):
        mid = (a + b) / 2.0
        yield mid
        if abs(b - a) < tolerance:
            break
        if target_function(a) * target_function(mid) < 0:
            b = mid
        else:
            a = mid

def false_position_generator(interval_start, interval_end, tolerance=1e-6, iterations=20):
    a, b = interval_start, interval_end
    c = a
    for i in range(iterations):
        fa, fb = target_function(a), target_function(b)
        c = (a * fb - b * fa) / (fb - fa)
        yield c
        if abs(target_function(c)) < tolerance:
            break
        if fa * target_function(c) < 0:
            b = c
        else:
            a = c

def secant_generator(first_guess, second_guess, tolerance=1e-6, iterations=20):
    x0, x1 = first_guess, second_guess
    for i in range(iterations):
        f0, f1 = target_function(x0), target_function(x1)
        if f1 == f0:
            break
        x_next = x1 - f1 * (x1 - x0) / (f1 - f0)
        yield x_next
        if abs(x_next - x1) < tolerance:
            break
        x0, x1 = x1, x_next

def newton_generator(initial_guess, tolerance=1e-6, iterations=20):
    x = initial_guess
    for i in range(iterations):
        f_value = target_function(x)
        df_value = derivative(x)
        if df_value == 0:
            break
        x_next = x - f_value / df_value
        yield x_next
        if abs(x_next - x) < tolerance:
            break
        x = x_next

class RootFindingManager:
    def __init__(self):
        self.methods = {
            "Bisection Method": (bisection_generator, {"interval_start": 0.0, "interval_end": 1.0}),
            "False Position": (false_position_generator, {"interval_start": 0.0, "interval_end": 1.0}),
            "Secant Method": (secant_generator, {"first_guess": 0.0, "second_guess": 1.0}),
            "Newton-Raphson": (newton_generator, {"initial_guess": 0.5})
        }
        self.results = {}

    def run_all_methods(self):
        for method_name, (generator_func, params) in self.methods.items():
            error_history = []
            for approximation in generator_func(**params):
                error_history.append(percent_error(approximation))
            self.results[method_name] = error_history

    def visualize_results(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_yscale('log')
        ax.set_xlabel('Iteration Number')
        ax.set_ylabel('Relative Error (%)')
        ax.set_title('Convergence Comparison of Root-Finding Methods')
        ax.grid(True, alpha=0.3, linestyle=':')

        for method_name, errors in self.results.items():
            iterations = range(1, len(errors) + 1)
            ax.plot(iterations, errors, label=method_name, marker='.')

        ax.legend(frameon=True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    manager = RootFindingManager()
    manager.run_all_methods()
    manager.visualize_results()