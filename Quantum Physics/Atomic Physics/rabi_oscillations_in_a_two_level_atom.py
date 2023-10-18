# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Edy Alberto Flores Leal
# Date:    17/October/2023
# CONTEXT
# Practical example of the usage of the Fourth-Order Runge-Kutta Method. In this code, we provide a numerical approach to simulate the probability evolution through
# time in a two-level atom. For more details about the theoretical and numerical approach, please consult the pdf available in this folder. 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Libraries
import numpy as np
import matplotlib.pyplot as plt

# Physical parameters
Ω = 1  # Rabi frequency 
Δ = 1  # Detuning

# Time parameters
t_0 = 0                        # Initial time (s)
t_f = 10                       # Final time   (s)
Δt = 0.01                      # Step size    (s)
n = int((t_f - t_0) / Δt) + 1  # Iterations
t = np.linspace(t_0, t_f, n)   # Time vector  (s)

# Fourth-Order Runge-Kutta function
def RK4(f, x0, y0, z0, h):
    k1y = h * f(x0, y0, z0)[0]
    k1z = h * f(x0, y0, z0)[1]
    #
    k2y = h * f(x0 + h / 2, y0 + k1y / 2, z0 + k1z / 2)[0]
    k2z = h * f(x0 + h / 2, y0 + k1y / 2, z0 + k1z / 2)[1]
    #
    k3y = h * f(x0 + h / 2, y0 + k2y / 2, z0 + k2z / 2)[0]
    k3z = h * f(x0 + h / 2, y0 + k2y / 2, z0 + k2z / 2)[1]
    #
    k4y = h * f(x0 + h, y0 + k3y, z0 + k3z)[0]
    k4z = h * f(x0 + h, y0 + k3y, z0 + k3z)[1]

    # Approximation
    y1 = y0 + (k1y + 2 * k2y + 2 * k3y + k4y) / 6
    z1 = z0 + (k1z + 2 * k2z + 2 * k3z + k4z) / 6
    return y1, z1

# Differential Equations
# dc1/dt = -i * Ω * exp(i * Δ * t) * c2
# dc2/dt = -i * Ω * exp(-i * Δ * t) * c1

def f(t, c1, c2):
    return complex(-1j * Ω * np.exp(1j * Δ * t) * c2), complex(-1j * Ω * np.exp(-1j * Δ * t) * c1)

# Initial arrays
c1, c1[0] = np.zeros(n, dtype = 'complex'), 1 + 0j
c2, c2[0] = np.zeros(n, dtype = 'complex'), 0 + 0j

# RK4 method evaluation
for i in range(n - 1):
    c1[i + 1], c2[i + 1] = RK4(f, t[i], c1[i], c2[i], Δt)

# Plot parameters
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 16})

# Plot
plt.plot(t, np.abs(c1)**2, label = r'$|\tilde{c}_{1}(t)|$')
plt.plot(t, np.abs(c2)**2, label = r'$|\tilde{c}_{2}(t)|$')
plt.plot(t, np.abs(c1)**2 + np.abs(c2)**2, label = r'$|\tilde{c}_{1}(t)| + |\tilde{c}_{1}(t)|$')
plt.title(r'\textbf{Rabi oscillations in an optically-driven two-level atom}', y = 1.1, fontsize = 24)
plt.xlabel(r'$t(s)$', fontsize = 20)
plt.ylabel(r'Population', fontsize = 20)
plt.legend()
plt.tight_layout()
plt.show()
