# Libraries
import numpy as np
import matplotlib.pyplot as plt

# Plot parameters
plt.rcParams.update({
    'lines.linewidth': 2,     # linewidth
    'text.usetex': True,      # LaTeX font
    'font.family': 'serif',   # Serif family
    'font.size': 16,          # font size
    'axes.titlesize': 20,     # title size
    'axes.grid': True,        # grid
    'grid.linestyle': "-.",   # grid style
})

# Physical parameters
Ω = 1  # Rabi frequency 
Δ = 1  # Detuning

# Time parameters
t_0 = 0                        # initial time (s)
t_f = 10                       # final time   (s)
Δt = 0.01                      # step size    (s)
n = int((t_f - t_0) / Δt) + 1  # iterations
t = np.linspace(t_0, t_f, n)   # time vector  (s)

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
def f(t, c1, c2):
    return complex(-1j * Ω * np.exp(1j * Δ * t) * c2), complex(-1j * Ω * np.exp(-1j * Δ * t) * c1)

# Initial arrays
c1, c1[0] = np.zeros(n, dtype = 'complex'), 1 + 0j
c2, c2[0] = np.zeros(n, dtype = 'complex'), 0 + 0j

# RK4 method evaluation
for i in range(n - 1):
    c1[i + 1], c2[i + 1] = RK4(f, t[i], c1[i], c2[i], Δt)

# Plot
plt.figure(figsize = (10, 4.5))
plt.plot(t, np.abs(c1)**2, label = r'$|\tilde{c}_{1}(t)|^2$')
plt.plot(t, np.abs(c2)**2, label = r'$|\tilde{c}_{2}(t)|^2$')
plt.plot(t, np.abs(c1)**2 + np.abs(c2)**2, label = r'$|\tilde{c}_{1}(t)|^2 + |\tilde{c}_{1}(t)|^2$')
plt.title(r'Rabi oscillations in an optically-driven two-level atom')
plt.xlabel(r'$t(s)$')
plt.ylabel(r'Population')
plt.legend(loc = 'lower right', borderpad = 0.2)
plt.tight_layout()
plt.savefig('pyplot2.pdf')