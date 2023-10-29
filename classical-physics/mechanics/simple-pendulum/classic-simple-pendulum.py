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
g = 9.81 # gravity (kg/m^2)
l = 1    # length  (m)

# Initial conditions
θ0 = np.pi/4   # initial angle            (rad)
ω0 = 3.0       # initial angular velocity (rad/s)

# Time parameters
t_0 = 0                       # initial time (s)
t_f = 10                      # final time   (s)
Δt = 0.01                     # step size    (s)
n = int((t_f - t_0) / Δt) + 1 # iterations
t = np.linspace(t_0, t_f, n)  # time array  (s)

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
    y1 = float(y0 + (k1y + 2 * k2y + 2 * k3y + k4y) / 6)
    z1 = float(z0 + (k1z + 2 * k2z + 2 * k3z + k4z) / 6)
    return y1, z1

# Differential Equations
def f(t, θ, ω):
    return ω, -(g / l) * np.sin(θ)

# Initial arrays
θ, θ[0] = np.zeros(n), θ0
ω, ω[0] = np.zeros(n), ω0

# RK4 method evaluation
for i in range(n - 1):
    θ[i + 1], ω[i + 1] = RK4(f, t[i], θ[i], ω[i], Δt)

plt.figure(figsize = (10, 4.5))

# Plot
plt.subplot(1, 2, 1)
plt.plot(t, θ, 'b-')
plt.xlabel(r'$t$ (s)')
plt.ylabel(r'$\theta(t)$ [rad]')
plt.title(r'Angle')

plt.subplot(1, 2, 2)
plt.plot(t, ω, 'r-')
plt.xlabel(r'$t$ (s)')
plt.ylabel(r'$\omega(t)$ [rad/s]')
plt.title(r'Angular velocity')
plt.tight_layout()
plt.gcf()
plt.savefig("pyplot2.pdf")
