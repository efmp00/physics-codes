# Libraries
import numpy as np
import matplotlib.pyplot as plt

# Physical parameters
m = 1    # mass    (kg)
g = 9.81 # gravity (kg/m^2)
l = 1    # length  (m)

# Initial conditions
θ0 = np.pi/4 # Initial angle            (rad)
ω0 = 3.0       # Initial angular velocity (rad/s)

# Time parameters
t_0 = 0                       # Initial time (s)
t_f = 10                      # Final time   (s)
Δt = 0.01                     # Step size    (s)
n = int((t_f - t_0) / Δt) + 1 # Iterations
t = np.linspace(t_0, t_f, n)    # Time vector  (s)

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

# Differential equations
def f(t, θ, ω):
    return ω, -(m * g / l) * np.sin(θ)

# Initial arrays
θ, θ[0] = np.zeros(n), θ0
ω, ω[0] = np.zeros(n), ω0

# RK4 method evaluation
for i in range(n - 1):
    θ[i + 1], ω[i + 1] = RK4(f, t[i], θ[i], ω[i], Δt)

print(θ[1])

# Plot parameters
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 16})
# Plot
plt.plot(t, θ, label = r'Angle $\theta(t)$')
plt.plot(t, ω, label = r'Angular velocity $\omega(t)$')
plt.title(r'\textbf{Simple pendulum}', y = 1.1, fontsize = 24)
plt.xlabel(r'$t$ (s)', fontsize = 20)
plt.legend(loc = 'upper right', frameon = False)
plt.xlim((min(t), max(t)))
plt.ylim((min(ω), max(ω) + 4))
plt.tight_layout()
plt.gcf()
plt.savefig("pyplot.pdf")