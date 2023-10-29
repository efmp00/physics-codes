# Libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sc

# Plot parameters
plt.rcParams.update({
    'lines.linewidth': 2,     # linewidth
    'text.usetex': True,      # LaTeX font
    'font.family': 'serif',   # Serif family
    'font.size': 20,          # font size
    'axes.titlesize': 20,     # title size
    'axes.grid': True,        # grid
    'grid.linestyle': "-.",   # grid style
})

# Physical parameters
# 2a = 0.5 mm, 2d = 2.0 mm, λ = 0.55 μm
a = 2.5e-4 # width                     (m)
λ = 5.5e-7 # wavelength                (m)
d = 1.0e-3 # interior edges separation (m) 

# Irradiance evaluation function
def irradiance(xmax, λ, D, d):
    x = np.linspace(-xmax, xmax, 1000) # domain of integration 
    X = x/xmax                         # normalized distance
    # Fresnel parameters
    α = np.sqrt(2 / (λ * D)) * (x + d + 2 * a)
    β = np.sqrt(2 / (λ * D)) * (x + d)
    γ = np.sqrt(2 / (λ * D)) * (x - d)
    δ = np.sqrt(2 / (λ * D)) * (x - d - 2 * a)
    # Fresnel integrals
    Sa, Ca = sc.fresnel(α)
    Sb, Cb = sc.fresnel(β)
    Sg, Cg = sc.fresnel(γ)
    Sd, Cd = sc.fresnel(δ)
    return X, 0.25 * (np.power(Ca + Cg - Cb - Cd, 2) + np.power(Sa + Sg - Sb - Sd, 2))

# --- Cases ---
# Case 1: xmax = 0.00175 m ; D = 0.016 m
X1, I1 = irradiance(1.75e-3, λ, 1.6e-2, d)

# Case 2: xmax = 0.00440 m ; D = 1.62 m
X2, I2 = irradiance(4.45e-3, λ, 1.62, d)

# Case 3: xmax = 0.01780 m ; D = 6.46 m
X3, I3 = irradiance(1.78e-2, λ, 6.46, d)

# Plot
plt.figure(figsize = (16, 5.5))

plt.subplot(1, 3, 1)
plt.plot(X1, I1, 'r-')
plt.xlabel(r'$x/x_{\mathrm{max}}$')
plt.ylabel(r'$I/I_{0}$')
plt.ylim(-0.05, 1)
plt.title(r'$x_{\mathrm{max}} = 1.75\times 10^{-3}\:\mathrm{m}$, $D = 0.016\:\mathrm{m}$')

plt.subplot(1, 3, 2)
plt.plot(X2, I2, 'b-')
plt.xlabel(r'$x/x_{\mathrm{max}}$')
plt.ylabel(r'$I/I_{0}$')
plt.ylim(-0.05, 1)
plt.title(r'$x_{\mathrm{max}} = 4.45\times 10^{-3}\:\mathrm{m}$, $D = 1.62\:\mathrm{m}$')

plt.subplot(1, 3, 3)
plt.plot(X3, I3, 'g-')
plt.xlabel(r'$x/x_{\mathrm{max}}$')
plt.ylabel(r'$I/I_{0}$')
plt.ylim(-0.05, 1)
plt.title(r'$x_{\mathrm{max}} = 1.78\times 10^{-2}\:\mathrm{m}$, $D = 6.46\:\mathrm{m}$')

plt.suptitle(r'Diffracted irradiance', size = 24)
plt.tight_layout()
plt.gcf()
plt.savefig('Scipy-pyplot.pdf')
