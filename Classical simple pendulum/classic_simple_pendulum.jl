# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Edy Alberto Flores Leal
# Date:   16/October/2023
# CONTEXT
# Practical example of the usage of the Fourth-Order Runge-Kutta Method. In this code, we provide a numerical approach to the Classical Simple Pendulum system. We
# solve the differential equation θ'' + (mg/l)sin(θ) = 0 obtained from the Euler-Lagrange equations using this numerical method. For more details about the
# theoretical and numerical approach, please consult the pdf available in this folder. 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Libraries
using LinearAlgebra
using Plots, ColorSchemes, LaTeXStrings

# Physical parameters
m = 1    # mass    (kg)
g = 9.81 # gravity (kg/m^2)
l = 1    # length  (m)

# Initial conditions
θ₀ = π/4 # initial angle            (rad)
ω₀ = 3.0 # initial angular velocity (rad/s)

# Time parameters
t₀ = 0                        # initial time (s)
t₁ = 10                       # final time   (s)
Δt = 0.01                     # step size    (s)
n = Int64((t₁ - t₀) / Δt) + 1 # iterations
t = t₀:Δt:(t₁ - Δt)           # time array   (s)

# Fourth-Order Runge-Kutta function
function RK4(f, x₀, y₀, z₀, h)
    k1y = h * f(x₀, y₀, z₀)[1]
    k1z = h * f(x₀, y₀, z₀)[2]
    #
    k2y = h * f(x₀ + h / 2, y₀ + k1y / 2, z₀ + k1z / 2)[1]
    k2z = h * f(x₀ + h / 2, y₀ + k1y / 2, z₀ + k1z / 2)[2]
    #
    k3y = h * f(x₀ + h / 2, y₀ + k2y / 2, z₀ + k2z / 2)[1]
    k3z = h * f(x₀ + h / 2, y₀ + k2y / 2, z₀ + k2z / 2)[2]
    #
    k4y = h * f(x₀ + h, y₀ + k3y, z₀ + k3z)[1]
    k4z = h * f(x₀ + h, y₀ + k3y, z₀ + k3z)[2]

    # Approximation
    y1 = Float64(y₀ + (k1y + 2 * k2y + 2 * k3y + k4y) / 6)
    z1 = Float64(z₀ + (k1z + 2 * k2z + 2 * k3z + k4z) / 6)
    return y1, z1
end

# Differential equations
# Remember that we start with a Nonlinear Second-Order ODE θ'' + (mg/l)sin(θ) = 0. To use the RK4 method, we transform this ODE into two First-Order ODE:
# dθ/dt = ω
# dω/dt = d^2θ/dt^2 = -(mg/l)sin(θ)

function f(t, θ, ω)
    return ω, -(m * g / l) * sin(θ)
end

# Initial arrays
θ, θ[1] = zeros(n), θ₀
ω, ω[1] = zeros(n), ω₀

# RK4 method evaluation
for i in 1:(n - 1)
    θ[i + 1], ω[i + 1] = RK4(f, t[i], θ[i], ω[i], Δt)
end

# Plot
plot(t, [θ, ω], grid = true, label = [L" Angle $\theta(t)$" L"Angular velocity $\omega(t)$"], linewidth = 2)
plot!(legend =:topright)
xlabel!(L"$\theta(t)$")
xlims!(0, 10)
ylims!(-4,6)
title!(L"$\mathbf{Simple}\:\,\mathbf{pendulum}$")
savefig("juliaplot.pdf")
