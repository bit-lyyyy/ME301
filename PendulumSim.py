# Aurthors: Adiv Ish-Shalom & Pavan Kapoor
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
g = 9.81  # Gravity (m/s^2)
L = 0.31   # Pendulum length (m)
T_end = 10  # Simulation time (s)

# Initial conditions for small (10°) and large (45°) angles
theta0_small = np.radians(10)
theta0_large = np.radians(45)
omega0 = 0  # Initial angular velocity

# Define the nonlinear and linearized equations of motion
def nonlinear_pendulum(t, y, L):
    theta, omega = y
    dydt = [omega, - (g / L) * np.sin(theta)]
    return dydt

def linear_pendulum(t, y, L):
    theta, omega = y
    dydt = [omega, - (g / L) * theta]  # Linearized sin(theta) ≈ theta
    return dydt

# Time span
t_span = (0, T_end)
t_eval = np.linspace(0, T_end, 1000)

# Solve for small angle (10°)
sol_nl_small = solve_ivp(nonlinear_pendulum, t_span, [theta0_small, omega0], t_eval=t_eval, args=(L,))
sol_l_small = solve_ivp(linear_pendulum, t_span, [theta0_small, omega0], t_eval=t_eval, args=(L,))

# Solve for large angle (45°)
sol_nl_large = solve_ivp(nonlinear_pendulum, t_span, [theta0_large, omega0], t_eval=t_eval, args=(L,))
sol_l_large = solve_ivp(linear_pendulum, t_span, [theta0_large, omega0], t_eval=t_eval, args=(L,))

# Plot results for small angle
plt.figure(figsize=(10,5))
plt.plot(sol_nl_small.t, np.degrees(sol_nl_small.y[0]), label='Nonlinear (10°)', linestyle='-', color='b')
plt.plot(sol_l_small.t, np.degrees(sol_l_small.y[0]), label='Linear (10°)', linestyle='--', color='r')
plt.xlabel('Time (s)')
plt.ylabel('Angular Displacement (degrees)')
plt.title('Pendulum Motion Numerical Analysis: Small Initial Angle (10°)')
plt.legend()
plt.grid()
plt.show()

# Plot results for large angle
plt.figure(figsize=(10,5))
plt.plot(sol_nl_large.t, np.degrees(sol_nl_large.y[0]), label='Nonlinear (45°)', linestyle='-', color='b')
plt.plot(sol_l_large.t, np.degrees(sol_l_large.y[0]), label='Linear (45°)', linestyle='--', color='r')
plt.xlabel('Time (s)')
plt.ylabel('Angular Displacement (degrees)')
plt.title('Pendulum Motion Numerical Analysis: Large Initial Angle (45°)')
plt.legend()
plt.grid()
plt.show()
