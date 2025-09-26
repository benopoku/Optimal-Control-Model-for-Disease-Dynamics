import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# This code implements an optimal control model for a disease.
# It uses an ODE system to model the population dynamics in different compartments
# (Susceptible, Protected, Carrier, Infectious, Recovered) and aims to minimize
# the number of infectious individuals through vaccination and treatment strategies.

# Parameters
Lambda = 1e5  # Recruitment rate (new individuals entering the susceptible population)
pi = 0.3      # Proportion of newborns directly protected (e.g., through maternal immunity or immediate vaccination)
eta = 0.1     # Rate of loss of immunity (from recovered state)
varphi = 0.05 # Rate of loss of protection (from protected state)
mu = 0.01     # Natural mortality rate (in all compartments)
lambda_A = 0.02 # Rate of acquiring carrier status (from susceptible or protected)
theta_A = 0.5  # Transition rate from protected to carrier (additional risk for protected individuals)
q_A = 0.1      # Proportion of carriers becoming infectious
gamma_A = 0.7  # Fraction of carriers progressing to infectious state (the rest recover)
delta_1 = 0.05 # Recovery rate for treated infectious individuals
d_A = 0.03     # Disease-induced mortality (in infectious state)
initial_conditions = [1e5, 5e4, 1e4, 1e3, 0]  # Initial population in [S, P, C_A, I_A, R] compartments

T = 80  # Simulation time (in days or other time units)
t_eval = np.linspace(0, T, T+1)  # Time points for evaluation of the ODE solution

# ODE system defining the rate of change in each compartment
def disease_model(t, y, controls, params):
    # y is a list or array representing the population in each compartment at time t
    S, P, C_A, I_A, R = y
    # controls is a function that returns the control values (omega and tau) at time t
    omega, tau = controls(t)
    # params is a list or array of the model parameters
    Lambda, pi, eta, varphi, mu, lambda_A, theta_A, q_A, gamma_A, delta_1, d_A = params

    # Differential equations for each compartment
    dSdt = (1 - pi) * Lambda + eta * R + varphi * P - (mu + omega + lambda_A) * S
    dPdt = pi * Lambda + omega * S - (varphi + mu + theta_A * lambda_A) * P
    dCAdt = lambda_A * S + lambda_A * theta_A * P - (mu + q_A) * C_A
    dIAdt = q_A * gamma_A * C_A - (mu + d_A + delta_1 + tau) * I_A
    dRdt = (delta_1 + tau) * I_A + q_A * (1 - gamma_A) * C_A - (mu + eta) * R

    return [dSdt, dPdt, dCAdt, dIAdt, dRdt]

# Cost function to be minimized
def cost_function(controls, solution, params, control_costs):
    # Transpose solution to get time series for each compartment
    solution = solution.T
    # Extract population dynamics over time for each compartment
    S, P, C_A, I_A, R = solution[:, 0], solution[:, 1], solution[:, 2], solution[:, 3], solution[:, 4]
    # Evaluate controls over the time points to get omega and tau values
    omega_vals, tau_vals = controls(t_eval)
    # Calculate the cost of infectious individuals over time (area under the curve)
    cost_infectious = np.trapz(I_A, dx=1)  # Minimize infectious individuals
    # Calculate the cost of applying controls (quadratic cost)
    cost_controls = np.trapz(omega_vals**2 + tau_vals**2, dx=1) * control_costs
    return cost_infectious + cost_controls # Total cost is the sum of infectious cost and control cost

# Control functions: interpolate control values at any given time t
def controls(t, omega_vals, tau_vals):
    # np.interp linearly interpolates the control values at the given time points
    omega = np.interp(t, t_eval, omega_vals)
    tau = np.interp(t, t_eval, tau_vals)
    return omega, tau

# Objective function for the optimization process
def objective(u):
    # u is the vector of control values to be optimized
    omega_vals = u[:len(t_eval)] # First half of u is vaccination rates
    tau_vals = u[len(t_eval):] # Second half of u is treatment rates
    # Create a control function using the current control values
    control_func = lambda t: controls(t, omega_vals, tau_vals)
    params = [Lambda, pi, eta, varphi, mu, lambda_A, theta_A, q_A, gamma_A, delta_1, d_A]

    # Solve the ODE system with the current control function
    sol = solve_ivp(
        lambda t, y: disease_model(t, y, control_func, params),
        [0, T], initial_conditions, t_eval=t_eval, vectorized=True
    )
    # Calculate the cost for the current solution
    return cost_function(control_func, sol.y, params, control_costs=0.1)

# Initial guess for the control functions (constant low rates)
omega0 = np.zeros_like(t_eval) + 0.01  # Initial vaccination rate guess
tau0 = np.zeros_like(t_eval) + 0.01    # Initial treatment rate guess
u0 = np.concatenate([omega0, tau0]) # Combine initial guesses into a single vector

# Bounds for the control variables (vaccination and treatment rates are between 0 and 0.1)
bounds = [(0, 0.1)] * len(t_eval) + [(0, 0.1)] * len(t_eval)

# Minimize the objective function using the L-BFGS-B method
result = minimize(objective, u0, bounds=bounds, method='L-BFGS-B')

# Extract optimal controls from the optimization result
u_optimal = result.x
omega_optimal = u_optimal[:len(t_eval)]
tau_optimal = u_optimal[len(t_eval):]

# Create the optimal control function
control_func_opt = lambda t: controls(t, omega_optimal, tau_optimal)

# Simulate the disease dynamics using the optimal controls
params = [Lambda, pi, eta, varphi, mu, lambda_A, theta_A, q_A, gamma_A, delta_1, d_A]
solution = solve_ivp(
    lambda t, y: disease_model(t, y, control_func_opt, params),
    [0, T], initial_conditions, t_eval=t_eval, vectorized=True
)

# Extract population dynamics for plotting
S, P, C_A, I_A, R = solution.y

# Plot the results
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(t_eval, S, label='Susceptible')
plt.plot(t_eval, P, label='Protected')
plt.plot(t_eval, C_A, label='Carrier_A')
plt.plot(t_eval, I_A, label='Infectious_A')
plt.plot(t_eval, R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.legend()
plt.title('Population Dynamics')
plt.savefig('population_dynamics.png')  # Save the first plot
plt.show()


plt.figure(figsize=(12, 8))
plt.plot(t_eval, omega_optimal, label='Vaccination Rate (omega)', color='orange')
plt.xlabel('Time (days)')
plt.ylabel('Control: Vaccination Rate')
plt.legend()
plt.savefig('vaccination_rate.png')  # Save the second plot (corrected filename)
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(t_eval, tau_optimal, label='Treatment Rate (tau)', color='red')
plt.xlabel('Time (days)')
plt.ylabel('Control: Treatment Rate')
plt.legend()
plt.savefig('treatment_rate.png')  # Save the third plot

#plt.tight_layout() # This was commented out, keeping it that way
plt.show()
