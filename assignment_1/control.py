"""
Example control_approx_linearization.py
Author: Nathan Duncan
Adapted from:
    Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
    GitHub: https://github.com/botprof/agv-examples
"""
### SIMULATION SETUP ###

import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from models import FourWheelSteered
from integration import rk_four

# dimensionality
n = 4
m = 2

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 20.0
T = 0.04

# Create an array of time values [s]
t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)

### COMPUTE THE REFERENCE TRAJECTORY ###
# desired start point
x_0_d = 0
y_0_d = 0
# desired velocity of the centre point (x,y)
V_DES = 20/3.6      # [m/s] 
# desired heading of the robot
THETA_DES = np.pi/4 # [rad]
# desired steering angle
PHI_DES = 0         # [rad]

V_F_DES = V_DES*np.cos(PHI_DES)

# Pre-compute the desired trajectory
x_d = np.zeros((n, N))
u_d = np.zeros((m, N))
x_d[0, 0] = x_0_d
x_d[1, 0] = y_0_d
x_d[2, 0] = THETA_DES
x_d[3, 0] = PHI_DES
for k in range(1, N):
    x_d[0, k] = x_d[0, 0] + V_DES * t[k] * np.cos(THETA_DES)
    x_d[1, k] = x_d[1, 0] + V_DES * t[k] * np.sin(THETA_DES)
    x_d[2, k] = THETA_DES
    x_d[3, k] = PHI_DES

### VEHICLE SETUP ###
# Set the track length of the vehicle
ELL_T = 0.5 # [m]
ELL_W = 1.5

# Create a vehicle of type FourSteer
vehicle = FourWheelSteered(ELL_W, ELL_T)

### SIMULATE THE CLOSED-LOOP SYSTEM ###
# initial conditions
# q_0 = np.array([0, 0, 3*np.pi/2.0, 0])
q_0 = np.array([0, 0, 0, 0])

# Setup some arrays
x = np.zeros((n, N))
u = np.zeros((m, N))
x[:, 0] = q_0

for k in range(1, N):
    # Simulate the vehicle motion
    x[:, k] = rk_four(vehicle.f, x[:, k - 1], u[:, k - 1], T)

    # Compute the approximate linearization 
    A = np.array(
        [
            [0, 0, -V_F_DES*np.cos(PHI_DES)*np.sin(THETA_DES), -V_F_DES*np.sin(PHI_DES)*np.cos(THETA_DES)],
            [0, 0,  V_F_DES*np.cos(PHI_DES)*np.cos(THETA_DES), -V_F_DES*np.sin(PHI_DES)*np.sin(THETA_DES)],
            [0, 0,                                          0,  V_F_DES*np.cos(PHI_DES)/ELL_T],
            [0, 0, 0, 0]
        ]
    )
    B = np.array(
        [
            [np.cos(PHI_DES)*np.cos(THETA_DES), 0],
            [np.cos(PHI_DES)*np.sin(THETA_DES), 0],
            [            np.sin(PHI_DES)/ELL_T, 0],
            [                                0, 1]
        ]
    )

    # Compute the gain matrix to place poles of (A - BK) at p
    p = [-1.0, -2.0, -0.2, -0.5]
    K = signal.place_poles(A, B, p)

    # Compute the controls (v_F, v_2)
    u[:, k] = u_d[:, k] - K.gain_matrix @ (x[:, k - 1] - x_d[:, k - 1])

### MAKE PLOTS ###
# Change some plot settings (optional)
# plt.rc("text", usetex=True)
# plt.rc("text.latex", preamble=r"\usepackage{cmbright,amsmath,bm}")
# plt.rc("savefig", format="pdf")
# plt.rc("savefig", bbox="tight")

# Plot the states as a function of time
fig1 = plt.figure(1)
fig1.set_figheight(6.4)
ax1a = plt.subplot(411)
plt.plot(t, x_d[0, :], "C1--")
plt.plot(t, x[0, :], "C0")
plt.grid(color="0.95")
plt.ylabel(r"$x$ [m]")
plt.setp(ax1a, xticklabels=[])
plt.legend(["Desired", "Actual"])
ax1b = plt.subplot(412)
plt.plot(t, x_d[1, :], "C1--")
plt.plot(t, x[1, :], "C0")
plt.grid(color="0.95")
plt.ylabel(r"$y$ [m]")
plt.setp(ax1b, xticklabels=[])
ax1c = plt.subplot(413)
plt.plot(t, x_d[2, :] * 180.0 / np.pi, "C1--")
plt.plot(t, x[2, :] * 180.0 / np.pi, "C0")
plt.grid(color="0.95")
plt.ylabel(r"$\theta$ [deg]")
plt.setp(ax1c, xticklabels=[])
ax1d = plt.subplot(414)
plt.step(t, u[0, :], "C2", where="post", label="$v_L$")
plt.step(t, u[1, :], "C3", where="post", label="$v_R$")
plt.grid(color="0.95")
plt.ylabel(r"$\bm{u}$ [m/s]")
plt.xlabel(r"$t$ [s]")
plt.legend()

# Save the plot
# plt.savefig("control_approx_linearization_fig1.pdf")

# Plot the position of the vehicle in the plane
fig2 = plt.figure(2)
plt.plot(x_d[0, :], x_d[1, :], "C1--", label="Desired")
plt.plot(x[0, :], x[1, :], "C0", label="Actual")
plt.axis("equal")
X_BL, Y_BL, X_BR, Y_BR, X_FL, Y_FL, X_FR, Y_FR, X_BD, Y_BD = vehicle.draw(x[:, 0])
plt.fill(X_BL, Y_BL, "k")
plt.fill(X_BR, Y_BR, "k")
plt.fill(X_FL, Y_FL, "k")
plt.fill(X_FR, Y_FR, "k")
plt.fill(X_BD, Y_BD, "C2", alpha=0.5, label="Start")
X_BL, Y_BL, X_BR, Y_BR, X_FL, Y_FL, X_FR, Y_FR, X_BD, Y_BD = vehicle.draw(x[:, N - 1])
plt.fill(X_BL, Y_BL, "k")
plt.fill(X_BR, Y_BR, "k")
plt.fill(X_FL, Y_FL, "k")
plt.fill(X_FR, Y_FR, "k")
plt.fill(X_BD, Y_BD, "C3", alpha=0.5, label="End")
plt.xlabel(r"$x$ [m]")
plt.ylabel(r"$y$ [m]")
plt.legend()

# Save the plot
# plt.savefig("control_approx_linearization_fig2.pdf")

# Show the plots to the screen
plt.show()

### MAKE AN ANIMATION ###

# Create the animation
ani = vehicle.animate_trajectory(x, x_d, T)

# Create and save the animation
# ani = vehicle.animate_trajectory(
#     x, x_d, T, True, "../agv-book/gifs/ch4/control_approx_linearization.gif"
# )

# Show all the plots to the screen
plt.show()