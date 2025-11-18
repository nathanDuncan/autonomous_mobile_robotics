"""
Simulation of a Luenberger-like observer for a differential drive robot.
Author: Nathan Duncan <20ntd1@queensu.ca>
ADAPTED FROM:
    Example fws_beacons_observer.py
    Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
    GitHub: https://github.com/botprof/agv-examples 
"""

# %%
# SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from integration import rk_four
from models import DiffDrive

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 20.0
T = 0.1

# Create an array of time values [s]
t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)

# Noise Flag
NOISE = True
sigma = np.sqrt(0.02)  # Standard deviation from sigma^2 = 0.02

# %%
# VEHICLE SETUP

# Set the wheelbase and track of the vehicle [m]
ELL_W = 0.25
ELL_T = 0.25

# Let's now use the class Differential Drive for plotting
vehicle = DiffDrive(ELL_W)

# Desired inputs for the specified trajectory
v_L_des = 1.95/8.0 # m/s
v_R_des = 2.05/8.0 # m/s 

# %%
# CREATE A MAP OF FEATURES

# Set the minimum number of features in the map that achieves observability
N_BEACONS = 10

# Set the size [m] of a square map
D_MAP = 30.0

# Create a map of randomly placed feature locations (first two have know locations)
f_map = np.zeros((2, N_BEACONS))
f_map[:, 0] = (3.0, 4.0)
f_map[:, 1] = (-7.0, 3.0)
if N_BEACONS > 2:
    for i in range(2, N_BEACONS):
        f_map[:, i] = D_MAP * (np.random.rand(2) - 0.5)


# %%
# FUNCTION TO MODEL RANGE TO FEATURES

def range_sensor(q, f_map):
    """
    Function to model the range sensor.

    Parameters
    ----------
    q : ndarray
        An array of length 2 representing the robot's position.
    f_map : ndarray
        An array of size (2, N_BEACONS) containing map feature locations.

    Returns
    -------
    ndarray
        The range to each feature in the map.
    """

    # Compute the range to each feature from the current robot position
    r = np.zeros(N_BEACONS)
    for j in range(0, N_BEACONS):
        r[j] = np.sqrt((f_map[0, j] - q[0]) ** 2 + (f_map[1, j] - q[1]) ** 2)
        if NOISE:
            epsilon = np.random.randn()    # draw from N(0,1)
            r[j] = r[j] + sigma * epsilon

    # Return the array of measurements
    return r


# %%
# FUNCTION TO IMPLEMENT THE OBSERVER

def fws_observer(q, u, r, f_map):
    """
    Function to implement an observer for the robot's pose.

    Parameters
    ----------
    q : ndarray
        An array of length 4 representing the (last) robot's pose.
    u : ndarray
        An array of length 2 representing the robot's inputs.
    r : ndarray
        An array of length N_FEATURES representing the range to each feature.
    f_map : ndarray
        An array of size (2, N_FEATURES) containing map feature locations.

    Returns
    -------
    ndarray
        The estimated pose of the robot.
    """

    # Compute the Jacobian matrices (i.e., linearize about current estimate)
    F = np.zeros((3, 3))
    F = np.eye(3) + T * np.array(
        [
            [
                0,
                0,
                -1/2*np.sin(q[2])*(u[0] + u[1]),
            ],
            [
                0,
                0,
                1/2*np.cos(q[2])*(u[0] + u[1]),
            ],
            [0, 0, 0]
        ]
    )
    H = np.zeros((N_BEACONS, 3))
    for j in range(0, N_BEACONS):
        H[j, :] = np.array(
            [
                -(f_map[0, j] - q[0]) / range_sensor(q, f_map)[j],
                -(f_map[1, j] - q[1]) / range_sensor(q, f_map)[j],
                0,
            ]
        )

    # Check the observability of this system
    observability_matrix = H
    for j in range(1, 3):
        observability_matrix = np.concatenate(
            (observability_matrix, H @ np.linalg.matrix_power(F, j)), axis=0
        )
    if np.linalg.matrix_rank(observability_matrix) < 3:
        raise ValueError("System is not observable!")

    # Set the desired poles at lambda_z (change these as desired)
    lambda_z = np.array([0.9, 0.8, 0.7])
    # Compute the observer gain
    if N_BEACONS > 2:
        # Use the pseudo-inverse to compute the observer gain (when overdetermined)
        L = signal.place_poles(F.T, np.eye(3), lambda_z).gain_matrix @ np.linalg.pinv(H)
    else:
        L = signal.place_poles(F.T, H.T, lambda_z).gain_matrix.T

    # Predict the state using the inputs and the robot's kinematic model
    q_new = q + T * vehicle.f(q, u)
    # Correct the state using the range measurements
    q_new = q_new + L @ (r - range_sensor(q, f_map))

    # Return the estimated state
    return q_new


# %%
# RUN SIMULATION

# Initialize arrays that will be populated with our inputs and states
q = np.zeros((3, N))
u = np.zeros((2, N))
q_hat = np.zeros((3, N))

# Set the initial pose [m, m, rad, rad], velocities [m/s, rad/s]
q[0, 0] = 3.0
q[1, 0] = -2.0
q[2, 0] = -np.pi / 6.0
u[0, 0] = v_L_des
u[1, 0] = v_R_des

# Just drive around and try to localize!
for k in range(1, N):
    # Measure the actual range to each feature
    r = range_sensor(q[:, k - 1], f_map)
    # Use the range measurements to estimate the robot's state
    q_hat[:, k] = fws_observer(q_hat[:, k - 1], u[:, k - 1], r, f_map)
    # Choose some new inputs
    u[0, k] = v_L_des
    u[1, k] = v_R_des

    # Simulate the robot's motion
    q[:, k] = rk_four(vehicle.f, q[:, k - 1], u[:, k - 1], T)

# %%
# MAKE SOME PLOTS

# Function to wrap angles to [-pi, pi]
def wrap_to_pi(angle):
    """Wrap angles to the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

# Change some plot settings (optional)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{cmbright,amsmath,bm}")
plt.rc("savefig", format="pdf")
plt.rc("savefig", bbox="tight")

# Plot the position of the vehicle in the plane
fig1 = plt.figure(1)
plt.plot(f_map[0, :], f_map[1, :], "C4*", label="Feature")
plt.plot(q[0, :], q[1, :])
plt.axis("equal")
X_L, Y_L, X_R, Y_R, X_BD, Y_BD, X_C, Y_C = vehicle.draw(q[0, 0],q[1, 0],q[2, 0])
plt.fill(X_L, Y_L, "k")
plt.fill(X_R, Y_R, "k")
plt.fill(X_C, Y_C, "k")
plt.fill(X_BD, Y_BD, "C2", alpha=0.5, label="Start")
X_L, Y_L, X_R, Y_R, X_BD, Y_BD, X_C, Y_C = vehicle.draw(q[0, N - 1], q[1, N - 1], q[2, N - 1])
plt.fill(X_L, Y_L, "k")
plt.fill(X_R, Y_R, "k")
plt.fill(X_C, Y_C, "k")
plt.fill(X_BD, Y_BD, "C3", alpha=0.5, label="End")
plt.xlabel(r"$x$ [m]")
plt.ylabel(r"$y$ [m]")
plt.legend()
fig1.savefig("vehicle_position.png", dpi=300, bbox_inches="tight")

# Plot the states as a function of time
fig2 = plt.figure(2)
fig2.set_figheight(6.3)
ax2a = plt.subplot(311)
plt.plot(t, q[0, :], "C0", label="Actual")
plt.plot(t, q_hat[0, :], "C1--", label="Estimated")
plt.grid(color="0.95")
plt.ylabel(r"$x$ [m]")
plt.setp(ax2a, xticklabels=[])
plt.legend()
ax2b = plt.subplot(312)
plt.plot(t, q[1, :], "C0", label="Actual")
plt.plot(t, q_hat[1, :], "C1--", label="Estimated")
plt.grid(color="0.95")
plt.ylabel(r"$y$ [m]")
plt.setp(ax2b, xticklabels=[])
ax2c = plt.subplot(313)
plt.plot(t, wrap_to_pi(q[2, :]) * 180.0 / np.pi, "C0", label="Actual")
plt.plot(t, wrap_to_pi(q_hat[2, :]) * 180.0 / np.pi, "C1--", label="Estimated")
plt.ylabel(r"$\theta$ [deg]")
plt.grid(color="0.95")
plt.setp(ax2c)
plt.xlabel(r"$t$ [s]")
fig2.savefig("states_over_time.png", dpi=300, bbox_inches="tight")

# Show all the plots to the screen
plt.show()