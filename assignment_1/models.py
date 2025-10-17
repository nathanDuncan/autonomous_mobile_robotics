"""
Python module models.py for various vehicle models.
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples

This module contains classes for various vehicle models, including:
    * 1D vehicle class (i.e., a simple cart) [Cart]
"""

import numpy as np
import graphics
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import chi2
from matplotlib import patches


class FourWheelSteered:
    """Four-wheel steered vehicle class.

    Parameters
    ----------
    ell_W : float
        The wheelbase of the vehicle [m].
    ell_T : float
        The vehicle's track length [m].
    """

    def __init__(self, ell_W, ell_T):
        """Constructor method."""
        self.ell_W = ell_W
        self.ell_T = ell_T

    def f(self, x, u):
        """Four-wheel steered vehicle kinematic model.

        Parameters
        ----------
        x : ndarray of length 4
            The vehicle's state (x, y, theta, phi).
        u : ndarray of length 2
            The vehicle's front wheel speed and steering angle rate.

        Returns
        -------
        f : ndarray of length 4
            The rate of change of the vehicle states.
        """
        f = np.zeros(4)
        f[0] = u[0] * np.cos(x[2]) * np.cos(x[3])
        f[1] = u[0] * np.sin(x[2]) * np.cos(x[3])
        f[2] = u[0] * 1.0 / (self.ell_T) * np.sin(x[3])
        f[3] = u[1]
        return f

    def ackermann(self, x):
        """Computes the Ackermann steering angles.

        Parameters
        ----------
        x : ndarray of length 4
            The vehicle's state (x, y, theta, phi).

        Returns
        -------
        ackermann_angles : ndarray of length 2
            The left and right wheel angles (phi_L, phi_R).
        """
        phi_L = np.arctan(
            2 * self.ell_W * np.tan(x[3]) / (2 * self.ell_W - self.ell_T * np.tan(x[3]))
        )
        phi_R = np.arctan(
            2 * self.ell_W * np.tan(x[3]) / (2 * self.ell_W + self.ell_T * np.tan(x[3]))
        )
        ackermann_angles = np.array([phi_L, phi_R])
        return ackermann_angles

    def draw(self, x):
        """Finds points that draw a four-wheel steered vehicle.

        Parameters
        ----------
        x : ndarray of length 4
            The vehicle's state (x, y, theta, phi).

        The geometric centre of the vehicle is (x, y), the body has orientation
        theta, effective steering angle phi, wheelbase ell_W and track length
        ell_T.

        Returns X_BL, Y_BL, X_BR, Y_BR, X_FL, Y_FL, X_FR, Y_FR, X_BD, Y_BD,
        where L denotes left, R denotes right, B denotes back, F denotes front,
        and BD denotes the vehicle's body.
        """
        # Left and right back wheels
        X_BL, Y_BL = graphics.draw_rectangle(
            x[0] - 0.5 * self.ell_W * np.cos(x[2]) - 0.5 * self.ell_T * np.sin(x[2]),
            x[1] - 0.5 * self.ell_W * np.sin(x[2]) + 0.5 * self.ell_T * np.cos(x[2]),
            0.5 * self.ell_T,
            0.25 * self.ell_T,
            x[2] - self.ackermann(x)[0],
        )
        X_BR, Y_BR = graphics.draw_rectangle(
            x[0] - 0.5 * self.ell_W * np.cos(x[2]) + 0.5 * self.ell_T * np.sin(x[2]),
            x[1] - 0.5 * self.ell_W * np.sin(x[2]) - 0.5 * self.ell_T * np.cos(x[2]),
            0.5 * self.ell_T,
            0.25 * self.ell_T,
            x[2] - self.ackermann(x)[1],
        )
        # Left and right front wheels
        X_FL, Y_FL = graphics.draw_rectangle(
            x[0] + 0.5 * self.ell_W * np.cos(x[2]) - self.ell_T / 2 * np.sin(x[2]),
            x[1] + 0.5 * self.ell_W * np.sin(x[2]) + self.ell_T / 2 * np.cos(x[2]),
            0.5 * self.ell_T,
            0.25 * self.ell_T,
            x[2] + self.ackermann(x)[0],
        )
        X_FR, Y_FR = graphics.draw_rectangle(
            x[0] + 0.5 * self.ell_W * np.cos(x[2]) + self.ell_T / 2 * np.sin(x[2]),
            x[1] + 0.5 * self.ell_W * np.sin(x[2]) - self.ell_T / 2 * np.cos(x[2]),
            0.5 * self.ell_T,
            0.25 * self.ell_T,
            x[2] + self.ackermann(x)[1],
        )
        # Body
        X_BD, Y_BD = graphics.draw_rectangle(
            x[0],
            x[1],
            2.0 * self.ell_W,
            2.0 * self.ell_T,
            x[2],
        )
        # Return the arrays of points
        return X_BL, Y_BL, X_BR, Y_BR, X_FL, Y_FL, X_FR, Y_FR, X_BD, Y_BD

    def animate(
        self,
        x,
        T,
        save_ani=False,
        filename="animate_fourwheelsteered.gif",
    ):
        """Create an animation of an Ackermann steered (car-like) vehicle.

        Returns animation object for array of vehicle positions x with time
        increments T [s], wheelbase ell_W [m], and track ell_T [m].

        To save the animation to a GIF file, set save_ani to True and give a
        filename (default 'animate_ackermann.gif').
        """
        fig, ax = plt.subplots()
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$y$ [m]")
        plt.axis("equal")
        (line,) = ax.plot([], [], "C0")
        (BLwheel,) = ax.fill([], [], color="k")
        (BRwheel,) = ax.fill([], [], color="k")
        (FLwheel,) = ax.fill([], [], color="k")
        (FRwheel,) = ax.fill([], [], color="k")
        (body,) = ax.fill([], [], color="C0", alpha=0.5)
        time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

        def init():
            """A function that initializes the animation."""
            line.set_data([], [])
            BLwheel.set_xy(np.empty([5, 2]))
            BRwheel.set_xy(np.empty([5, 2]))
            FLwheel.set_xy(np.empty([5, 2]))
            FRwheel.set_xy(np.empty([5, 2]))
            body.set_xy(np.empty([5, 2]))
            time_text.set_text("")
            return line, BLwheel, BRwheel, FLwheel, FRwheel, body, time_text

        def movie(k):
            """The function called at each step of the animation."""
            # Draw the path followed by the vehicle
            line.set_data(x[0, 0 : k + 1], x[1, 0 : k + 1])
            # Draw the Ackermann steered drive vehicle
            X_BL, Y_BL, X_BR, Y_BR, X_FL, Y_FL, X_FR, Y_FR, X_BD, Y_BD = self.draw(
                x[:, k]
            )
            BLwheel.set_xy(np.transpose([X_BL, Y_BL]))
            BRwheel.set_xy(np.transpose([X_BR, Y_BR]))
            FLwheel.set_xy(np.transpose([X_FL, Y_FL]))
            FRwheel.set_xy(np.transpose([X_FR, Y_FR]))
            body.set_xy(np.transpose([X_BD, Y_BD]))
            # Add the simulation time
            time_text.set_text(r"$t$ = %.1f s" % (k * T))
            # Dynamically set the axis limits
            ax.set_xlim(x[0, k] - 10 * self.ell_W, x[0, k] + 10 * self.ell_W)
            ax.set_ylim(x[1, k] - 10 * self.ell_W, x[1, k] + 10 * self.ell_W)
            ax.figure.canvas.draw()
            # Return the objects to animate
            return line, BLwheel, BRwheel, FLwheel, FRwheel, body, time_text

        # Create the animation
        ani = animation.FuncAnimation(
            fig,
            movie,
            np.arange(1, len(x[0, :]), max(1, int(1 / T / 10))),
            init_func=init,
            interval=T * 1000,
            blit=True,
            repeat=False,
        )
        if save_ani is True:
            ani.save(filename, fps=min(1 / T, 10))
        # Return the figure object
        return ani