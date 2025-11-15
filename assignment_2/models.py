import numpy as np
import graphics

class DiffDrive:
    """Differential-drive vehicle class.

    Parameters
    ----------
    ell : float
        The track length of the vehicle [m].
    """

    def __init__(self, ell):
        """Constructor method."""
        self.ell = ell

    def f(self, x, u):
        """Differential drive kinematic vehicle kinematic model.

        Parameters
        ----------
        x : ndarray of length 3
            The vehicle's state (x, y, theta).
        u : ndarray of length 2
            The left and right wheel speeds (v_L, v_R).

        Returns
        -------
        f : ndarray of length 3
            The rate of change of the vehicle states.
        """
        f = np.zeros(3)
        f[0] = 0.5 * (u[0] + u[1]) * np.cos(x[2])
        f[1] = 0.5 * (u[0] + u[1]) * np.sin(x[2])
        f[2] = 1.0 / self.ell * (u[1] - u[0])
        return f

    def uni2diff(self, u_in):
        """
        Convert speed and angular rate inputs to differential drive wheel speeds.

        Parameters
        ----------
        u_in : ndarray of length 2
            The speed and turning rate of the vehicle (v, omega).

        Returns
        -------
        u_out : ndarray of length 2
            The left and right wheel speeds (v_L, v_R).
        """
        v = u_in[0]
        omega = u_in[1]
        v_L = v - self.ell / 2 * omega
        v_R = v + self.ell / 2 * omega
        u_out = np.array([v_L, v_R])
        return u_out

    def draw(self, x, y, theta):
        """
        Finds points that draw a differential drive vehicle.

        The centre of the wheel axle is (x, y), the vehicle has orientation
        theta, and the vehicle's track length is ell.

        Returns X_L, Y_L, X_R, Y_R, X_BD, Y_BD, X_C, Y_C, where L is for the
        left wheel, R for the right wheel, B for the body, and C for the caster.
        """
        # Left and right wheels
        X_L, Y_L = graphics.draw_rectangle(
            x - 0.5 * self.ell * np.sin(theta),
            y + 0.5 * self.ell * np.cos(theta),
            0.5 * self.ell,
            0.25 * self.ell,
            theta,
        )
        X_R, Y_R = graphics.draw_rectangle(
            x + 0.5 * self.ell * np.sin(theta),
            y - 0.5 * self.ell * np.cos(theta),
            0.5 * self.ell,
            0.25 * self.ell,
            theta,
        )
        # Body
        X_BD, Y_BD = graphics.draw_circle(x, y, self.ell)
        # Caster
        X_C, Y_C = graphics.draw_circle(
            x + 0.5 * self.ell * np.cos(theta),
            y + 0.5 * self.ell * np.sin(theta),
            0.125 * self.ell,
        )
        # Return the arrays of points
        return X_L, Y_L, X_R, Y_R, X_BD, Y_BD, X_C, Y_C

    def animate(self, x, T, save_ani=False, filename="animate_diffdrive.gif"):
        """Create an animation of a differential drive vehicle.

        Returns animation object for array of vehicle positions x with time
        increments T [s], track ell [m].

        To save the animation to a GIF file, set save_ani to True and provide a
        filename (default 'animate_diffdrive.gif').
        """
        fig, ax = plt.subplots()
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$y$ [m]")
        plt.axis("equal")
        (line,) = ax.plot([], [], "C0")
        (leftwheel,) = ax.fill([], [], color="k")
        (rightwheel,) = ax.fill([], [], color="k")
        (body,) = ax.fill([], [], color="C0", alpha=0.5)
        (castor,) = ax.fill([], [], color="k")
        time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

        def init():
            """Function that initializes the animation."""
            line.set_data([], [])
            leftwheel.set_xy(np.empty([5, 2]))
            rightwheel.set_xy(np.empty([5, 2]))
            body.set_xy(np.empty([36, 2]))
            castor.set_xy(np.empty([36, 2]))
            time_text.set_text("")
            return line, leftwheel, rightwheel, body, castor, time_text

        def movie(k):
            """Function called at each step of the animation."""
            # Draw the path followed by the vehicle
            line.set_data(x[0, 0 : k + 1], x[1, 0 : k + 1])
            # Draw the differential drive vehicle
            X_L, Y_L, X_R, Y_R, X_B, Y_B, X_C, Y_C = self.draw(
                x[0, k], x[1, k], x[2, k]
            )
            leftwheel.set_xy(np.transpose([X_L, Y_L]))
            rightwheel.set_xy(np.transpose([X_R, Y_R]))
            body.set_xy(np.transpose([X_B, Y_B]))
            castor.set_xy(np.transpose([X_C, Y_C]))
            # Add the simulation time
            time_text.set_text(r"$t$ = %.1f s" % (k * T))
            # Dynamically set the axis limits
            ax.set_xlim(x[0, k] - 10 * self.ell, x[0, k] + 10 * self.ell)
            ax.set_ylim(x[1, k] - 10 * self.ell, x[1, k] + 10 * self.ell)
            ax.figure.canvas.draw()
            # Return the objects to animate
            return line, leftwheel, rightwheel, body, castor, time_text

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

    def animate_trajectory(
        self, x, xd, T, save_ani=False, filename="animate_diffdrive.gif"
    ):
        """Create an animation of a differential drive vehicle with plots of
        actual and desired trajectories.

        Returns animation object for array of vehicle positions and desired
        positions x with time increments T [s], track ell [m].

        To save the animation to a GIF file, set save_ani to True and provide a
        filename (default 'animate_diffdrive.gif').
        """
        fig, ax = plt.subplots()
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$y$ [m]")
        plt.axis("equal")
        (desired,) = ax.plot([], [], "--C1")
        (line,) = ax.plot([], [], "C0")
        (leftwheel,) = ax.fill([], [], color="k")
        (rightwheel,) = ax.fill([], [], color="k")
        (body,) = ax.fill([], [], color="C0", alpha=0.5)
        (castor,) = ax.fill([], [], color="k")
        time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

        def init():
            """Function that initializes the animation."""
            desired.set_data([], [])
            line.set_data([], [])
            leftwheel.set_xy(np.empty([5, 2]))
            rightwheel.set_xy(np.empty([5, 2]))
            body.set_xy(np.empty([36, 2]))
            castor.set_xy(np.empty([36, 2]))
            time_text.set_text("")
            return desired, line, leftwheel, rightwheel, body, castor, time_text

        def movie(k):
            """Function called at each step of the animation."""
            # Draw the desired trajectory
            desired.set_data(xd[0, 0 : k + 1], xd[1, 0 : k + 1])
            # Draw the path followed by the vehicle
            line.set_data(x[0, 0 : k + 1], x[1, 0 : k + 1])
            # Draw the differential drive vehicle
            X_L, Y_L, X_R, Y_R, X_B, Y_B, X_C, Y_C = self.draw(
                x[0, k], x[1, k], x[2, k]
            )
            leftwheel.set_xy(np.transpose([X_L, Y_L]))
            rightwheel.set_xy(np.transpose([X_R, Y_R]))
            body.set_xy(np.transpose([X_B, Y_B]))
            castor.set_xy(np.transpose([X_C, Y_C]))
            # Add the simulation time
            time_text.set_text(r"$t$ = %.1f s" % (k * T))
            # Dynamically set the axis limits
            ax.set_xlim(x[0, k] - 10 * self.ell, x[0, k] + 10 * self.ell)
            ax.set_ylim(x[1, k] - 10 * self.ell, x[1, k] + 10 * self.ell)
            ax.figure.canvas.draw()
            # Return the objects to animate
            return desired, line, leftwheel, rightwheel, body, castor, time_text

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

    def animate_estimation(
        self,
        x,
        x_hat,
        P_hat,
        alpha,
        T,
        save_ani=False,
        filename="animate_diffdrive.gif",
    ):
        """Create an animation of a differential drive vehicle with plots of
        estimation uncertainty.

        Returns animation object for array of vehicle positions x with time
        increments T [s], track ell [m].

        To save the animation to a GIF file, set save_ani to True and provide a
        filename (default 'animate_diffdrive.gif').
        """
        fig, ax = plt.subplots()
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$y$ [m]")
        plt.axis("equal")
        (estimated,) = ax.plot([], [], "--C1")
        (line,) = ax.plot([], [], "C0")
        (leftwheel,) = ax.fill([], [], color="k")
        (rightwheel,) = ax.fill([], [], color="k")
        (body,) = ax.fill([], [], color="C0", alpha=0.5)
        (castor,) = ax.fill([], [], color="k")
        time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)
        s2 = chi2.isf(alpha, 2)

        def init():
            """Function that initializes the animation."""
            estimated.set_data([], [])
            line.set_data([], [])
            leftwheel.set_xy(np.empty([5, 2]))
            rightwheel.set_xy(np.empty([5, 2]))
            body.set_xy(np.empty([36, 2]))
            castor.set_xy(np.empty([36, 2]))
            time_text.set_text("")
            return estimated, line, leftwheel, rightwheel, body, castor, time_text

        def movie(k):
            """Function called at each step of the animation."""
            # Draw the desired trajectory
            estimated.set_data(x_hat[0, 0 : k + 1], x_hat[1, 0 : k + 1])
            # Draw the path followed by the vehicle
            line.set_data(x[0, 0 : k + 1], x[1, 0 : k + 1])
            # Draw the differential drive vehicle
            X_L, Y_L, X_R, Y_R, X_B, Y_B, X_C, Y_C = self.draw(
                x[0, k], x[1, k], x[2, k]
            )
            leftwheel.set_xy(np.transpose([X_L, Y_L]))
            rightwheel.set_xy(np.transpose([X_R, Y_R]))
            body.set_xy(np.transpose([X_B, Y_B]))
            castor.set_xy(np.transpose([X_C, Y_C]))
            # Compute eigenvalues and eigenvectors to find axes for covariance ellipse
            W, V = np.linalg.eig(P_hat[0:2, 0:2, k])
            # Find the index of the largest and smallest eigenvalues
            j_max = np.argmax(W)
            j_min = np.argmin(W)
            ell = patches.Ellipse(
                (x_hat[0, k], x_hat[1, k]),
                2 * np.sqrt(s2 * W[j_max]),
                2 * np.sqrt(s2 * W[j_min]),
                angle=np.arctan2(V[j_max, 1], V[j_max, 0]) * 180 / np.pi,
                alpha=0.2,
                color="C1",
            )
            ax.add_artist(ell)
            # Add the simulation time
            time_text.set_text(r"$t$ = %.1f s" % (k * T))
            # Dynamically set the axis limits
            ax.set_xlim(x[0, k] - 10 * self.ell, x[0, k] + 10 * self.ell)
            ax.set_ylim(x[1, k] - 10 * self.ell, x[1, k] + 10 * self.ell)
            ax.figure.canvas.draw()
            # Return the objects to animate
            return estimated, line, leftwheel, rightwheel, body, castor, time_text

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
