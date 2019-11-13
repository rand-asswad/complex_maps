import numpy as np
from matplotlib import pyplot as plt


def init_grid(xlim=(-1, 1), ylim=(-1, 1), step=0.1, nb_pts=100, separate_axes=True):
    """
    Initializes gridline complex points
    :param xlim: tuple (xmin, xmax)
    :param ylim: tuple (ymin, ymax)
    :param step: interval between gridlines
    :param nb_pts: number of points in each line
    :param seperate_axes: boolean value for separating horizontal and vertical lines
    :returns tuple of lists along each axis if :param serparate_axes is true else a single list of lines
    """

    # plot along real axis
    x_pts = np.linspace(xlim[0], xlim[1], nb_pts, dtype=complex)
    horz = [x_pts + j * 1j for j in np.arange(ylim[0], ylim[1] + step, step)]

    # plot along imaginary axis
    y_pts = np.linspace(ylim[0] * 1j, ylim[1] * 1j, nb_pts, dtype=complex)
    vert = [y_pts + i for i in np.arange(xlim[0], xlim[1] + step, step)]

    if separate_axes:
        return horz, vert
    return horz + vert


def init_polar(rlim=(0, 1), angle_lim=(0, 2*np.pi), r_step=0.1, angle_step=np.pi/12, nb_pts=100, separate_axes=True):
    """
    Initializes polar lines complex points
    :param rlim: tuple (rmin, rmax) for radius range
    :param angle_lim: tuple (angle_min, angle_max) for angle range
    :param r_step: radius interval
    :param angle_step: angle interval
    :param nb_pts: number of points in each line
    :param seperate_axes: boolean value for separating radius and angle lines
    :returns tuple of lists along each axis if :param serparate_axes is true else a single list of lines
    """

    # plot half lines
    r_pts = np.linspace(rlim[0], rlim[1], nb_pts, dtype=complex)
    lines = [r_pts * np.exp(1j * theta) for theta in np.arange(angle_lim[0], angle_lim[1], angle_step)]

    # plot circles
    theta_pts = np.linspace(angle_lim[0], angle_lim[1], nb_pts, dtype=complex)
    circles = [r * np.exp(1j * theta_pts) for r in np.arange(rlim[0], rlim[1] + r_step, r_step)]

    if separate_axes:
        return lines, circles
    return lines + circles


def plot_map(curves, map=None, ax=None, **kwargs):
    """
    Plots complex map image along given curves
    :param curves: list of curves or tuple of lists
    :param map: complex function (default: identity map)
    :param ax: matplotlib ax (default: new ax)
    :param kwargs: keyword arguments to pass to matplotlib.axes.Axes.plot
    """
    if not ax:
        fig = plt.figure()
        ax = fig.subplots()
    ax.axes.set_aspect('equal')

    if isinstance(curves, tuple):
        color = iter(plt.cm.bwr(np.linspace(0,1,len(curves))))
        for group in curves:
            curr_color = next(color)
            for c in group:
                w = np.array([map(z) for z in c]) if map else c
                ax.plot(w.real, w.imag, c=curr_color, **kwargs)
    else:
        color = iter(plt.cm.rainbow(np.linspace(0,1,len(curves))))
        for c in curves:
            w = np.array([map(z) for z in c]) if map else c
            curr_color = next(color)
            ax.plot(w.real, w.imag, c=curr_color, **kwargs)
