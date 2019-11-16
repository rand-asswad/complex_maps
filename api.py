import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection as LC

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
rcParams['text.usetex'] = True
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica']


def init_grid(xlim=(-1, 1), ylim=(-1, 1), step=0.1, nb_pts=1000, separate_axes=True):
    """
    Initializes gridline complex points
    :param xlim: tuple (xmin, xmax)
    :param ylim: tuple (ymin, ymax)
    :param step: interval between gridlines
    :param nb_pts: number of points in each line
    :param seperate_axes: boolean value for separating horizontal and vertical lines
    :returns tuple of lists along each axis if :param serparate_axes is true else a single list of lines
    """
    Nx = int(round((xlim[1] - xlim[0]) / step))
    Ny = int(round((ylim[1] - ylim[0]) / step))

    # plot along real axis
    x_pts = np.linspace(xlim[0], xlim[1], nb_pts, dtype=complex)
    horz = [x_pts + j * 1j for j in np.linspace(ylim[0], ylim[1], Ny)]

    # plot along imaginary axis
    y_pts = np.linspace(ylim[0] * 1j, ylim[1] * 1j, nb_pts, dtype=complex)
    vert = [y_pts + i for i in np.linspace(xlim[0], xlim[1], Nx)]

    if separate_axes:
        return horz, vert
    return horz + vert


def init_polar(rlim=(0, 1), angle_lim=(0, 2*np.pi), r_step=0.1, angle_step=np.pi/12, nb_pts=1000, separate_axes=True):
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
    Nr = int(round((rlim[1] - rlim[0]) / r_step))
    Nt = int(round((angle_lim[1] - angle_lim[0]) / angle_step))

    # plot half lines
    r_pts = np.linspace(rlim[0], rlim[1], nb_pts, dtype=complex)
    lines = [r_pts * np.exp(1j * theta) for theta in np.linspace(angle_lim[0], angle_lim[1], Nt)]

    # plot circles
    theta_pts = np.linspace(angle_lim[0], angle_lim[1], nb_pts, dtype=complex)
    circles = [r * np.exp(1j * theta_pts) for r in np.linspace(rlim[0], rlim[1], Nr)]

    if separate_axes:
        return lines, circles
    return lines + circles


def plot_map(curves, map, plot_domain=True, align='horizontal', **kwargs):
    """
    Plots complex map image along given curves
    :param curves: list of curves or tuple of lists
    :param map: complex function (default: identity map)
    :param kwargs: keyword arguments to pass to matplotlib.axes.Axes.plot
    """
    fig = plt.figure()
    if map:
        if plot_domain:
            if align.lower() == 'vertical':
                domain, ax = fig.subplots(nrows=2, ncols=1)
                fig.subplots_adjust(hspace=.4)
            else:
                domain, ax = fig.subplots(nrows=1, ncols=2)
                fig.subplots_adjust(wspace=.4)
            domain.axes.set_aspect('equal')
            ax.axes.set_aspect('equal')
        else:
            ax = fig.subplots()
            ax.axes.set_aspect('equal')
            domain = None
    else:
        domain = fig.subplots()
        domain.axes.set_aspect('equal')
        ax = None

    # set title
    title = kwargs.pop('title', None)
    if title:
        fig.suptitle(title)
    
    # set axes labels
    axes_label = kwargs.pop('axis_label', True)
    func = kwargs.pop('func', 'f')
    var = kwargs.pop('var', 'z')
    if axes_label:
        label = '$\\mathrm{}({})$'
        if ax:
            ax.set_xlabel(label.format('{Re}', f'{func}\\left({var}\\right)'))
            ax.set_ylabel(label.format('{Im}', f'{func}\\left({var}\\right)'))
        if domain:
            domain.set_xlabel(label.format('{Re}', var))
            domain.set_ylabel(label.format('{Im}', var))
    
    # color mode
    if kwargs.pop('color_each', False) and isinstance(curves, tuple):
        c_list = []
        for group in curves:
            c_list += group
        curves = c_list
    
    # set default linewidth
    kwargs.setdefault('linewidth', 1)

    if isinstance(curves, tuple):
        color = iter(plt.cm.bwr(np.linspace(0,1,len(curves))))
        for group in curves:
            curr_color = next(color)
            for c in group:
                if ax:
                    w = np.array([map(z) for z in c], dtype=complex)
                    w = w[np.where(np.isfinite(w))]
                    ax.plot(w.real, w.imag, c=curr_color, **kwargs)
                if domain:
                    domain.plot(c.real, c.imag, c=curr_color, **kwargs)
    else:
        color = iter(plt.cm.rainbow(np.linspace(0,1,len(curves))))
        for c in curves:
            curr_color = next(color)
            if ax:
                w = np.array([map(z) for z in c], dtype=complex)
                w = w[np.where(np.isfinite(w))]
                ax.plot(w.real, w.imag, c=curr_color, **kwargs)
            if domain:
                domain.plot(c.real, c.imag, c=curr_color, **kwargs)
    if domain:
        if ax:
            return domain, ax
        return domain
    return ax


def plot_gradient_line(x, y, norm=None, ax=None, **kwargs):
    if not ax:
        fig, ax = plt.subplots()
        ax.axes.set_aspect('equal')
        ax.set_xlim(x.min() -1, x.max()+1)
        ax.set_ylim(y.min() -1, y.max()+1)
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
    
    # prepare points
    if norm is None:
        norm = np.linspace(0, 1, len(x))

    # defaults
    cmap = kwargs.pop('cmap', plt.get_cmap('rainbow'))
    
    # draw line
    lc = LC(segments, array=norm, cmap=cmap, **kwargs)
    line = ax.add_collection(lc)

    # colorbar
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='5%', pad=0.1)
    cbar = fig.colorbar(line, cax=cax)

    return ax, cbar
