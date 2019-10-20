import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def process_default_kwargs(kwargs, default_kwargs):
    if kwargs is None:
        kwargs = {}
    for kwarg, value in default_kwargs.items():
        if kwarg not in kwargs:
            kwargs[kwarg] = value
    return kwargs

def heat_map(arr, XY=None, figure_kwargs=None, pcolormesh_kwargs=None):
    '''Plot a heat map of the values of an array with a colorbar.

    '''
    vmin = arr.min()
    vmax = arr.max()
    default_figure_kwargs = {'figsize': (6, 4)}
    figure_kwargs = process_default_kwargs(figure_kwargs,
                                           default_figure_kwargs)
    default_pcolormesh_kwargs = {'rasterized': True,
                                 'shading': 'gouraud',
                                 'vmin': vmin,
                                 'vmax': vmax}
    pcolormesh_kwargs = process_default_kwargs(pcolormesh_kwargs,
                                               default_pcolormesh_kwargs)
    fig = plt.figure(**figure_kwargs)
    gs  = mpl.gridspec.GridSpec(1, 2, width_ratios=[1, 0.25])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    if XY is None:
        cax = ax0.pcolormesh(arr, **pcolormesh_kwargs)
    else:
        cax = ax0.pcolormesh(*XY, arr, **pcolormesh_kwargs)
    cbar = fig.colorbar(cax, cax=ax1)
    cbar.set_ticks([pcolormesh_kwargs['vmin'], pcolormesh_kwargs['vmax']])
    return fig, [ax0, ax1]

def visualize_real_array(arr, figure_kwargs=None, matshow_kwargs=None):
    '''Visualize a real matrix.

    '''
    vmin = -np.abs(arr).max()
    vmax = np.abs(arr).max()

    default_figure_kwargs = {'figsize': (6, 4)}
    figure_kwargs = process_default_kwargs(figure_kwargs,
                                           default_figure_kwargs)

    default_matshow_kwargs = {'cmap': mpl.cm.RdBu,
                              'vmin': vmin,
                              'vmax': vmax}
    matshow_kwargs = process_default_kwargs(matshow_kwargs,
                                            default_matshow_kwargs)

    fig = plt.figure(**figure_kwargs)
    gs  = mpl.gridspec.GridSpec(1, 2, width_ratios=[1, 0.25])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    cax = ax0.matshow(arr, **matshow_kwargs)
    cbar = fig.colorbar(cax, cax=ax1)
    cbar.set_ticks([matshow_kwargs['vmin'], 0, matshow_kwargs['vmax']])
    return fig, [ax0, ax1]

def visualize_complex_array(arr, figure_kwargs=None, matshow_kwargs=None):
    '''Visualize a complex matrix.

    '''
    arr_real = np.real(arr)
    arr_imag = np.imag(arr)
    vmin = -max(np.abs(arr_real).max(), np.abs(arr_imag).max())
    vmax = max(np.abs(arr_real).max(), np.abs(arr_imag).max())

    default_figure_kwargs = {'figsize': (10, 4)}
    figure_kwargs = process_default_kwargs(figure_kwargs,
                                           default_figure_kwargs)

    default_matshow_kwargs = {'cmap': mpl.cm.RdBu,
                              'vmin': vmin,
                              'vmax': vmax}
    matshow_kwargs = process_default_kwargs(matshow_kwargs,
                                            default_matshow_kwargs)

    fig = plt.figure(**figure_kwargs)
    gs  = mpl.gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.25])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    ax0.matshow(arr_real, **matshow_kwargs)
    cax = ax1.matshow(arr_imag, **matshow_kwargs)
    cbar = fig.colorbar(cax, cax=ax2)
    cbar.set_ticks([matshow_kwargs['vmin'], 0, matshow_kwargs['vmax']])
    ax1.set_yticks([])
    return fig, [ax0, ax1, ax2]

def sphere_plot_lat_long(fn, lat_count=30, long_count=60, ax=None, cmap=None,
        plot_surface_kwargs=None, norm_kwargs=None):
    Thetas, Phis = np.mgrid[0:np.pi:1.j*lat_count,0:2*np.pi:1.j*long_count]
    Xs = np.sin(Thetas) * np.cos(Phis)
    Ys = np.sin(Thetas) * np.sin(Phis)
    Zs = np.cos(Thetas)
    values = fn(Thetas, Phis)
    vmax = np.abs(values).max()
    vmin = -vmax

    if cmap is None:
        cmap = mpl.cm.RdBu

    default_plot_surface_kwargs = {'rasterized': True,
                                   'rcount': lat_count,
                                   'ccount': long_count}
    plot_surface_kwargs = process_default_kwargs(plot_surface_kwargs,
                                                 default_plot_surface_kwargs)
    default_norm_kwargs = {'vmin': vmin, 'vmax': vmax}
    norm_kwargs = process_default_kwargs(norm_kwargs, default_norm_kwargs)

    return_figax = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        return_figax = True
    max_mag = np.abs(values).max()
    norm = mpl.colors.Normalize(**norm_kwargs)
    ax.plot_surface(Xs, Ys, Zs, facecolors=cmap(norm(values.real)),
                    **plot_surface_kwargs)
    ax.set_aspect('equal')
    mpl.rcParams['savefig.dpi'] = 300
    return (fig, ax) if return_figax else None
