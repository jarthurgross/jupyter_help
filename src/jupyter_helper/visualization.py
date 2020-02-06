import itertools as it
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import colorcet as cc

def process_default_kwargs(kwargs, default_kwargs):
    """Update a default kwarg dict with user-supplied values

    """
    if kwargs is None:
        kwargs = {}
    for kwarg, value in kwargs.items():
        default_kwargs[kwarg] = value

def heat_map(arr, XY=None, figure_kwargs=None, pcolormesh_kwargs=None):
    '''Plot a heat map of the values of an array with a colorbar.

    '''
    vmin = arr.min()
    vmax = arr.max()
    default_figure_kwargs = {'figsize': (6, 4)}
    process_default_kwargs(figure_kwargs, default_figure_kwargs)
    default_pcolormesh_kwargs = {'rasterized': True,
                                 'shading': 'gouraud',
                                 'vmin': vmin,
                                 'vmax': vmax}
    process_default_kwargs(pcolormesh_kwargs, default_pcolormesh_kwargs)
    fig = plt.figure(**default_figure_kwargs)
    gs  = mpl.gridspec.GridSpec(1, 2, width_ratios=[1, 0.25])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    if XY is None:
        cax = ax0.pcolormesh(arr, **default_pcolormesh_kwargs)
    else:
        cax = ax0.pcolormesh(*XY, arr, **default_pcolormesh_kwargs)
    cbar = fig.colorbar(cax, cax=ax1)
    cbar.set_ticks([default_pcolormesh_kwargs['vmin'],
                    default_pcolormesh_kwargs['vmax']])
    return fig, [ax0, ax1]

def visualize_real_array(arr, figure_kwargs=None, matshow_kwargs=None):
    '''Visualize a real matrix.

    '''
    vmax = np.abs(arr).max()
    vmin = -vmax

    default_figure_kwargs = {'figsize': (6, 4)}
    process_default_kwargs(figure_kwargs, default_figure_kwargs)

    default_matshow_kwargs = {'cmap': mpl.cm.RdBu,
                              'vmin': vmin,
                              'vmax': vmax}
    process_default_kwargs(matshow_kwargs, default_matshow_kwargs)

    fig = plt.figure(**default_figure_kwargs)
    gs  = mpl.gridspec.GridSpec(1, 2, width_ratios=[1, 0.25])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    cax = ax0.matshow(arr, **default_matshow_kwargs)
    cbar = fig.colorbar(cax, cax=ax1)
    cbar.set_ticks([default_matshow_kwargs['vmin'], 0,
                    default_matshow_kwargs['vmax']])
    return fig, [ax0, ax1]

def visualize_complex_array(arr, figure_kwargs=None, matshow_kwargs=None):
    '''Visualize a complex matrix.

    '''
    arr_real = np.real(arr)
    arr_imag = np.imag(arr)
    vmin = -max(np.abs(arr_real).max(), np.abs(arr_imag).max())
    vmax = max(np.abs(arr_real).max(), np.abs(arr_imag).max())

    default_figure_kwargs = {'figsize': (10, 4)}
    process_default_kwargs(figure_kwargs, default_figure_kwargs)

    default_matshow_kwargs = {'cmap': mpl.cm.RdBu,
                              'vmin': vmin,
                              'vmax': vmax}
    process_default_kwargs(matshow_kwargs, default_matshow_kwargs)

    fig = plt.figure(**default_figure_kwargs)
    gs  = mpl.gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.25])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    ax0.matshow(arr_real, **default_matshow_kwargs)
    cax = ax1.matshow(arr_imag, **default_matshow_kwargs)
    cbar = fig.colorbar(cax, cax=ax2)
    cbar.set_ticks([default_matshow_kwargs['vmin'], 0,
                    default_matshow_kwargs['vmax']])
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
    process_default_kwargs(plot_surface_kwargs, default_plot_surface_kwargs)
    default_norm_kwargs = {'vmin': vmin, 'vmax': vmax}
    process_default_kwargs(norm_kwargs, default_norm_kwargs)

    return_figax = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        return_figax = True
    max_mag = np.abs(values).max()
    norm = mpl.colors.Normalize(**default_norm_kwargs)
    ax.plot_surface(Xs, Ys, Zs, facecolors=cmap(norm(values.real)),
                    **default_plot_surface_kwargs)
    ax.set_aspect('equal')
    mpl.rcParams['savefig.dpi'] = 300
    return (fig, ax) if return_figax else None

def interior_convex_combs(a, b, N):
    return [((N - n)*a + n*b)/N for n in range(1, N)]

def face_interior_convex_combs(a, b, c, N):
    return [interior_convex_combs(((N - n)*a + n*b)/N, ((N - n)*a + n*c)/N, n)
            for n in range(2, N)]

class Point:
    def __init__(self, coords):
        self.coords = np.array(coords)

    def set_idx(self, idx):
        self.idx = idx

    def __add__(self, other):
        return Point(self.coords + other.coords)

    def __rmul__(self, other):
        return Point(other * self.coords)

    def __truediv__(self, other):
        return Point(self.coords / other)

    def __repr__(self):
        return 'Point(' + self.coords.__repr__() + ')'

    def __iter__(self):
        return self.coords.__iter__()

class Deltahedron:
    '''A class for making refined triangulations of polyhedra

    '''
    def __init__(self, vertices, faces, N):
        self.vertices = list(map(Point, vertices))
        self.faces = faces
        self.edges = []
        for a, b, c in self.faces:
            for pair in [[a, b], [b, c], [c, a]]:
                if sorted(pair) not in self.edges:
                    self.edges.append(sorted(pair))
        self.edges_interior_points = {(a, b): interior_convex_combs(self.vertices[a],
                                                                    self.vertices[b], N)
                                      for a, b in self.edges}
        self.faces_interior_points = {(a, b, c): face_interior_convex_combs(self.vertices[a],
                                                                            self.vertices[b],
                                                                            self.vertices[c], N)
                                      for a, b, c in self.faces}
        self.corner_triangles = []
        for face in self.faces:
            for j in range(3):
                a = face[j]
                b = face[(j+1)%3]
                c = face[(j+2)%3]
                ab = self.get_edge_from_vertex_idxs(a, b)[0]
                ac = self.get_edge_from_vertex_idxs(a, c)[0]
                self.corner_triangles.append([self.vertices[a], ab, ac])
                if j == 0:
                    self.corner_triangles.append([self.faces_interior_points[face][0][0], ab, ac])
                if j == 1:
                    self.corner_triangles.append([self.faces_interior_points[face][-1][0], ab, ac])
                if j == 2:
                    self.corner_triangles.append([self.faces_interior_points[face][-1][-1], ab, ac])
        self.edge_triangles = []
        for a, b, c in self.faces:
            ab = self.get_edge_from_vertex_idxs(a, b)
            ac = self.get_edge_from_vertex_idxs(a, c)
            bc = self.get_edge_from_vertex_idxs(b, c)
            for j in range(len(self.faces_interior_points[a,b,c])):
                self.edge_triangles.append([ab[j], ab[j+1],
                                            self.faces_interior_points[a,b,c][j][0]])
                self.edge_triangles.append([ac[j], ac[j+1],
                                            self.faces_interior_points[a,b,c][j][-1]])
                self.edge_triangles.append([bc[j], bc[j+1],
                                            self.faces_interior_points[a,b,c][-1][j]])
            for j in range(len(self.faces_interior_points[a,b,c]) - 1):
                self.edge_triangles.append([ab[j+1],
                                            self.faces_interior_points[a,b,c][j][0],
                                            self.faces_interior_points[a,b,c][j+1][0]])
                self.edge_triangles.append([ac[j+1],
                                            self.faces_interior_points[a,b,c][j][-1],
                                            self.faces_interior_points[a,b,c][j+1][-1]])
                self.edge_triangles.append([bc[j+1],
                                            self.faces_interior_points[a,b,c][-1][j],
                                            self.faces_interior_points[a,b,c][-1][j+1]])
        self.face_triangles = []
        for face in self.faces:
            for j in range(len(self.faces_interior_points[face]) - 1):
                for k in range(len(self.faces_interior_points[face][j])):
                    self.face_triangles.append([self.faces_interior_points[face][j][k],
                                                self.faces_interior_points[face][j+1][k],
                                                self.faces_interior_points[face][j+1][k+1]])
            for j in range(1, len(self.faces_interior_points[face]) - 1):
                for k in range(len(self.faces_interior_points[face][j]) - 1):
                    self.face_triangles.append([self.faces_interior_points[face][j][k],
                                                self.faces_interior_points[face][j][k+1],
                                                self.faces_interior_points[face][j+1][k+1]])


    def get_edge_from_vertex_idxs(self, a, b):
        '''Return a view of the edge oriented according to the order of the vertices given.

        '''
        if a < b:
            return self.edges_interior_points[a,b]
        elif a > b:
            return self.edges_interior_points[b,a][::-1]
        else:
            raise ValueError

    def get_triangulation(self):
        points = it.chain(self.vertices,
                          *it.chain(self.edges_interior_points.values()),
                          *it.chain(*self.faces_interior_points.values()))
        coords = []
        for n, point in enumerate(points):
            point.set_idx(n)
            coords.append(point.coords)
        triangles = [[a.idx, b.idx, c.idx] for a, b, c in it.chain(self.corner_triangles,
                                                                   self.edge_triangles,
                                                                   self.face_triangles)]
        return np.array(coords), np.array(triangles)

class Tetrahedron(Deltahedron):
    def __init__(self, N):
        A = np.sqrt(2)
        vertices = [[-1, A, 0], [-1, -A, 0], [1, 0, A], [1, 0, -A]]
        faces = [(0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)]
        super().__init__(vertices, faces, N)

class Octahedron(Deltahedron):
    def __init__(self, N):
        vertices = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, 0], [0, 0, -1],
                    [-1, 0, 0]]
        faces = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1), (1, 4, 5),
                 (2, 1, 5), (3, 2, 5), (4, 3, 5)]
        super().__init__(vertices, faces, N)

class Icosahedron(Deltahedron):
    def __init__(self, N):
        phi = (1 + np.sqrt(5))/2
        vertices = [[0, 1, phi], [phi, 0, 1], [1, phi, 0],
                    [0, -1, phi], [phi, 0, -1], [-1, phi, 0],
                    [0, -1, -phi], [-phi, 0, -1], [-1, -phi, 0],
                    [0, 1, -phi], [-phi, 0, 1], [1, -phi, 0]]
        faces = [(0, 3, 1), (0, 3, 10), (1, 4, 2), (1, 4, 11),
                 (2, 5, 0), (2, 5, 9), (6, 9, 7), (6, 9, 4),
                 (7, 10, 8), (7, 10, 5), (8, 11, 6), (8, 11, 3),
                 (0, 1, 2), (0, 10, 5), (3, 1, 11), (3, 10, 8),
                 (6, 7, 8), (6, 4, 11), (9, 7, 5), (9, 4, 2)]
        super().__init__(vertices, faces, N)

def make_LSC_from_rgb_list(name, rgb_list):
    rs = rgb_list[:,0]
    gs = rgb_list[:,1]
    bs = rgb_list[:,2]
    xs = np.linspace(0, 1, rs.shape[0] - 1)
    cdict = {'red': [(x, y0, y1) for x, y0, y1 in zip(xs, rs[:-1], rs[1:])],
             'green': [(x, y0, y1) for x, y0, y1 in zip(xs, gs[:-1], gs[1:])],
             'blue': [(x, y0, y1) for x, y0, y1 in zip(xs, bs[:-1], bs[1:])]}
    return mpl.colors.LinearSegmentedColormap(name, cdict)

def lightness(rgbas):
    # From https://stackoverflow.com/a/596243/1236650 which references
    # http://alienryderflex.com/hsp.html
    A = np.array([0.299, 0.587, 0.114, 0])
    return np.sqrt(np.tensordot(A, rgbas**2, ([0], [-1])))

def linearize_divergent_cmap(name, decr_vals, middle_val, incr_vals, N=256):
    decr_incr_vals = np.vstack([decr_vals[::-1], middle_val, incr_vals])
    decr_lightness_diffs = np.diff(lightness(np.vstack([middle_val, decr_vals])))
    incr_lightness_diffs = np.diff(lightness(np.vstack([middle_val, incr_vals])))
    linearizing_xs = np.hstack([0, np.cumsum(np.hstack([decr_lightness_diffs[::-1], incr_lightness_diffs]))])
    linear_interp = interp1d(linearizing_xs, decr_incr_vals, axis=0)
    if lightness(decr_incr_vals[0]) > lightness(decr_incr_vals[-1]):
        x1 = linearizing_xs[0]
        x2 = linearizing_xs[len(decr_vals)]
        y_star = float(lightness(decr_incr_vals[-1]))
        y1 = float(lightness(linear_interp(x1)))
        y2 = float(lightness(linear_interp(x2)))
        m = (y2 - y1)/(x2 - x1)
        x_star = (y_star - y1)/m + x1
        linear_cmap = make_LSC_from_rgb_list(
                name, linear_interp(np.linspace(x_star, linearizing_xs[-1], N)))
    else:
        x1 = linearizing_xs[len(decr_vals)]
        x2 = linearizing_xs[-1]
        y_star = float(lightness(decr_incr_vals[0]))
        y1 = float(lightness(linear_interp(x1)))
        y2 = float(lightness(linear_interp(x2)))
        m = (y2 - y1)/(x2 - x1)
        x_star = (y_star - y1)/m + x1
        linear_cmap = make_LSC_from_rgb_list(
                name, linear_interp(np.linspace(linearizing_xs[0], x_star, N)))
    return linear_cmap

kbc_vals = cc.m_kbc(np.linspace(0, 1, 256))
kry_vals = kbc_vals[:,[2,1,0,3]]
yrkbc = linearize_divergent_cmap('yrkbc', kry_vals, [0, 0, 0, 1], kbc_vals, N=257)

def set_lut_with_cmap(surf, cmap, N=257):
    cmap_vals = cmap(np.linspace(0, 1, N))
    new_lut = np.round(255*cmap_vals)
    surf.module_manager.scalar_lut_manager.lut.number_of_colors = len(cmap_vals)
    surf.module_manager.scalar_lut_manager.lut.table = new_lut
