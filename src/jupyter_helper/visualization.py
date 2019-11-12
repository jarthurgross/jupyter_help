import itertools as it
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def process_default_kwargs(kwargs, default_kwargs):
    """Update a default kwarg dict with user-supplied values

    """
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
    vmax = np.abs(arr).max()
    vmin = -vmax

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

class Octahedron(Deltahedron):
    def __init__(self, N):
        vertices = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, 0], [0, 0, -1],
                    [-1, 0, 0]]
        faces = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1), (1, 4, 5),
                 (2, 1, 5), (3, 2, 5), (4, 3, 5)]
        super().__init__(vertices, faces, N)
