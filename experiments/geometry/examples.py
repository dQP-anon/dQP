import numpy as np
import scipy as sp
import igl

import sys
sys.path.append('../../')

from matplotlib.tri import Triangulation
from matplotlib import pyplot as plt

# ant and cross models from the dataset compiled at https://github.com/duxingyi-charles/Locally-Injective-Mappings-Benchmark

class SquareMesh():
    def __init__(self,nsides):
        v, f = igl.triangulated_grid(nsides, nsides)
        self.v = v
        self.f = f
        self.b = None

        self.nv = np.shape(v)[0]

        self.L = np.max(v[:, 0]) - np.min(v[:, 0])
        self.i_corner = np.argmin(np.linalg.norm(v - np.array([self.L,0]), axis=1))

        self.l_boundary = igl.boundary_loop(self.f)
        self.l_interior = np.setdiff1d(np.arange(0, self.nv), self.l_boundary)

        self.nb = len(self.l_boundary)
        self.P_boundary = sp.sparse.coo_matrix((np.ones(self.nb), (np.arange(self.nb), self.l_boundary)),
                                     shape=(self.nb, self.nv))  # boundary projector

        return None

    def get_corner_constraint(self,p_corner,p_edge):
        # assumes given a square
        v_corner = np.array(self.L*np.array([(1-p_corner),p_corner]))

        i_bot_edge = np.where(np.logical_and(self.v[:,0] - (1- p_edge)*self.L >= 0,self.v[:,1] == 0))[0]
        i_right_edge = np.where(np.logical_and(self.v[:,0] == self.L,self.v[:,1] - (p_edge)*self.L <= 0))[0]
        v_bot = self.v[i_bot_edge, :]
        v_right = self.v[i_right_edge, :]
        i_bot_pt = np.argmin(np.linalg.norm(v_bot - np.array([(1 - p_edge) * self.L, 0]), axis=1))
        i_right_pt = np.argmin(np.linalg.norm(v_right - np.array([self.L, (p_edge) * self.L]), axis=1))
        N_bot = len(i_bot_edge)
        N_right = len(i_right_edge)
        bot_sort = np.argsort(self.v[i_bot_edge, 0])
        right_sort =  np.argsort(self.v[i_right_edge, 1])[::-1]

        # consruct interpolants between edge vertices at constrained points
        f_bot = sp.interpolate.interp1d(np.array([0,1]),np.array([[v_bot[i_bot_pt,0],v_corner[0]],[0,v_corner[1]]]))
        f_right = sp.interpolate.interp1d(np.array([0,1]),np.array([[self.L,v_corner[0]],[v_right[i_right_pt,1],v_corner[1]]]))

        # interpolate
        v_bot[bot_sort,:] = f_bot(np.linspace(0,1,N_bot)).T
        v_right[right_sort,:] = f_right(np.linspace(0,1,N_right)).T

        v_new = self.v.copy()
        v_new[i_bot_edge,:] = v_bot
        v_new[i_right_edge,:] = v_right
        b = self.P_boundary @ v_new

        self.b = b
        return None

    def get_point_constraint(self):

        # b = mapping.P_boundary @ mapping.v
        # b[10, 0] = 0.8
        # b[10, 1] = 0.2
        # self.b = b

        v_new = self.v
        v_new[self.i_corner,:] = np.array([0.4,0.6])
        self.b = self.P_boundary @ v_new

        return None

def plot_domain(option,v,f):
    boundary_inds = igl.boundary_loop(f)
    b = v[boundary_inds,:]
    b_rep = np.vstack((b, b[0, :]))
    fig, ax = plt.subplots()
    plt.plot(b_rep[:, 0], b_rep[:, 1], 'k', linewidth=2)
    ax.triplot(Triangulation(v[:, 0], v[:, 1], triangles=f), color="k", linewidth=0.5,
                  alpha=0.5)
    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.savefig('results/' + option + '/initial.pdf')
    plt.savefig('results/' + option + '/initial.png',dpi=300,transparent=True,bbox_inches='tight')
    plt.close(fig)

def load_mesh(example,N=None):

    if example == "corner":
        assert(N is not None)
        p_corner = 0.65
        p_edge = 1
        square = SquareMesh(N)
        square.get_corner_constraint(p_corner,p_edge)
        v, f, b = square.v,square.f,square.b
    elif example == "parameterization":
        path = "ant"
        v, f = igl.read_triangle_mesh(path + "/result.obj")
        v = v[:, 0:2]
        boundary_inds = igl.boundary_loop(f)
        b = v[boundary_inds, :]
    elif example == "cross":
        path = "cross"
        v, f = igl.read_triangle_mesh(path + "/result.obj")
        v = v[:,0:2]
        boundary_inds = igl.boundary_loop(f)
        b = v[boundary_inds,:]
        v, f = igl.read_triangle_mesh(path + "/input.obj")

    # ps.init()
    # ps.register_surface_mesh("mesh", v, f)
    # ps.show()

    plot_domain(example,v,f)

    return v,f,b