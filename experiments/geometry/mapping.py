import numpy as np
import scipy as sp
from scipy.sparse import diags,csc_matrix,lil_matrix,coo_matrix
import igl,robust_laplacian
from mapping_layer import QPLaplacianLayer
from visualization import visualize_mapping,visualize_optimization,inversion_zoom,manual_polyscope,export_to_MATLAB
from examples import load_mesh

from qpsolvers import solve_qp
import sys
sys.path.append('../../')
sys.path.append('../../src') # to see qp_diagnostic if needed

from matplotlib import pyplot as plt

from src import sparse_helper

import torch

import time

import argparse

#############################
# shapely for point-in-polygon check internal/external angle
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
############################

class mapping_problem:
    '''
    Class for a harmonic mapping problem
    '''

    def __init__(self,v,f,optimize_L,video,lambda_reg,option):

        self.optimize_L = optimize_L
        self.video = video
        self.lambda_reg = lambda_reg
        self.option = option

        # mesh data
        self.v = v # Mesh vertices v and faces f
        self.nv = np.shape(v)[0]
        self.f = f
        self.nf = np.shape(f)[0]
        self.e = igl.edges(f)
        self.dim = 2

        # topological data
        self.T = None
        self.L = None
        self.nb = None
        self.ni = None
        self.P_boundary = None
        self.P_interior = None
        self.l_boundary = None
        self.l_interior = None

        self.precompute_mesh_operators()

        # dirchilet boundary data
        self.nc = None
        self.P_cone = None
        self.l_cone = None
        self.N = None
        self.N_con_proj = None
        self.theta = None
        self.b = None
        self.bvec = None

        # mapping solution
        self.x_mapping_free = None
        self.x_mapping = None
        self.dxdn_free = None
        self.dxdn = None
        self.inverted_free = None
        self.inverted = None

    def precompute_mesh_operators(self):
        '''
        Precompute Laplacian and domain projectors
        '''
        # The following operators precomputed on the CPU using sparse scipy and stored as tensors on the desired device.
        # T = mapping_problem.lift_mapping(self)
        # self.T = T

        # geometric Laplacian - our code
        # Mass = mapping_problem.mass_matrix(self)
        # Tt = T.transpose()
        # TtM = Tt @ Mass
        # L = TtM @ T
        # self.L = L

        # geometric Laplacian - external code
        # L,_ = robust_laplacian.mesh_laplacian(np.hstack((self.v,np.zeros((self.nv,1)))),self.f) # have to embed in 3D...

        # combinatorial Laplacian
        A = -igl.adjacency_matrix(self.f)
        A_sum = np.array(A.sum(axis=1)).squeeze()
        A.setdiag(np.abs(A_sum))
        L = A

        self.L_block = L.tocoo()
        self.L_block.sum_duplicates() # canonicalize
        self.L = sp.sparse.block_diag((L,L),format='coo')
        self.L.sum_duplicates()

        # with np.printoptions(precision=3,formatter={'float_kind':'{:1.2f}'.format}):
            # print(np.array2string(L.todense()[:,:]))
            # print(self.v)
            # print(self.f)

        # get boundary and set constraint
        self.l_boundary = igl.boundary_loop(self.f)
        self.l_interior = np.setdiff1d(np.arange(0, self.nv), self.l_boundary)

        self.nb = len(self.l_boundary)
        self.ni = len(self.l_interior)
        self.P_boundary = coo_matrix((np.ones(self.nb), (np.arange(self.nb), self.l_boundary)),
                                     shape=(self.nb, self.nv))  # boundary projector
        self.P_interior = coo_matrix((np.ones(self.ni), (np.arange(self.ni), self.l_interior)),
                                     shape=(self.ni, self.nv))  # interior projector

        return None

    def process_dirichlet_data(self,b):
        '''
        Compute cone projectors from dirichlet constraints
        '''

        self.b = b

        # internal angles of constraints
        b_pad = np.vstack((b[-1, :], b, b[0, :]))
        theta = np.zeros(self.nb)
        polygon = Polygon(list(map(tuple, b)))
        N = np.zeros((self.nb, 4))  # normals
        d_bisect = np.zeros((self.nb,2))
        # plt.plot(*polygon.exterior.xy) # for checking

        R = np.array([[0, -1], [1, 0]]) # 90 degree rotation
        for ii in np.arange(1, self.nb + 1):
            v_curr = b_pad[ii, :]
            dl_1 = b_pad[ii - 1, :] - v_curr
            dl_2 = b_pad[ii + 1, :] - v_curr
            dl_1 = dl_1 / np.linalg.norm(dl_1)
            dl_2 = dl_2 / np.linalg.norm(dl_2)

            theta[ii - 1] = np.arccos(np.clip(np.dot(dl_1, dl_2),-1+1e-6,1-1e-6))
            assert(not np.isnan(theta[ii-1]))
            N[(ii - 1), 0:2] = -np.matmul(R, dl_1)
            N[(ii - 1), 2:4] = np.matmul(R, dl_2)

            barycenter = np.mean(np.vstack((v_curr, b_pad[ii - 1, :], b_pad[ii + 1, :])), 0)

            point = Point(barycenter[0], barycenter[1])
            if not bool(polygon.contains(point)) and np.abs(theta[ii - 1] - np.pi) > 1e-2: # don't take as inside if on edge
                # plt.scatter(barycenter[0], barycenter[1], color='r')
                # plt.plot([barycenter[0],v_curr[0]],[barycenter[1],v_curr[1]],color='r')
                theta[ii - 1] = 2 * np.pi - theta[ii - 1]
            # else:
                # plt.scatter(barycenter[0], barycenter[1], color='k')
                # plt.plot([barycenter[0],v_curr[0]],[barycenter[1],v_curr[1]],color='k')

            d_bisect[ii - 1, :] = (dl_1 + dl_2)
            if np.linalg.norm(d_bisect[ii - 1, :]) > 1e-6:
                d_bisect[ii - 1, :] = d_bisect[ii - 1, :] / np.linalg.norm(d_bisect[ii - 1, :])
            if theta[ii -1] > np.pi:
                d_bisect[ii-1,:] *= -1

            theta[ii - 1] = theta[ii - 1] % (2 * np.pi)

        # plt.show()

        self.theta = theta

        in_cone = theta > np.pi + 1e-4 # if close to flat, don't take as constrained ... had to tune manually
        self.nc = np.sum(in_cone)
        self.l_cone = [i for i, x in enumerate(in_cone) if x]
        self.P_cone = coo_matrix((np.ones(self.nc), (np.arange(self.nc), self.l_boundary[in_cone])),
                            shape=(self.nc, self.nv))  # cone condition projector

        N = -N  # sign-reversal to match PSD L
        d_bisect = -d_bisect # no perturbation for cone constrained

        self.N = N
        self.d_bisect = d_bisect

        # get cone half-space normals matrix for constraints (2nc x 2nc)
        N_cone_proj = np.zeros((2 * self.nc, 2 * self.nc))
        N_cone = N[in_cone, :]

        for ii in np.arange(0, self.nc):
            N_cone_proj[2 * ii, ii] = N_cone[ii, 0]  # nL x
            N_cone_proj[2 * ii + 1, ii] = N_cone[ii, 2]  # nR x
            N_cone_proj[2 * ii, ii + self.nc] = N_cone[ii, 1]  # nL y
            N_cone_proj[2 * ii + 1, ii + self.nc] = N_cone[ii, 3]  # nR y

        self.N_cone_proj = N_cone_proj

        # expand projectors to account for vectorization of mapping vec(x,y)
        self.P_cone = sp.sparse.bmat([[self.P_cone, None], [None, self.P_cone]],format='csc')
        self.P_boundary = sp.sparse.bmat([[self.P_boundary, None], [None, self.P_boundary]],format='csc')
        self.P_interior = sp.sparse.bmat([[self.P_interior, None], [None, self.P_interior]],format='csc')
        self.bvec = b.copy().reshape((2 * self.nb, 1), order='F')

    def compute_mapping(self):
        '''
        Compute harmonic mapping
        '''

        # free
        x_mapping_free = solve_qp(P=self.L, q=np.zeros(self.dim * self.nv), A=self.P_boundary, b=self.bvec,
                                  solver="gurobi", verbose=False)
        self.x_mapping_free = np.expand_dims(x_mapping_free, -1)
        self.dxdn_free = self.L @ x_mapping_free
        self.inverted_free = self.compute_inversion(self.x_mapping_free)

        # constrained
        if self.optimize_L:
            self.optimize_mapping()

        x_mapping = solve_qp(P=self.L, q=np.zeros(self.dim*self.nv), A=self.P_boundary, b=self.bvec, G=-self.N_cone_proj @ self.P_cone @ self.L,
                             h=np.zeros(self.dim*self.nc), solver="gurobi", verbose=False)
        if x_mapping is None:
            solve_qp(P=self.L, q=np.zeros(self.dim * self.nv), A=self.P_boundary, b=self.bvec,
                                 G=-self.N_cone_proj @ self.P_cone @ self.L,
                                 h=np.zeros(self.dim * self.nc), solver="mosek", verbose=True) # use mosek because gives better read-out
            print("Solver failed to return a solution for constrained problem.")
        else:
            self.x_mapping = np.expand_dims(x_mapping, -1)
            self.dxdn = self.L @ x_mapping
            self.inverted = self.compute_inversion(self.x_mapping)

    def optimize_mapping(self):
        A = self.P_boundary
        b = self.bvec
        G_precompute = -self.N_cone_proj @ self.P_cone # - sign due to convention of qpsolvers

        laplacian_layer = QPLaplacianLayer(self.L_block,self.e,self.nv,self.nc,self.nb,A=A,b=b,G_precompute=G_precompute)
        laplacian_layer.train()
        optimizer = torch.optim.Adam(laplacian_layer.parameters(),lr=1e-2) # 1e-2

        opt_data = {}

        counter_cone_satisfied = 0
        iter = 0
        L_init = torch.tensor(self.L.data,dtype=torch.float64)
        L_init = L_init / torch.linalg.vector_norm(L_init)
        while iter < 10000:
            iter_time = time.time()
            L,x_star,nu_star,dQP_forward_time,optnet_time,scqpth_time,qplayer_time = laplacian_layer()
            total_forward_time = time.time() - iter_time
            # print("Total forward time: " + str(total_forward_time))

            if x_star is None:
                break

            L_prev = L

            optimizer.zero_grad()
            L_curr = L.to_sparse_coo().coalesce().values()
            L_curr = L_curr / torch.linalg.norm(L_curr)
            loss_dual = torch.linalg.vector_norm(nu_star)
            loss_reg = self.lambda_reg*torch.linalg.vector_norm(L_curr - L_init,np.inf) # scale invariant difference

            loss = loss_dual + loss_reg
            print("Loss: " + str(loss.detach().numpy()))

            print(nu_star)
            nInactive = torch.sum(nu_star < 1e-4) # NOTE STRICT ACTIVE TOLERANCE ... BUT INNACURATE SOLUTION UP TO TOLERANCE=1e-4
            print(self.nc*self.dim - nInactive)
            if nInactive == self.nc*self.dim: # if all inactive
                if self.lambda_reg == 0:
                    break
                else:
                    break
                    # print("HERE")
                    # counter_cone_satisfied += 1
                    # if counter_cone_satisfied > 10: # reduce regularizer loss even if satisfy cone constraint
                    #     break

            back_time = time.time()
            # with torch.autograd.profiler.profile(with_modules=True) as prof:
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(laplacian_layer.parameters(), 0.01)
            optimizer.step()
            total_backward_time = time.time() - back_time
            # print("Total backward time: " + str(total_backward_time))
            # print(prof.key_averages().table(sort_by="self_cpu_time_total"))

            dQP_backward_time = np.loadtxt("results/profiling/t_diff.dat", dtype=float)

            curr_data = {
                "loss_dual": loss_dual.detach().numpy(),
                "loss_reg": loss_reg.detach().numpy(),
                "L" : sparse_helper.csc_torch_to_scipy(L_prev),
                "x_star": x_star.detach().numpy(),
                "dQP_forward_time" : dQP_forward_time,
                "dQP_backward_time" : dQP_backward_time,
                "total_forward_time" : total_forward_time,
                "total_backward_time" : total_backward_time,
                "optnet_time" : optnet_time,
                "scqpth_time" : scqpth_time,
                "qplayer_time": qplayer_time,
                "nActive": self.nc*self.dim - nInactive
            }

            opt_data[str(iter)] = curr_data

            # break
            iter += 1
            # if np.mod(iter,1000) == 0: # annealing
            #     self.lambda_reg *= 0.1

        self.L = sparse_helper.csc_torch_to_scipy(L) # solve_qp takes csc matrix

        x_mapping = solve_qp(P=self.L, q=np.zeros(self.dim * self.nv), A=self.P_boundary, b=self.bvec,
                             G=-self.N_cone_proj @ self.P_cone @ self.L,
                             h=np.zeros(self.dim * self.nc), solver="gurobi", verbose=False)
        x_mapping = np.expand_dims(x_mapping, -1)

        curr_data = {
            "loss_dual": None,
            "loss_reg": None,
            "L": self.L,
            "x_star": x_mapping,
            "dQP_forward_time": None,
            "dQP_backward_time": None,
            "total_forward_time": None,
            "total_backward_time": None,
            "optnet_time": None,
            "scqpth_time": None,
            "qplayer_time": None,
            "nActive":None
        }

        opt_data[str(iter)] = curr_data

        # print(self.L.todense())
        if len(opt_data) > 1:
            visualize_optimization(mapping=self,opt_data=opt_data)

    def lift_mapping(self):
        '''
        Obtain map from vertices to differentials
        '''

        # Construct matrix T (#faces * dim^2 x #verts * dim) mapping column-stacked vertices to row-stacked differentials in order given by face indices f

        v,nv,f,nf,dim= self.v,self.nv,self.f,self.nf,self.dim

        T = lil_matrix((nf*dim*dim,nv*dim))

        for ff in range(nf): # Computing differentials per triangle
            r = np.array([v[f[ff,0],0],v[f[ff,1],0],v[f[ff,2],0],v[f[ff,0],1],v[f[ff,1],1],v[f[ff,2],1]])
            Telem = mapping_problem.affine_differential(self,r)

            # Inserting into T
            for vv in range(3):
                for tt in range(dim**2):
                    T[ff*dim**2+tt,f[ff,vv]] = Telem[tt,vv]
                    T[ff*dim**2+tt,f[ff, vv]+nv] = Telem[tt,vv+3]

        return csc_matrix(T)

    def affine_differential(self,r):
        '''
        Differential per-triangle in 2D
        '''

        # Stacked differential T = vec(A) of affine transformation F(r) = Ar + d on a single triangle
        # x = vec(x1,x2,x3,y1,y2,y3) stacked vertex coordinates ; A = vec(a,b,c,d) stacked differential

        x1, x2, x3, y1, y2, y3 = r
        T = np.array([[y3-y2, y1-y3, y2-y1], [x2-x3, x3-x1, x1-x2]]) / (x2*y1-x3*y1-x1*y2+x3*y2+x1*y3-x2*y3)
        return sp.linalg.block_diag(T,T)

    def mass_matrix(self):
        '''
        Triangle areas
        '''

        v, f= self.v, self.f
        A = igl.doublearea(v, f) # why not / 2?
        A = np.hstack((A, A, A, A))

        return csc_matrix(diags(A))

    def compute_inversion(self, x):
        x = np.reshape(x, (self.nv, self.dim), order='F')
        x = x[self.f,:]
        d = np.zeros((self.nf,2,self.dim))
        d[:,0,:] = x[:,1,:] - x[:,0,:]
        d[:,1,:] = x[:,2,:] - x[:,0,:]
        flipped = -np.sign(sp.linalg.det(d))
        return flipped

def main(optimize_L=True,video=True,lambda_reg=0,option="cross"):
    # options in examples
    # "corner", "parameterization", "cross"
    # lambda_reg = 1e2 good choice

    if option == "corner":
        for N in np.int64(np.sqrt(np.power(10,np.arange(1.1,4.2,0.1)))): # equal sample in log_10 vertices from 10^0 to 10^3

            print("Number of vertices:" + str(N**2))
            print("Log 10 number of vertices:" + str(np.log10(N**2)))

            v,f,b = load_mesh(option,N)

            mapping = mapping_problem(v, f, optimize_L=optimize_L, video=video, lambda_reg=lambda_reg, option=option)
            mapping.process_dirichlet_data(b)
            mapping.compute_mapping()
            # visualize_mapping(mapping,map_type="free")
            visualize_mapping(mapping,map_type="constrained")
            plt.close()
    else:
        N = None
        v, f, b = load_mesh(option, N)
        mapping = mapping_problem(v, f, optimize_L=optimize_L, video=video, lambda_reg=lambda_reg, option=option)
        mapping.process_dirichlet_data(b)
        mapping.compute_mapping()
        export_to_MATLAB(mapping)
        # visualize_mapping(mapping, map_type="free")
        # visualize_mapping(mapping, map_type="constrained")

        # manual_polyscope(mapping)
        # if option == "parameterization":
        #     inversion_zoom(mapping)

parser = argparse.ArgumentParser()
parser.add_argument('example', metavar='example', type=str)
parser.add_argument('lambda_reg',metavar='lambda_reg', type=float)
args = parser.parse_args()
main(lambda_reg=args.lambda_reg,option=args.example)

# note, the ant set-up takes a long time (setting up BCS, constructing matrices, etc) but could be improved significantly, we only focus on profiling the QP solve.
# main(lambda_reg=0,option="corner") # to isolate dQP backward timing, need to enable directly in dQP backward inside src
# main(lambda_reg=0,option="cross")
# main(lambda_reg=10,option="cross")
# main(lambda_reg=0,option="parameterization")