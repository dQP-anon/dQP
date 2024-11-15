import numpy as np

import sys
sys.path.append('../../')
sys.path.append('../')

from src import dQP,sparse_helper
import torch_geometric

import torch
from torch import nn

################# Optnet & Tolerances #########################
time_alternatives = True # solve with other methods in parallel, but do not use for optimization
tolerance = 1e-5
eps_active = 1e-4
if time_alternatives:
    from qpth.qp import QPFunction
    from proxsuite.torch.qplayer import QPFunction as prox_qpfunction
    from deps.scqpth.control import scqpth_control
    from deps.scqpth.scqpth import SCQPTHNet
import time
################# Optnet & Tolerances #########################

class QPLaplacianLayer(nn.Module):
    ''' a differentiable QP layer for harmonic mapping with Laplacian deformation parameter
    '''

    def __init__(self,L_init,edges,nv,nc,nb,A,b,G_precompute):
        super().__init__()

        self.edges = torch.tensor(edges.T,dtype=torch.int64)
        self.symedges = torch.cat((self.edges,torch.flip(self.edges,dims=(0,))),1)
        self.nv = nv
        self.nc = nc
        self.nb = nb
        self.dim = 2

        Linds = np.hstack((np.expand_dims(L_init.row,-1),np.expand_dims(L_init.col,-1)))
        idx = np.where((Linds == edges[:, None]).all(-1))[1]
        Lvals = L_init.data[idx]
        self.Lvals = nn.Parameter(torch.tensor(Lvals,dtype=torch.float64)) # non-zero values of Laplacian are parameters of layer

        self.A = sparse_helper.csc_scipy_to_torch(A).requires_grad_(False)
        self.b = torch.tensor(b,requires_grad=False,dtype=torch.float64)
        self.G_precompute = torch.tensor(G_precompute,requires_grad=False,dtype=torch.float64).to_sparse_coo().coalesce() # note .to_sparse_coo() is new ; doesn't need grad but pretend to have one for spspmm
        self.q = torch.zeros(self.dim * self.nv,1,requires_grad=False,dtype=torch.float64)
        self.h = torch.zeros(self.dim * self.nc,1,requires_grad=False,dtype=torch.float64)

        dQP_settings = dQP.build_settings(qp_solver="piqp", lin_solver="scipy SPLU", solve_type="sparse", normalize_constraints=False, eps_abs=tolerance, eps_rel=0, eps_active=eps_active,empty_batch=False)

        self.dQP_layer = dQP.dQP_layer(settings=dQP_settings)
        self.ReLU = torch.nn.ReLU()

    def forward(self):
        Lvals = -self.ReLU(-self.Lvals) - 1e-2 # thresohld to be negative
        Lvals = torch.cat((Lvals,Lvals)) # duplicate for symmetry
        Linds, Lvals = torch_geometric.utils.get_laplacian(self.symedges, Lvals)
        Lvals = -Lvals # PD convention
        Lvals[Linds[0, :] == Linds[1, :]] += 1e-4 # perturb by eps*I to make PD for QP solver
        Linds = torch.cat((Linds,Linds+self.nv),1)
        Lvals = torch.cat((Lvals,Lvals)) # duplicate for block diagonal
        L = torch.sparse_coo_tensor(Linds, Lvals.double(), (self.nv*self.dim,self.nv*self.dim))
        G = torch.matmul(self.G_precompute,L)

        t = time.time()
        x_star,_,nu_star,_,_ = self.dQP_layer(Q=L.to_sparse_csc(), q=self.q, G=G.to_sparse_csc(), h=self.h, A=self.A, b=self.b)
        dQP_time = time.time() - t

        if time_alternatives and self.nv < 4000: # hard-coded cut-off at ~ 1s
            # print(x_star)

            L_dense = L.to_dense().detach() # don't time conversion
            G_dense = G.to_dense()
            A_dense = self.A.to_dense().detach()

            l_qplayer = -1.0e20 * torch.ones(self.nc * self.dim, dtype=torch.float64)
            qp_layer = lambda Q, q, G, h, A, b: prox_qpfunction(structural_feasibility=True, eps=tolerance)(
                Q, q, A, b, G, l_qplayer, h
            )  # assumes feasibility

            control_scqpth = scqpth_control(eps_abs=tolerance, eps_rel=0)
            lb_scqpth = -1.0e20 * torch.ones((self.nc * self.dim + 2 * self.nb * self.dim, 1), dtype=torch.float64) # 2 cone constraints per vertex ; 2 sets of inequalities for equalities on boundary
            scqpth_layer = lambda Q, q, G, h: SCQPTHNet(control_scqpth)(Q=Q, p=q, A=G, lb=lb_scqpth, ub=h)

            if self.nv < 4000: # hard-coded cut-off at ~ 1s
                t = time.time()
                out = QPFunction(eps=tolerance, notImprovedLim=2,maxIter=60, check_Q_spd=False)(L_dense, self.q.squeeze(), G_dense, self.h.squeeze(), A_dense, self.b.squeeze())
                optnet_time = time.time() - t
                # print(out)
                print("Optnet forward: " + str(time.time() - t))
            else:
                optnet_time = -1

            if self.nv < 600: # hard-coded cut-off at ~ 1s
                t = time.time()
                out = qp_layer(L_dense, self.q.squeeze(), G_dense,self.h.squeeze(), A_dense, self.b.squeeze())
                qplayer_time = time.time() - t
                # print(out)
                print("QPLayer forward: " + str(time.time() - t))
            else:
                qplayer_time = -1

            if self.nv < 2000:
                A,b = A_dense.unsqueeze(0), self.b.unsqueeze(0)
                G,h = G_dense.unsqueeze(0), self.h.unsqueeze(0)
                G = torch.cat((torch.cat((G, A), dim=-2), -A), dim=-2)
                h = torch.cat((torch.cat((h, b), dim=1), -b), dim=1)
                t = time.time() # have to reshape and put equalities into the inequalities
                out = scqpth_layer(L_dense.unsqueeze(0), self.q.unsqueeze(0), G, h)
                scqpth_time = time.time() - t
                # print(out)
                print("Scqpth forward: " + str(time.time() - t))
            else:
                scqpth_time = -1
        else:
            optnet_time = -1
            scqpth_time = -1
            qplayer_time = -1

        return L.to_sparse_csc(),x_star, nu_star, dQP_time, optnet_time, scqpth_time, qplayer_time