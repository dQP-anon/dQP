#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch 
import math
import time 
from torch import nn
import pandas as pd
import numpy as np
import scipy.sparse as spa
import matplotlib.pyplot as plt
import networkx as nx
sys.path.append('../../')
from src import dQP
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer as CvxpyLayer_fun
from proxsuite.torch.qplayer import QPFunction as prox_qpfunction

from deps.qpth_dual.qp import QPFunction
from deps.scqpth_dual.control import scqpth_control
from deps.scqpth_dual.scqpth import SCQPTHNet
from deps.Alt_Diff.numerical_experiment.opt_layer import alt_diff

from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph

#-------------------------------Data generation------------------------------ 

def generate_random_qp(nprob,dim,nIneq,nEq,seed=1,dtype="torch"):
    
    torch.manual_seed(seed)
    P = torch.rand(nprob,dim,dim,dtype=torch.float64)
    Q = torch.bmm(P,P.transpose(1,2)) + 1e-4 * torch.eye(dim).unsqueeze(0).expand(nprob,dim,dim)
    q = torch.rand(nprob,dim,dtype=torch.float64)
    
    G = torch.rand(nprob,nIneq, dim,dtype=torch.float64)
    s0 = torch.ones(nprob,nIneq,1,dtype=torch.float64)
    z0 = torch.ones(nprob,dim,1,dtype=torch.float64)
    h = torch.bmm(G,z0) + s0
    
    
    A = torch.rand(nprob,nEq, dim,dtype=torch.float64)
    b = torch.bmm(A,z0)

    Q.requires_grad_(True)
    G.requires_grad_(True)
    A.requires_grad_(True)
    q.squeeze_(-1).requires_grad_(True)
    h.squeeze_(-1).requires_grad_(True)
    b.squeeze_(-1).requires_grad_(True)
    if dtype=="numpy":
        return Q.detach().numpy(),q.detach().numpy(),G.detach().numpy(),h.detach().numpy(),A.detach().numpy(),b.detach().numpy()
    return Q,q,G,h,A,b


def generate_random_kLaplican(n,seed):
    n_features = 2  
    centers = 3  
    X, _ = make_blobs(n_samples=n, n_features=n_features, centers=centers, random_state=seed)
    
    k = 3  
    knn_graph = kneighbors_graph(X, k, mode='connectivity', include_self=False)
    
    G = nx.from_scipy_sparse_array(knn_graph)
    # csr matrix 
    L = nx.laplacian_matrix(G)
    return L

def generate_sparse_mat(m,dim):
    density = 5e-4
    nnz = int(np.ceil(density * m * dim))
    idx = np.random.randint(0,m,size=nnz)
    miss_idx = set(range(0,m)) - set(idx)
    if miss_idx:
        idx = np.hstack((idx,list(miss_idx)))
    jdx = np.random.randint(0,dim,size=nnz+len(miss_idx))
    mat = spa.coo_matrix((np.random.randn(len(idx)),(idx,jdx)),shape=(m,dim))
    return mat
def generate_random_spaqp(dim,nIneq,nEq, seed=1):

    torch.manual_seed(seed)
    np.random.seed(seed)
    # m = nEq + nIneq
    # csr matrix 
    L = generate_random_kLaplican(dim,seed)
    Q = torch.sparse_coo_tensor(L.nonzero(),L.data,L.shape,dtype=torch.float64)
    # s = max(np.absolute(np.linalg.eigvals(L)))
    # L += (abs(s) + 1e-02) * spa.eye(dim)
    q = torch.randn(dim,dtype=torch.float64)
    
    
    
    A = generate_sparse_mat(nEq,dim)
    G = generate_sparse_mat(nIneq,dim)
    A = torch.sparse_coo_tensor(A.nonzero(),A.data,A.shape,dtype=torch.float64)
    G = torch.sparse_coo_tensor(G.nonzero(),G.data,G.shape,dtype=torch.float64)
    
    s0 = torch.ones(nIneq,1,dtype=torch.float64)
    z0 = torch.ones(dim,1,dtype=torch.float64)
    
    h = G @ z0 + s0
    b = A @ z0

    Q.requires_grad_(True)
    G.requires_grad_(True)
    A.requires_grad_(True)
    q.requires_grad_(True)
    h.squeeze_(-1).requires_grad_(True)
    b.squeeze_(-1).requires_grad_(True)
    return Q,q,G,h,A,b
def __Ab_conversion(model_name,Q,A,b):
    # Only refers to the case when A,b is None
    if model_name in ['OptNet','QPLayer']:
        A = torch.autograd.Variable(torch.Tensor())
        b = torch.autograd.Variable(torch.Tensor())

    # elif model_name=='QPLayer':
    #     # Take batch_size==1
    #     A = torch.zeros((1,1,Q.shape[-1]),requires_grad=False).type(Q.type())
    #     b = torch.zeros((1,1),requires_grad=False).type(Q.type())
  
    return A,b

def __sparse_eye(n):
    idx = torch.arange(n)
    val = torch.ones(n)
    mat = torch.sparse_coo_tensor(
       torch.stack([idx, idx]),
       val,
       (n, n),dtype=torch.float64)
    mat.requires_grad_(True)
    return mat.to_sparse_csc()


def load_model(model_name,dim,nIneq,nEq,eps,eps_active,prob_type='dense'):
    # OptNet
    if  model_name=='OptNet':
        model = QPFunction(verbose=False,eps=eps)
    # QPLayer
    # settings
    maxIter = 100
    omp_parallel = False
    
    # QPLayer
    if model_name=='QPLayer':
        l_qplayer = -1.0e20 * torch.ones(nIneq, dtype=torch.float64)
        
        model = lambda Q,q,G,h,A,b : prox_qpfunction(structural_feasibility=True,maxIter=maxIter, eps=eps, omp_parallel=omp_parallel)(
            Q, q, A, b, G, l_qplayer, h
        ) # assumes feasibility
    
    # Cvxpy
    if model_name=='Cvxpy':
        Q_sqrt_ = cp.Parameter(( dim , dim ) )
        q_ = cp.Parameter( dim )
        G_ = cp.Parameter(( nIneq , dim ) )
        h_ = cp.Parameter( nIneq )
        x_ = cp.Variable( dim )
        obj = cp.Minimize(0.5* cp.sum_squares( Q_sqrt_ @ x_ ) + q_.T @ x_)

        if nEq>0:           
            A_ = cp.Parameter(( nEq , dim ) )
            b_ = cp.Parameter( nEq )
            
            # obj = cp.Minimize(0.5* x_.T @ Q @ x_ + q_.T @ x_ )
            cons = [ A_ @ x_ == b_ , G_ @ x_ <= h_ ]
            prob = cp.Problem( obj , cons)
            
            model = CvxpyLayer_fun(prob, parameters=[Q_sqrt_, q_, G_, h_, A_, b_], variables=[x_])
        else:
            cons = [G_ @ x_ <= h_]
            prob = cp.Problem( obj , cons)            
            model = CvxpyLayer_fun(prob, parameters=[Q_sqrt_, q_, G_, h_], variables=[x_])
    
    if model_name=='SCQPTH':
        control_scqpth = scqpth_control(eps_abs=eps,eps_rel=0)
        lb_scqpth = -1.0e20 * torch.ones((1,nIneq+2*nEq,1), dtype=torch.float64)
        model = lambda Q,q,G,h: SCQPTHNet(control_scqpth)(Q=Q,p=q,A=G,lb=lb_scqpth,ub=h)

    
    # Alt-Diff
    if model_name=='Alt-Diff':
        model = lambda Q,q,G,h,A,b: alt_diff(Q,q,A,b,G,h,thres=eps)
        
    # dQP
    if model_name[:3]=='dQP':
        dQP_fw_solver = model_name[4:]
        if prob_type=='sparse':
            lin_solver = 'scipy SPLU'            
        else:
            lin_solver = 'scipy LU'
        anet_settings = dQP.build_settings(qp_solver=dQP_fw_solver,solve_type=prob_type,lin_solver=lin_solver,normalize_constraints=False,time=False,eps_abs=eps,eps_rel=0,eps_active=eps_active) # intentionally set higher tolerance
        model = dQP.dQP_layer(anet_settings)
    return model

def forward_solve(model_name,model,prob_type,if_eq,Q,q,G,h,A,b):
    start_time = time.time()
    x,mu,nu = None,None,None
    #-------------------OptNet-------------------
    if model_name=='OptNet':
        x,(mu,nu) = model(Q,q,G,h,A,b)
    #-------------------QPLayer-------------------
    elif model_name=='QPLayer':
        x,mu,nu = model(Q,q,G,h,A,b)
        x.unsqueeze_(-1)
        if if_eq:
            mu.unsqueeze_(-1)
        nu.unsqueeze_(-1)
        
    #-------------------dQP-------------------
    
    elif model_name[:3]=='dQP':
        # Q1,q1,G1,h1 = Q[0],q[0],G[0],h[0]
        # if A is not None:
        #     A1,b1 = A[0],b[0]
        # else:
        #     A1,b1 = None,None
            
        
        if prob_type=='sparse':
             
            if A is not None:
                if Q.dim()==3 and Q.shape[0]==1:
                    Q1,q1,G1,h1,A1,b1 = Q[0].to_sparse_csc(), q[0].unsqueeze(-1), G[0].to_sparse_csc(), h[0].unsqueeze(-1), A[0].to_sparse_csc(), b[0].unsqueeze(-1)
                else:
                    Q1,q1,G1,h1,A1,b1 = Q.to_sparse_csc(), q.unsqueeze(-1), G.to_sparse_csc(), h.unsqueeze(-1), A.to_sparse_csc(), b.unsqueeze(-1)

            else:
                if Q.dim()==3 and Q.shape[0]==1:
                    Q1,q1,G1,h1 = Q[0].to_sparse_csc(), q[0].unsqueeze(-1), G[0].to_sparse_csc(), h[0].unsqueeze(-1)
                else:
                    Q1,q1,G1,h1 = Q.to_sparse_csc(), q.unsqueeze(-1), G.to_sparse_csc(), h.unsqueeze(-1)

                A1,b1 = None, None
        else:
            if A is not None:
                if Q.dim()==3 and Q.shape[0]==1:
                    Q1,q1,G1,h1,A1,b1 = Q[0],q[0].unsqueeze(-1),G[0],h[0].unsqueeze(-1),A[0],b[0].unsqueeze(-1)
                else:
                    Q1,q1,G1,h1,A1,b1 = Q,q.unsqueeze(-1),G,h.unsqueeze(-1),A,b.unsqueeze(-1)

            else:
                if Q.dim()==3 and Q.shape[0]==1:
                    Q1,q1,G1,h1 = Q[0],q[0].unsqueeze(-1),G[0],h[0].unsqueeze(-1)

                else:
                    Q1,q1,G1,h1 = Q,q.unsqueeze(-1),G,h.unsqueeze(-1)
                A1,b1 = None, None
        start_time = time.time()
        x,mu,nu,_,_ = model(Q1,q1,G1,h1,A1,b1)
        x.unsqueeze_(-1)
        if if_eq:
            mu.unsqueeze_(-1)
        nu.unsqueeze_(-1)
        
        
        
    #-------------------Cvxpy-------------------
    elif model_name=='Cvxpy':
        Q_sqrt = torch.linalg.cholesky(Q,upper=True)
        x, = model(Q_sqrt, q, G, h, A, b)
    #-------------------SCQPTH-------------------
    elif model_name=='SCQPTH':
        if Q.dim()==2:
            Q_new,q_new,G_new,h_new = Q.unsqueeze(0),q.unsqueeze(0),G.unsqueeze(0),h.unsqueeze(0)
        else:
            Q_new,q_new,G_new,h_new = Q,q,G,h
        nIneq = G.shape[1]
        if if_eq:
            if Q.dim()==2:
                A_new, b_new = A.unsqueeze(0),b.unsqueeze(0)
            else:
                A_new,b_new = A,b
                
            G_new = torch.cat( (torch.cat((G_new,A_new),dim=-2),-A_new),dim=-2)
            h_new = torch.cat( (torch.cat((h_new,b_new),dim=-1),-b_new),dim=-1)
        else:
            G_new,h_new = G,h
        start_time = time.time()
        x,nu = model(Q_new,q_new.unsqueeze(-1),G_new,h_new.unsqueeze(-1))
        
       
    forward_time = time.time() - start_time
    return x,mu,nu,forward_time


#-------------------------------Benchmark------------------------------ 
def eq_residual(z,A,b,if_eq):
    if not if_eq:
        return None
    else:
        
        b = b.unsqueeze(-1)

        if A.dim()==3:
            val = torch.max(torch.abs(torch.bmm(A,z) - b))
        elif A.dim()==2:
            if (z.dim()==3) and (z.shape[0]==1):
                z = z.squeeze(0)
            val = torch.max(torch.abs(A @ z - b))
        return val.detach().numpy()

def ineq_residual(z,G,h):
    h = h.unsqueeze(-1)

    if G.dim()==3:
        val = torch.maximum(torch.max(torch.bmm(G,z) - h), torch.zeros(1)).squeeze()
    elif G.dim()==2:
        if (z.dim()==3) and (z.shape[0]==1):
            z = z.squeeze(0)
        val = torch.maximum(torch.max(G@z - h), torch.zeros(1)).squeeze()

    return val.detach().numpy()
def dual_residual(x,y,z,Q,q,G,h,A,b,if_eq):
    q = q.unsqueeze(-1)
    h = h.unsqueeze(-1)
    
    if Q.dim()==3:

        if not if_eq:
            r_d = torch.bmm(Q,x) + q + torch.bmm(G.transpose(1,2),y) 
        else:
            b = b.unsqueeze(-1)
            r_d = torch.bmm(Q,x) + q + torch.bmm(G.transpose(1,2),y) + torch.bmm(A.transpose(1,2),z)  
    elif Q.dim()==2:
        if (x.dim()==3) and (x.shape[0]==1):
            x = x.squeeze(0)
            y = y.squeeze(0)  
            if if_eq:
                z = z.squeeze(0)  
        if not if_eq:
            r_d = Q @ x + q + G.transpose(-1,-2) @ y 
        else:
            b = b.unsqueeze(-1)
            r_d = Q @ x + q + G.transpose(-1,-2) @ y  + A.transpose(-1,-2) @ z  
        
    return torch.max(torch.abs(r_d)).detach().numpy()


def duality_gap(x, y, z, Q, q, G, h, A, b, if_eq):
    q = q.unsqueeze(-1)
    h = h.unsqueeze(-1)

    if Q.dim() == 3:
        if not if_eq:
            r_g = torch.bmm(torch.bmm(x.transpose(1, 2), Q), x) + \
                torch.bmm(q.transpose(1, 2), x) + \
                torch.bmm(h.transpose(1, 2), y)
        else:
            b = b.unsqueeze(-1)
            r_g = torch.bmm(torch.bmm(x.transpose(1, 2), Q), x) + torch.bmm(q.transpose(
                1, 2), x) + torch.bmm(b.transpose(1, 2), z) + torch.bmm(h.transpose(1, 2), y)
    elif Q.dim() == 2:
        if (x.dim()==3) and (x.shape[0]==1):
            x = x.squeeze(0)
            y = y.squeeze(0)  
            if if_eq:
                z = z.squeeze(0)  
        if not if_eq:
            r_g = x.transpose(-1, -2) @ Q @ x + q.T @ x + h.T @ y
        else:
            b = b.unsqueeze(-1)
            r_g = x.transpose(-1, -2) @ Q @ x + q.T @ x + h.T @ y + b.T @ z
    return torch.max(torch.abs(r_g)).detach().numpy()


def objective_value(x, Q, q):
    q = q.unsqueeze(-1)
    if Q.dim()==3:    
        val = 1/2 * torch.bmm(torch.bmm(x.transpose(1,2),Q),x) + torch.bmm(q.transpose(1,2),x)
    elif Q.dim()==2:
        x = x.squeeze(0)
        val = 1/2 * x.T @ Q @ x + q.T @ x
    return val.squeeze().detach().numpy()



def plot_benchmark(model_names,eq_reses,ineq_reses,dual_reses,dual_gaps,title,data_name):
    fig, axs = plt.subplots(2, 2,figsize=(10,8))

    for i in range(len(model_names)):
        axs[0,0].plot(eq_reses[:,i])
        axs[0,1].plot(ineq_reses[:,i])
        axs[1,0].plot(dual_reses[:,i])
        axs[1,1].plot(dual_gaps[:,i])
    axs[0,0].set_title('Equality residual')
    axs[0,0].set_xlabel('Data id')
    axs[0,0].legend(model_names,loc='best')  
    axs[0,0].set_yscale('log')

    axs[0,1].set_title('Inequality residual')
    axs[0,1].set_xlabel('Data id')
    axs[0,1].set_yscale('log')


    axs[1,0].set_title('Dual residual')
    axs[1,0].set_xlabel('Data id')
    axs[1,0].set_yscale('log')

    axs[1,1].set_title('Duality gap')
    axs[1,1].set_xlabel('Data id')
    axs[1,1].set_yscale('log')

    fig.tight_layout()
    plt.savefig('./results/' + data_name + '_res.png')
    st = fig.suptitle(title, fontsize=14)
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
    
