#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys
import os
import numpy as np
import exp_utils
import scipy.sparse as spa

def generate_randomqp_dense(): 
    
    for dim in dims:
        nIneq = dim
        nEq = int(np.ceil(dim/2))
        for i in range(nprob):
            Q,q,G,h,A,b = exp_utils.generate_random_qp(1,dim,nIneq,nEq,seed=i)
            prob_name = 'n' + str(dim) + '_ineq' + str(nIneq) + '_eq' + str(nEq) + '_' + str(i)
            np.savez(file_path + '/' + prob_name + '.npz',
                      dataset=data_name,
                      Q=Q[0].detach().numpy(),
                      q=q[0].detach().numpy(),
                      G=G[0].detach().numpy(),
                      h=h[0].detach().numpy(),
                      A=A[0].detach().numpy(),
                      b=b[0].detach().numpy())

def __coo_to_spa(M):
    M = M.detach().coalesce()
    ind = M.indices().numpy()
    val = M.values().numpy()
    shape = tuple(M.size())
    Mspa = spa.csc_matrix((val,(ind[0],ind[1])),shape=shape)
    return Mspa

def generate_randomqp_sparse():

    for dim in dims:
        nIneq = dim
        nEq = int(np.ceil(dim/2))
        for i in range(nprob):
            Q,q,G,h,A,b = exp_utils.generate_random_spaqp(dim,nIneq,nEq,seed=i)
            Q,G,A = __coo_to_spa(Q),__coo_to_spa(G),__coo_to_spa(A)
            q,h,b = q.detach().numpy(),h.detach().numpy(),b.detach().numpy()
            prob_name = 'n' + str(dim) + '_ineq' + str(nIneq) + '_eq' + str(nEq) + '_' + str(i)
            np.savez(file_path + '/' + prob_name + '.npz',
                      dataset=data_name,
                      Q=Q,
                      q=q,
                      G=G,
                      h=h,
                      A=A,
                      b=b)
def generate_random_projection():

    for dim in dims:
        Q = spa.eye(dim,format="csc")
        G = - spa.eye(dim,format="csc")
        h = np.zeros(dim)
        A = spa.csc_matrix(np.ones((1,dim)))
        b = np.ones(1)
        for i in range(nprob):
            np.random.seed(i)
            q = np.random.randn(dim)
            prob_name = 'n' + str(dim) + '_' + str(i)
            np.savez(file_path + '/' + prob_name + '.npz',
                      dataset=data_name,
                      Q=Q,
                      q=q,
                      G=G,
                      h=h,
                      A=A,
                      b=b)
def generate_chain():

    N = 100
    for n in ns:
        

        Q = 0.5 * spa.eye(N * n)
        

        
        G0 = spa.lil_matrix((N, N))
        L = spa.lil_matrix((N, N))

        # Set the diagonal to 1
        for i in range(N):
            G0[i, i] = 1
            L[i, i] = 2
        
        # Create the non-zero entries for G0 (off-diagonal)
        row, col = np.diag_indices(N)
        
        row1 = row[0:N-1] 
        col1 = col[0:N-1] + 1
        G0[row1, col1] = -1
        G0[-1, 0] = -1
        
        L[row1, col1] = -1
        
        row2 = row[0:N-1] + 1 
        col2 = col[0:N-1] 
        L[row2, col2] = -1
        
        L1 = spa.kron(L,spa.eye(n))
        BHL = L1.T @ L1 
        lamb = 1
        
        Q += lamb * BHL
        
        # Convert G0 to CSC format after modifications
        G0 = G0.tocsc()
        Q = Q.tocsc()
        
        I_n = spa.eye(n, format="csc")
        
        G1 = spa.kron(G0, I_n, format="csc")
        G = spa.vstack((G1, -G1))

        h = np.ones(2*N*n)

        A = None 
        b = None 
        for i in range(nprob):
            np.random.seed(i)
            pt = 10 * np.random.randn(N, n)
            x = pt.reshape(-1)  
            q = -x
            dim = N*n
            prob_name = 'n' + str(dim) + '_' + str(i)
            np.savez(file_path + '/' + prob_name + '.npz',
                      dataset=data_name,
                      Q=Q,
                      q=q,
                      G=G,
                      h=h,
                      A=A,
                      b=b)
      
        
        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_name', metavar='data_name', type=str)

    args = parser.parse_args()
    data_name = args.data_name
   
    file_path = './data/' + data_name 
    
    data_names = ["randomqp_dense",
                  "randomqp_sparse",
                  "random_projection",
                  "chain"]
    if not (data_name in data_names):
        raise Exception("Wrong data_name, which should be one of " + str(data_names))
    else:
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if data_name=="randomqp_dense":
            nprob = 50
            dims = [10, 20, 50, 100, 220, 450, 1000, 2100, 4600]
            generate_randomqp_dense()
        elif data_name=="randomqp_sparse":
            nprob = 100
            dims = [100, 220, 450, 1000, 2100, 4600]
            generate_randomqp_sparse()
            
            nprob= 25
            dims = [10000]
            generate_randomqp_sparse()
        elif data_name=="random_projection":
            nprob = 50
            dims = [10, 20, 50, 100, 220, 450, 1000, 2100, 4600]
            generate_random_projection()
            
            nprob= 25
            dims = [10000,100000]
            generate_random_projection()
            
        elif data_name=="chain":
            nprob = 50
            ns = [2,5,10,20,40]
            generate_chain()
            
            nprob = 25
            ns = [100,1000]
            generate_chain()
        
        
        
        print(data_name + " :data generation is finished.")
        
        
        
    
        