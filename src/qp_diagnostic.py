import numpy as np
import scipy as sp
import sys

import qpsolvers
import os
import time

import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('../../src')
import dQP
import lin_solvers

# use in debugging mode in forward of dQP
# example:
# from src.qp_diagnostic import save_for_benchmark

# save_for_benchmark("cross",sparse_helper.csc_torch_to_scipy(Q),q.detach().numpy(),sparse_helper.csc_torch_to_scipy(G),h.detach().numpy(),sparse_helper.csc_torch_to_scipy(A),b.detach().numpy())
# save_for_benchmark("sudoku3x3",Q.detach().numpy(),q.detach().numpy(),G.detach().numpy(),h.detach().numpy(),A.detach().numpy(),b.detach().numpy())
# save_for_benchmark("random_reduced_vs_full",Q.detach().numpy(),q.detach().numpy(),G.detach().numpy(),h.detach().numpy(),A.detach().numpy(),b.detach().numpy())
def save_for_benchmark(save_name,Q,q,G,h,A,b):
    file_path = os.path.dirname(os.path.realpath(__file__)) # reference this file
    save_path = file_path + "/../experiments/diagnostic/data/" + save_name + ".npz"
    np.savez(save_path, Q=Q,q=q,G=G,h=h,A=A,b=b)
    return

def test_qp(Q,q,G,h,A,b,n_sample,available_qp_solvers):
    n_solvers = len(available_qp_solvers)
    main_kwargs = {"verbose": False}

    if "gurobi" in available_qp_solvers:
        ref_solver = "gurobi"
    else:
        ref_solver = "cvxopt"
    ref_solver_kwargs = dQP.set_solver_tolerance(main_kwargs.copy(), ref_solver, 1e-8, 1e-8)
    ref_solver_kwargs["solver"] = ref_solver
    ref_solution = dQP.call_single_qpsolvers(Q, q, G, h, A, b, ref_solver_kwargs)
    x_ref = ref_solution.x

    t_qp = np.zeros((3,n_solvers,n_sample))
    e_qp = t_qp.copy()

    eps_abs_list = [1e-8, 1e-5, 1e-2]
    eps_rel_list = eps_abs_list
    for ii in range(n_solvers):
        solver = available_qp_solvers[ii]
        solver_kwargs = main_kwargs.copy() # copy otherwise will change main_kwargs through solver_kwargs
        solver_kwargs["solver"] = solver
        # print("------------------------------------------")
        # print(solver)
        # print("------------------------------------------")
        for jj in range(3):
            # print("------------------------------------------")
            # print(eps_abs_list[jj])
            # print("------------------------------------------")
            solver_kwargs = dQP.set_solver_tolerance(solver_kwargs, solver, eps_abs_list[jj], eps_rel_list[jj])
            for kk in range(n_sample):
                # print("Sample:" + str(kk))
                t0 = time.time()

                try:
                    solution = dQP.call_single_qpsolvers(Q, q, G, h, A, b, solver_kwargs)
                    t1 = time.time()
                except Exception as e:
                    print(repr(e))
                    t_qp[jj,ii,kk] = np.infty
                    e_qp[jj,ii,kk] = np.infty
                    break

                t_qp[jj,ii,kk] = t1-t0
                e_qp[jj,ii,kk] = np.linalg.norm(solution.x-x_ref)/np.linalg.norm(x_ref)

    return np.nanmean(t_qp,axis=-1),np.nanstd(t_qp,axis=-1),np.nanmean(e_qp,axis=-1)

def test_linear(Q,q,G,h,A,b,n_sample,available_linear_solvers,available_qp_solvers):
    if b is not None:
        nEq = b.size
    else:
        nEq = 0
    dim = q.size
    nIneq = h.size

    n_solvers = len(available_linear_solvers)
    main_kwargs = {"verbose": False}

    if "scipy LU" in available_linear_solvers:
        ref_linear_solver = "scipy LU"
        is_sparse = False
        lin_solver = lin_solvers.dense_solve
    else:
        is_sparse = True
        lin_solver = lin_solvers.sparse_solve
        ref_linear_solver = "scipy SPLU"

    if "gurobi" in available_qp_solvers: # QP included
        ref_qp_solver = "gurobi" # TODO : maybe need to choose a solver more sensitive to tolerance choice?
        # ref_qp_solver = "piqp"
    else:
        ref_qp_solver = "cvxopt" # the difference between cvxopt and dapq is significant! O(1) error v.s. O(1e-13)
        # ref_qp_solver = "daqp"


    normalize = True
    if normalize:
        G, h, A, b = normalize_for_derivative(G, h, A, b, is_sparse)
        
    main_kwargs["solver"] = ref_qp_solver
    ref_solver_kwargs = dQP.set_solver_tolerance(main_kwargs.copy(), ref_qp_solver, 1e-8, 1e-8)
    ref_solution = dQP.call_single_qpsolvers(Q, q, G, h, A, b, ref_solver_kwargs)
    x_ref = np.expand_dims(ref_solution.x,-1)
    mu_ref = np.expand_dims(ref_solution.y,-1)
    nu_ref = np.expand_dims(ref_solution.z,-1)

    D_ref = get_full_D(Q, q, G, h, A, b, x_ref, nu_ref, is_sparse)
    if is_sparse:
        sv_max = sp.sparse.linalg.svds(D_ref, return_singular_vectors=False, k=1, which='LM')[0]
        sv_min = sp.sparse.linalg.svds(D_ref, return_singular_vectors=False, k=1, which='SM')[0]
        D_cond = sv_max / sv_min
    else:
        D_cond = np.linalg.cond(D_ref)

    t = np.zeros((3,n_solvers,n_sample))
    e = t.copy()
    t_full = t.copy()
    e_full = t.copy()

    incoming_grad = np.random.randn(D_ref.shape[-1],1)
    grad_ref,_ = lin_solver(D_ref,incoming_grad,linear_solver=ref_linear_solver)

    Q_pattern = None if not is_sparse else Q.nonzero()
    G_pattern = None if not is_sparse else G.nonzero()
    A_pattern = None if not is_sparse else A.nonzero()

    grads_ref = extract_grads(x_ref, mu_ref, nu_ref, None, grad_ref, dim ,nIneq, nEq, "full",
                                                                   Q_pattern=Q_pattern, G_pattern=G_pattern,A_pattern=A_pattern)

    eps_list = [1e-8, 1e-5, 1e-2]
    for ii in range(n_solvers):
        for jj in range(3):
            solver_kwargs = dQP.set_solver_tolerance(main_kwargs.copy(), main_kwargs["solver"], eps_list[jj], eps_list[jj])
            for kk in range(n_sample):
                # print(available_linear_solvers[ii])
                try:
                    solution = dQP.call_single_qpsolvers(Q, q, G, h, A, b, solver_kwargs)
                    x = np.expand_dims(solution.x, -1)
                    mu = np.expand_dims(solution.y,-1)
                    nu = np.expand_dims(solution.z, -1)

                    D = get_full_D(Q, q, G, h, A, b, x, nu, is_sparse)
                    t2 = time.time()
                    grad_full, _ = lin_solver(D, incoming_grad, linear_solver=ref_linear_solver)
                    t3 = time.time()
                    grads_full = extract_grads(x, mu, nu, None, grad_full, dim, nIneq, nEq, "full",
                                               Q_pattern=Q_pattern, G_pattern=G_pattern, A_pattern=A_pattern)

                    K,active = get_reduced_KKT(Q,q,G,h,A,b,x,nu,is_sparse,1e-7)
                    nu = nu[active,:]
                    # print(np.sum(active))
                    G_pattern =  None if not is_sparse else (G.tocsr()[active, :]).nonzero()

                    incoming_grad_active = incoming_grad[np.concatenate((np.ones(dim + nEq,dtype=np.bool_),active)),:]
                    if available_linear_solvers[ii] in available_qp_solvers or available_linear_solvers[ii] == "cholespy":
                        QP_form = [K[0:dim, 0:dim], -incoming_grad_active[0:dim], K[dim:, 0:dim], incoming_grad_active[dim:]]
                        t0 = time.time()
                        grad, _= lin_solver(K,incoming_grad_active,linear_solver=available_linear_solvers[ii],QP_form=QP_form)
                        t1 = time.time()
                    else:
                        t0 = time.time()
                        grad, _ = lin_solver(K, incoming_grad_active, linear_solver=available_linear_solvers[ii], QP_form=None)
                        t1 = time.time()

                    if grad is None:
                        t[jj, ii, kk] = np.infty
                        e[jj, ii, kk] = np.infty
                        break

                except Exception as e:
                    print(repr(e))
                    t[jj,ii,kk] = np.infty
                    e[jj,ii,kk] = np.infty
                    break

                t[jj,ii,kk] = t1-t0
                t_full[jj,ii,kk] = t3-t2

                grads = extract_grads(x, mu, nu, active, grad, dim ,nIneq, nEq, "reduced",Q_pattern=Q_pattern,G_pattern=G_pattern,A_pattern=A_pattern)

                if is_sparse:
                    for mm in range(6):
                        if mm % 2 == 0:
                            e[jj,ii,kk] += sp.sparse.linalg.norm(grads_ref[mm]-grads[mm])/sp.sparse.linalg.norm(grads_ref[mm])
                            e_full[jj,ii,kk] += sp.sparse.linalg.norm(grads_ref[mm]-grads_full[mm])/sp.sparse.linalg.norm(grads_ref[mm])
                        else:
                            e[jj,ii,kk] += np.linalg.norm(grads_ref[mm]-grads[mm])/np.linalg.norm(grads_ref[mm])
                            e_full[jj,ii,kk] += np.linalg.norm(grads_ref[mm]-grads_full[mm])/np.linalg.norm(grads_ref[mm])
                else:
                    for mm in range(6):
                        e[jj,ii,kk] += np.linalg.norm(grads_ref[mm]-grads[mm])/np.linalg.norm(grads_ref[mm])
                        e_full[jj,ii,kk] += np.linalg.norm(grads_ref[mm]-grads_full[mm])/np.linalg.norm(grads_ref[mm])

                print(e[jj,ii,kk])

    return np.nanmean(t,axis=-1),np.nanstd(t,axis=-1),np.nanmean(e,axis=-1), np.nanmean(t_full,axis=-1), np.nanstd(t_full,axis=-1), np.nanmean(e_full,axis=-1), D_cond, ref_qp_solver

def normalize_for_derivative(G,h,A,b,is_sparse):
    if is_sparse:
        N_G = np.expand_dims(sp.sparse.linalg.norm(G,ord=2,axis=1),-1)
        G = G._divide(N_G)
        h = np.divide(h,N_G)
        if A is not None and b is not None:
            N_A = np.expand_dims(sp.sparse.linalg.norm(A, ord=2, axis=1),-1)
            A = A._divide(N_A)
            b = np.divide(b, N_A)
    else:
        N_G = np.expand_dims(sp.linalg.norm(G, ord=2, axis=1), -1)
        G = np.divide(G, N_G)
        h = np.divide(h, N_G)
        if A is not None and b is not None:
            N_A = np.expand_dims(sp.linalg.norm(A, ord=2, axis=1), -1)
            A = np.divide(A, N_A)
            b = np.divide(b, N_A)

    return G,h,A,b

# TODO : for reduced expand Q,G,A to be (1) sparse matrices instead of vectors of nonzero entries (2) put in 0 inactive rows for G and h
# TODO : then iterate through both grads, take their norm difference, and then sum ... done

def extract_grads(x,mu,nu,active,grad,dim,nIneq,nEq,type,Q_pattern=None,G_pattern=None,A_pattern=None):
    dx = -grad[0:dim, :]
    dmu = -grad[dim:(dim + nEq), :]
    dnu = -grad[(dim + nEq):, :]

    if type == "full":
        diag_nu = sp.sparse.diags_array(nu.squeeze())
    elif type == "reduced":
        diag_nu = sp.sparse.eye(nu.size)

    if Q_pattern is None:
        grad_Q = np.outer(dx, x)
        grad_Q = 1 / 2 * (grad_Q + grad_Q.T)
    else:
        i_row = Q_pattern[0]
        i_col = Q_pattern[1]
        grad_Q = 1 / 2 * (np.multiply(dx[i_row], x[i_col]) + np.multiply(x[i_row], dx[i_col])).squeeze()
        grad_Q = sp.sparse.coo_matrix((grad_Q,(i_row,i_col)), shape=(dx.size,dx.size))

    if G_pattern is None:
        grad_G = diag_nu @  np.outer(dnu, x) + np.outer(nu, dx)
        if type == "reduced":
            grad_G_complete = np.zeros((nIneq,dim))
            grad_G_complete[active,:] = grad_G
            grad_G = grad_G_complete
    else:
        i_row = G_pattern[0]
        i_col = G_pattern[1]
        if type == "full":
            grad_G = (np.multiply(np.multiply(nu[i_row],dnu[i_row]), x[i_col]) + np.multiply(nu[i_row], dx[i_col])).squeeze()
        else:
            grad_G = (np.multiply(dnu[i_row], x[i_col]) + np.multiply(nu[i_row], dx[i_col])).squeeze()
        grad_G = sp.sparse.coo_matrix((grad_G,(i_row,i_col)), shape=(dnu.size,dx.size))
        if type == "reduced":
            grad_G_complete = sp.sparse.coo_matrix((nIneq,dim)).tocsr()
            grad_G_complete[active,:] = grad_G
            grad_G = grad_G_complete


    if A_pattern is None:
        grad_A = np.outer(dmu, x) + np.outer(mu, dx)
    else:
        i_row = A_pattern[0]
        i_col = A_pattern[1]
        grad_A = (np.multiply(dmu[i_row], x[i_col]) + np.multiply(mu[i_row], dx[i_col])).squeeze()
        grad_A = sp.sparse.coo_matrix((grad_A,(i_row,i_col)), shape=(dmu.size,dx.size))
    
    grad_q = dx
    grad_h = diag_nu @ -dnu
    if type == "reduced":
        grad_h_complete = np.zeros((nIneq,1))
        grad_h_complete[active,:] = grad_h
        grad_h = grad_h_complete

    grad_b = -dmu
    
    return grad_Q,grad_q,grad_G,grad_h,grad_A,grad_b
    
# get the reduced KKT
# TODO ; add a function preceding this for refinement algorithm
def get_reduced_KKT(Q,q,G,h,A,b,x,nu,is_sparse,eps_active):
    if b is not None:
        nEq = b.size
    else:
        nEq = 0

    if is_sparse:
        solve_type = "sparse"
    else:
        solve_type = "dense"

    if is_sparse:
        G = G.tocsr()

    h_approx = G.dot(x)
    r_pri = h - h_approx  # h - Gx^*
    r_pri = r_pri.squeeze()

    active = np.squeeze(np.logical_and(r_pri < eps_active, nu.squeeze() > eps_active))
    nActive = np.sum(active, axis=-1)
    nEq_reduce = nActive + nEq

    # form the effective equality constraints
    if nEq == 0:
        A_reduce = G[active, :]

    else:
        if solve_type == "dense":
            A_reduce = np.vstack((A, G[active,:]))
        elif solve_type == "sparse":
            A_reduce = sp.sparse.vstack((A, G[active,:]), format="csc")

    if solve_type == "dense":
        K = np.bmat([[Q, np.transpose(A_reduce)],
                                   [A_reduce, np.zeros((nEq_reduce, nEq_reduce))]])
    elif solve_type == "sparse":
        K = sp.sparse.bmat([[Q, np.transpose(A_reduce)], [A_reduce, None]], format="csc")

    return K,active


# get the full derivative matrix
def get_full_D(Q,q,G,h,A,b,x,nu,is_sparse):
    if b is not None:
        nEq = b.size
    else:
        nEq = 0
    nIneq = h.size

    h_approx = G.dot(x)
    r_pri = h - h_approx  # h - Gx^*
    r_pri = r_pri.squeeze()

    if nEq > 0:
        if not is_sparse:
            D = np.bmat([
                [Q, np.transpose(A), np.transpose(G) @ sp.sparse.diags_array(nu.squeeze())],
                [A, np.zeros((nEq,nEq+nIneq))],
                [G, np.zeros((nIneq,nEq)),-np.diag(r_pri)]
            ])
        elif is_sparse:
            D = sp.sparse.bmat([
                [Q, np.transpose(A),np.transpose(G) @ sp.sparse.diags_array(nu.squeeze())],
                [A, None, None],
                [G, None, -sp.sparse.diags_array(r_pri)]
            ], format="csc")
    else:
        if not is_sparse:
            D = np.bmat([
                [Q,  np.transpose(G) @ sp.sparse.diags_array(nu.squeeze())],
                [G, -np.diag(r_pri)]
            ])
        elif is_sparse:
            D = sp.sparse.bmat([
                [Q, np.transpose(G) @ sp.sparse.diags_array(nu.squeeze())],
                [G, -sp.sparse.diags_array(r_pri)]
            ], format="csc")

    return D


def visualize_qp(t,t_sd,e,solvers,save_name):
    i_sort = np.argsort(t[0,:]) # sort solvers by time for highest accuracy tolerance
    t = t[:,i_sort]
    t_sd = t_sd[:,i_sort]
    e = e[:,i_sort]
    solvers = [solvers[i] for i in i_sort.tolist()]

    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(111)

    e_label = [[],[],[]]
    for i in range(3):
        for j in range(len(i_sort)):
            if np.abs(e[i,j]) < 1e-16:
                e_label[i] += ["ref"]
            else:
                # e_label[i] += ["%0.1E" % e[i,j]]
                e_label[i] += [str(int(np.floor(np.log10(e[i,j])))) if e[i,j] != np.inf else "inf"]

    data = {
        '1e-8': [t[0,:],t_sd[0,:],e_label[0]],
        '1e-5': [t[1,:],t_sd[1,:],e_label[1]],
        '1e-2': [t[2,:],t_sd[2,:],e_label[2]],
    }

    x = np.arange(len(solvers))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    for attribute, measurement in data.items():
        offset = width * multiplier
        ax.bar(x + offset, measurement[0], width, label=attribute)

        for ii in range(len(x)):
            ax.text(
                x[ii] + offset + width / 2, 1.3 * measurement[0][ii], measurement[2][ii], ha="center", va="bottom"
            )

        ax.errorbar(x + offset,measurement[0], measurement[1],
                     fmt='.', color='Black', elinewidth=2, capthick=10,
                     errorevery=1, alpha=0.5, ms=4,capsize=2)

        multiplier += 1

    # plt.tight_layout()
    ax.set_xticks(x + width, tuple(solvers))
    ax.legend(loc='upper left', ncols=3)
    ax.set_xlabel("QP Solver")
    ax.set_ylabel("Time (s)")
    ax.set_yscale('log')

    file_path = os.path.dirname(os.path.realpath(__file__))  # reference this file
    fig_path = file_path + "/../experiments/diagnostic/results/" + save_name + "_qp.pdf"
    plt.savefig(fig_path)
    # plt.show()

    return None

def visualize_linear(t,t_sd,e,cond,solvers,save_name,is_full,ref_qp_solver,fig=None):
    i_sort = np.argsort(t[0,:]) # sort solvers by time for highest accuracy tolerance
    t = t[:,i_sort]
    t_sd = t_sd[:,i_sort]
    e = e[:,i_sort]
    solvers = [solvers[i] for i in i_sort.tolist()]

    if fig is None:
        fig = plt.figure(figsize=(13, 7))
        ax = fig.add_subplot(211)
    else:
        ax = fig.add_subplot(212)

    e_label = [[],[],[]]
    for i in range(3):
        for j in range(len(i_sort)):
            if np.abs(e[i,j]) < 1e-16:
                e_label[i] += ["ref"]
            else:
                # e_label[i] += ["%0.1E" % e[i,j]]
                e_label[i] += [str(int(np.floor(np.log10(e[i,j])))) if e[i,j] != np.inf else "inf"]

    data = {
        '1e-8': [t[0,:],t_sd[0,:],e_label[0]],
        '1e-5': [t[1,:],t_sd[1,:],e_label[1]],
        '1e-2': [t[2,:],t_sd[2,:],e_label[2]],
    }

    x = np.arange(len(solvers))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    for attribute, measurement in data.items():
        offset = width * multiplier
        ax.bar(x + offset, measurement[0], width, label=attribute)

        for ii in range(len(x)):
            ax.text(
                x[ii] + offset + width / 2, measurement[0][ii], measurement[2][ii], ha="center", va="bottom"
            )

        ax.errorbar(x + offset,measurement[0], measurement[1],
                     fmt='.', color='Black', elinewidth=2, capthick=10,
                     errorevery=1, alpha=0.5, ms=4,capsize=2)

        multiplier += 1

    plt.tight_layout()
    ax.set_xticks(x + width, tuple(solvers))
    ax.legend(loc='upper left', ncols=3)
    ax.set_xlabel("Linear Solver")
    ax.set_ylabel("Time (s)")
    ax.set_yscale('log')
    # ax.set_ylim(1e-5,1e-1) # cross
    ax.set_ylim(1e-3,1e-0) # sudoku 3x3

    if is_full:
        ax.set_title("Full Derivative Fixed QP Solver: " + ref_qp_solver + " Ref. D conditioning: " + str(cond))
    else:
        ax.set_title("Reduced Derivative Fixed QP Solver: " + ref_qp_solver + " Ref. D conditioning: " + str(cond))

    file_path = os.path.dirname(os.path.realpath(__file__))  # reference this file
    # if is_full:
    fig_path = file_path + "/../experiments/diagnostic/results/" + save_name + "_qp-" + ref_qp_solver + "_linear.pdf"
    # else:
    #     fig_path = file_path + "/../experiments/diagnostic/results/" + save_name + "_qp-" + ref_qp_solver + "_reduced_linear.pdf"
    plt.savefig(fig_path)
    # plt.show()

    return fig

def benchmark(save_name):
    file_path = os.path.dirname(os.path.realpath(__file__))  # reference this file
    save_path = file_path + "/../experiments/diagnostic/data/" + save_name + ".npz"
    data = np.load(save_path,allow_pickle=True)

    Q = data["Q"][()]
    q = data["q"][()]
    G = data["G"][()]
    h = data["h"][()]
    A = data["A"][()]
    b = data["b"][()]
    is_sparse = sp.sparse.issparse(Q)

    # dQP doesn't have this constraint, but this copy of the code does?
    if h.ndim== 1:
        h = np.expand_dims(h,-1)

    if q.ndim == 1:
        q = np.expand_dims(q,-1)

    if b is not None and b.ndim == 1:
        b = np.expand_dims(b,-1)

    if is_sparse:
        available_qp_solvers = qpsolvers.sparse_solvers

    else:
        available_qp_solvers = qpsolvers.dense_solvers

    t_qp, t_sd_qp, e_qp = test_qp(Q,q,G,h,A,b,30,available_qp_solvers)
    visualize_qp(t_qp,t_sd_qp,e_qp,available_qp_solvers,save_name)

    ################################################################################################3

    # simplify:
    # if is_sparse:
    #     available_lin_solvers = lin_solvers.get_sparse_solvers()
    #
    # else:
    #     available_lin_solvers = lin_solvers.get_dense_solvers()
    #
    # t_d, t_sd_d, e_d, t_full_d, t_sd_full_d, e_full_d, D_cond, ref_qp_solver = test_linear(Q,q,G,h,A,b,30,available_lin_solvers,available_qp_solvers)
    # fig = visualize_linear(t_full_d,t_sd_full_d,e_full_d,D_cond,available_lin_solvers,save_name,is_full=True,ref_qp_solver=ref_qp_solver)
    # visualize_linear(t_d,t_sd_d,e_d,D_cond,available_lin_solvers,save_name,is_full=False,ref_qp_solver=ref_qp_solver,fig=fig)

    # solve it and use the stored class data in dQP

    return

if __name__ == "__main__":
    benchmark("cross")
    # benchmark("sudoku3x3")
    # benchmark("random_reduced_vs_full")