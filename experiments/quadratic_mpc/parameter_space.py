import numpy as np

import sys
sys.path.append('../../')

from src import dQP

import torch
import time

from matplotlib import pyplot as plt
import matplotlib as mpl

from integer_pairing import cantor
from sklearn import svm

from src.qp_diagnostic import get_full_D

import qpsolvers

# https://stackoverflow.com/questions/51495819/how-to-plot-svm-decision-boundary-in-sklearn-python
def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

# On the Facet-to-Facet Property of Solutions to Convex Parametric Quadratic Programs
def simple_example(dQP_layer,full_qp_kwargs,N_grid,active_tol_order,refine,solver):
    dim = 3
    nIneq = 6
    nEq = 0

    # Fixed parameters
    Q = torch.eye(dim, dtype=torch.float64)
    q = torch.zeros((dim, 1), dtype=torch.float64)
    G = torch.tensor([[1, 0, -1], [-1, 0, -1], [0, 1, -1], [0, -1, -1], [3 / 4, 16 / 25, -1], [-3 / 4, -16 / 25, -1]],
                     dtype=torch.float64)
    A = None
    b = None

    # Grid
    theta_1_sweep = np.linspace(-1.5,1.5,N_grid)
    theta_2_sweep = theta_1_sweep.copy()
    active_label = []
    full_cond = []
    red_cond = []
    n_active = []
    t_solve_reduced = []
    t_solve_full = []
    # n_sample = 10
    # incoming_grad = np.random.randn(dim + nIneq, 1)

    for theta_1 in theta_1_sweep:
        for theta_2 in theta_2_sweep:
            theta = torch.tensor([theta_1,theta_2],dtype=torch.float64).requires_grad_(True)
            h = -torch.ones((nIneq,1),dtype=torch.float64) + torch.vstack((theta[0],-theta[0],-theta[1],theta[1],theta[0],-theta[0]))

            x, mu, nu, _, _ = dQP_layer(Q, q, G, h, A, b)

            if np.sum(dQP_layer.active) == 0:
                active_label += [0]
            elif np.sum(dQP_layer.active) == 1:  # assume indices themselves alone don't appear in Cantor?
                active_label += [np.argwhere(dQP_layer.active).squeeze().tolist()]
            elif np.sum(dQP_layer.active) <= dim:
                active_label += [cantor.pair(*np.argwhere(
                    dQP_layer.active).squeeze().tolist())]  # unique label for active set ; assumes that output of np.argwhere is ordered***
            else:
                active_label += [None]  # unique label for active set ; assumes that output of np.argwhere is ordered***

            solution = dQP.call_single_qpsolvers(Q.detach().numpy(), q.detach().numpy(), G.detach().numpy(), h.detach().numpy(), None, None, full_qp_kwargs )
            D = get_full_D(Q.detach().numpy(), q.detach().numpy(), G.detach().numpy(), h.detach().numpy(), None, None, np.expand_dims(solution.x, -1), np.expand_dims(solution.z, -1), False)
            K = dQP_layer.KKT_A_np

            t_tot_full = 0
            t_tot_reduced = 0
            # for ii in range(n_sample):
            #     ti = time.time()
            #     np.linalg.solve(D,incoming_grad)
            #     t_tot_full += time.time() - ti
            #
            #     incoming_grad_active = incoming_grad[np.concatenate((np.ones(dim + nEq, dtype=np.bool_), dQP_layer.active)), :]
            #     ti = time.time()
            #     np.linalg.solve(K, incoming_grad_active)
            #     t_tot_reduced += time.time() - ti
            #
            # t_solve_full += [t_tot_full/n_sample]
            # t_solve_reduced += [t_tot_reduced/n_sample]

            full_cond += [np.linalg.cond(D)]
            red_cond += [np.linalg.cond(K)]
            n_active += [np.sum(dQP_layer.active)]

    visualize(theta_1_sweep,theta_2_sweep,active_label,full_cond,red_cond,n_active,N_grid,active_tol_order,"MPC",solver,refine,t_solve_full,t_solve_reduced)

def visualize(theta_1_sweep,theta_2_sweep,active_label,full_cond,red_cond,n_active,N_grid,active_tol_order,example_label,solver,refine,t_solve_full,t_solve_reduced):
    th1, th2 = np.meshgrid(theta_1_sweep, theta_2_sweep)


    # create SVM's for contour plots
    X = np.hstack(
        (np.reshape(th1, (N_grid ** 2, 1), order='F'), np.reshape(th2, (N_grid ** 2, 1), order='F')))  # fix th1 first
    Y = np.array(active_label, dtype=float)  # float explicit to make None --> nan
    Y[np.isnan(Y)] = int(np.nanmean(Y))
    # make NaN's the mean (NaN's are weakly active, we want to ignore them here because if there are many constraints then cantor pairing blows up and makes color hard to see
    clf_set = svm.SVC()
    clf_set.fit(X, Y)

    # slice
    red_cond = np.array(red_cond,dtype=float)
    full_cond = np.array(full_cond, dtype=float)
    t_red = np.array(t_solve_reduced,dtype=float)
    t_full = np.array(t_solve_full,dtype=float)
    dx = theta_1_sweep[1] - theta_1_sweep[0]
    min_th = np.min(theta_1_sweep)
    max_th = np.max(theta_1_sweep)
    cut = np.abs(X[:,1]-max_th/3) < dx/2
    slice_theta = X[cut,0] # horizontal slice, fixed Y
    slice_red_cond = red_cond[cut]
    slice_full_cond = full_cond[cut]
    # slice_t_red = t_red[cut]
    # slice_t_full = t_full[cut]
    assert(np.size(slice_theta) == N_grid)

    active_label = np.reshape(active_label, (N_grid, N_grid), order='F')  # fix th1 last
    full_cond = np.reshape(full_cond,(N_grid,N_grid),order='F')
    n_active = np.array(n_active,dtype=int)
    red_cond[n_active > 3] = np.nan # remove any weakly active from condition calculation
    red_cond = np.reshape(red_cond,(N_grid,N_grid),order='F')
    # n_active = np.reshape(np.array(n_active,dtype=int),(N_grid,N_grid),order='F')

    # phase plot
    fig, axs = plt.subplots(1,3,sharex=True)
    plot_contours(axs[0], clf_set, th1, th2, alpha=0.8)
    axs[0].scatter(th1, th2, c=active_label)
    plt.hlines(max_th/3,min_th,max_th,'w')
    CS1 = axs[1].contourf(th1,th2,full_cond,alpha=0.8,norm=mpl.colors.LogNorm(),vmin=np.nanmin(full_cond),vmax=np.nanmax(full_cond))
    CS2 = axs[2].contourf(th1,th2,red_cond,alpha=0.8,norm=mpl.colors.LogNorm(),vmin=np.nanmin(red_cond),vmax=np.nanmax(red_cond))

    axs[0].set_xlim([min_th, max_th])
    axs[0].set_ylim([min_th, max_th])
    axs[0].set_aspect('equal', 'box')
    axs[0].set_title('Active Sets')
    axs[1].set_xlim([min_th, max_th])
    axs[1].set_ylim([min_th, max_th])
    axs[1].set_aspect('equal', 'box')
    axs[1].set_title('Full Cond #')
    axs[2].set_xlim([min_th, max_th])
    axs[2].set_ylim([min_th, max_th])
    axs[2].set_aspect('equal', 'box')
    axs[2].set_title('Red. Cond #')
    axs[0].set_ylabel('theta_2')
    axs[0].set_xlabel('theta_1')
    axs[1].set_xlabel('theta_1')
    axs[2].set_xlabel('theta_1')

    plt.colorbar(CS1,ax=axs[1],fraction=0.046, pad=0.04)
    plt.colorbar(CS2,ax=axs[2],fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig('results/phase_plot_ex_' + example_label  + '_' + solver + '_r' + str(int(refine))  + '_tol_e-' + str(active_tol_order) + '.pdf')
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(slice_theta,slice_full_cond, 'k', linewidth=0.5,label='Full')
    ax.plot(slice_theta, slice_red_cond, 'r', linewidth=0.5,label='Reduced')
    if np.size(slice_red_cond[slice_red_cond > 1e5]) > 0:
        plt.vlines(slice_red_cond[slice_red_cond > 1e5],0,np.max(slice_full_cond),'r',label='Weakly Active')
    plt.yscale('log')
    plt.ylim([1,1e5])
    plt.legend()
    plt.xlabel('theta_1')
    plt.ylabel('Condition')
    plt.savefig('results/condition_slice_ex_' + example_label  + '_' + solver + '_r' + str(int(refine)) + '_tol_e-' + str(active_tol_order) + '.pdf')
    plt.close(fig)

    # fig, ax = plt.subplots()
    # ax.plot(slice_theta, slice_t_full, 'k', linewidth=0.5, label='Full')
    # ax.plot(slice_theta, slice_t_red, 'r', linewidth=0.5, label='Reduced')
    # if np.size(slice_red_cond[slice_red_cond > 1e5]) > 0:
    #     plt.vlines(slice_red_cond[slice_red_cond > 1e5], 0, np.max(slice_full_cond), 'r', label='Weakly Active')
    # plt.yscale('log')
    # plt.legend()
    # plt.xlabel('theta_1')
    # plt.ylabel('Time')
    # plt.savefig('../../results/quadratic_mpc/time_slice_ex_' + example_label + '_' + solver + '_r' + str(
    #     int(refine)) + '_tol_e-' + str(active_tol_order) + '.pdf')
    # plt.close(fig)


def main():
    qp_kwargs = None
    eps_abs = 1e-4
    N_grid = 10

    for solver in qpsolvers.dense_solvers:
        for active_tol_order in [7]:
            for refine in [True, False]:
                # need accurate IPM with strict tolerance
                dQP_settings = dQP.build_settings(eps_active=10 ** (-active_tol_order), eps_abs=eps_abs, eps_rel=0, time=False,
                                                            solve_type="dense",
                                                            qp_solver=solver,
                                                            qp_solver_keywords=qp_kwargs, refine_active=refine)
                dQP_layer = dQP.dQP_layer(settings=dQP_settings)

                if qp_kwargs is not None:  # for solving externally from dQP_layer to get D
                    full_qp_kwargs = dict(**{"solver": solver, "verbose": False, "initvals": None}, **qp_kwargs)
                else:
                    full_qp_kwargs = {"solver": solver, "verbose": False, "initvals": None}

                simple_example(dQP_layer,full_qp_kwargs,N_grid,active_tol_order,refine,solver)

# run from this directory, python parameter_space.py
main()