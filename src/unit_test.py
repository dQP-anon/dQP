import unittest
import dQP
import numpy as np
import scipy.sparse as spa
import torch

from proxsuite.torch.qplayer import QPFunction as prox_qpfunction

from sparse_helper import initialize_torch_from_npz

import sys,os
src_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(src_dir)
sys.path.append(parent_dir)

def Print(output):
   print(output)
   # return None

# This is a coarse unit test, primarily checking basic compatibilities

class TestdQPLayerOptions(unittest.TestCase):
    # @unittest.skip("Skip for fast debugging.")


    def test_00_primal_nBatch_type_nEq(self):
        dim = 25
        for seed in [6,9]:
            for to_sparse in [False,True]:
                for equalities in [False,True]:
                    for nBatch in [1,2]:
                        # if to_sparse and equalities and nBatch > 1:
                        #     Print("stop")
                        Print("--------Primal: dQP: S:" + str(to_sparse) + " Eq:" + str(equalities) + " B:" + str(nBatch > 1) + "----------\n")
                        Q, q, G, h, A, b = generate_random_qp(dim=dim, nBatch=nBatch, seed=seed, to_sparse=to_sparse, equalities=equalities)
                        if to_sparse:
                            solve_type = "sparse"
                        else:
                            solve_type = "dense"
                        settings = dQP.build_settings(solve_type=solve_type, empty_batch=True)
                        dQP_layer =  dQP.dQP_layer(settings)
                        if to_sparse:
                            if nBatch == 1:
                                if equalities:
                                    x_active, _, _, _, _ = dQP_layer(Q.to_sparse_csc(), q, G.to_sparse_csc(), h, A.to_sparse_csc(), b)
                                else:
                                    x_active, _, _, _, _ = dQP_layer(Q.to_sparse_csc(), q, G.to_sparse_csc(), h, A, b)

                            elif nBatch > 1:
                                if equalities:
                                    x_active, _, _, _, _ = dQP_layer([Q[i, :, :].to_sparse_csc() for i in range(nBatch)], q,
                                                                     [G[i, :, :].to_sparse_csc() for i in range(nBatch)], h,
                                                                     [A[i, :, :].to_sparse_csc() for i in range(nBatch)], b)
                                else:
                                    x_active, _, _, _, _ = dQP_layer(
                                        [Q[i, :, :].to_sparse_csc() for i in range(nBatch)], q,
                                        [G[i, :, :].to_sparse_csc() for i in range(nBatch)], h,
                                        A, b)
                        else:
                            x_active, _, _, _, _ = dQP_layer(Q, q, G, h, A, b)


                        Print("x_star:")
                        Print(x_active)
                        Print("\n")
                        x_active.sum().backward()
                        Print("d sum(x) /dh:")
                        Q_active_grad = Q.grad
                        G_active_grad = G.grad
                        h_active_grad = h.grad
                        Print(h_active_grad)
                        # Print(G_active_grad)
                        Print("\n")

                        ###########################################################

                        Print("QPLayer: \n")
                        Q, q, G, h, A, b = generate_random_qp(dim=dim, nBatch=nBatch, seed=seed, to_sparse=False, equalities=equalities) # sparse always false for QPLayer
                        if not equalities:
                            A = torch.empty(0)
                            b = torch.empty(0)
                        l = -1.0e20 * torch.ones(h.size(), dtype=torch.float64)
                        QPLayer = lambda Q, q, G, h, A, b: prox_qpfunction(structural_feasibility=True, omp_parallel=False)(
                            Q, q, A, b, G, l, h
                        )
                        x_qplayer, _, _ = QPLayer(Q, q, G, h, A, b)

                        Print("x_star:")
                        Print(x_qplayer)
                        Print("\n")
                        x_qplayer.sum().backward()
                        Print("d sum(x) /dh:")
                        Q_qplayer_grad = Q.grad
                        G_qplayer_grad = G.grad
                        h_qplayer_grad = h.grad
                        Print(h_qplayer_grad)
                        # Print(G_qplayer_grad)
                        Print("\n")
                        Print("---------------------------------------\n")

                        assert (x_qplayer.size() == x_active.size())  # check shapes match QPLayer ... but don't check soln/derivative since tolerance dependent
                        assert (Q_qplayer_grad.size() == Q_active_grad.size())
                        assert (G_qplayer_grad.size() == G_active_grad.size())
                        assert (h_qplayer_grad.size() == h_active_grad.size())

        return

    def test_01_weird_one_constraint(self):
        dim = 25
        for seed in [6, 9]:
            for to_sparse in [False, True]:
                for equalities in [False, True]:
                    for nBatch in [1, 2]:
                        # if to_sparse and equalities and nBatch > 1:
                        #     Print("stop")
                        Print("--------1 Constraint: dQP: S:" + str(to_sparse) + " Eq:" + str(equalities) + " B:" + str(
                            nBatch > 1) + "----------\n")
                        Q, q, G, h, A, b = generate_random_qp(dim=dim, nBatch=nBatch, seed=seed, to_sparse=to_sparse,
                                                              equalities=equalities,one_constraint=True)
                        if to_sparse:
                            solve_type = "sparse"
                        else:
                            solve_type = "dense"
                        settings = dQP.build_settings(solve_type=solve_type, empty_batch=True)
                        dQP_layer =  dQP.dQP_layer(settings)
                        if to_sparse:
                            if nBatch == 1:
                                if equalities:
                                    x_active, _, _, _, _ = dQP_layer(Q.to_sparse_csc(), q, G.to_sparse_csc(), h,
                                                                     A.to_sparse_csc(), b)
                                else:
                                    x_active, _, _, _, _ = dQP_layer(Q.to_sparse_csc(), q, G.to_sparse_csc(), h, A, b)

                            elif nBatch > 1:
                                if equalities:
                                    x_active, _, _, _, _ = dQP_layer(
                                        [Q[i, :, :].to_sparse_csc() for i in range(nBatch)], q,
                                        [G[i, :, :].to_sparse_csc() for i in range(nBatch)], h,
                                        [A[i, :, :].to_sparse_csc() for i in range(nBatch)], b)
                                else:
                                    x_active, _, _, _, _ = dQP_layer(
                                        [Q[i, :, :].to_sparse_csc() for i in range(nBatch)], q,
                                        [G[i, :, :].to_sparse_csc() for i in range(nBatch)], h,
                                        A, b)
                        else:
                            x_active, _, _, _, _ = dQP_layer(Q, q, G, h, A, b)

                        Print("x_star:")
                        Print(x_active)
                        Print("\n")
                        x_active.sum().backward()
                        Print("d sum(x) /dh:")
                        Q_active_grad = Q.grad
                        G_active_grad = G.grad
                        h_active_grad = h.grad
                        Print(h_active_grad)
                        Print("\n")

                        ###########################################################

                        Print("QPLayer: \n")
                        Q, q, G, h, A, b = generate_random_qp(dim=dim, nBatch=nBatch, seed=seed, to_sparse=False,
                                                              equalities=equalities,one_constraint=True)  # sparse always false for QPLayer
                        if not equalities:
                            A = torch.empty(0)
                            b = torch.empty(0)
                        l = -1.0e20 * torch.ones(h.size(), dtype=torch.float64)
                        QPLayer = lambda Q, q, G, h, A, b: prox_qpfunction(structural_feasibility=True,
                                                                           omp_parallel=False)(
                            Q, q, A, b, G, l, h
                        )
                        x_qplayer, _, _ = QPLayer(Q, q, G, h, A, b)

                        Print("x_star:")
                        Print(x_qplayer)
                        Print("\n")
                        x_qplayer.sum().backward()
                        Print("d sum(x) /dh:")
                        Q_qplayer_grad = Q.grad
                        G_qplayer_grad = G.grad
                        h_qplayer_grad = h.grad
                        Print(h_qplayer_grad)
                        Print("\n")
                        Print("---------------------------------------\n")

                        assert(x_qplayer.size() == x_active.size())  # check shapes match QPLayer ... but don't check soln/derivative since tolerance dependent
                        assert(Q_qplayer_grad.size() == Q_active_grad.size())
                        assert(G_qplayer_grad.size() == G_active_grad.size())
                        assert(h_qplayer_grad.size() == h_active_grad.size())


        return

    Print("Need to address grad = None w.r.t. nu_star if nActive = 0... happens because empty nu[active] = empty is passed; would have to rewrite how this is done")
    # @unittest.skip("Need to address grad = None w.r.t. nu_star if nActive = 0... happens because empty nu[active] = empty is passed; would have to rewrite how this is done")
    def test_02_dual_nBatch_type_nEq(self):
        dim = 15
        for seed in [6,7]:
            for to_sparse in [False,True]:
                for equalities in [False,True]:
                    for nBatch in [1,2]:
                        # if to_sparse and equalities and nBatch > 1:
                        #     Print("stop")
                        Print("--------Dual: dQP: S:" + str(to_sparse) + " Eq:" + str(equalities) + " B:" + str(nBatch > 1) + "----------\n")
                        Q, q, G, h, A, b = generate_random_qp(dim=dim, nBatch=nBatch, seed=seed, to_sparse=to_sparse, equalities=equalities)
                        if to_sparse:
                            solve_type = "sparse"
                        else:
                            solve_type = "dense"
                        settings = dQP.build_settings(solve_type=solve_type, empty_batch=True, verbose=True)
                        dQP_layer =  dQP.dQP_layer(settings)
                        if to_sparse:
                            if nBatch == 1:
                                if equalities:
                                    _, _, nu_active, _, _ = dQP_layer(Q.to_sparse_csc(), q, G.to_sparse_csc(), h,
                                                                     A.to_sparse_csc(), b)
                                else:
                                    _, _, nu_active, _, _ = dQP_layer(Q.to_sparse_csc(), q, G.to_sparse_csc(), h, A, b)

                            elif nBatch > 1:
                                if equalities:
                                    _, _, nu_active, _, _ = dQP_layer(
                                        [Q[i, :, :].to_sparse_csc() for i in range(nBatch)], q,
                                        [G[i, :, :].to_sparse_csc() for i in range(nBatch)], h,
                                        [A[i, :, :].to_sparse_csc() for i in range(nBatch)], b)
                                else:
                                    _, _, nu_active, _, _ = dQP_layer(
                                        [Q[i, :, :].to_sparse_csc() for i in range(nBatch)], q,
                                        [G[i, :, :].to_sparse_csc() for i in range(nBatch)], h,
                                        A, b)
                        else:
                            _, _, nu_active, _, _ = dQP_layer(Q, q, G, h, A, b)

                        Print("nu_star:")
                        Print(nu_active)
                        Print("\n")
                        nu_active.sum().backward()
                        Print("d sum(nu) /dh:")
                        Q_active_grad = Q.grad
                        G_active_grad = G.grad
                        h_active_grad = h.grad
                        Print(h_active_grad)
                        Print("\n")

                        ###########################################################

                        Print("QPLayer: \n")
                        Q, q, G, h, A, b = generate_random_qp(dim=dim, nBatch=nBatch, seed=seed, to_sparse=False, equalities=equalities) # sparse always false for QPLayer
                        if not equalities:
                            A = torch.empty(0)
                            b = torch.empty(0)
                        l = -1.0e20 * torch.ones(h.size(), dtype=torch.float64)
                        QPLayer = lambda Q, q, G, h, A, b: prox_qpfunction(structural_feasibility=True, omp_parallel=False)(
                            Q, q, A, b, G, l, h
                        )
                        _, _, nu_qplayer = QPLayer(Q, q, G, h, A, b)

                        Print("nu_star:")
                        Print(nu_qplayer)
                        Print("\n")
                        nu_qplayer.sum().backward()
                        Print("d sum(nu) /dh:")
                        Q_qplayer_grad = Q.grad
                        G_qplayer_grad = G.grad
                        h_qplayer_grad = h.grad
                        Print(h_qplayer_grad)
                        Print("\n")
                        Print("---------------------------------------\n")

                        assert(nu_qplayer.size() == nu_active.size()) # check shapes match QPLayer ... but don't check soln/derivative since tolerance dependent
                        assert(Q_qplayer_grad.size() == Q_active_grad.size())
                        assert(G_qplayer_grad.size() == G_active_grad.size())
                        assert(h_qplayer_grad.size() == h_active_grad.size())


        return

    def test_03_shape_change(self):
        dim = 25
        seed = 9
        to_sparse = False
        solve_type = "dense"
        equalities = True
        for nBatch in [1,2]:
            Q, q, G, h, A, b = generate_random_qp(dim=dim, nBatch=nBatch, seed=seed, to_sparse=to_sparse,
                                                  equalities=equalities)
            settings = dQP.build_settings(solve_type=solve_type)
            dQP_layer =  dQP.dQP_layer(settings)
            x_active_1, _, _, _, _ = dQP_layer(Q, q, G, h, A, b)

            Q, q, G, h, A, b = generate_random_qp(dim=dim, nBatch=nBatch, seed=seed, to_sparse=to_sparse,
                                                  equalities=equalities)
            settings = dQP.build_settings(solve_type=solve_type)
            dQP_layer =  dQP.dQP_layer(settings)
            x_active_2, _, _, _, _ = dQP_layer(Q, q.unsqueeze(-1), G, h.unsqueeze(-1), A, b.unsqueeze(-1))

            Q, q, G, h, A, b = generate_random_qp(dim=dim, nBatch=nBatch, seed=seed, to_sparse=to_sparse,
                                                  equalities=equalities)
            settings = dQP.build_settings(solve_type=solve_type)
            dQP_layer =  dQP.dQP_layer(settings)
            x_active_3, _, _, _, _ = dQP_layer(Q, q.unsqueeze(-1), G, h.unsqueeze(-1), A, b)

            Print(x_active_1)
            Print(x_active_2)
            Print(x_active_3)

            assert (torch.all(x_active_1 == x_active_2))
        return

    def test_04_has_dual(self):
        dim = 25
        seed = 9
        to_sparse = False
        solve_type = "dense"
        equalities = True
        nBatch = 1
        Q, q, G, h, A, b = generate_random_qp(dim=dim, nBatch=nBatch, seed=seed, to_sparse=to_sparse,
                                              equalities=equalities)
        settings = dQP.build_settings(solve_type=solve_type, empty_batch=True, dual_available=False)
        dQP_layer =  dQP.dQP_layer(settings)
        x_active, _, _, _, _ = dQP_layer(Q, q, G, h, A, b)
        return

    def test_05_refinement(self):
        dim = 25
        seed = 9
        to_sparse = False
        solve_type = "dense"
        equalities = True
        nBatch = 1

        for eps in [1e-3,1e-6,1e-8]:
            for qp_solver in ["cvxopt","piqp"]: # piqp responds well to change in tolerance
                Q, q, G, h, A, b = generate_random_qp(dim=dim, nBatch=nBatch, seed=seed, to_sparse=to_sparse,
                                                      equalities=equalities)
                settings = dQP.build_settings(solve_type=solve_type, empty_batch=True, dual_available=False, refine_active=True, eps_abs=eps, eps_rel=eps, qp_solver=qp_solver)
                dQP_layer =  dQP.dQP_layer(settings)
                x_active, _, _, _, _ = dQP_layer(Q, q, G, h, A, b)
                Print(x_active)
                x_active.sum().backward()
                Print("d sum(x) /dh:")
                h_active_grad = h.grad
                Print(h_active_grad)
                Print("\n")

                Q, q, G, h, A, b = generate_random_qp(dim=dim, nBatch=nBatch, seed=seed, to_sparse=to_sparse,
                                                      equalities=equalities)
                settings = dQP.build_settings(solve_type=solve_type, empty_batch=True, dual_available=True, refine_active=True, eps_abs=eps, eps_rel=eps, qp_solver=qp_solver)
                dQP_layer =  dQP.dQP_layer(settings)
                x_active, _, _, _, _ = dQP_layer(Q, q, G, h, A, b)
                Print(x_active)
                x_active.sum().backward()
                Print("d sum(x) /dh:")
                h_active_grad = h.grad
                Print(h_active_grad)
                Print("\n")
        return

    def test_06_omp_parallel(self):
        dim = 100 # need none of the 64 to be singular, so make a little larger so fill in ratio is enough to avoid zero rows
        seed = 9
        to_sparse = False
        solve_type = "dense"
        equalities = True
        nBatch = 64

        Q, q, G, h, A, b = generate_random_qp(dim=dim, nBatch=nBatch, seed=seed, to_sparse=to_sparse,
                                              equalities=equalities)
        settings = dQP.build_settings(solve_type=solve_type, empty_batch=True, omp_parallel=True, verbose=True)
        dQP_layer =  dQP.dQP_layer(settings)
        x_active_1, _, _, _, _ = dQP_layer(Q, q, G, h, A, b)
        return

    def test_07_test_sparse_grad_size(self):
        solve_type = "sparse"
        Q, q, G, h, A, b = initialize_torch_from_npz("../experiments/mega_test/data/mm/DUAL2.npz")

        print([Q,q,G,h,A,b])
        settings = dQP.build_settings(solve_type=solve_type, empty_batch=True, verbose=False)
        dQP_layer =  dQP.dQP_layer(settings)
        x_active, _, _, _, _ = dQP_layer(Q, q, G, h, A, b)
        Print(x_active)
        x_active.sum().backward()
        Print("d sum(x) /dh:")
        h_active_grad = h.grad
        Print(h_active_grad)
        Print("\n")

        Q, q, G, h, A, b = initialize_torch_from_npz("../experiments/mega_test/data/mm/DUAL2.npz")

        Q = Q.to_dense()
        G = G.to_dense()
        if A is not None:
            A = A.to_dense()

        Print("QPLayer: \n")
        l = -1.0e20 * torch.ones(h.size(), dtype=torch.float64)
        QPLayer = lambda Q, q, G, h, A, b: prox_qpfunction(structural_feasibility=True, omp_parallel=False)(
            Q, q, A, b, G, l, h
        )
        x_qplayer, _, _ = QPLayer(Q, q, G, h, A, b)

        Print("x_star:")
        Print(x_qplayer)
        Print("\n")
        x_qplayer.sum().backward()
        Print("d sum(x) /dh:")
        h_qplayer_grad = h.grad
        Print(h_qplayer_grad)
        Print("\n")
        Print("---------------------------------------\n")

        return

    def test_08_test_available_sparse_linear_solvers(self):

        from lin_solvers import get_sparse_solvers
        for lin_solver in get_sparse_solvers():
            Q, q, G, h, A, b = initialize_torch_from_npz("../experiments/diagnostic/data/cross.npz")
            Print("--------------------------------------------------")
            Print(lin_solver)
            settings = dQP.build_settings(solve_type="sparse",lin_solver=lin_solver)
            dQP_layer =  dQP.dQP_layer(settings)
            x_active, _, _, _, _ = dQP_layer(Q, q, G, h, A, b)
            Print(x_active)
            x_active.sum().backward()
            Print("d sum(x) /dh:")
            h_active_grad = h.grad
            Print(h_active_grad)
            Print("\n")

        return

    # @unittest.skip("Not implemented")
    def test_09_test_ill_conditioned_least_squares(self):
        Q, q, G, h, A, b = initialize_torch_from_npz("../experiments/mega_test/data/mm/QADLITTL.npz")

        settings = dQP.build_settings(solve_type="sparse",empty_batch=True,verbose=True)
        dQP_layer =  dQP.dQP_layer(settings)
        x_active, _, _, _, _ = dQP_layer(Q, q, G, h, A, b)
        Print(x_active)
        x_active.sum().backward()
        Print("d sum(x) /dh:")
        h_active_grad = h.grad
        Print(h_active_grad)
        Print("\n")

        Q, q, G, h, A, b = initialize_torch_from_npz("../experiments/mega_test/data/mm/QADLITTL.npz")

        Q = Q.to_dense()
        G = G.to_dense()
        if A is not None:
            A = A.to_dense()

        Print("QPLayer: \n")
        l = -1.0e20 * torch.ones(h.size(), dtype=torch.float64)
        QPLayer = lambda Q, q, G, h, A, b: prox_qpfunction(structural_feasibility=True, omp_parallel=False)(
            Q, q, A, b, G, l, h
        )
        x_qplayer, _, _ = QPLayer(Q, q, G, h, A, b)

        Print("x_star:")
        Print(x_qplayer)
        Print("\n")
        x_qplayer.sum().backward()
        Print("d sum(x) /dh:")
        h_qplayer_grad = h.grad
        Print(h_qplayer_grad)
        Print("\n")
        Print("---------------------------------------\n")

        return










# Adapted from proxsuite: https://github.com/Simple-Robotics/proxsuite/blob/main/examples/python/init_dense_qp.py
def generate_random_qp(dim=10, nBatch=1, seed=1, to_sparse=False, equalities=True,one_constraint=False):
    # note, the shape of the outputs, including in special cases, was carefully selected to be compatible; and
    # in the shape check, should see what happens when these change
    # also, this doesn't restrict G,A to be full rank. They often can have rows of just 0's, so may need to tune seeds
    np.random.seed(seed)

    if one_constraint:
        nEq = 1
        nIneq = 1
    else:
        nEq = int(dim / 4)
        nIneq = int(dim / 4)

    nCons = nEq + nIneq

    Q = np.zeros((nBatch, dim, dim))
    q = np.zeros((nBatch, dim))
    G = np.zeros((nBatch, nIneq, dim))
    h = np.zeros((nBatch, nIneq))
    A = np.zeros((nBatch, nEq, dim))
    b = np.zeros((nBatch, nEq))

    for ii in range(nBatch):
        # Q
        P = spa.random(
            dim, dim, density=0.2, data_rvs=np.random.randn, format="csc"
        ).toarray()
        P = (P + P.T) / 2.0
        s = max(np.absolute(np.linalg.eigvals(P)))
        P += (abs(s) + 1e-02) * spa.eye(dim)
        P = spa.coo_matrix(P)

        # Both inequality and equality constraints
        C = spa.random(nCons, dim, density=0.15, data_rvs=np.random.randn, format="csc").toarray()
        v = np.random.randn(dim)  # Fictitious solution
        u = C @ v

        Q[ii, :, :] = P.toarray()
        q[ii, :] = np.random.randn(dim)
        G[ii, :, :] = C[nIneq:, :]
        h[ii, :] = u[nIneq:]
        A[ii, :, :] = C[:nEq, :]
        b[ii, :] = u[:nEq]

    if nBatch == 1:
        Q = torch.tensor(Q[0, :, :], dtype=torch.float64, requires_grad=True)
        G = torch.tensor(G[0, :, :], dtype=torch.float64, requires_grad=True)
        A = torch.tensor(A[0, :, :], dtype=torch.float64, requires_grad=True)
        q = torch.tensor(q[0, :], dtype=torch.float64, requires_grad=True)
        h = torch.tensor(h[0, :], dtype=torch.float64, requires_grad=True)
        b = torch.tensor(b[0, :], dtype=torch.float64, requires_grad=True)
    else:
        Q = torch.tensor(Q, dtype=torch.float64, requires_grad=True)
        G = torch.tensor(G, dtype=torch.float64, requires_grad=True)
        A = torch.tensor(A, dtype=torch.float64, requires_grad=True)
        q = torch.tensor(q, dtype=torch.float64, requires_grad=True)
        h = torch.tensor(h, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(b, dtype=torch.float64, requires_grad=True)

    if not equalities:
        A = None
        b = None

    return Q,q,G,h,A,b


if __name__ == '__main__':
    unittest.main()