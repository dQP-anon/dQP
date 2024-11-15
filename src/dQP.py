import numpy as np
import scipy as sp

import torch
from torch import nn

# qp solver
import qpsolvers
from scipy.sparse import csc_matrix, bmat

import sys
import os
import warnings

# sys.path.append('../')

src_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(src_dir)
sys.path.append(parent_dir)

from src.set_solver_tolerance import set_solver_tolerance

import src.sparse_helper as sparse_helper
import src.lin_solvers as lin_solvers
import time

# CPU parallelism
from joblib import Parallel, delayed


# Note, internal conventions for QP variables and parameters are different from the paper
# (P,q,C,d,A,b) = (Q,q,G,h,A,b)
# (z,\lambda,\mu) = (x,\mu,\nu)

####################### Notes for TODO items
# (0) Batching is w.r.t. all parameters. Requires duplication if some parameters are fixed. ==> TODO: remove duplication
# (1) if nActive = 0 then back-propagation through nu_star will yield dL/dG, dL/dh = None. ==> TODO: output zeros
#######################

class dQP_layer(nn.Module):
    ''' solves and differentiates
    x^* = argmin_x 1/2 x^T Q x + q^T x
             s.t.  G x <= h
                   A x  = b
    (including dual variables mu^*,nu^*)
    Q dim x dim ; q dim x 1 ; G nIneq x dim ; h nIneq x 1 ; A nEq x dim ; b nEq x 1
    if batched, first dimension is nBatch. Serial via for loops unless multiple CPUs/OMP is turned on

    input:
        torch parameters Q,q,G,h,A,b
        if sparse, need to be CSC type
    output:
        x_star,mu_star,nu_star,time

    see README for information about options

    internal variables
        -dim        : # dim
        -nEq        : # equalities
        -nIneq      : # inequalities
        -nBatch     : batch size, interpreted from input
        -active     : active set
        -nActive    : # active
        -A copy of Q,q,G,h,A,b,x_star,mu_star,nu_star as numpy variables denoted _np
        -r_pri_np    : primal residual h - Gx^*
        -A_reduce_np : equality/active inequality constraints
        -nEq_reduce  : # equality/active inequality constraints
        -KKT_A_np    : reduced KKT
        -KKT_A_np_factors   : factorizations of reduced KKT
        -KKT_b_np           : RHS of reduced KKT
        -non_differentiable : determines whether least-squares is used in differentiate_QP

        -differentiate_QP  : differentiates solution
        -sparse_row_norm   : differentiable matrix vecnorm
        -sparse_row_normalize :  differentiable matrix normalization by row

    '''

    def __init__(self,settings=None):
        super().__init__()

        if settings is None:
            settings = build_settings() # call with defaults

        for k, v in settings.items():
            setattr(self, k, v)

        self.dim = None
        self.nIneq = None
        self.nEq = None
        self.nBatch = None

        self.nActive = None
        self.active = None
        self.nEq_reduce = None

        self.differentiate_QP = differentiate_QP.apply # set-up differentiation through active constraints

        # TODO : note that if these are not reset and A = None changes for different input, may cause problem
        self.N_A = None # torch variables
        self.N_G = None

        # initialize numpy variables which are carried implicitly ; torch version is not stored
        self.x_star_np = None
        self.mu_star_np = None
        self.nu_star_np = None
        self.nu_star_with_inactive_np = None

        self.Q_np = None
        self.q_np = None
        self.G_np = None
        self.h_np = None
        self.A_np = None
        self.b_np = None

        self.r_pri_np = None
        self.A_reduce_np = None
        self.KKT_A_np = None
        self.KKT_A_np_factors = None
        self.KKT_b_np = None

        self.non_differentiable = False # flag for non-differentiable weakly active constraints ; perform LSQ if True

        # custom differentiable sparse normalization functions
        self.sparse_row_norm = sparse_helper.sparse_row_norm.apply
        self.sparse_row_normalize = sparse_helper.sparse_row_normalize.apply

    def forward(self, Q, q, G, h, A=None, b=None):
        # check shapes, extract dim,nIneq,nEq,nBatch
        self.reset_parameters()
        Q,q,G,h,A,b = self.get_shapes(Q, q, G, h, A, b)

        if not self.training: # check if evaluation mode is turned on
            # warm-starts with previous problem's x_star stored in the class layer
            if self.warm_start_from_previous:  # false if nBatch > 1
                initvals = self.x_star_np
            else:
                initvals = None
            kwargs_fixed = dict(**{"solver": self.qp_solver, "verbose": self.verbose, "initvals": initvals},
                                **self.qp_solver_keywords)
            # do not normalize, this may be carried out in the solver if needed
            self.data_to_np(Q,q,G,h,A,b)
            x = call_single_qpsolvers(self.Q_np,self.q_np,self.G_np,self.h_np,self.A_np,self.b_np,kwargs_fixed)
            return torch.from_numpy(x.x),torch.from_numpy(x.y),torch.from_numpy(x.z), None, None # should time
        else:
            if self.time:
                if self.normalize_constraints:
                    # normalize
                    start_normalize = time.time()
                    G,h,A,b = self.normalize(G,h,A,b)
                    normalize_time = time.time() - start_normalize

                    print("### Time normalize: " + str(normalize_time))

                # convert and store in numpy as dense or sparse
                start_convert = time.time()
                self.data_to_np(Q,q,G,h,A,b)
                convert_time = time.time() - start_convert

                print("### Time conversion: " + str(convert_time))

                # solve and time QP ; solution is stored
                start_solve = time.time()
                self.solve()
                solve_time = time.time() - start_solve

                self.nu_star_with_inactive_np = self.nu_star_np.copy()

                print("### Time QP Solve: " + str(solve_time))

                # differentiate and time
                start_setup_diff = time.time()
                x_star, mu_star, nu_star = self.setup_diff(Q=Q,q=q,G=G,h=h,A=A,b=b)
                setup_diff_time = time.time() - start_setup_diff

                total_forward_time = normalize_time + convert_time + solve_time + setup_diff_time

                print("### Time Setup Differentiation Time: " + str(setup_diff_time))
            else:
                if self.normalize_constraints:
                    G, h, A, b = self.normalize(G, h, A, b)
                self.data_to_np(Q, q, G, h, A, b)
                self.solve()
                self.nu_star_with_inactive_np = self.nu_star_np.copy()
                x_star, mu_star, nu_star = self.setup_diff(Q=Q, q=q, G=G, h=h, A=A, b=b)
                solve_time = None
                total_forward_time = None

            # fill in inequality dual variables to include nu = 0 for inactive ; add empty batch dimension (typical convention)
            if self.nBatch == 1:
                # nu_star_with_inactive = torch.zeros(self.nIneq,dtype=torch.float64)
                nu_star_with_inactive = torch.tensor(self.nu_star_with_inactive_np,dtype=torch.float64)
                if self.nActive > 0:
                    nu_star_with_inactive[self.active] = nu_star

                if self.empty_batch: # include empty batch dimension
                    x_star = x_star.unsqueeze(0)
                    mu_star = mu_star.unsqueeze(0)
                    nu_star_with_inactive = nu_star_with_inactive.unsqueeze(0)

            else:
                x_star = torch.vstack(x_star) # TODO : throughout, never work with N x 1 vectors and instead just use N, . This is more noticeable in batching and when fed into other NN
                if self.nEq > 0:
                    mu_star = torch.vstack(mu_star)
                nu_star_with_inactive = torch.tensor(self.nu_star_with_inactive_np,dtype=torch.float64)
                for i in range(self.nBatch):
                    if self.nActive[i] > 0:
                        nu_star_with_inactive[i][self.active[i]] = nu_star[i]

            if self.normalize_constraints:
                if self.N_A is not None:
                        mu_star = torch.div(mu_star, self.N_A)
                nu_star_with_inactive = torch.div(nu_star_with_inactive,self.N_G)

            return x_star, mu_star, nu_star_with_inactive, solve_time, total_forward_time

    def reset_parameters(self):
        '''
        may not be necessary ; just a backup measure in case previously stored conditions are kept in error or not
        over-writtten
        currently think it might matter if nEq != 0 and then nEq = 0

        also ; in some cases, may want to store and re-use these
        '''
        self.dim,self.nIneq,self.nEq,self.nBatch = None, None, None, None
        self.nActive, self.active,self.nEq_reduce = None, None, None
        self.N_A, self.N_G = None, None
        self.x_star_np, self.mu_star_np, self.nu_star_np = None, None, None
        self.Q_np, self.q_np, self.G_np, self.h_np, self.A_np, self.b_np = None, None, None, None, None, None
        self.r_pri_np, self.A_reduce_np, self.KKT_A_np, self.KKT_A_np_factors, self.KKT_b_np = None, None, None, None, None
        self.non_differentiable = False


    def get_shapes(self,Q,q,G,h,A,b):
        '''
        extract dim,nIneq,nEq,nBatch and standardize shapes
        note, these checks will certainly overlap with checks in qpsolvers and their calls
        can find ways to improve the last dimension = 1 for vectors condition ; may or may not need to rewrite other code
        another option is just to let other parts fail and just extract, without checking
        '''

        # if batch sparse, extract nBatch and dimensions from first entry TODO : don't skip compatibility checks among matrices
        if self.solve_type == "sparse" and (isinstance(Q,list) and isinstance(G,list)):
            self.nBatch = len(Q)
            self.nIneq,self.dim = G[0].size()
            if A is not None and b is not None:
                self.nEq, _ = A[0].size()
            else:
                self.nEq = 0
        elif (self.solve_type == "sparse" and Q.layout == torch.sparse_csc) or (self.solve_type == "dense" and Q.layout == torch.strided):
            # check matrix sizes first
            Qsz = Q.size()
            Gsz = G.size()

            assert((Qsz[-1] == Qsz[-2]) and (Qsz[-1] == Gsz[-1])) # Q square, dim consistent

            self.dim,self.nIneq = Gsz[-1:-3:-1] # get last two sizes, backwards

            if (Q.dim() == 3) and (G.dim() == 3):
                assert((Qsz[0] == Gsz[0]))  # nBatch consistent
                self.nBatch = Qsz[0]
            elif (Q.dim() == 2) and (G.dim() == 2):
                self.nBatch = 1
            else:
                raise Exception("Q,G must have consistent number of dimensions, either 2 or 3")

            # handle A,b similarly
            if A is not None and b is not None:
                Asz = A.size()
                assert(Qsz[-1] == Asz[-1])  # dim consistent
                if A.dim() == 3:
                    assert(Qsz[0] == Asz[0]) # nBatch consistent
                    self.nEq = Asz[1]
                else:
                    assert(self.nBatch == 1) # ensure Q,G also not batched

                _, self.nEq = Asz[-1:-3:-1]  # get last two sizes, backwards
            elif A is None and b is None:
                self.nEq = 0
            else:
                raise Exception("Both A,b must either be None or not")
        else:
            raise Exception("Solve type must match input.") # TODO : automatically adapt settings to solve type?

        # ensure vector sizes match ; modify dimensions if necessary
        qsz = q.size()
        hsz = h.size()

        if (q.dim() == 3) and (h.dim() == 3):
            assert (qsz[0] == self.nBatch and hsz[0] == self.nBatch)  # nBatch consistent
            assert (qsz[-1:-3:-1] == torch.Size((1, self.dim)) and hsz[-1:-3:-1] == torch.Size(
                (1, self.nIneq)))  # vector, dim, nIneq consistent

            if self.nBatch == 1:
                q = q.squeeze(0)
                h = h.squeeze(0)
                q = q.squeeze(-1)
                h = h.squeeze(-1)
            if self.nBatch > 1:
                q = q.squeeze(-1)
                h = h.squeeze(-1)
        elif (q.dim() == 2) and (h.dim() == 2):
            if (self.nBatch == 1) and (qsz == torch.Size((self.nBatch, self.dim))) and (hsz == torch.Size((self.nBatch, self.nIneq))):
                q = q.squeeze(0)
                h = h.squeeze(0)
            elif (self.nBatch == 1) and (qsz == torch.Size((self.dim,1))) and (hsz == torch.Size((self.nIneq,1))):
                q = q.squeeze(-1)
                h = h.squeeze(-1)
        elif (q.dim() == 1) and (h.dim() == 1):
            assert (self.nBatch == 1)  # nBatch consistent
        else:
            raise Exception("Both q,h must have consistent number of dimensions, 1, 2, or 3")

        if b is not None:
            # ensure vector sizes match ; expand dimensions if needed
            bsz = b.size()

            if (b.dim() == 3):
                assert (bsz[0] == self.nBatch)  # nBatch consistent
                assert (bsz[-1:-3:-1] == torch.Size((1, self.nEq)))  # vector, dim, nIneq consistent

                if self.nBatch == 1:
                    b = b.squeeze(0)
                    b = b.squeeze(-1)
                if self.nBatch > 1:
                    b = b.squeeze(-1)
            elif (b.dim() == 2):
                if (self.nBatch == 1) and (bsz == torch.Size((self.nBatch, self.nEq))):
                    b = b.squeeze(0)
                elif (self.nBatch == 1) and (bsz == torch.Size((self.nEq, 1))):
                    b = b.squeeze(-1)
            elif (b.dim() == 1):
                assert (self.nBatch == 1)  # nBatch consistent
            else:
                raise Exception("b must have consistent number of dimensions, 1, 2, or 3")

        if self.nBatch > 1:
            assert(not self.warm_start_from_previous)
            assert(not self.refine_active)
            assert(self.dual_available)

        return Q,q,G,h,A,b

    def normalize(self,G,h,A,b):
        '''
        (differentiable) normalize the constraints
        is batched
        '''

        if self.solve_type == "sparse":
            if self.nBatch == 1:
                # normalize inequality constraints
                self.N_G = self.sparse_row_norm(G,2).squeeze(-1)

                # assert(torch.all(self.N_G > 1e-8)) # TODO: remove constraints if automatically satisfied? 0 <= h_i ; fail if infeasible? ... or don't check
                zero_rows = self.N_G < 1e-7
                N_G = self.N_G.clone()
                N_G[[zero_rows.clone()]] = 1
                self.N_G = N_G

                G = self.sparse_row_normalize(G,self.N_G.unsqueeze(-1))
                h = torch.div(h, self.N_G)

                # normalize equality constraints
                if A is not None and b is not None:
                    self.N_A = self.sparse_row_norm(A,2).squeeze(-1)

                    # assert (torch.all(self.N_A > 1e-8))
                    zero_rows = self.N_A < 1e-7
                    N_A = self.N_A.clone()
                    N_A[[zero_rows.clone()]] = 1
                    self.N_A = N_A

                    A = self.sparse_row_normalize(A, self.N_A.unsqueeze(-1))
                    b = torch.div(b, self.N_A)
                else:
                    self.N_A = None
            else:
                self.N_G = torch.zeros((self.nBatch,self.nIneq),dtype=torch.float64)

                for i in range(self.nBatch):
                    self.N_G[i,:] = self.sparse_row_norm(G[i].to_sparse_csc(),2).squeeze(-1)

                    zero_rows = self.N_G[i, :] < 1e-7
                    N_Gi = self.N_G[i,:].clone()
                    N_Gi[[zero_rows.clone()]] = 1
                    self.N_G[i,] = N_Gi

                    G[i] = self.sparse_row_normalize(G[i].to_sparse_csc(),self.N_G[i, :].unsqueeze(-1))
                h = torch.div(h, self.N_G)
                if A is not None and b is not None:
                    self.N_A = torch.zeros((self.nBatch, self.nEq), dtype=torch.float64)

                    for i in range(self.nBatch):
                        self.N_A[i,:] = self.sparse_row_norm(A[i].to_sparse_csc(), 2).squeeze(-1)

                        zero_rows = self.N_A[i, :] < 1e-7
                        N_Ai = self.N_A[i, :].clone()
                        N_Ai[[zero_rows.clone()]] = 1
                        self.N_A[i,] = N_Ai

                        A[i] = self.sparse_row_normalize(A[i].to_sparse_csc(), self.N_A[i, :].unsqueeze(-1))
                    b = torch.div(b, self.N_A)
                else:
                    self.N_A = None

        elif self.solve_type == "dense":
            # normalize inequality constraints
            self.N_G = torch.linalg.vector_norm(G, ord=2, dim=-1)

            # assert(torch.all(self.N_G > 1e-8))
            zero_rows = self.N_G < 1e-7
            N_G = self.N_G.clone()
            N_G[[zero_rows.clone()]] = 1
            self.N_G = N_G

            G = torch.div(G, self.N_G.unsqueeze(-1))
            h = torch.div(h, self.N_G)

            # normalize equality constraints
            if A is not None and b is not None:
                self.N_A = torch.linalg.vector_norm(A, ord=2, dim=-1)

                # assert (torch.all(self.N_A > 1e-8))
                zero_rows = self.N_A < 1e-7
                N_A = self.N_A.clone()
                N_A[[zero_rows.clone()]] = 1
                self.N_A = N_A

                A = torch.div(A, self.N_A.unsqueeze(-1))
                b = torch.div(b, self.N_A)
            else:
                self.N_A = None

        return G,h,A,b

    def data_to_np(self,Q,q,G,h,A,b):
        '''
        Convert torch QP parameters to numpy and scipy variables, accounting for sparsity
        Stores variables in the class and optionally saves them
        '''

        if self.solve_type == "dense":
            self.Q_np = Q.detach().numpy()
            self.G_np = G.detach().numpy()

            if A is not None:
                self.A_np = A.detach().numpy()
            else:
                self.A_np = None
        elif self.solve_type == "sparse":
            if self.nBatch == 1:
                self.Q_np = sparse_helper.csc_torch_to_scipy(Q)
                self.G_np = sparse_helper.csc_torch_to_scipy(G)
                if A is not None:
                    self.A_np = sparse_helper.csc_torch_to_scipy(A)
            elif self.nBatch > 1:
                self.Q_np = []
                self.G_np = []
                if A is not None:
                    self.A_np = []
                    for i in range(self.nBatch):
                        self.Q_np += [sparse_helper.csc_torch_to_scipy(Q[i])]
                        self.G_np += [sparse_helper.csc_torch_to_scipy(G[i])]
                        self.A_np += [sparse_helper.csc_torch_to_scipy(A[i])]
                else:
                    for i in range(self.nBatch):
                        self.Q_np += [sparse_helper.csc_torch_to_scipy(Q[i])]
                        self.G_np += [sparse_helper.csc_torch_to_scipy(G[i])]

        self.q_np = q.detach().numpy()
        self.h_np = h.detach().numpy()
        if b is not None:
            self.b_np = b.detach().numpy()
        else:
            self.b_np = None

        # check symmetric Q #
        if self.solve_type == "dense":
            assert((np.linalg.norm(self.Q_np - (self.Q_np + np.moveaxis(self.Q_np,-1,-2)) / 2) < 1e-8))
        elif self.solve_type == "sparse" and self.nBatch == 1: # TODO: check even when nBatch > 1
            assert((sp.sparse.linalg.norm(self.Q_np - (self.Q_np + self.Q_np.T) / 2) < 1e-8))

        # check PSD Q
        if self.check_PSD:
            if self.solve_type == "dense":
                try:
                    np.linalg.cholesky(self.Q_np)
                except:
                    raise Exception("Q not PSD")
            elif self.solve_type == "sparse":
                raise NotImplementedError("Can consider using CHOLESPY here")

        return

    def solve(self):
        '''
        Solve the QP using qpsolvers
        data_to_np must be called before
        '''

        # warm-starts with previous problem's x_star stored in the class layer
        if self.warm_start_from_previous: # false if nBatch > 1
            initvals = self.x_star_np
        else:
            initvals = None

        if self.nBatch == 1:
            kwargs_main = {
                "problem" : qpsolvers.Problem(P=self.Q_np, q=self.q_np, G=self.G_np, h=self.h_np, A=self.A_np, b=self.b_np),
                "solver" : self.qp_solver,
                "verbose" : self.verbose,
                "initvals" : initvals
            }
            solution = qpsolvers.solve_problem(**dict(**kwargs_main,**self.qp_solver_keywords))

            if solution.x is None:
                print("Solver failed to return a solution. Re-solving with verbose and exiting.")
                kwargs_main["verbose"] = True
                qpsolvers.solve_problem(**dict(**kwargs_main, **self.qp_solver_keywords))

                raise Exception("Exiting")

            self.x_star_np = solution.x
            self.mu_star_np = None if solution.y is None else solution.y  # duals optional
            self.nu_star_np = None if solution.z is None else solution.z
            self.nu_star_with_inactive_np = self.nu_star_np # keep a complete copy


        elif self.nBatch > 1:
            # initialize
            self.x_star_np = np.zeros((self.nBatch, self.dim))
            if self.nEq > 0:
                self.mu_star_np = np.zeros((self.nBatch, self.nEq))
            self.nu_star_np = np.zeros((self.nBatch, self.nIneq))

            if self.omp_parallel: # TODO : we need to review if this is the best ; for example, does it start and restart the cores each time? If so, does it matter?
                kwargs_fixed = dict(**{"solver": self.qp_solver,"verbose": self.verbose,"initvals": initvals},**self.qp_solver_keywords)
                if self.nEq > 0:
                    solution = Parallel(n_jobs=self.n_cpu,prefer="processes")(
                        delayed(call_single_qpsolvers)(self.Q_np[i], self.q_np[i], self.G_np[i],
                                                   self.h_np[i], self.A_np[i], self.b_np[i], kwargs_fixed) for i in range(self.nBatch))
                else:
                    solution = Parallel(n_jobs=self.n_cpu, prefer="processes")(
                        delayed(call_single_qpsolvers)(self.Q_np[i], self.q_np[i], self.G_np[i],
                                                       self.h_np[i], None, None, kwargs_fixed) for i in range(self.nBatch))

                # Parallel outputs a list of the solutions, extract now:
                for i in range(self.nBatch):
                    self.x_star_np[i] = solution[i].x
                    # implicitly assumes if nBatch > 1 then duals are available:
                    if self.nEq > 0:
                        self.mu_star_np[i] =  solution[i].y  # TODO : make duals optional by setting = None throughout
                    self.nu_star_np[i] = solution[i].z
            else:
                for i in range(self.nBatch):
                    if self.nEq > 0:
                        kwargs_main = {
                            "problem": qpsolvers.Problem(P=self.Q_np[i], q=self.q_np[i], G=self.G_np[i],
                                                         h=self.h_np[i], A=self.A_np[i], b=self.b_np[i]),
                            "solver": self.qp_solver,
                            "verbose": self.verbose,
                            "initvals": initvals
                        }
                    else:
                        kwargs_main = {
                            "problem": qpsolvers.Problem(P=self.Q_np[i], q=self.q_np[i], G=self.G_np[i],
                                                         h=self.h_np[i], A=None, b=None),
                            "solver": self.qp_solver,
                            "verbose": self.verbose,
                            "initvals": initvals
                        }

                    solution = qpsolvers.solve_problem(**dict(**kwargs_main, **self.qp_solver_keywords))

                    if solution.x is None:
                        print("Solver failed to return a solution. Re-solving with verbose and exiting.")
                        kwargs_main["verbose"] = True
                        qpsolvers.solve_problem(**dict(**kwargs_main, **self.qp_solver_keywords))
                        raise Exception("Exiting")

                    self.x_star_np[i] = solution.x
                    # implicitly assumes if nBatch > 1 then duals are available:
                    if self.nEq > 0:
                        self.mu_star_np[i] = solution.y # TODO : make duals optional by setting = None throughout
                    self.nu_star_np[i] = solution.z

        return None

    def setup_diff(self,Q,q,G,h,A,b):
        '''
        Form the reduced KKT and set-up derivatives through x_star
        Sets non_differentiable check
        '''

        x_mu_nu_star = self.get_x_mu_nu_star()

        if self.nBatch == 1:
            self.non_differentiable = (self.nEq_reduce > self.dim or # quick check for linear dependence
            np.any(self.nu_star_np < 1e-8) ) # weakly active (note nu_star_np by now only contains active nu)

            diff_params = {
                "x_mu_nu_star": x_mu_nu_star,  # differentiable x_star
                "KKT_A_np": self.KKT_A_np,
                "KKT_A_np_factors": self.KKT_A_np_factors,
                "solve_type": self.solve_type,
                "qp_solver": self.qp_solver,
                "lin_solver": self.lin_solver,
                "available_qp_solvers" : self.available_qp_solvers, # to check if lin_solver is a QP method
                "dim": self.dim,
                "nEq": self.nEq,
                "nIneq": self.nIneq,
                "nActive": self.nActive,
                "non_differentiable": self.non_differentiable,
                # internally converts csc --> coo to get indices
                "Q_pattern": None if (Q.layout is torch.strided) else self.Q_np.nonzero(),
                "G_pattern": None if (G.layout is torch.strided) else (self.G_np[self.active, :]).nonzero(),
                "A_pattern": None if ((A is None) or (A.layout is torch.strided)) else self.A_np.nonzero()
            }

            if self.solve_type == "sparse": # does not compute gradient w.r.t. zero entries
                # torch.index_select and .values() only work with COO sparse matrices
                Q = Q.to_sparse_coo().coalesce().values()
                G = G.to_sparse_coo().coalesce()
                G_ind = G._indices()
                G = G.values()
                if self.nEq > 0:
                    A = A.to_sparse_coo().coalesce().values()

            if self.solve_type == "dense":
                G = G[self.active,:]
            elif self.solve_type == "sparse":
                G = G[self.active[G_ind[0,:].numpy()]]
            h = h[self.active]

            x_mu_nu_star = self.differentiate_QP(Q,q,G,h,A,b,diff_params)

            x_star = x_mu_nu_star[0:self.dim]
            mu_star = x_mu_nu_star[self.dim:(self.nEq + self.dim)]
            nu_star = x_mu_nu_star[(self.nEq + self.dim):]
        else:
            x_star = []
            mu_star = []
            nu_star = []
            for i in range(self.nBatch):
                non_differentiable = (self.nEq_reduce[i] > self.dim or  # quick check for linear dependence
                                           np.any(self.nu_star_np[i] < 1e-8))  # weakly active (note nu_star_np by now only contains active nu)

                diff_params = {
                    "x_mu_nu_star": x_mu_nu_star[i],  # differentiable x_star
                    "KKT_A_np": self.KKT_A_np[i],
                    "KKT_A_np_factors": None,
                    "solve_type": self.solve_type,
                    "qp_solver": self.qp_solver,
                    "lin_solver": self.lin_solver,
                    "available_qp_solvers": self.available_qp_solvers,  # to check if lin_solver is a QP method
                    "dim": self.dim,
                    "nEq": self.nEq,
                    "nIneq": self.nIneq,
                    "nActive": self.nActive[i],
                    "non_differentiable": non_differentiable,
                    # internally converts csc --> coo to get indices
                    "Q_pattern": None if (Q[i].layout is torch.strided) else self.Q_np[i].nonzero(),
                    "G_pattern": None if (G[i].layout is torch.strided) else (self.G_np[i][self.active[i,:], :]).nonzero(),
                    "A_pattern": None if ((A is None) or (A[i].layout is torch.strided)) else self.A_np[i].nonzero()
                }

                if self.solve_type == "dense":
                    Q_i = Q[i]
                    G_i = G[i,self.active[i]]

                    if self.nEq > 0:
                        A_i = A[i]
                        b_i = b[i]
                    else:
                        A_i = None
                        b_i = None
                if self.solve_type == "sparse":  # does not compute gradient w.r.t. zero entries
                    # torch.index_select and .values() only work with COO sparse matrices
                    Q_i = Q[i].to_sparse_coo().coalesce().values()

                    G_i = G[i].to_sparse_coo().coalesce()
                    G_ind = G_i._indices()
                    G_i = G_i.values()
                    G_i = G_i[self.active[i,G_ind[0, :].numpy()]]

                    if self.nEq > 0:
                        A_i = A[i].to_sparse_coo().coalesce().values()
                        b_i = b[i]
                    else:
                        A_i = None
                        b_i = None

                h_i = h[i,self.active[i]]

                x_mu_nu_star[i] = self.differentiate_QP(Q_i, q[i], G_i, h_i, A_i, b_i, diff_params)

                x_star += [x_mu_nu_star[i][0:self.dim]]
                mu_star += [x_mu_nu_star[i][self.dim:(self.nEq + self.dim)]]
                nu_star += [x_mu_nu_star[i][(self.nEq + self.dim):]]

        return x_star,mu_star,nu_star

    def get_x_mu_nu_star(self):
        '''
        Constructs torch parameter x_mu_nu_star
        Solve for or retrieve mu_star via the external solver or fully/partially solving the reduced KKT
        Optional refinement of the active set
        '''

        # initial estimated active set before any changes
        self.get_active()
        if self.verbose:
            print("# Active: " + str(self.nActive))

        if self.dual_available:
            # note, does not overwrite any of the solution despite hard-threshold projection onto constraints

            if self.refine_active:
                grad_f = (self.Q_np.dot(self.x_star_np) + self.q_np)
                # don't use mu_nu from refinement, just use to determine tolerances on active
                self.refine(grad_f)

            # mu_star_np, nu_star_np already set without solving ; forget inactive duals (will be 0 later)
            if self.nBatch == 1:
                self.nu_star_np = self.nu_star_np[self.active]
            else:
                nu_star_temp = self.nu_star_np
                self.nu_star_np = []
                for i in range(self.nBatch):
                    self.nu_star_np += [nu_star_temp[i,self.active[i,:]]]

            self.get_reduced_KKT()

        else:
            # Compute the dual variable ; attempt to solve reduced KKT and project (over-write) solution.
            # If disagrees, 2 options:
            # (1) estimate dual variable decoupled from primal with least-squares (which also supports under-determined)
            # (2) improve active set until agrees best

            self.get_reduced_KKT()

            try:
                x_mu_reduced_np = self.solve_KKT_for_dual()

                # check re-obtained remotely similar x_star
                dx_star = np.linalg.norm(self.x_star_np - x_mu_reduced_np [0:self.dim]) / (np.linalg.norm(self.x_star_np))

                if self.verbose:
                    print("reduced solve dx_star: " + str(dx_star))
                assert( dx_star < self.eps_abs*10)

                # if agrees, over-write; essentially project the solution
                # the factorization is re-used in backward for free (TODO: check)
                self.x_star_np = x_mu_reduced_np[0:self.dim] # over-write
                if self.nEq > 0:
                    self.mu_star_np = x_mu_reduced_np[self.dim:(self.nEq + self.dim)] # over-write
                self.nu_star_np = x_mu_reduced_np[(self.nEq + self.dim):] # set
            except  Exception as e:
                print('Reduced KKT solution disagrees; active set choice may be suboptimal or generally unstable: ', repr(e))
                self.KKT_A_np_factors = None # TODO : maybe don't get rid of, but instead check inside backprop in case can still be reused but failed for other reason
                # precompute for optimality conditions and dual residual
                grad_f = (self.Q_np.dot(self.x_star_np) + self.q_np)

                # use only initial active set
                if not self.refine_active:

                    # set-up reduced equality constraints
                    if self.A_np is None:
                        G_aset = self.G_np[self.active, :]
                    else:
                        if self.solve_type == "dense":
                            G_aset = np.vstack((self.A_np, self.G_np[self.active, :]))
                        elif self.solve_type == "sparse":
                            G_aset = sp.sparse.vstack((self.A_np, self.G_np[self.active, :]))

                    # solve for the dual variable
                    if self.solve_type == "dense":
                        mu_nu_star, _ = lin_solvers.dense_LSQ(G_aset.T, -grad_f)
                    elif self.solve_type == "sparse":
                        mu_nu_star, _ = lin_solvers.sparse_LSQ(G_aset.T, -grad_f)

                # refine active set
                elif self.refine_active:
                    mu_nu_star = self.refine(grad_f)

                # save
                if self.nEq > 0:
                    self.mu_star_np = mu_nu_star[:(self.nEq)] # over-write
                self.nu_star_np = mu_nu_star[self.nEq:] # set

        # if self.nBatch == 1:
        #     assert (np.all(self.nu_star_np[self.nEq:, :]) > 0)  # check active dual variable sign
        # TODO : loop to check if nBatch > 1 ; or remove check altogether

        if self.nBatch == 1:
            x_mu_nu_star = torch.tensor(self.x_star_np, dtype=torch.float64)
            if self.mu_star_np is not None:
                mu_star = torch.tensor(self.mu_star_np, dtype=torch.float64)
                x_mu_nu_star = torch.concatenate((x_mu_nu_star,mu_star))

            if self.nActive > 0:
                nu_star = torch.tensor(self.nu_star_np, dtype=torch.float64)
                x_mu_nu_star = torch.concatenate((x_mu_nu_star,nu_star))
        else:
            x_mu_nu_star = []
            for i in range(self.nBatch):
                x_mu_nu_star_temp = torch.tensor(self.x_star_np[i], dtype=torch.float64)
                if self.mu_star_np is not None:
                    mu_star = torch.tensor(self.mu_star_np[i], dtype=torch.float64)
                    x_mu_nu_star_temp = torch.concatenate((x_mu_nu_star_temp, mu_star))

                if self.nActive[i] > 0:
                    nu_star = torch.tensor(self.nu_star_np[i], dtype=torch.float64)
                    x_mu_nu_star_temp = torch.concatenate((x_mu_nu_star_temp, nu_star))
                x_mu_nu_star += [x_mu_nu_star_temp]

        return x_mu_nu_star

    def get_active(self):
        '''
        Determines the active constraints at the solution x_star_np
        '''

        self.get_r_pri()
        # one-sided check on residual
        active = self.r_pri_np < self.eps_active
        self.nActive = np.sum(active,axis=-1)
        self.nEq_reduce = self.nActive + self.nEq
        self.active = active

        return None

    def get_r_pri(self):
        '''
        Determines the primal residual using stored x_star_np, G_np, h_np
        '''

        x,G,h = self.x_star_np, self.G_np, self.h_np

        if self.solve_type == "sparse" and self.nBatch > 1:
            h_approx = h.copy()
            for i in range(self.nBatch):
                h_approx[i] = G[i] @ x[i]
        else:
            h_approx = (G @ np.expand_dims(x,-1)).squeeze(-1)

        self.r_pri_np = h - h_approx  # h - Gx^*

        return None

    def solve_KKT_for_dual(self):
        '''
        Get the current reduced KKT from stored active set and solve it for x_star, and active nu_star
        '''

        if self.time:
            start_reduced_KKT = time.time()

        if self.lin_solver in self.available_qp_solvers:
            QP_form = [self.Q_np, self.q_np, self.KKT_A_np[self.dim:, 0:self.dim], self.KKT_b_np[self.dim:]]
        else:
            QP_form = None

        # solve and store factorization for backwards if available TODO: Make sure that if KKT changes that A_np_factors changes too
        if self.solve_type == "dense":
            x_mu_reduced_np, self.KKT_A_np_factors = lin_solvers.dense_solve(self.KKT_A_np, self.KKT_b_np,
                                                                          linear_solver=self.lin_solver,
                                                                          QP_form=QP_form, x_warmstart=None)
        elif self.solve_type == "sparse":
            x_mu_reduced_np, self.KKT_A_np_factors = lin_solvers.sparse_solve(self.KKT_A_np, self.KKT_b_np,
                                                                           linear_solver=self.lin_solver,
                                                                           QP_form=QP_form, x_warmstart=None)

        if self.time:
            reduced_KKT_time = time.time() - start_reduced_KKT
            print("### Time 1st KKT solve: " + str(reduced_KKT_time))

        return x_mu_reduced_np

    def refine(self,grad_f):
        '''
        Refines the active set by simultaneously minimizing the primal and dual residuals w.r.t. different active sets;
        The solution is fixed
        Sets active, nActive, nEqreduce, and the reduced KKT
        Returns active mu,nu
        '''

        active = self.active
        self.get_r_pri()
        r_pri = self.r_pri_np

        # get ordering of numerically inactive constraints
        i_ord = np.argsort(r_pri, axis=0) # note: if using duals to determine active set by something like nu > eps, the selected duals may not be the least ordered primal
        iterate = True
        prev_tot_viol = 1e10
        prev_mu_nu_star = 0
        iter = 0

        while iterate:
            print("iter refinement: " + str(iter))

            # set-up reduced equality constraints
            if self.A_np is None:
                G_aset = self.G_np[active, :]
            else:
                if self.solve_type == "dense":
                    G_aset = np.vstack((self.A_np, self.G_np[active, :]))
                elif self.solve_type == "sparse":
                    G_aset = sp.sparse.vstack((self.A_np, self.G_np[active, :]))

            # solve for the dual variable
            if self.solve_type == "dense":
                mu_nu_star, r_dual = lin_solvers.dense_LSQ(G_aset.T, -grad_f, lsq_solver=self.qp_solver, eps_abs=self.eps_abs, eps_rel=self.eps_rel)
            elif self.solve_type == "sparse":
                mu_nu_star, r_dual = lin_solvers.sparse_LSQ(G_aset.T, -grad_f, lsq_solver=self.qp_solver, eps_abs=self.eps_abs, eps_rel=self.eps_rel)

            # check violation of KKT and update active set
            tot_viol = np.sqrt(np.linalg.norm(r_pri[i_ord[0:self.nActive]]) ** 2 + r_dual ** 2)
            print("\| r \|_2: " + str(tot_viol))

            if tot_viol < self.eps_active:
                iterate = False
            elif tot_viol < prev_tot_viol:
                print("Change constraints.")
                self.KKT_A_np_factors = None  # KKT has changed, release factorization
                prev_tot_viol = tot_viol
                prev_mu_nu_star = mu_nu_star
                if self.nActive < self.nIneq:
                    active[i_ord[self.nActive]] = True
                    self.nActive += 1
                else:
                    break
            else:
                print("Keep previous constraints.")
                self.nActive = self.nActive - 1  # revert to previous activity
                active[i_ord[self.nActive]] = False

                mu_nu_star = prev_mu_nu_star  # keep previous mu_star
                iterate = False

            iter += 1

        if self.verbose:
            print("# Active Refined: " + str(self.nActive))
        assert (self.nActive == np.sum(self.active))

        # update active set
        self.active = active
        self.nEq_reduce = self.nActive + self.nEq
        self.get_reduced_KKT()

        return mu_nu_star


    def get_reduced_KKT(self):
        '''
        Form the reduced KKT only in np form given the active constraints
        '''

        self.A_reduce_np = []
        self.KKT_A_np = []
        self.KKT_b_np = []

        # append fake batch 1 dimension to reduce code: # TODO ... different way?
        if self.nBatch == 1:
            self.q_np, self.h_np = [self.q_np],[self.h_np]
            self.Q_np,self.G_np = [self.Q_np],[self.G_np]

            self.active = np.expand_dims(self.active,0)
            self.nActive = np.expand_dims(self.nActive,0)
            self.nEq_reduce = np.expand_dims(self.nEq_reduce,0)

            if self.nEq > 0:
                self.b_np = [self.b_np]
                self.A_np = [self.A_np]

        for i in range(self.nBatch):
            # form the effective equality constraints

            if self.nEq_reduce[i] == 0:
                self.A_reduce_np += [None]
                self.KKT_A_np += [self.Q_np[i]]
                self.KKT_b_np += [-self.q_np[i]]
            else:
                if self.nEq == 0:
                    # self.A_reduce_np += [self.G_np[i][self.active[i]] if not (self.nIneq == 1 and self.nActive[i] == 1) else self.G_np[i][self.active[i]]]
                    self.A_reduce_np += [self.G_np[i][self.active[i]]]

                else:
                    if self.solve_type == "dense":
                        # self.A_reduce_np += [np.vstack((self.A_np[i], self.G_np[i][self.active[i]] if not (self.nIneq == 1 and self.nActive[i] == 1) else self.G_np[i][self.active[i]]))]
                        self.A_reduce_np += [np.vstack((self.A_np[i], self.G_np[i][self.active[i]]))]

                    elif self.solve_type == "sparse":
                        # self.A_reduce_np += [sp.sparse.vstack((self.A_np[i], self.G_np[i][self.active[i]] if not (self.nIneq == 1 and self.nActive[i] == 1) else self.G_np[i][self.active[i]]), format="csc")]
                        self.A_reduce_np += [sp.sparse.vstack((self.A_np[i], self.G_np[i][self.active[i]]), format="csc")]

                # get np version for calculations, including sparse if necessary
                if self.solve_type == "dense":
                    self.KKT_A_np += [np.bmat([[self.Q_np[i], np.transpose(self.A_reduce_np[i])], [self.A_reduce_np[i], np.zeros((self.nEq_reduce[i],self.nEq_reduce[i]))]])]
                elif self.solve_type == "sparse":
                    self.KKT_A_np += [bmat([[self.Q_np[i], np.transpose(self.A_reduce_np[i])], [self.A_reduce_np[i], None]], format="csc")]
                if self.nEq == 0:
                    self.KKT_b_np += [np.concatenate((-self.q_np[i],self.h_np[i][self.active[i]]))]
                else:
                    self.KKT_b_np += [np.concatenate((-self.q_np[i],self.b_np[i],self.h_np[i][self.active[i]]))]

        # squeeze the extra dimensions
        if self.nBatch == 1:
            self.Q_np, self.q_np, self.G_np, self.h_np = self.Q_np[0],self.q_np[0],self.G_np[0],self.h_np[0]
            self.active = np.squeeze(self.active, 0)
            self.nEq_reduce = np.squeeze(self.nEq_reduce, 0)
            self.nActive = np.squeeze(self.nActive,0)
            if self.nEq > 0:
                self.A_np, self.b_np = self.A_np[0],self.b_np[0]

            self.KKT_A_np = self.KKT_A_np[0]
            self.KKT_b_np = self.KKT_b_np[0]
            self.A_reduce_np = self.A_reduce_np[0]

        return None

def call_single_qpsolvers(Q,q,G,h,A,b,kwargs):
    kwargs_problem = {
        "problem": qpsolvers.Problem(P=Q, q=q, G=G,h=h, A=A, b=b)
    }

    solution = qpsolvers.solve_problem(**dict(**kwargs_problem, **kwargs))

    if solution.x is None:
        print("Solver failed to return a solution. Re-solving with verbose and exiting.")
        kwargs["verbose"] = True
        qpsolvers.solve_problem(**dict(**kwargs_problem, **kwargs))
        raise Exception("Exiting")

    return solution

class differentiate_QP(torch.autograd.Function):
    '''
    Differentiate the QP explicitly using the reduced KKT formed with the active constraints
    '''

    @staticmethod
    def forward(ctx, Q, q, G, h, A, b, params):
        '''
        Just return known solution as differentiable parameter. Store data for backwards.
        '''
        ctx.KKT_A = params["KKT_A_np"]
        ctx.pre_factorization = params["KKT_A_np_factors"]
        ctx.solve_type = params["solve_type"]
        ctx.qp_solver = params["qp_solver"]
        ctx.lin_solver = params["lin_solver"]
        ctx.available_qp_solvers = params["available_qp_solvers"]
        ctx.dim = params["dim"]
        ctx.nEq = params["nEq"]
        ctx.nIneq = params["nIneq"]
        ctx.nActive = params["nActive"]
        ctx.non_diff = params["non_differentiable"] # have to not use torch keyword --> non_diff
        ctx.x_mu_nu_star = params["x_mu_nu_star"].detach().numpy()

        specific_requires_grad = [] # handle None case
        for var in [Q,q,G,h,A,b]:
            if var is None:
                specific_requires_grad += [False]
            else:
                specific_requires_grad += [var.requires_grad]

        ctx.specific_requires_grad = specific_requires_grad
        ctx.patterns = [params["Q_pattern"],params["G_pattern"],params["A_pattern"]]

        return params["x_mu_nu_star"] # just return x_star

    @staticmethod
    def backward(ctx,grad_output):
        '''
        Computes dl/dQ, dl/dq, dl/dG, dl/dh, dl/dA, dl/db
        '''

        # For geometry scaling, profile the backward through QP alone
        # t = time.time() # TODO: note that this should be removed in final product
        # For geometry scaling, profile the backward through QP alone

        KKT_A = ctx.KKT_A
        KKT_b = grad_output.numpy()

        if ctx.lin_solver in ctx.available_qp_solvers:
            QP_form = [KKT_A[0:ctx.dim,0:ctx.dim],-KKT_b[0:ctx.dim],KKT_A[ctx.dim:,0:ctx.dim],KKT_b[ctx.dim:]]
        else:
            QP_form = None

        try:
            if ctx.non_diff:
                raise Exception("Weakly active anticipated. Back-propagate using least-squares.")
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    if ctx.solve_type == "dense":
                        grad_b,_ = lin_solvers.dense_solve(KKT_A, KKT_b,linear_solver=ctx.lin_solver,pre_factorization=ctx.pre_factorization,QP_form=QP_form)
                    elif ctx.solve_type == "sparse":
                        grad_b,_ = lin_solvers.sparse_solve(KKT_A, KKT_b,linear_solver=ctx.lin_solver,pre_factorization=ctx.pre_factorization,QP_form=QP_form)

            assert(grad_b is not None)

        # if linear solve fails, do least-squares. This occurs when there are weakly active constraints.
        except Exception as e:
            print('Linear solve failed. Back-propagate using least-squares: ', repr(e))

            print("Use QP solver to solve least-squares. Users can change solver and tolerances on L1135.")
            if ctx.solve_type == "dense":
                # grad_b,_ = lin_solvers.dense_LSQ(KKT_A, KKT_b, lsq_solver=ctx.qp_solver, eps_abs=1e-5, eps_rel=1e-5) # TODO: PIQP failed on sudoku... potentially because output empty arrays or vectors without dim 1 at end
                grad_b,_ = lin_solvers.dense_LSQ(KKT_A, KKT_b, lsq_solver="scipy", eps_abs=1e-5, eps_rel=1e-5)
            elif ctx.solve_type == "sparse":
                # grad_b,_ = lin_solvers.sparse_LSQ(KKT_A, KKT_b, lsq_solver=ctx.qp_solver, eps_abs=1e-5, eps_rel=1e-5)
                grad_b,_ = lin_solvers.sparse_LSQ(KKT_A, KKT_b, lsq_solver="scipy", eps_abs=1e-5, eps_rel=1e-5)

        if len(grad_b.shape) == 2:
            grad_b = grad_b.squeeze(-1)

        x = ctx.x_mu_nu_star[0:ctx.dim]
        mu = ctx.x_mu_nu_star[ctx.dim:(ctx.dim+ctx.nEq)]
        nu = ctx.x_mu_nu_star[(ctx.dim+ctx.nEq):]
        dx =  -grad_b[0:ctx.dim]
        dmu = -grad_b[ctx.dim:(ctx.dim+ctx.nEq)]
        dnu = -grad_b[(ctx.dim+ctx.nEq):]

        grad_Q,grad_q,grad_G,grad_h,grad_A,grad_b = None,None,None,None,None,None
        if ctx.specific_requires_grad[0]:
            if ctx.patterns[0] is None:
                grad_Q = np.outer(dx,x)
                grad_Q = torch.from_numpy(1/2*(grad_Q + grad_Q.T))
            else:
                i_row = ctx.patterns[0][0]
                i_col = ctx.patterns[0][1]
                grad_Q = torch.from_numpy(1/2*(np.multiply(dx[i_row],x[i_col]) + np.multiply(x[i_row],dx[i_col]))).squeeze(-1)
                if grad_Q.dim() == 2:
                    grad_Q = grad_Q.squeeze(-1)
                elif grad_Q.dim() == 0:
                    grad_Q = grad_Q.unsqueeze(0)
        if ctx.specific_requires_grad[1]:
            grad_q = torch.from_numpy(dx)
        if ctx.specific_requires_grad[2]:
            if ctx.patterns[1] is None:
                grad_G = torch.from_numpy(np.outer(dnu, x) + np.outer(nu, dx))
            else:
                i_row = ctx.patterns[1][0]
                i_col = ctx.patterns[1][1]
                grad_G = torch.from_numpy(np.multiply(dnu[i_row],x[i_col]) + np.multiply(nu[i_row],dx[i_col])).squeeze(-1)
                if grad_G.dim() == 2:
                    grad_G = grad_G.squeeze(-1)
                elif grad_G.dim() == 0:
                    grad_G = grad_G.unsqueeze(0)
        if ctx.specific_requires_grad[3]:
            grad_h = torch.from_numpy(-dnu)
        if ctx.specific_requires_grad[4]:
            if ctx.patterns[2] is None:
                grad_A = torch.from_numpy(np.outer(dmu, x)) + np.outer(mu, dx)
            else:
                i_row = ctx.patterns[2][0]
                i_col = ctx.patterns[2][1]
                grad_A = torch.from_numpy(np.multiply(dmu[i_row],x[i_col]) + np.multiply(mu[i_row],dx[i_col])).squeeze(-1)
                if grad_A.dim() == 2:
                    grad_A = grad_A.squeeze(-1)
                elif grad_A.dim() == 0:
                    grad_A = grad_A.unsqueeze(0)

        if ctx.specific_requires_grad[5]:
            grad_b = torch.from_numpy(-dmu)

        # For geometry scaling, profile the backward through QP alone
        # t_diff = time.time() - t # TODO: note that this should be removed in final product
        # f = open("../experiments/geometry/results/profiling/t_diff.dat","w+")
        # print(t_diff,file=f)
        # For geometry scaling, profile the backward through QP alone

        return grad_Q,grad_q,grad_G,grad_h,grad_A,grad_b,None

def build_settings(check_PSD=False,time=False,solve_type="dense",dual_available=None,normalize_constraints=False,empty_batch=True,warm_start_from_previous=False,omp_parallel=False,n_cpu=None, # general arguments
                   eps_active=1e-5,refine_active=False, # active arguments
                   qp_solver=None,verbose=False,qp_solver_keywords=None,eps_abs=1e-6,eps_rel=0, # qp solver arguments ... default to solver preference
                   lin_solver=None): # linear solver arguments

    available_qp_solvers = qpsolvers.available_solvers

    if verbose:
        print("Available QP solvers:\n" + str(qpsolvers.available_solvers))
        print("Available linear solvers:\n" + str(lin_solvers.get_dense_solvers() + lin_solvers.get_sparse_solvers() + qpsolvers.available_solvers))

    if solve_type == "dense":
        if qp_solver is None: # get first available qp solver
            if "cvxopt" in qpsolvers.dense_solvers:
                qp_solver = "cvxopt"
            else:
                qp_solver = qpsolvers.dense_solvers[0]
        if lin_solver is None:
            lin_solver = "scipy LU"
        # assert(qp_solver in qpsolvers.dense_solvers) # qpsolvers labeling of dense/sparse is not a strict classification
        assert(lin_solver in lin_solvers.get_dense_solvers())
    elif solve_type == "sparse":
        if qp_solver is None: # get first available qp solver
            if "gurobi" in qpsolvers.sparse_solvers:
                qp_solver = "gurobi"
            else:
                qp_solver = qpsolvers.sparse_solvers[0]
        if lin_solver is None:
            lin_solver = "scipy SPLU"
        # assert(qp_solver in qpsolvers.sparse_solvers)
        assert(lin_solver in lin_solvers.get_sparse_solvers())

    if dual_available is None:
        if qp_solver in ["clarabel","cvxopt","daqp","ecos","gurobi","highs","hpipm","mosek","osqp","piqp","proxqp","qpalm","qpoases","qpswift","quadprog","scs"]:
            if verbose:
                print("Solver is in base qpsolvers, duals are available.")
            dual_available = True
        else:
            raise("Solver is not in base qpsolvers, user must specify (T/F) if duals are available")


    if omp_parallel and n_cpu is None:
        n_cpu = os.cpu_count()
        print("No CPU count given, using all available: " + str(n_cpu))

    if qp_solver_keywords is None:
        qp_solver_keywords = {}
        qp_solver_keywords = set_solver_tolerance(qp_solver_keywords,qp_solver,eps_abs,eps_rel)

    else:
        print("eps_abs,eps_rel ignored if custom keywords given")

    if verbose:
        print("qp_solver: " + str(qp_solver))
        print("lin_solver: " + str(lin_solver))
        print("qp_solver_keywords: " + str(qp_solver_keywords))

    assert(isinstance(qp_solver_keywords,dict))

    settings = {
        "verbose" : verbose,
        "check_PSD" : check_PSD,
        "time" : time,
        "available_qp_solvers" : available_qp_solvers,
        "solve_type" : solve_type,
        "qp_solver": qp_solver,
        "dual_available": dual_available,
        "normalize_constraints": normalize_constraints,
        "empty_batch": empty_batch,
        "warm_start_from_previous": warm_start_from_previous,
        "qp_solver_keywords" : qp_solver_keywords,
        "eps_active": eps_active,
        "eps_abs": eps_active,
        "eps_rel": eps_active,
        "lin_solver": lin_solver,
        "refine_active": refine_active,
        "omp_parallel" : omp_parallel,
        "n_cpu" : n_cpu
    }

    return settings