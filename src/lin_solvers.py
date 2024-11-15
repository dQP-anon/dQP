import numpy as np
import scipy as sp

import qpsolvers

# linear solvers
dense_solvers = ["scipy LU","scipy LDL"]
sparse_solvers = ["scipy SPLU"]

from src.set_solver_tolerance import set_solver_tolerance
from scipy.sparse import csc_matrix # qp solvers uses csc

try:
    import pypardiso
    sparse_solvers += ["pardiso"]
except:
    print("skipping pardiso")

try:
    import qdldl
    sparse_solvers += ["qdldl"]
except:
    print("skipping qdldl")

import sys
sys.path.append('../')

def get_dense_solvers():
    return dense_solvers

def get_sparse_solvers():
    return sparse_solvers

def dense_solve(A,b,linear_solver="scipy LU",pre_factorization=None,QP_form=None,x_warmstart=None):
    '''
    Wrapper for dense linear solves for matrices with KKT form (symmetric indefinite)
    '''

    if linear_solver == "scipy LU":
        if pre_factorization is not None:
            x = sp.linalg.lu_solve(pre_factorization, b)
            return x,pre_factorization

        # SciPy's LU factorization and solve
        lu, piv = sp.linalg.lu_factor(A)
        A_factors = (lu,piv)
        x = sp.linalg.lu_solve(A_factors, b)
        return x, A_factors
    elif linear_solver == "scipy LDL":
        assert(np.linalg.norm(A - (A + A.T)/2) < 1e-18)

        if pre_factorization is not None:
            lu,d,perm,P = pre_factorization
            x = P.T @ sp.linalg.solve_triangular(lu[perm, :].T, sp.linalg.solve(d, sp.linalg.solve_triangular(lu[perm, :],P @ b,lower=True)))
            return x,pre_factorization

        # SciPy's LDL factorization and solve
        n = np.shape(A)[0]
        lu, d, perm = sp.linalg.ldl(A, lower=1)  # Use the upper part
        P = sp.sparse.csr_matrix((np.ones(n, dtype=int), (np.arange(0, n), perm)), shape=(n, n))
        x = P.T @ sp.linalg.solve_triangular(lu[perm, :].T, sp.linalg.solve(d, sp.linalg.solve_triangular(lu[perm, :],P @ b,lower=True)))
        A_factors = (lu,d,perm,P)
        return x, A_factors
    elif linear_solver in qpsolvers.dense_solvers and QP_form is not None:
        # Use a QP solver to solve the KKT for a solely equality-constrained problem
        Q,q,A,b = QP_form
        x = qpsolvers.solve_problem(qpsolvers.Problem(P=Q, q=q, A=A, b=b),solver=linear_solver,initvals=x_warmstart)
        if x.x is None:
            return None, None
        x = np.concatenate((x.x, x.y))
        x = np.expand_dims(x, -1)
        return x, None

def sparse_solve(A,b,linear_solver="pardiso",pre_factorization=None,QP_form=None,x_warmstart=None):
    '''
    Wrapper for sparse linear solves for matrices with KKT form (symmetric indefinite)
    '''

    if linear_solver == "pardiso":
        # Pardiso's sparse LU factorization (internally saves it into the pypardiso object) and solve
        x = pypardiso.spsolve(A,b)
        x = np.expand_dims(x,axis=1)
        return x,None
    elif linear_solver == "scipy SPLU":
        if pre_factorization is not None:
            x = pre_factorization.solve(b)
            return x, pre_factorization

        # SciPy's sparse solve
        # x = sp.sparse.linalg.spsolve(A,b)
        A_factors = sp.sparse.linalg.splu(A)
        x = A_factors.solve(b)
        return x,A_factors
    elif linear_solver == "qdldl":

        # assert(sp.sparse.linalg.norm(A - (A + A.T)/2) < 1e-18)

        if pre_factorization is not None:
            x = pre_factorization.solve(b)
            x = np.expand_dims(x, axis=1)
            return x, pre_factorization

        # raise NotImplementedError
        A_factors = qdldl.Solver(csc_matrix(A) + 1e-16*sp.sparse.eye(A.shape[0])) # why is the perturbation necessary to prevent elimination tree failure?
        x = A_factors.solve(b)
        x = np.expand_dims(x,axis=1)
        return x, A_factors
    elif linear_solver in qpsolvers.sparse_solvers and QP_form is not None:
        # Use a QP solver to solve the KKT for a solely equality-constrained problem

        Q, q, A, b = QP_form

        x = qpsolvers.solve_problem(qpsolvers.Problem(P=Q, q=q, A=A, b=b), solver=linear_solver,initvals=x_warmstart)
        if x.x is None:
            return None, None

        x = np.concatenate((x.x, x.y))
        x = np.expand_dims(x, -1)
        return x, None

def dense_LSQ(A,b,lsq_solver="scipy",eps_abs=1e-5,eps_rel=1e-5): # TODO : include warm-start, especially for refinement
    '''
    Wrapper for dense least-squares
    '''

    if lsq_solver in qpsolvers.dense_solvers:
        if lsq_solver in ["cvxopt","ecos","proxqp"]:
            lsq_solver = "piqp" # TODO : better choice ; but for now removing QP solvers that need full rank

        qp_solver_keywords = {}
        qp_solver_keywords = set_solver_tolerance(qp_solver_keywords, lsq_solver, eps_abs, eps_rel)
        kwargs_main = {
            "R": A,
            "s": b,
            "solver": lsq_solver,
            "sparse_conversion": False
        }
        x = qpsolvers.solve_ls(**dict(**kwargs_main, **qp_solver_keywords))
        r = np.linalg.norm(b - A @ x)
    elif lsq_solver == "scipy":
        # SciPy's dense least-squares
        x = sp.linalg.lstsq(A, b)[0]
        r = np.linalg.norm(b - A @ x)

    print("LSQ residual: " + str(r))

    return x,r

def sparse_LSQ(A,b,lsq_solver="scipy",eps_abs=1e-5,eps_rel=1e-5):
    '''
    Wrapper for sparse least-squares
    '''

    if lsq_solver in qpsolvers.sparse_solvers:
        qp_solver_keywords = {}
        qp_solver_keywords = set_solver_tolerance(qp_solver_keywords, lsq_solver, eps_abs, eps_rel)
        kwargs_main = {
            "R": A,
            "s": b,
            "solver": lsq_solver,
            "sparse_conversion": True
        }
        x = qpsolvers.solve_ls(**dict(**kwargs_main, **qp_solver_keywords))
        r = np.linalg.norm(b - A @ x)
    elif lsq_solver == "scipy":
        # SciPy's sparse iterative least-squares
        x, _, _, r = sp.sparse.linalg.lsqr(A, b, atol=eps_rel, btol=eps_rel)[:4]
        x = np.expand_dims(x, axis=-1)

    print("LSQ residual: " + str(r))

    return x,r