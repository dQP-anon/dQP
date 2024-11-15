# Adapted from qpbenchmark https://github.com/qpsolvers/qpbenchmark/blob/main/qpbenchmark/solver_settings.py
import numpy as np
def set_solver_tolerance(qp_solver_keywords,qp_solver,eps_abs,eps_rel):
    """Set absolute tolerances for solvers that support it.
        Args:
            eps_abs: Absolute primal, dual and duality-gap tolerance.

        Notes:
            When we set an absolute tolerance :math:`\epsilon_{abs}` on
            residuals, we ask the solver to find an approximation of the
            optimum such that the primal residual, dual residual and duality
            gap are below :math:`\epsilon_{abs}`, that is:

            .. math::

                \begin{align}
                r_p := \max(\| A x - b \|_\infty, [G x - h]^+, [lb - x]^+,
                [x - ub]^+) & \leq \epsilon_{abs} \\
                r_d := \| P x + q + A^T y + G^T z + z_{box} \|_\infty &
                \leq \epsilon_{abs} \\
                r_g := | x^T P x + q^T x + b^T y + h^T z + lb^T z_{box}^- +
                ub^T z_{box}^+ | & \leq \epsilon_{abs}
                \end{align}

            were :math:`v^- = \min(v, 0)` and :math:`v^+ = \max(v, 0)`. The
            tolerance on the primal residual is called "feasibility tolerance"
            by some solvers, for instance CVXOPT and ECOS. See `this note
            <https://scaron.info/blog/optimality-conditions-and-numerical-tolerances-in-qp-solvers.html>`__
            for more details.
    """

    if eps_abs is None or eps_rel is None:
        print("If a tolerance is None, leaves empty and lets the solvers choose their default.")

    if qp_solver ==  "clarabel":
        if eps_abs is not None:
            qp_solver_keywords["tol_feas"] = eps_abs
            qp_solver_keywords["tol_gap_abs"] = eps_abs
        if eps_rel is not None:
            qp_solver_keywords["tol_gap_rel"] = eps_rel
    elif qp_solver == "cvxopt":
        if eps_abs is not None:
            qp_solver_keywords["feastol"] = eps_abs
    elif qp_solver == "daqp":
        if eps_abs is not None:
            qp_solver_keywords["dual_tol"] = eps_abs
            qp_solver_keywords["primal_tol"] = eps_abs
    elif qp_solver == "ecos":
        if eps_abs is not None:
            qp_solver_keywords["feastol"] = eps_abs
    elif qp_solver == "gurobi":
        if eps_abs is not None:
            qp_solver_keywords["FeasibilityTol"] = eps_abs
            qp_solver_keywords["OptimalityTol"] = eps_abs
    elif qp_solver == "highs":
        if eps_abs is not None:
            qp_solver_keywords["dual_feasibility_tolerance"] = eps_abs
    elif qp_solver == "hpipm":
        if eps_abs is not None:
            qp_solver_keywords["tol_comp"] = eps_abs
            qp_solver_keywords["tol_stat"] = eps_abs
            qp_solver_keywords["tol_eq"] = eps_abs
            qp_solver_keywords["tol_ineq"] = eps_abs
    elif qp_solver == "osqp":
        if eps_abs is not None:
            qp_solver_keywords["eps_abs"] = eps_abs
        if eps_rel is not None:
            qp_solver_keywords["eps_rel"] = eps_rel
    elif qp_solver == "piqp":
        qp_solver_keywords["check_duality_gap"] = True
        if eps_abs is not None:
            qp_solver_keywords["eps_abs"] = eps_abs
            qp_solver_keywords["eps_duality_gap_abs"] = eps_abs
        if eps_rel is not None:
            qp_solver_keywords["eps_duality_gap_rel"] = eps_rel
            qp_solver_keywords["eps_rel"] = eps_rel
    elif qp_solver == "proxqp":
        if eps_abs is not None:
            qp_solver_keywords["eps_abs"] = eps_abs
            qp_solver_keywords["eps_duality_gap_abs"] = eps_abs
        if eps_rel is not None:
            qp_solver_keywords["eps_duality_gap_rel"] = eps_rel
            qp_solver_keywords["eps_rel"] = eps_rel
    elif qp_solver == "qpalm":
        if eps_abs is not None:
            qp_solver_keywords["eps_abs"] = eps_abs
        if eps_rel is not None:
            qp_solver_keywords["eps_rel"] = eps_rel
    elif qp_solver == "qpswift":
        if eps_abs is not None:
            qp_solver_keywords["RELTOL"] = eps_abs * np.sqrt(3.0) # TODO : check? this is what qpbenchmarkhas in solver_settings.py Line 116
    elif qp_solver == "scs":
        if eps_abs is not None:
            qp_solver_keywords["eps_abs"] = eps_abs
        if eps_rel is not None:
            qp_solver_keywords["eps_rel"] = eps_rel

    return qp_solver_keywords