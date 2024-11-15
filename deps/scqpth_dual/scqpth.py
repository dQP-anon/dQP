import torch
import torch.nn as nn
from .lu_layer import TorchLU


class SCQPTHNet(nn.Module):
    def __init__(self, control):
        super().__init__()
        self.control = control

    def forward(self, Q, p, A, lb, ub):
        unroll = self.control.get('unroll', False)
        if unroll:
            x = scqpth_solve(Q=Q, p=p, A=A, lb=lb, ub=ub, control=self.control)
        else:
            x,y = SCQPTHLayer.apply(Q, p, A, lb, ub, self.control)
        return x,y


class SCQPTHLayer(torch.autograd.Function):
    """
    Autograd function for forward solving and backward differentiating constraint QP
    """

    @staticmethod
    def forward(ctx, Q, p, A, lb, ub, control):
        """
        ADMM algorithm for forward solving constraint QP
        """

        # --- forward solve
        # --- solve QP:
        sol = scqpth_solve(Q=Q, p=p, A=A, lb=lb, ub=ub, control=control)
        x = sol.get('x')
        z = sol.get('z')
        y = sol.get('y')
        rho = sol.get('rho')
        if not torch.is_tensor(rho):
            rho = torch.tensor(rho)

        # --- save for backwards:
        ctx.eps = control.get('eps', 1e-6)
        ctx.save_for_backward(x, z, y, Q, A, lb, ub, rho)

        return x, y.detach()

    @staticmethod
    def backward(ctx, dl_dz, _):
        """
        Fixed point backward differentiation
        """
        eps = ctx.eps
        x, z, y, Q, A, lb, ub, rho = ctx.saved_tensors
        grads = scqpth_grad(dl_dz=dl_dz, x=x, z=z, y=y, Q=Q, A=A, lb=lb, ub=ub, eps=eps)
        return grads


class SCQPTH:
    def __init__(self, Q, p, A, lb, ub, control):
        # --- input space:
        self.Q = Q
        self.p = p
        self.A = A
        self.lb = lb
        self.ub = ub
        self.control = control

        # --- solution storage:
        self.sol = {}

    def solve(self):
        # --- for warm-start:
        x = self.sol.get('x')
        z = self.sol.get('z')
        y = self.sol.get('y')
        if self.control.get('warm_start'):
            rho = self.sol.get('rho', self.control.get('rho'))
            self.control['rho'] = rho
        # --- solve QP:
        sol = scqpth_solve(Q=self.Q, p=self.p, A=self.A, lb=self.lb, ub=self.ub, control=self.control, x=x, z=z, y=y)
        # --- return solution
        if self.control.get('unroll', False):
            return sol
        else:
            self.sol = sol
            x = sol.get('x')
            return x

    def update(self, Q=None, p=None, A=None, lb=None, ub=None, control=None):
        if Q is not None:
            self.Q = Q
        if p is not None:
            self.p = p
        if A is not None:
            self.A = A
        if lb is not None:
            self.lb = lb
        if ub is not None:
            self.ub = ub
        if control is not None:
            self.control = control
        return None


def scqpth_solve(Q, p, A, lb, ub, control, x=None, z=None, y=None):
    #######################################################################
    # Solve a batch QP in form:
    #   x_star =   argmin_x 0.5*x^TQx + p^Tx
    #             subject to lb <= Ax <= ub
    #
    # Q:  A (n_batch,n_x,n_x) SPD tensor
    # p:  A (n_batch,n_x,1) tensor.
    # A:  A (n_batch,n_eq, n_x) tensor.
    # b:  A (n_batch,n_eq) tensorr.
    # lb: A (n_batch,n_x,1) tensor
    # ub: A (n_batch,n_x,1) tensor
    # Returns: x_star:  A (n_batch,n_x,1) tensor
    #######################################################################

    # --- unpacking control:
    max_iters = control.get('max_iters', 10_000)
    eps_abs = control.get('eps_abs', 1e-3)
    eps_rel = control.get('eps_rel', 1e-3)
    eps_infeas = control.get('eps_infeas', 1e-4)
    check_solved = control.get('check_solved', 25)
    check_feasible = control.get('check_feasible', check_solved)
    check_feasible = max(round(check_feasible/check_solved), 1)*check_solved
    alpha = control.get('alpha', 1.2)
    alpha_iter = control.get('alpha_iter', 100)
    rho = control.get('rho')
    rho_min = control.get('rho_min', 1e-6)
    rho_max = control.get('rho_max', 1e6)
    adaptive_rho = control.get('adaptive_rho', False)
    adaptive_rho_tol = control.get('adaptive_rho_tol', 5)
    adaptive_rho_iter = control.get('adaptive_rho_iter', 100)
    adaptive_rho_iter = max(round(adaptive_rho_iter/check_solved), 1)*check_solved
    adaptive_rho_max_iter = control.get('adaptive_rho_max_iter', 10_000)
    sigma = control.get('sigma', 1e-6)
    sigma = max(sigma, 0)
    verbose = control.get('verbose', False)
    scale = control.get('scale', True)
    beta = control.get('beta')
    warm_start = control.get('warm_start', False)
    unroll = control.get('unroll', False)

    # --- prep:
    n_batch = Q.shape[0]
    n_A = A.shape[1]
    n_x = A.shape[2]
    if p is None:
        p = torch.zeros((n_batch, n_x,1))
    p_norm = torch.linalg.norm(p, ord=torch.inf, dim=1, keepdim=True)
    any_lb = torch.max(lb) > -float("inf")
    any_ub = torch.min(ub) < float("inf")

    # --- scaling and pre-conditioning:
    if scale:
        if Q is not None:
            # --- compute Q_norm:
            Q_norm = torch.linalg.norm(Q, ord=torch.inf, dim=1)
            idx = Q_norm <= 0.0
            if torch.any(idx):
                Q_norm_min = torch.clamp(Q_norm.mean(dim=1), min=1e-6)
                Q_norm_clamp = torch.clamp(Q_norm, min=Q_norm_min.unsqueeze(1))
                Q_norm[idx] = Q_norm_clamp[idx]
            # --- compute D:
            D = torch.sqrt(1 / Q_norm)
            if beta is None:
                v = torch.quantile(D, q=torch.tensor([0.10, 0.90], dtype=D.dtype), dim=1)
                beta = 1 - v[[0]]/v[[1]]
                beta = beta.T
            D = (1 - beta) * D + beta * D.mean(dim=1, keepdim=True)
            Q = (D.unsqueeze(2) * Q * D.unsqueeze(1))
        else:
            D = 1.0
        # --- scale p:
        p = D.unsqueeze(2) * p
        # --- A, lb, ub scaling:
        AD = A * D.unsqueeze(1)
        AD_norm = torch.linalg.norm(AD, ord=torch.inf, dim=2)
        idx = AD_norm <= 0.0
        if torch.any(idx):
            AD_norm_min = torch.clamp(AD_norm.mean(dim=1), min=1e-6)
            AD_norm_clamp = torch.clamp(AD_norm, min=AD_norm_min.unsqueeze(1))
            AD_norm[idx] = AD_norm_clamp[idx]
        E = 1 / AD_norm
        E = E.unsqueeze(2)
        D = D.unsqueeze(2)
        A = E * AD
        lb = E * lb
        ub = E * ub
    else:
        D = 1.0
        E = 1.0

    # --- finite lb and ub for infeasibility checks:
    lb_inf = torch.isinf(lb)
    ub_inf = torch.isinf(ub)
    if torch.any(lb_inf).item():
        lb_finite = lb * 1.0
        lb_finite[lb_inf] = 1.0
    else:
        lb_finite = lb
    if torch.any(ub_inf).item():
        ub_finite = ub * 1.0
        ub_finite[ub_inf] = 1.0
    else:
        ub_finite = ub

    # --- storage AT for reuse later:
    AT = torch.transpose(A, dim0=1, dim1=2)
    ATA = torch.matmul(AT, A)

    # --- rho parameter selection:
    if rho is None:
        A_norm = torch.linalg.matrix_norm(ATA, keepdim=True)
        if Q is not None:
            Q_norm = torch.linalg.matrix_norm(Q, keepdim=True)
            rho = Q_norm / A_norm
        else:
            rho = 1 / A_norm
        rho = rho * (n_A / n_x)**0.5
        rho = torch.clamp(rho, min=rho_min, max=rho_max)

    # --- warm-starting:
    has_x = x is not None
    has_z = z is not None
    has_y = y is not None
    if warm_start and has_x and has_z and has_y:
        x = 0.95 * (x / D)
        z = 0.95 * (E * z)
        y = 0.95 * (y / E)
        u = y / rho
    else:
        x = torch.zeros((n_batch, n_x, 1), dtype=Q.dtype)
        z = torch.zeros((n_batch, n_A, 1), dtype=Q.dtype)
        u = torch.zeros((n_batch, n_A, 1), dtype=Q.dtype)

    # --- LU factorization:
    if Q is not None:
        M = Q + rho * ATA
    else:
        M = rho * ATA
    if sigma > 0:
        diag = M.diagonal(dim1=1, dim2=2) + sigma
        M[:, range(n_x), range(n_x)] = diag
    with torch.no_grad():
        LU, P = torch.linalg.lu_factor(M)
    if unroll:
        LUModel = TorchLU(A=M, LU=LU, P=P)

    # --- preambles
    is_optimal = False
    is_infeas = torch.tensor(False)
    is_primal_infeas = False
    is_dual_infeas = False
    primal_error = dual_error = Ax_norm = z_norm = ATy_norm = Qx_norm = iter = 1.0

    # --- main loop
    for i in range(max_iters):
        # --- adaptive rho:
        if adaptive_rho and i % adaptive_rho_iter == 0 and 0 < i < adaptive_rho_max_iter:
            num = primal_error / tol_primal_rel_norm
            denom = dual_error / tol_dual_rel_norm
            denom = torch.clamp(denom, min=1e-12)
            ratio = (num / denom) ** 0.5
            update_rho_1 = (ratio > adaptive_rho_tol).sum() > 0
            update_rho_2 = (ratio < (1 / adaptive_rho_tol)).sum() > 0
            update_rho = update_rho_1.item() or update_rho_2.item()
            if update_rho:
                rho_new = rho * ratio
                rho = rho * do_stop + rho_new * torch.logical_not(do_stop)
                rho = torch.clamp(rho, min=rho_min, max=rho_max)
                if Q is not None:
                    M = Q + rho * ATA
                else:
                    M = rho * ATA
                if sigma > 0:
                    diag = M.diagonal(dim1=1, dim2=2) + sigma
                    M[:, range(n_x), range(n_x)] = diag
                LU, P = torch.linalg.lu_factor(M)
                if unroll:
                    LUModel = TorchLU(A=M, LU=LU, P=P)

        # --- projection to sub-space:
        x_prev = x
        rhs = -p + rho * torch.matmul(AT, z - u) + sigma * x_prev
        if unroll:
            x = LUModel(A=M, b=rhs)
        else:
            x = torch.linalg.lu_solve(LU, P, rhs)
        if i > alpha_iter:
            x = alpha * x + (1 - alpha) * x_prev

        # --- proximal projection:
        Ax = torch.matmul(A, x)
        z = Ax + u
        if any_lb:
            z = torch.maximum(z, lb)
        if any_ub:
            z = torch.minimum(z, ub)

        # --- update residual:
        r = Ax - z
        u_prev = u
        u = u_prev + r

        # --- check solved:
        if i % check_solved == 0 or i >= (max_iters - 1):
            # --- update dual variable:
            y_prev = rho * u_prev
            y = rho * u
            # --- dual residual elements:
            ATy = torch.matmul(AT, y)
            if Q is not None:
                Qx = torch.matmul(Q, x)
            else:
                Qx = 0.0
            s = Qx + p + ATy

            # --- running sum of residuals or dual variables:
            primal_error = torch.linalg.norm(r / E, ord=torch.inf, dim=1, keepdim=True)
            dual_error = torch.linalg.norm(s / D, ord=torch.inf, dim=1, keepdim=True)
            if verbose:
                is_feas = torch.logical_not(is_infeas)
                primal_error_max = (primal_error * is_feas).max()
                dual_error_max = (dual_error * is_feas).max()
                print('iteration = {:d}'.format(i))
                print('|| primal_error|| = {:f}'.format(primal_error_max.item()))
                print('|| dual_error|| = {:f}'.format(dual_error_max.item()))

            # --- primal:
            Ax_norm = torch.linalg.norm(Ax / E, ord=torch.inf, dim=1, keepdim=True)
            z_norm = torch.linalg.norm(z / E, ord=torch.inf, dim=1, keepdim=True)

            # --- dual:
            ATy_norm = torch.linalg.norm(ATy / D, ord=torch.inf, dim=1, keepdim=True)
            if Q is not None:
                Qx_norm = torch.linalg.norm(Qx / D, ord=torch.inf, dim=1, keepdim=True)
            else:
                Qx_norm = 0.0

            tol_primal_rel_norm = torch.maximum(Ax_norm, z_norm)
            tol_dual_rel_norm = torch.maximum(torch.maximum(ATy_norm, Qx_norm), p_norm)
            tol_primal = eps_abs + eps_rel * tol_primal_rel_norm
            tol_dual = eps_abs + eps_rel * tol_dual_rel_norm
            # --- check for optimality
            is_optimal = torch.logical_and(primal_error < tol_primal, dual_error < tol_dual)
            if torch.all(is_optimal).item():
                break

        # --- check for feasibility:
        if i % check_feasible == 0 or i >= (max_iters-1):
            # --- delta y and delta x
            dy = y - y_prev
            dy_neg = torch.clamp(dy, max=0)#was negative this
            dy_pos = torch.clamp(dy, min=0)
            dx = x - x_prev
            dy_norm = torch.linalg.norm(E * dy, ord=torch.inf, dim=1, keepdim=True)
            dx_norm = torch.linalg.norm(D * dx, ord=torch.inf, dim=1, keepdim=True)

            # --- feasibility thresholds:
            tol_primal_feas = eps_infeas * dy_norm
            tol_dual_feas = eps_infeas * dx_norm

            # --- pirmal infeasibility:
            primal_infeas_1 = torch.linalg.norm(torch.matmul(AT, dy) / D, ord=torch.inf, dim=1, keepdim=True)
            is_primal_infeas_1 = primal_infeas_1 < tol_primal_feas
            if torch.any(is_primal_infeas_1).item():
                if any_lb:
                    lb_dy = (lb_finite * dy_neg).sum(dim=1, keepdim=True)
                else:
                    lb_dy = 0.0
                if any_ub:
                    ub_dy = (ub_finite * dy_pos).sum(dim=1, keepdim=True)
                else:
                    ub_dy = 0.0
                primal_infeas_2 = lb_dy + ub_dy
                is_primal_infeas_2 = primal_infeas_2 < tol_primal_feas
                is_primal_infeas = torch.logical_and(is_primal_infeas_1, is_primal_infeas_2)
            else:
                is_primal_infeas = is_primal_infeas_1

            # --- dual infeasibility:
            if Q is not None:
                dual_infeas_1 = torch.linalg.norm(torch.matmul(Q, dx) / D, ord=torch.inf, dim=1, keepdim=True)
            else:
                dual_infeas_1 = 0.0
            dual_infeas_2 = torch.linalg.norm(p * dx, ord=torch.inf, dim=1, keepdim=True)
            dual_infeas_3 = torch.linalg.norm(torch.matmul(A, dx) / E, ord=torch.inf, dim=1, keepdim=True)

            is_dual_infeas = torch.logical_and(dual_infeas_1 < tol_dual_feas, dual_infeas_2 < tol_dual_feas)
            is_dual_infeas = torch.logical_and(is_dual_infeas, dual_infeas_3 < tol_dual_feas)

            # --- check for primal or dual infeasibility:
            is_infeas = torch.logical_or(torch.logical_or(is_primal_infeas, is_dual_infeas), is_infeas)
            # --- stop if all optimal and/or infeasible
            do_stop = torch.logical_or(is_infeas, is_optimal)
            if torch.all(do_stop).item():
                break

    # --- reverse the scaling:
    iter = i
    x = D * x
    z = z / E
    y = E * y
    y_neg = torch.clamp(y, max=0)
    y_pos = torch.clamp(y, min=0)
    # --- return solution:
    if unroll:
        sol = x
    else:
        is_optimal = is_optimal[:, 0, 0].numpy()
        is_infeas = is_infeas[:, 0, 0].numpy()
        is_primal_infeas = is_primal_infeas[:, 0, 0].numpy()
        is_dual_infeas = is_dual_infeas[:, 0, 0].numpy()
        status = is_optimal.astype(int)
        sol = {"x": x, "z": z, "y": y, "y_pos": y_pos, "y_neg": y_neg,
               "primal_error": primal_error, "dual_error": dual_error, "rho": rho, "iter": i,
               "is_optimal": is_optimal, "status": status, "is_infeas": is_infeas,
               "is_primal_infeas": is_primal_infeas, "is_dual_infeas": is_dual_infeas,
               "ATA": ATA, "LU": LU, "P": P}

    return sol


# --- if scale is true then we can't re-use ATA, LU and P
def scqpth_grad(dl_dz, x, z, y, Q, A, lb, ub, rho=None, ATA=None, LU=None, P=None, eps=1e-6):
    # --- prep:
    dtype = x.dtype
    n_batch = Q.shape[0]
    n_x = A.shape[2]
    n_A = A.shape[1]
    y_neg = torch.clamp(y, max=0)
    y_pos = torch.clamp(y, min=0)
    if rho is None:
        rho = 1.0
    u = y / rho

    # --- create data:
    AT = torch.transpose(A, dim0=1, dim1=2)
    if ATA is None:
        ATA = torch.matmul(AT, A)
    ATA[:, range(n_x), range(n_x)] = ATA.diagonal(dim1=1, dim2=2) + eps

    # --- derivative of the projection operator:
    xt = torch.transpose(x, 1, 2)
    s_x_u = torch.matmul(A, x) + u

    dpi_dx = torch.ones((n_batch, n_A, 1), dtype=dtype)
    dpi_dx[s_x_u > ub] = 0
    dpi_dx[s_x_u < lb] = 0

    # --- dl_dx: chain rule
    dl_dx = torch.matmul(A, torch.linalg.solve(ATA, dl_dz)) * dpi_dx

    # --- cache factorizations is important for efficiency
    if LU is None or P is None:
        if Q is not None:
            M = Q + rho * ATA
        else:
            M = rho * ATA
        LU, P = torch.linalg.lu_factor(M)

    # --- Adjoint with regularization term
    lhs = torch.linalg.lu_solve(LU, P, AT * (2 * dpi_dx.squeeze(2).unsqueeze(1) - 1))

    lhs = -rho * torch.matmul(A, lhs)
    diag = lhs.diagonal(dim1=1, dim2=2) + dpi_dx.squeeze(2)
    lhs[:, range(n_A), range(n_A)] = diag + eps

    d_nu = torch.linalg.solve(torch.transpose(lhs, dim0=1, dim1=2), -dl_dx)

    # --- compute d_x
    d_x = torch.linalg.lu_solve(LU, P, torch.matmul(AT, d_nu))
    d_x_t = torch.transpose(d_x, 1, 2)

    # --- dl_dp:
    dl_dp = d_x

    # --- dl_dQ:
    if Q is not None:
        dl_dQ1 = torch.matmul(0.50 * d_x, xt)
        dl_dQ = dl_dQ1 + torch.transpose(dl_dQ1, 1, 2)
    else:
        dl_dQ = None

    # --- dl_dA:
    Ax_lb = z - lb
    Ax_ub = z - ub
    Adx = torch.matmul(A, d_x)
    lb_active = y_neg < 0
    lb_not_active = torch.logical_and(torch.logical_not(lb_active), Ax_lb > eps)
    ub_active = y_pos > 0
    ub_not_active = torch.logical_and(torch.logical_not(ub_active), Ax_ub > eps)
    dy_neg = torch.zeros(y_neg.shape, dtype=dtype)
    dy_pos = torch.zeros(y_pos.shape, dtype=dtype)
    dy_neg[lb_not_active] = Adx[lb_not_active] / Ax_lb[lb_not_active]
    dy_pos[ub_not_active] = Adx[ub_not_active] / Ax_ub[ub_not_active]

    # --- solve system:
    kkt = torch.matmul(A, -dl_dz - torch.matmul(Q, d_x))
    AAT = torch.matmul(A, AT)
    AAT[:, range(n_A), range(n_A)] = AAT.diagonal(dim1=1, dim2=2) + eps
    dy = torch.linalg.solve(AAT, kkt)
    dy_neg[lb_active] = dy[lb_active] / y_neg[lb_active]
    dy_pos[ub_active] = dy[ub_active] / y_pos[ub_active]

    d_y = y_neg * dy_neg + y_pos * dy_pos
    # --- dl_dA, dl_dlb, dl_dub
    dl_dA = torch.matmul(d_y, xt) + torch.matmul(y, d_x_t)
    dl_dlb = -y_neg * dy_neg
    dl_dub = -y_pos * dy_pos

    # --- return grads:
    grads = (dl_dQ, dl_dp, dl_dA, dl_dlb, dl_dub, None)

    return grads


def scqpth_grad_kkt(dl_dz, x, y, Q, A, lb, ub):
    # --- prep:
    n_batch = x.shape[0]
    n_x = x.shape[1]
    n_A = A.shape[1]
    n_G = 2 * n_A

    # --- compute lams:
    lams_neg = torch.threshold(-y, 0, 0)
    lams_pos = torch.threshold(y, 0, 0)
    lams = torch.cat((lams_neg, lams_pos), 1)

    # --- make h and G
    G = torch.cat((-A, A), dim=1)
    h = torch.cat((-lb, ub), dim=1)
    slacks = h - torch.matmul(G, x)
    slacks = torch.clamp(slacks, 10 ** -12)
    lams = torch.clamp(lams, 10 ** -12)

    # --- make inversion matrix:
    lhs_1 = torch.cat((Q, torch.transpose(G, 1, 2) * torch.transpose(lams, 1, 2)), 2)
    lhs_2 = torch.cat((G, torch.diag_embed(-slacks.squeeze(2))), 2)
    lhs = torch.cat((lhs_1, lhs_2), 1)

    # --- Compute differentials:
    rhs = torch.cat((-dl_dz, torch.zeros(n_batch, 2 * n_A, 1)), dim=1)
    back_sol = torch.linalg.solve(lhs, rhs)

    # --- unpack solution:
    dx = back_sol[:, :n_x, :]
    dlam = back_sol[:, -n_G:, :]

    # --- compute gradients
    xt = torch.transpose(x, 1, 2)

    # --- dl_dp
    dl_dp = dx

    # --- dl_dQ
    # dl_dQ = 0.5 * (torch.matmul(dx, xt) + torch.matmul(x, dxt))
    dl_dQ1 = torch.matmul(0.50 * dx, xt)
    dl_dQ = dl_dQ1 + torch.transpose(dl_dQ1, 1, 2)

    # --- inequality
    dl_dG = torch.matmul(lams * dlam, xt) + torch.matmul(lams, torch.transpose(dx, 1, 2))
    dl_dA = -dl_dG[:, :n_A, :] + dl_dG[:, -n_A:, :]
    dl_dh = -lams * dlam
    dl_dlb = -dl_dh[:, :n_A, :]
    dl_dub = dl_dh[:, -n_A:, :]

    return dl_dQ, dl_dp, dl_dA, dl_dlb, dl_dub
