def scqpth_control(max_iters=10_000, eps_abs=1e-3, eps_rel=1e-3, eps_infeas=1e-4, check_solved=25, check_feasible=25,
                   alpha=1.2, alpha_iter=100, rho=None, rho_min=1e-6, rho_max=1e6, adaptive_rho=True, adaptive_rho_tol=10,
                   adaptive_rho_iter=50, adaptive_rho_max_iter=1000, sigma=0.0, verbose=False,
                   scale=True, beta=None, warm_start=False, eps=1e-6, unroll=False, **kwargs):
    control = {"max_iters": max_iters,
               "eps_abs": eps_abs,
               "eps_rel": eps_rel,
               "eps_infeas": eps_infeas,
               "check_solved": check_solved,
               "check_feasible": check_feasible,
               "alpha": alpha,
               "alpha_iter": alpha_iter,
               "rho": rho,
               "rho_min": rho_min,
               "rho_max": rho_max,
               "adaptive_rho": adaptive_rho,
               "adaptive_rho_tol": adaptive_rho_tol,
               "adaptive_rho_iter": adaptive_rho_iter,
               "adaptive_rho_max_iter": adaptive_rho_max_iter,
               "sigma": sigma,
               "verbose": verbose,
               "scale": scale,
               "beta": beta,
               "warm_start": warm_start,
               "unroll": unroll,
               "eps": eps
               }
    control.update(**kwargs)
    return control
