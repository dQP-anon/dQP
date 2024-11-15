# import dQP and a tool to load example differentiable parameters
from src import dQP
from src.sparse_helper import initialize_torch_from_npz

# initialize dQP and parameters (assumes Gurobi and QDLDL installed)
settings = dQP.build_settings(solve_type="sparse",qp_solver="gurobi",lin_solver="qdldl")
dQP_layer = dQP.dQP_layer(settings=settings)
P,q,C,d,A,b = initialize_torch_from_npz("experiments/diagnostic/data/cross.npz")

# == solve QP ==
z_star,lambda_star,mu_star,_,_ = dQP_layer(P,q,C,d,A,b)

# == form a scalar loss and differentiate ==
z_star.sum().backward()

print(z_star) # optimal point $$z^*$$
print(d.grad) # gradient (w.r.t. d)