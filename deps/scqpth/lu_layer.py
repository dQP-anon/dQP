import torch
import torch.nn as nn


class TorchLU(nn.Module):
    def __init__(self, A=None, LU=None, P=None):
        super().__init__()
        if LU is None or P is None:
            with torch.no_grad():
                LU, P = torch.linalg.lu_factor(A)
        self.LU = LU
        self.P = P

    def forward(self, A, b):
        x = TorchLULayer.apply(A, b, self.LU, self.P)
        return x


class TorchLULayer(torch.autograd.Function):
    """
    Autograd function for forward solving and backward LU solver
    """

    @staticmethod
    def forward(ctx, A, b, LU=None, P=None):
        """
        Forward solving
        """
        if LU is None or P is None:
            with torch.no_grad():
                LU, P = torch.linalg.lu_factor(A)


        x = torch.linalg.lu_solve(LU, P, b)

        # --- save for backwards:
        ctx.save_for_backward(LU, P, x)

        return x

    @staticmethod
    def backward(ctx, dl_dx):
        """
        Fixed point backward differentiation. Note this only works for symmetric A.
        """
        LU, P, x = ctx.saved_tensors

        # --- backward solve:
        if len(x.shape) < 3:
            xt = x.transpose
        else:
            xt = torch.transpose(x, 1, 2)
        dx = torch.linalg.lu_solve(LU, P, -dl_dx)
        dl_dA = torch.matmul(dx,xt)
        dl_db = -dx

        grads = (dl_dA, dl_db, None, None)

        return grads