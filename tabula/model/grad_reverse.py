import torch
from torch.autograd import Function


class GradReverse(Function):
    '''
    Domain adaptation loss for batch correction.
    Sourced from: https://github.com/bowang-lab/scGPT/blob/main/scgpt/model/grad_reverse.py
    '''

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambd)
