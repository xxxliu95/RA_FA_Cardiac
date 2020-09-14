import torch.nn as nn
import torch.autograd

class RoundingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        output = x.round()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors

        return grad_output


class RoundLayer(nn.Module):
    def __init__(self):
        super(RoundLayer, self).__init__()


    def forward(self, x):
        return RoundingFunction.apply(x)
