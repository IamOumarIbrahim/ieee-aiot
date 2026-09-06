# safe_linear.py - Workaround for Windows NVIDIA cuBLAS Sgemm internal errors on Ada Lovelace
import torch
import torch.nn.functional as F

def safe_mm(a, b):
    if a.is_cuda and a.dim() == 2 and b.dim() == 2:
        inp = a.t().unsqueeze(0)
        w = b.t().unsqueeze(-1)
        return F.conv1d(inp, w).squeeze(0).t()
    return torch.mm(a, b)

class SafeLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight)
        out = torch._C._nn.linear(input, weight, bias)
        return out.clone()

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if weight.dim() == 1:
            if len(ctx.needs_input_grad) > 0 and ctx.needs_input_grad[0]:
                grad_input = grad_output.unsqueeze(-1) * weight
            if len(ctx.needs_input_grad) > 1 and ctx.needs_input_grad[1]:
                in_2d = input.reshape(-1, weight.shape[0])
                g_2d = grad_output.reshape(-1, 1)
                grad_weight = (g_2d * in_2d).sum(dim=0)
            if len(ctx.needs_input_grad) > 2 and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum()
        else:
            in_features = weight.shape[1]
            out_features = weight.shape[0]
            input_2d = input.reshape(-1, in_features)
            grad_output_2d = grad_output.reshape(-1, out_features)

            if len(ctx.needs_input_grad) > 0 and ctx.needs_input_grad[0]:
                grad_input_2d = safe_mm(grad_output_2d, weight)
                grad_input = grad_input_2d.reshape(input.shape)

            if len(ctx.needs_input_grad) > 1 and ctx.needs_input_grad[1]:
                grad_weight = safe_mm(grad_output_2d.t().contiguous(), input_2d)

            if len(ctx.needs_input_grad) > 2 and ctx.needs_input_grad[2]:
                grad_bias = grad_output_2d.sum(dim=0)

        if len(ctx.needs_input_grad) == 2:
            return grad_input, grad_weight
        return grad_input, grad_weight, grad_bias

def apply_safe_linear_patch():
    F.linear = SafeLinearFunction.apply
