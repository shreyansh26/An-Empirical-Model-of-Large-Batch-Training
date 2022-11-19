import torch

def get_grad(model):
    parameters = list(model.parameters())
    grads = [param.grad.flatten().view(-1,1) for param in parameters if not type(param.grad) == type(None)]
    grad = torch.cat(grads)
    return grad

def exp_mov_avg(mov_avg, x, beta, step):
    if mov_avg is None: 
        mov_avg = 0
    mov_avg = beta * mov_avg + (1 - beta) * x
    return mov_avg, mov_avg/(1-beta**(step+1))