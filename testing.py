import numpy as np
import torch

x = torch.rand(20)
print(x)
val, ind = torch.topk(x.abs(),10,dim=0)
val = x[ind]
x *= 0
x[ind] = val
print(x)
