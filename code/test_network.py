
#%%
import torch
from networks import FullyConnected
import torch.nn as nn
from networks import SPU
from transformer import NetworkTransformer, LinearDummy

DEVICE = 'cpu'
INPUT_SIZE = 2
OUTPUT_SIZE = 1

fc_layers = [2, 2]
net = FullyConnected(DEVICE, INPUT_SIZE, fc_layers).to(DEVICE)


weights = torch.Tensor([[1, 1], [-1, 1]]).reshape(2,2)
bias = torch.Tensor([0, 0])
linear_layer0 = LinearDummy(weights, bias)

weights = torch.Tensor([[1, 1], [-1, 1]]).reshape(2,2)
bias = torch.Tensor([0, 0])
linear_layer1 = LinearDummy(weights, bias)

weights = torch.Tensor([[1, 1], [-1, 1]]).reshape(2,2)
bias = torch.Tensor([0, 0])
linear_layer2 = LinearDummy(weights, bias)


net.layers = nn.Sequential(*[linear_layer0, SPU(), linear_layer1])

nt = NetworkTransformer(net, fc_layers)

inputs = torch.Tensor([0.5,0.5]).reshape(1,2)
lb = torch.Tensor([-1, -1]).reshape(1,2)
ub = torch.Tensor([1,1]).reshape(1,2)
x_out0, lb_out0, ub_out0 = nt.forward_pass(inputs, lb, ub)
lb_out1, ub_out1 = nt.backsub_pass(heuristic='0')
tmp = 0

# %%
