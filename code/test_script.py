#%%
import argparse
from pickle import NEWFALSE
from numpy.core.records import get_remaining_size
import torch
from networks import FullyConnected
from transformer import NetworkTransformer
#from utils import get_input_bounds

def get_input_bounds(input, eps, l=0, u=1):
    lb = torch.clamp(input - eps, min = l)
    ub = torch.clamp(input + eps, max = u)
    return lb, ub


DEVICE = 'cpu'
INPUT_SIZE = 28

net_name = 'net0_fc1'
example = 'example_img0_0.01800'
filename = 'test_cases/' + net_name +'/' + example + '.txt'
with open(filename, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(example.split('_')[-1])


if net_name.endswith('fc1'):
    fc_layers = [50, 10]
    net = FullyConnected(DEVICE, INPUT_SIZE, fc_layers).to(DEVICE)

net.load_state_dict(torch.load('mnist_nets/' + net_name + '.pt', map_location=torch.device(DEVICE)))

inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
outs = net(inputs)
pred_label = outs.max(dim=1)[1].item()
assert pred_label == true_label

lb, ub = get_input_bounds(inputs, eps)

nt = NetworkTransformer(net, fc_layers)
x_out0, lb_out0, ub_out0 = nt.forward_pass(inputs, lb, ub)

lb_out1, ub_out1 = nt.backsub_pass()



#%%
span_box = torch.sum(ub_out0 - lb_out0).item()
span_x = torch.sum(ub_out1 - lb_out1).item()

print('Span box: ' + str(span_box))
print('Span x: ' + str(span_x))

verified_box = sum((lb_out0[0,true_label] > ub_out0[0,:])).item()==9
verified_x = sum((lb_out1[0,true_label] > ub_out1[0,:])).item()==9

print('Result box: ' + str(verified_box))
print('Result x: ' + str(verified_x))

# %%
