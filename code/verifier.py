import argparse
import torch
from networks import FullyConnected
import torch.nn as nn

from transformer import NetworkTransformer, INPUT_SIZE
from utils import get_input_bounds

DEVICE = 'cpu'
HEURISTICS = ['0','x','midpoint', '0', 'try']

def analyze(net, inputs, eps, true_label, fc_layers): 

    lb, ub = get_input_bounds(inputs, eps)
    nt = NetworkTransformer(net, fc_layers, true_label=true_label, input_dim=lb.shape[0])
    x_out, lb_out, ub_out = nt.forward_pass(inputs, lb, ub)
    verified = sum((lb_out[0,true_label] > ub_out[0,:])).item()==9
    if verified: return True

    # single backsub, individual p_l
    lb_out0, ub_out0 = nt.backsub_pass()
    verified = torch.all(lb_out0[0,:] > 0).item()
    if verified: return True

    # single backsub, fixed p_l
    for i, heuristic in enumerate(HEURISTICS):
        lb_out1, ub_out1 = nt.backsub_pass(fix_heuristic=heuristic)
        lb_out0 = torch.maximum(lb_out0, lb_out1)
        ub_out0 = torch.minimum(ub_out0, ub_out1)
        verified = torch.all(lb_out0[0,:] > 0).item()
        if verified: return True
        
    # iterative backsub, individual p_l
    lb_out1, ub_out1 = nt.iterative_backsub()
    lb_out0 = torch.maximum(lb_out0, lb_out1)
    ub_out0 = torch.minimum(ub_out0, ub_out1)
    verified = torch.all(lb_out0[0,:] > 0).item()
    if verified: return True

    #iterative backsub, fixed p_l
    for i, heuristic in enumerate(HEURISTICS):
        lb_out1, ub_out1 = nt.iterative_backsub(fix_heuristic=heuristic)
        lb_out0 = torch.maximum(lb_out0, lb_out1)
        ub_out0 = torch.minimum(ub_out0, ub_out1)
        verified = torch.all(lb_out0[0,:] > 0).item()
        if verified: return True 

    if not verified: return False
    return 0


def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepPoly relaxation')
    parser.add_argument('--net',
                        type=str,
                        required=True,
                        help='Neural network architecture which is supposed to be verified.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    if args.net.endswith('fc1'):
        fc_layers = [50, 10]
        net = FullyConnected(DEVICE, INPUT_SIZE, fc_layers).to(DEVICE)
    elif args.net.endswith('fc2'):
        fc_layers = [100, 50, 10]
        net = FullyConnected(DEVICE, INPUT_SIZE, fc_layers).to(DEVICE)
    elif args.net.endswith('fc3'):
        fc_layers = [100, 100, 10]
        net = FullyConnected(DEVICE, INPUT_SIZE, fc_layers).to(DEVICE)
    elif args.net.endswith('fc4'):
        fc_layers = [100, 100, 50, 10]
        net = FullyConnected(DEVICE, INPUT_SIZE, fc_layers).to(DEVICE)
    elif args.net.endswith('fc5'):
        fc_layers = [100, 100, 100, 100, 10]
        net = FullyConnected(DEVICE, INPUT_SIZE, fc_layers).to(DEVICE)
    else:
        assert False

    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label, fc_layers):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
