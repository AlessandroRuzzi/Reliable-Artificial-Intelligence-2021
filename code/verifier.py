import argparse
import torch
from networks import FullyConnected
from transformer import NetworkTransformer
import time

DEVICE = 'cpu'
INPUT_SIZE = 28

def get_input_bounds(input, eps, l=0, u=1):
    lb = torch.clamp(input - eps, min = l)
    ub = torch.clamp(input + eps, max = u)
    return lb, ub


def analyze(net, inputs, eps, true_label):

    start = time.time()

    lb, ub = get_input_bounds(inputs, eps)
    x_out, lb_out, ub_out = net.forward_pass(inputs, lb, ub)

    end = time.time()
    print("Propagation done. Time : " + str(round(end-start,3)))

    lb_t = lb_out[0,true_label]
    cc = 0
    for k in range(10):
        if k != true_label and lb_t > ub_out[0,k]:
            cc +=1
    verified = cc == 9
    #verified= sum((lb_out[0,true_label] > ub_out[0,:])).item()==9

    return verified


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
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
    elif args.net.endswith('fc2'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
    elif args.net.endswith('fc3'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net.endswith('fc4'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
    elif args.net.endswith('fc5'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
    else:
        assert False

    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))

    abstract_net = NetworkTransformer(net, heuristic='box')

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(abstract_net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
