import torch
from networks import FullyConnected
import torch.nn as nn

from transformer import NetworkTransformer
from utils import get_input_bounds
from torchvision import datasets
from torchvision.transforms import ToTensor
import random
import torchattacks
from verifier import analyze

def get_net(net_name):
    if net_name.endswith('fc1'):
        fc_layers = [50, 10]
        net = FullyConnected(DEVICE, INPUT_SIZE, fc_layers).to(DEVICE)
    elif net_name.endswith('fc2'):
        fc_layers = [100, 50, 10]
        net = FullyConnected(DEVICE, INPUT_SIZE, fc_layers).to(DEVICE)
    elif net_name.endswith('fc3'):
        fc_layers = [100, 100, 10]
        net = FullyConnected(DEVICE, INPUT_SIZE, fc_layers).to(DEVICE)
    elif net_name.endswith('fc4'):
        fc_layers = [100, 100, 50, 10]
        net = FullyConnected(DEVICE, INPUT_SIZE, fc_layers).to(DEVICE)
    elif net_name.endswith('fc5'):
        fc_layers = [100, 100, 100, 100, 10]
        net = FullyConnected(DEVICE, INPUT_SIZE, fc_layers).to(DEVICE)
    else:
        assert False
    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % net_name, map_location=torch.device(DEVICE)))
    return net, fc_layers

def get_sample(batch_size=1):
    mnist_data = datasets.MNIST(root = '/tmp/mnist/data', train = False, transform = ToTensor(), download = True)
    mnist_loader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size , shuffle=True)
    images, true_labels = iter(mnist_loader).next()
    return images, true_labels

def attack(eps, net, images, true_labels):
    atk = torchattacks.PGD(net, eps=eps, alpha=2/255, steps=100, random_start=True)
    #atk = torchattacks.FGSM(net, eps=eps)
    #atk = torchattacks.CW(net, c=1, lr=0.01, steps=100, kappa=0)
    adv_images = atk(images, true_labels)
    outs =  net(adv_images)
    pred_label = torch.max(outs, 1)[1].tolist()
    return adv_images, pred_label

DEVICE = 'cpu'
INPUT_SIZE = 28
HEURISTICS = ['x','midpoint']
net_names = ['net0_fc1','net0_fc2','net0_fc3','net0_fc4','net0_fc5','net1_fc1','net1_fc2','net1_fc3','net1_fc4','net1_fc5']
eps = 0.07
batch_size=20 # number of adversarial examples we want to check

not_ver = 0
nvbv = 0
for net_name in net_names:
    net, fc_layers = get_net(net_name)
    images, true_labels = get_sample(batch_size=batch_size)
    adv_images, pred_label = attack(eps, net, images, true_labels)
    #print(pred_label, true_labels.tolist())
    #assert pred_label == true_labels.tolist()

    ## Pull out the indices that assert.
    asserted_indices = list(i[0] == i[1] for i in zip(pred_label, true_labels.tolist()))
    #print(asserted_indices)
    print(f"{sum(asserted_indices)} asserted labels.")

    for each_image, each_label, assertion in zip(adv_images, true_labels, asserted_indices):
        if analyze(net, each_image, eps, each_label, fc_layers):
            if not assertion : #If adversarial gave a different output
                print('verified unsound')
                not_ver+=1
            else:
                print('verified')
            
        else:
            if not assertion :
                print('not verified')
            else:
                print("not verified but verified")
                nvbv +=1
print(not_ver)
print(nvbv)
