from numpy.lib.function_base import append
import torch
import torch.nn as nn
import numpy as np

from networks import FullyConnected, Normalization
from networks import SPU

from utils import get_line_from_two_points, spu, dx_spu


class spuLayerTransformer(nn.Module):
    def __init__(self, layer, heuristic):
        super(spuLayerTransformer, self).__init__()
        self.heuristic = heuristic

    def forward(self, x, l_in, u_in):
        '''
        l, u, x: (1, d_in)
        '''
        dim = x.shape[1]
        x_out = SPU().forward(x)
        if self.heuristic == 'x':
            # choose p_l as x
            p_l = u_in - ((abs(l_in) + abs(u_in))/2)
        else:
            pass

        l_out = torch.zeros(x.shape)
        u_out = torch.zeros(x.shape)
        for i in range(dim):
            # TODO: Implementation needs to be changed to torch tensors, not floats
            if self.heuristic == 'box':
                lo, uo = self._1D_box_bounds(float(l_in[0, i]), float(u_in[0, i]))
            else:
                lo, uo = self._1D_bounds(float(l_in[0, i]), float(u_in[0, i]), float(p_l[0, i]))
            l_out[0, i] = lo
            u_out[0, i] = uo
    

        return x_out, l_out, u_out

    @staticmethod
    def _1D_box_bounds(l: float, u: float):
        if l >= 0:
            l = spu(l)
            u = spu(u)
        elif u<= 0:
            l = spu(u)
            u = spu(l)
        else:
            u = spu(u)
            l = -0.5
        return l,u

    @staticmethod
    def _1D_bounds(l: float, u: float, p_l: float):
        #p_l = torch.clamp(p_l, min=l, max=u)
        p_l = np.clip(p_l, a_min=l, a_max=u)
        if u > 0:
            # ub is fixed
            (ub_slope, ub_intercept) = get_line_from_two_points(l, spu(l), u, spu(u))

            if p_l >= 0:
                # lb is chosen as tangent at p_l
                lb_slope = dx_spu(p_l)
                lb_intercept = spu(p_l) + (-p_l)*lb_slope
            else:
                # lb is chosen as tight for x<0 
                (lb_slope, lb_intercept) = get_line_from_two_points(l, spu(l), 0, spu(0))
        elif u <= 0:
            # now ub based on tangent and lb fixed
            (lb_slope, lb_intercept) = get_line_from_two_points(l, spu(l), u, spu(u))
            ub_slope = dx_spu(p_l)
            ub_intercept = spu(p_l) + (-p_l)*ub_slope

        # coming up with the linear bounds doe not make sense if only scalar bounds are computed - box is better
        l_out = lb_intercept + (lb_slope > 0)*(l * lb_slope) + (lb_slope <= 0)*(u * lb_slope)
        u_out = ub_intercept + (ub_slope > 0)*(u * ub_slope) + (ub_slope <= 0)*(l * ub_slope)
     
        return l_out, u_out


class linearLayerTransformer(nn.Module):
    def __init__(self, layer: nn.Linear):
        super(linearLayerTransformer, self).__init__()
        self.weights = layer.weight
        self.d_in = self.weights.shape[1]
        self.d_out = self.weights.shape[0]
        self.bias = torch.reshape(layer.bias, (self.d_out,1))

    def forward(self, x, l_in, u_in):
        '''
        l, u, x: (1, d_in)
        '''
        w_p = torch.clamp(self.weights, min=0)
        w_m = torch.clamp(self.weights, max=0)
        u_out = torch.mm(w_p, u_in.t()) + torch.mm(w_m, l_in.t()) + self.bias
        l_out = torch.mm(w_p, l_in.t()) + torch.mm(w_m, u_in.t()) + self.bias
        x_out = torch.mm(self.weights, x.t()) + self.bias
        return x_out.t(), l_out.t(), u_out.t()


class NetworkTransformer(nn.Module):

    def __init__(self, net: FullyConnected, heuristic='x'):
        super(NetworkTransformer, self).__init__()
        
        self.net = net
        self.layers = []

        for layer in net.layers:
            if isinstance(layer, Normalization) or isinstance(layer, nn.Flatten):
                pass
            elif isinstance(layer, nn.Linear):
                self.layers.append(linearLayerTransformer(layer))
            elif isinstance(layer, SPU):
                self.layers.append(spuLayerTransformer(layer, heuristic))
            else:
                raise Exception("Layer type not recognized: " + str(type(layer)))
        
    
    def forward_pass(self, x, l, u):

        x = self.net.layers[0].forward(x)
        x = self.net.layers[1].forward(x)   
        l = self.net.layers[0].forward(l)
        l = self.net.layers[1].forward(l)   
        u = self.net.layers[0].forward(u)
        u = self.net.layers[1].forward(u)   
        for la in self.layers:
            x, l, u = la.forward(x, l, u)
        return x, l, u