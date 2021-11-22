from numpy.lib.function_base import append
import torch
import torch.nn as nn
import numpy as np

from networks import FullyConnected, Normalization
from networks import SPU

from utils import get_line_from_two_points, spu, dx_spu


class spuLayerTransformer(nn.Module):
    def __init__(self, dim, heuristic):
        super(spuLayerTransformer, self).__init__()
        self.heuristic = heuristic
        self.dim = dim
        # TODO: Not very safe, should be -+ Inf
        self.lb_slope = torch.zeros((1, dim))
        self.lb_intercept = torch.zeros((1, dim))
        self.ub_slope = torch.zeros((1, dim))
        self.ub_intercept = torch.zeros((1, dim))

# TODO: maybe decouple bounds computation from forward pass?
    def forward(self, x, l_in, u_in):
        '''
        l, u, x: (1, d_in)
        '''
        x_out = SPU().forward(x)
        if self.heuristic == 'x':
            # choose p_l as x
            p_l = x
        else:
            pass

        l_out = torch.zeros(x.shape)
        u_out = torch.zeros(x.shape)
        for i in range(self.dim):
            # TODO: Implementation needs to be changed to torch tensors, not floats     
            _,__ = self._compute_linear_bounds_1D(i, float(l_in[0, i]), float(u_in[0, i]), float(p_l[0, i]))
            lo, uo = self._1D_box_bounds(float(l_in[0, i]), float(u_in[0, i]))
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

    def _compute_linear_bounds_1D(self, idx: int, l: float, u: float, p_l: float):
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

        # safe linear bounds for backsubstitution
        self.lb_slope[0, idx] = lb_slope 
        self.lb_intercept[0, idx] = lb_intercept
        self.ub_slope[0, idx] = ub_slope 
        self.ub_intercept[0, idx] = ub_intercept
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

    def __init__(self, net: FullyConnected, layer_dim, heuristic='x'):
        super(NetworkTransformer, self).__init__()
        
        self.net = net
        self.layers = []

        for i, layer in enumerate(net.layers):
            if isinstance(layer, Normalization) or isinstance(layer, nn.Flatten):
                pass
            elif isinstance(layer, nn.Linear):
                self.layers.append(linearLayerTransformer(layer))
            elif isinstance(layer, SPU):
                self.layers.append(spuLayerTransformer(layer, layer_dim[i], heuristic))
            else:
                raise Exception("Layer type not recognized: " + str(type(layer)))
        
    def forward_pass(self, x, l, u):

        x = self._apply_initial_layers(x)  
        l = self._apply_initial_layers(l)  
        u = self._apply_initial_layers(u)  
        for la in self.layers:
            x, l, u = la.forward(x, l, u)
        return x, l, u

    def _apply_initial_layers(self, x):
        x = self.net.layers[0].forward(x)
        x = self.net.layers[1].forward(x)  
        return x

    def backsubstitution(self, from_layer: int, to_layer: int):
        '''
        compute new linear bounds for from_layer, by substituting linear bounds from previous layers 
        up until to_layer
        '''
        if not isinstance(self.layers[from_layer], SPU):
            raise Exception('Backsubstitution has to be strated from SPU layer.')

        if to_layer < 0:
            raise Exception('Backsubstitution beyond network input is not possible.')

        
