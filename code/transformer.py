import torch
import torch.nn as nn

from networks import FullyConnected
from networks import SPU

from utils import get_line_from_two_points, spu, dx_spu


class spuLayerTransformer(nn.Module):
    def __init__(self, layer):
        super(spuLayerTransformer, self).__init__()

    def forward_pass(self, x, l_in, u_in, heuristic: str = 'x'):
        '''
        l, u, x: (d_in,1)
        '''
        dim = x.shape[0]
        x_out = SPU(x)
        if heuristic == 'x':
            # choose p_l as x
            p_l = x
        else:
            pass

        l_out, u_out = torch.zeros(x.shape)
        for i in range(dim):
            (lb_slope, lb_intercept), (ub_slope, ub_intercept) = self._1D_bounds(l_in[i], u_in[i], p_l[i])
            l_out[i] = lb_intercept + (lb_slope > 0)*(l_in[i] * (-lb_slope)) + (lb_slope <= 0)*(u_in[i] * lb_slope)
            u_out[i] = ub_intercept + (ub_slope > 0)*(u_in[i] * ub_slope) + (ub_slope <= 0)*(l_in[i] * (-lb_slope))
        
        return x_out, l_out, u_out

    @staticmethod
    def _1D_bounds(l: float, u: float, p_l: float):
        p_l = torch.clamp(p_l, a_min=l, a_max=u)
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

        return (lb_slope, lb_intercept), (ub_slope, ub_intercept)


class linearLayerTransformer(nn.Module):
    def __init__(self, layer: nn.Linear):
        super(linearLayerTransformer, self).__init__()
        self.weights = layer.weight
        self.bias = layer.bias
        self.d_in = self.weights.shape[1]
        self.d_out = self.weights.shape[0]

    def forward_pass(self, x, l_in, u_in):
        '''
        l, u, x: (d_in,1)
        '''
        w_p = torch.clamp(self.weights, min=0)
        w_m = torch.clamp(self.weights, max=0)
        u_out = torch.multiply(w_p, u_in) + torch.multiply(w_m, l_in) + self.bias
        l_out = torch.multiply(w_p, l_in) + torch.multiply(w_m, u_in) + self.bias
        x_out = torch.multiply(self.weights, x) + self.bias
        return x_out, l_out, u_out


class NetworkTransformer(nn.Module):

    def __init__(self, net: FullyConnected):
        super(NetworkTransformer, self).__init__()
        
        self.net = net
        self.layers = []
        # TODO: Instantiate abstract layers
        

    
    def forward_pass(self, x, l, u):
        # TODO: apply normalization and flatten
        for layer in self.layers:
            x, l, u = layer.forward_pass(x, l, u)

        return x, l, u