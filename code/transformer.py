from numpy.lib.function_base import append
import torch
from torch._C import EnumType
import torch.nn as nn
import numpy as np
from typing import List

from networks import FullyConnected, Normalization
from networks import SPU

from utils import get_line_from_two_points, spu, dx_spu

INPUT_SIZE = 28
HEURISTICS = ['x', '0', 'midpoint']

def get_pl(x: float, l: float, u: float, heuristic: str):
    if heuristic == 'x':
        p_l = x
    elif heuristic == '0':
        p_l = 0
    elif heuristic == '0_mid':
        if u < 0 or l > 0:
            p_l = l + (u - l)/2
        else:
            p_l = 0
    elif heuristic == 'midpoint':
        p_l = l + (u - l)/2

    return p_l

class LinearDummy(nn.Module):
    def __init__(self, weights, bias):
        self.weight = weights
        self.bias = bias


class spuLayerTransformer(nn.Module):
    def __init__(self, dim: int):
        super(spuLayerTransformer, self).__init__()
        self.dim = dim
        self.lb_in = torch.zeros((dim,1))
        self.ub_in = torch.zeros((dim,1))
        self.heuristics = []

# TODO: maybe decouple bounds computation from forward pass?
    def forward(self, x, l_in, u_in):
        '''
        l, u, x: (1, d_in)
        '''
        x_out = SPU().forward(x)
        self.lb_in = l_in.t()
        self.ub_in = u_in.t()
        self.x = x.t()
        l_out = torch.zeros(x.shape)
        u_out = torch.zeros(x.shape)
        span_heur = torch.zeros(len(HEURISTICS))
        for i in range(self.dim):
            l_out[0, i] , u_out[0, i]  = self._1D_box_bounds(l_in[0, i].item(), u_in[0, i].item())
            
            for k, heur in enumerate(HEURISTICS):
                p_l = get_pl(x[0, i].item(), l_in[0, i].item(), u_in[0, i].item(), heur)
                l_h, u_h  = self._compute_bounds_1D(l_in[0, i].item(), u_in[0, i].item(), p_l)
                span_heur[k] = u_h - l_h
            min_span = torch.min(span_heur)
            min_heur = HEURISTICS[(span_heur == min_span).nonzero(as_tuple=True)[0][0]]
            self.heuristics.append(min_heur)
            

        return x_out, l_out, u_out


    @staticmethod
    def _1D_box_bounds(l: float, u: float):
        assert(l<=u)
        if l >= 0:
            l_out = spu(l)
            u_out = spu(u)
        elif u<= 0:
            l_out = spu(u)
            u_out = spu(l)
        else:
            u_out = max(spu(u),spu(l))
            l_out = -0.5
        return l_out,u_out

    
    def compute_linear_bounds(self, l_in: torch.Tensor, u_in: torch.Tensor, heuristics: List[str]):
        '''
        input shape: (n_dim, 1)
        '''
        slope_lb = torch.zeros_like(l_in)
        intercept_lb = torch.zeros_like(l_in)
        slope_ub = torch.zeros_like(u_in)
        intercept_ub = torch.zeros_like(u_in)
        n = l_in.shape[0]

        '''if heuristic == 'x':
            p_l = self.x
        elif heuristic == '0':
            p_l = torch.zeros_like(l_in)
        elif heuristic == 'midpoint':
            p_l = (u_in - l_in)/2'''

        # TODO: Is there a way to vectorize this in torch?
        for i in range(n):
            #print(heuristics[i])
            p_l = get_pl(self.x[i,0].item(), l_in[i,0].item(), u_in[i,0].item(), heuristics[i])
            slope_lb[i,0], intercept_lb[i,0], slope_ub[i,0], intercept_ub[i,0] = self._compute_linear_bounds_1D(l_in[i,0].item(), u_in[i,0].item(), p_l)
            
        return torch.diag(slope_lb[:,0]), intercept_lb, torch.diag(slope_ub[:,0]), intercept_ub

    def _compute_bounds_1D(self, l: float, u: float, p_l: float):
        lb_slope, lb_intercept, ub_slope, ub_intercept = self._compute_linear_bounds_1D(l, u, p_l)
        l_out = lb_intercept + (lb_slope > 0)*(l * lb_slope) + (lb_slope <= 0)*(u * lb_slope)
        u_out = ub_intercept + (ub_slope > 0)*(u * ub_slope) + (ub_slope <= 0)*(l * ub_slope)
        return l_out, u_out
        
    def _compute_linear_bounds_1D(self, l: float, u: float, p_l: float):
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
        #self.lb_slope[0, idx] = lb_slope 
        #self.lb_intercept[0, idx] = lb_intercept
        #self.ub_slope[0, idx] = ub_slope 
        #self.ub_intercept[0, idx] = ub_intercept
        # coming up with the linear bounds doe not make sense if only scalar bounds are computed - box is better
        return lb_slope, lb_intercept, ub_slope, ub_intercept


class linearLayerTransformer(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        super(linearLayerTransformer, self).__init__()
        self.weights = weight
        self.d_in = self.weights.shape[1]
        self.d_out = self.weights.shape[0]
        self.bias = torch.reshape(bias, (self.d_out,1))

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


class dummyLayer:
    def __init__(self, dim: int):
        self.lb_in = torch.zeros((dim,1))
        self.ub_in = torch.zeros((dim,1))
        self.dim = dim


class NetworkTransformer(nn.Module):

    def __init__(self, net: FullyConnected, layer_dim: List[int], true_label: int, input_dim=INPUT_SIZE*INPUT_SIZE):
        super(NetworkTransformer, self).__init__()
        
        self.net = net
        self.layers = [dummyLayer(input_dim)]
        self.true_label = true_label
        c = 0
        for layer in net.layers:
            if isinstance(layer, Normalization) or isinstance(layer, nn.Flatten):
                pass
            elif isinstance(layer, nn.Linear) or isinstance(layer, LinearDummy):
                self.layers.append(linearLayerTransformer(layer.weight, layer.bias))
            elif isinstance(layer, SPU):
                self.layers.append(spuLayerTransformer(layer_dim[c]))
                c += 1
            else:
                raise Exception("Layer type not recognized: " + str(type(layer)))
        
        self.layers.append(dummyLayer(layer_dim[-1]))
        self.n_layers = len(self.layers)

    def _apply_initial_layers(self, x):
        x = self.net.layers[0].forward(x)
        x = self.net.layers[1].forward(x)  
        return x
    
    def forward_pass(self, x, l, u):

        x = self._apply_initial_layers(x)  
        l = self._apply_initial_layers(l)  
        u = self._apply_initial_layers(u)  
        self.layers[0].lb_in = l
        self.layers[0].ub_in = u
        for la in self.layers[1:-1]:
            x, l, u = la.forward(x, l, u)
        return x, l, u

    def backsub_pass(self, fix_heuristic=''):
        # for now one pass through whole network
        lb_out, ub_out = self._backsubstitution(self.n_layers-1, 0, fix_heuristic, useSubtract=True)
        return lb_out, ub_out 
    
    def iterative_backsub(self, fix_heuristic=''):

        spu_layers = [i for i in range(self.n_layers - 1) if isinstance(self.layers[i], spuLayerTransformer)]

        for l_id in spu_layers[1:]:
            self._backsubstitution(l_id, 0, fix_heuristic, useSubtract=False)
        
        return self.backsub_pass(fix_heuristic)

    def _backsubstitution(self, from_layer: int, to_layer: int, fix_heuristic: str, useSubtract: bool = True):
        '''
        compute new linear bounds for from_layer, by substituting linear bounds from previous layers 
        up until to_layer
        '''

        layer0 = self.layers[from_layer]
        # TODO: Can start lyer be any layer or only spu?
        if not (isinstance(layer0, spuLayerTransformer) or (from_layer == self.n_layers-1)):
            raise Exception('Backsubstitution has to be started from SPU or output layer.')

        if to_layer < 0:
            raise Exception('Backsubstitution beyond network input is not possible.')

        dim0 = layer0.dim
        if useSubtract:
            W_init = torch.eye(dim0-1, dim0-1)*(-1)
            W_init= torch.cat([W_init[:, 0:self.true_label],
                              torch.ones(dim0-1, 1),
                              W_init[:, self.true_label:dim0]], 1)
            B_init = torch.zeros((dim0 - 1,1))
        else:
            W_init = torch.eye(dim0, dim0)
            B_init = torch.zeros((dim0,1))

        M_LB = W_init.clone()
        B_LB = B_init
        M_UB = W_init.clone()
        B_UB = B_init

        for layer in reversed(self.layers[to_layer:from_layer]): #TODO: Inlcude operation of from_layer?
            if isinstance(layer, linearLayerTransformer):
                w = layer.weights
                b = layer.bias
                B_LB = B_LB + torch.mm(M_LB,b)
                M_LB = torch.mm(M_LB, w)
                B_UB = B_UB + torch.mm(M_UB,b)
                M_UB = torch.mm(M_UB, w)

            elif isinstance(layer, spuLayerTransformer):
                if fix_heuristic == '':
                    heur = layer.heuristics
                else :
                    heur = [fix_heuristic for _ in range(layer.dim)]
                lb_slope, lb_intercept, ub_slope, ub_intercept = layer.compute_linear_bounds(layer.lb_in, layer.ub_in, heur)
                M_LB, B_LB = self._compute_spu_backsub_params(M_LB, B_LB, lb_slope, lb_intercept, ub_slope, ub_intercept)
                M_UB, B_UB = self._compute_spu_backsub_params(M_UB, B_UB, ub_slope, ub_intercept, lb_slope, lb_intercept)
                
        backsub_layer_lb = linearLayerTransformer(M_LB, B_LB)
        backsub_layer_ub = linearLayerTransformer(M_UB, B_UB)

        # TODO: Only allow to be spu layer?
        # TODO: Pass ro_layer bounds as input?
        l_in = self.layers[to_layer].lb_in
        u_in = self.layers[to_layer].ub_in
        x_dummy = torch.zeros_like(l_in)

        _, lb0, __ = backsub_layer_lb.forward(x_dummy, l_in, u_in)
        _, __, ub0 = backsub_layer_ub.forward(x_dummy, l_in, u_in)

        # TODO: Decide whether to compute input or output bounds for start_layer
        chg_count = 0
        if not useSubtract:
            for k in range(self.layers[from_layer].lb_in.shape[0]):
                if lb0[0, k] > self.layers[from_layer].lb_in[k,0]:
                    self.layers[from_layer].lb_in[k,0] = lb0[0, k]
                    chg_count +=1
            for k in range(self.layers[from_layer].ub_in.shape[0]):        
                if ub0[0, k] < self.layers[from_layer].ub_in[k,0]:
                    self.layers[from_layer].ub_in[k,0] = ub0[0, k]
                    chg_count +=1
            #print('Changed ' + str(chg_count) + ' bounds in layer ' + str(from_layer))
        else:
            return lb0, ub0



    @staticmethod
    def _compute_spu_backsub_params(M, B, lb_slope, lb_intercept, ub_slope, ub_intercept):
        '''Propagates the linear bounds through weight matrix M and bias B

        dimensions:
         lb_slope, ub_slope: (n_dim, n_dim) <- diagonal
         lb_intercept, ub_intercept: (n_dim, 1)
         M: (x, n_dim)
         B: (x, 1)
        '''
        M_pos = torch.clamp(M, min = 0)
        M_neg = torch.clamp(M, max = 0)
        # lb and ub will be switched for computing M_u, B_u
        M_l = torch.mm(M_pos, lb_slope) + torch.mm(M_neg, ub_slope)
        B_l = B + torch.mm(M_pos, lb_intercept) + torch.mm(M_neg, ub_intercept)

        return M_l, B_l


            

