import argparse
from pickle import NEWFALSE
from numpy.core.records import get_remaining_size
import torch
from networks import FullyConnected
from transformer import NetworkTransformer
from utils import get_input_bounds
import numpy as np
from utils import get_line_from_two_points, spu, dx_spu

HEURISTICS = ['0', 'midpoint','x']

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

def compute_linear_bounds_1D(l: float, u: float, p_l: float):
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
        
        return lb_slope, lb_intercept, ub_slope, ub_intercept

def main():
    for heur in HEURISTICS:
        print("Heuristic: " + str(heur))
        for _ in range(100000):
            l,u = np.random.randn(), np.random.randn()
        if l > u:
            l,u = u,l

        for x in np.linspace(l,u,10000):
            p_l = get_pl(x, l, u, heur)
            w_l,b_l,w_u,b_u = compute_linear_bounds_1D(l, u, p_l)
            assert spu(x) > w_l * x + b_l - 1e-5
            assert spu(x) < w_u * x + b_u + 1e-5


if __name__ == '__main__':
    main()