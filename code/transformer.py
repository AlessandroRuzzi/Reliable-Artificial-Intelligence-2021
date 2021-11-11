#%%
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def spu(x: float) -> float:
    return np.square(x) - 0.5 if x >= 0 else np.exp(-x)/(np.exp(-x) + 1) - 1

def dx_spu(x: float) -> float:
    return 2*x if x >= 0 else -np.exp(x)/np.square(np.exp(x) + 1)

def get_line_from_two_points(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
    slope = (y2 - y1)/(x2 - x1)
    intercept = y1 + (-x1)*slope
    return (slope, intercept)

def compute_spu_bounds(l: float, u: float, p_l: float) -> Tuple[Tuple[float, float],Tuple[float, float]]:
    ''' computes linear bounds based on linearization point p_l.
     p_l should be chosen by some heuristic
        - minimize area
        - input at original image
        - some other idea?'''
    if l <= 0 and u > 0:
        # ub is fixed
        (ub_slope, ub_intercept) = get_line_from_two_points(l, spu(l), u, spu(u))

        if p_l >= 0:
            # lb is chosen as tangent at p_l
            lb_slope = dx_spu(p_l)
            lb_intercept = spu(p_l) + (-p_l)*lb_slope
        else:
            # lb is chosen as tight for x<0 
            assert(l<0)
            (lb_slope, lb_intercept) = get_line_from_two_points(l, spu(l), 0, spu(0))
    elif u <= 0:
        # now ub based on tangent and lb fixed
        assert(p_l<=0)
        (lb_slope, lb_intercept) = get_line_from_two_points(l, spu(l), u, spu(u))
        ub_slope = dx_spu(p_l)
        ub_intercept = spu(p_l) + (-p_l)*ub_slope

    return (lb_slope, lb_intercept), (ub_slope, ub_intercept)


def compute_linear_bounds(l: np.array, u: np.array, w: np.array) -> Tuple[float, float]:
    '''Computes bounds for 1-D output of linear layer, i.e.
        lb <= w^T*x <= ub for l <= x <= u 
    '''
    l_in = l.reshape(-1,1)
    u_in = u.reshape(-1,1)
    weights = w.reshape(-1,1)
    w_plus = weights.clip(min=0)
    w_minus = weights.clip(max=0)
    ub = np.matmul(np.transpose(w_plus), u_in) + np.matmul(np.transpose(w_minus), l_in)
    lb = np.matmul(np.transpose(w_plus), l_in) + np.matmul(np.transpose(w_minus), u_in)
    return (lb, ub)



#%%
if __name__ == "__main__":

    # bounds for single spu unit
    l = -5
    u = 5
    x = np.linspace(l, u, 100)
    f = np.vectorize(spu)(x)
    p_l = 1

    (lb_slope, lb_intercept), (ub_slope, ub_intercept) = compute_spu_bounds(l,u, p_l)

    lb = lb_slope*x + lb_intercept
    ub = ub_slope*x + ub_intercept

    plt.figure(1)
    plt.plot(x, f)
    plt.plot(x, lb)
    plt.plot(x, ub)
    plt.show()
#%%  bounds for linear layer + spu unit
    # input bounds
    l = np.array([-1, -1])
    u = np.array([1, 1])
    # weights for linear layer
    w = np.array([1, -2]).reshape(2,1)
    # bounds for linear output
    l_1, u_1 = compute_linear_bounds(l, u, w)
    # bounds for spu unit
    (lb_slope, lb_intercept), (ub_slope, ub_intercept) = compute_spu_bounds(l_1,u_1, p_l=0)

    #plot results
    # linear + spu output
    def f(x1, x2):
        x_in = np.matmul(np.transpose(w), np.array([x1, x2]).reshape(2,1))
        return spu(x_in)
    def ub(x1,x2):
        x_in = np.matmul(np.transpose(w), np.array([x1, x2]).reshape(2,1))
        return ub_intercept + ub_slope*x_in 
    def lb(x1,x2):
        x_in = np.matmul(np.transpose(w), np.array([x1, x2]).reshape(2,1))
        return lb_intercept + lb_slope*x_in 

    x0 = np.linspace(l[0], u[0], 25)
    x1 = np.linspace(l[1], u[1], 25)
    X0,X1 = np.meshgrid(x0, x1)
    F = np.vectorize(f)(X0, X1)
    LB = np.vectorize(lb)(X0, X1)
    UB = np.vectorize(ub)(X0, X1)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X0, X1, LB)
    ax.plot_surface(X0, X1, F)
    ax.plot_surface(X0, X1, UB)
    plt.show()
