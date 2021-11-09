#%%
import numpy as np
import matplotlib.pyplot as plt

def spu(x):
    return np.square(x) - 0.5 if x >= 0 else np.exp(-x)/(np.exp(-x) + 1) - 1

def dx_spu(x):
    return 2*x if x >= 0 else -np.exp(x)/np.square(np.exp(x) + 1)

def get_line_from_two_points(x1, y1, x2, y2):
    slope = (y2 - y1)/(x2 - x1)
    intercept = y1 + (-x1)*slope
    return (slope, intercept)

def compute_spu_bounds(l, u, p_l):
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


#%%
l = -5
u = 5
x = np.linspace(l, u, 100)
f = np.vectorize(spu)(x)
p_l = 1

(lb_slope, lb_intercept), (ub_slope, ub_intercept) = compute_spu_bounds(l,u, p_l)

lb = lb_slope*x + lb_intercept
ub = ub_slope*x + ub_intercept

plt.figure()
plt.plot(x, f)
plt.plot(x, lb)
plt.plot(x, ub)
plt.show()


# %%
