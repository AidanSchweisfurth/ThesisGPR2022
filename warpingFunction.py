from cmath import inf
from tokenize import Double
import GPy
import numpy as np
from statistics import NormalDist
from scipy import stats


class WarpingFunction(GPy.util.input_warping_functions.InputWarpingFunction):
    def __init__(self, name="Warp_Probit"):
        super(WarpingFunction, self).__init__(name)

    def f(self, y, psi=None):
        #print("----------------------")
        #print(y)
        z = y.flatten().astype(float).tolist()
        for i in range(len(z)):
            if (z[i] == 0):
                z[i] = 0.000001
            elif (z[i] == 1):
                z[i] = 0.999999
        
        #print(z)
        t = stats.norm.ppf(z).reshape((-1, 1))
        #print(t)
        return t


    def fgrad_y(self, y, psi=None):
        #It is the reciprocal of the pdf composed with the quantile function
        C = y.astype(float)
        return 1/stats.norm._pdf(C)

    def update_grads(self, Y_untransformed, kiy):
        pass
        #self.d.gradient = self.fgrad_y(Y_untransformed)
        # grad_y_psi, grad_psi = self.fgrad_y_psi(Y_untransformed,
        #                                         return_covar_chain=True)
        # djac_dpsi = ((1.0 / grad_y[:, :, None, None]) * grad_y_psi).sum(axis=0).sum(axis=0)
        # dquad_dpsi = (Kiy[:, None, None, None] * grad_psi).sum(axis=0).sum(axis=0)

        # warping_grads = -dquad_dpsi + djac_dpsi

        # self.psi.gradient[:] = warping_grads[:, :-1]
        # self.d.gradient[:] = warping_grads[0, -1]
       
    def f_inv(self, z, max_iterations=250, y=None):
        c = z.astype(float)
        return stats.norm._cdf(c)