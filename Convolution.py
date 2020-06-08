# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 16:21:35 2020

@author: renli
"""

def convolution_univar(f1,f2):
    import numpy as np
    from scipy.integrate import quad
    def res_fcn(z):
        f=lambda x:f1(x)*f2(z-x)
        return quad(f,-np.inf,np.inf)[0]
    return res_fcn

def self_convolution(f,repeat_time):
    from itertools import repeat
    from functools import reduce
    res_fcn=reduce(convolution_univar,repeat(f,repeat_time))
    return res_fcn

def multi_convolution(f_list):
    from functools import reduce
    res_fcn=reduce(convolution_univar,f_list)
    return res_fcn

if __name__=='__main__':
    from scipy.stats import norm
    f=self_convolution(norm().pdf,10)
    import StructureNote as SN
    res=f(0)
    # res=SN.fplot(f,-1,1,10)