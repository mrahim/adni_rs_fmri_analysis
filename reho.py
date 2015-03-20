# -*- coding: utf-8 -*-
"""
ReHO : Kendall's W

W = 12 * S / m^2 (n^3 - n)

S = sum_1^n ( R_i - R )^2

R = 1/2 * m * (n + 1)

R_i = sum_j=1^m ( r_i,j )

n time-samples, m voxels


Created on Fri Mar 20 17:24:32 2015

@author: mehdi.rahim@cea.fr
"""

import numpy as np

def ReHo(r):
    """ Returns kendall W on r (time x voxels) n*m
    """

    # n:time, m:voxels
    n, m = r.shape
    
    # Ri: sum of all times (gives m voxels vector)
    Ri = np.sum(r, axis=0)
    
    # R: mean value
    R = .5 * m * (n + 1)
    
    # sum of squared deviation
    S = np.sum((Ri - R)**2)
    
    # kendall's W
    W = 12 * S / m**2 * (n**3 - n)
    
    return W