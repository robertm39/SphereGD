# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 15:15:06 2021

@author: rober
"""

import numpy as np

import sphere_gd

def sphere_gd_test_1():
    dims = 100
    
    # Start at (-1, 0, 0)
    x = np.array([-1.0] + [0.0] * (dims-1))
    
    # The function just adds the coordinates of x
    # so this is always the gradient
    grad = np.array([-1.0] * dims)
    
    alpha = 0.1
    
    # One-tenth of a right angle
    max_step = np.pi / 20
    
    num_steps = 50
    
    for i in range(1, num_steps+1):
        
        x = sphere_gd.spherical_gd_step(x, grad, alpha, max_step)
        
        if i % 10 == 0:
            gain = x[0] + x[1] + x[2]
            
            # print('x: {}'.format(x))
            print('gain: {:.5f}'.format(gain))
            print('')
    
    gain = x[0] + x[1] + x[2]
    
    print('x: {}'.format(x))
    print('gain: {:.5f}'.format(gain))

def main():
    sphere_gd_test_1()

if __name__ == '__main__':
    main()