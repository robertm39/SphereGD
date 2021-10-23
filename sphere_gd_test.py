# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 15:15:06 2021

@author: Robert Morton
"""

import numpy as np

import sphere_gd
import pack_spheres

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

def sphere_pack_test_1():
    # "pack" two circles around a center circle
    # only one circle will be allowed to move
    # they will start out very close
    
    fixed = np.array([1.0, 0.0])
    moveable = np.array([np.cos(0.1), np.sin(0.1)])
    prev_grad = np.zeros([2], dtype=np.float64)
    momentum = 0.9
    
    alpha = 0.1
    max_step = np.pi / 20
    
    num_steps = 100
    
    for i in range(1, num_steps+1):
        grad = pack_spheres.sphere_loss_grad(moveable, [fixed])
        
        # Apply momentum in the base grade space, not the constrained space
        grad = grad + momentum * prev_grad
        prev_grad = grad
        
        moveable = sphere_gd.spherical_gd_step(moveable, grad, alpha, max_step)
        
        if i % 10 == 0:
            dist = np.linalg.norm(moveable - fixed)
            print('sphere: {}'.format(moveable))
            print('distance: {:.3f}'.format(dist))
            print('')
    
    
    dist = np.linalg.norm(moveable - fixed)
    print('sphere: {}'.format(moveable))
    print('distance: {:.3f}'.format(dist))

def main():
    # sphere_gd_test_1()
    sphere_pack_test_1()

if __name__ == '__main__':
    main()