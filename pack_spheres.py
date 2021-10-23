# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 15:40:29 2021

@author: Robert Morton
"""

import numpy as np

import sphere_gd

def sphere_loss_grad(sphere, others, loss_power=1, min_dist=1.0):
    """
    Return the gradient of the loss wrt the first spere's coordinates.
    
    Parameters:
        sphere: numpy array
            The coordinates of the sphere to differentiate with respect to.
        others: list[numpy array]
            The coordinates of the other spheres.
        min_dist: double
            The minimum distance two spheres must have to not overlap.
    """
    
    dimensions = sphere.shape[0]
    grad = np.zeros((dimensions))
    
    # Go through all of the other spheres
    # if they're too close, add an appropriate loss gradient
    
    for other in others:
        # Get the distance between this sphere and the other sphere
        dist = np.linalg.norm(sphere - other)
        
        
        # If they're far enough apart, don't do anything
        if dist >= min_dist:
            continue
        
        # # draw spheres together (?)
        # if dist >= min_dist + 0.01:
        #     continue
        
        # They're too close
        # this derivative will be negative,
        # because increasing the distance decreases the loss
        
        # loss_wrt_dist_deriv = 2 * (dist - min_dist)
        loss_wrt_dist_deriv = dist - min_dist
        # loss_wrt_dist_deriv *= max(0.1,
        #                            abs(dist - min_dist) ** (loss_power - 1))
        
        # don't raise the loss to a power for now
        
        # sign = np.sign(dist - min_dist)
        # loss_wrt_dist_deriv += sign * abs((dist - min_dist) ** loss_power)
        
        # Compute the derivative of the distance wrt each coordinate
        # this turned out to be surprisingly nice
        dist_wrt_coords_deriv = sphere - other
        
        overall_deriv = np.multiply(loss_wrt_dist_deriv, dist_wrt_coords_deriv)
        
        
        grad += overall_deriv
    
    return grad

def get_min_dist(spheres):
    dists = list()
    num_overlaps = 0
    for i in range(len(spheres) - 1):
        for j in range(i+1, len(spheres)):
            sphere_1 = spheres[i]
            sphere_2 = spheres[j]
            dist = np.linalg.norm(sphere_1 - sphere_2)
            if dist < 1.0 - 1e-10:
                num_overlaps += 1
            dists.append(dist)
    return min(dists), num_overlaps

def print_max_grad_norm(grads):
    norms = list()
    for grad in grads:
        norm = np.linalg.norm(grad)
        norms.append(norm)
    max_norm = max(norms)
    print('max grad norm: {:5f}'.format(max_norm))

def pack_spheres(dimensions,
                  num_spheres,
                  loss_power,
                  alpha=0.1,
                  max_step=np.pi/20,
                  momentum=0.9,
                  num_steps=50,
                  print_freq=10):
    
    # First, get the spheres
    # they will be a list of np vectors
    # and will start out at random points on the surface of the origin sphere
    spheres = list()
    for _ in range(num_spheres):
        spheres.append(sphere_gd.get_random_normal_vector(dimensions))
    
    # Then, get the grads
    # they will also be a list of np vectors
    prev_grads = list()
    for _ in range(num_spheres):
        prev_grads.append(np.zeros([dimensions]))
    
    min_dist, overlaps = get_min_dist(spheres)
    print('min dist: {:.8f}, overlaps: {}'.format(min_dist, overlaps))
    
    for step in range(1, num_steps + 1):
        # Do one GD step
        
        # calculate the new gradients
        new_grads = list()
        for i in range(num_spheres):
            sphere = spheres[i]
            others = spheres[:i] + spheres[i+1:]
            grad = sphere_loss_grad(sphere, others, loss_power)
            
            prev_grad = prev_grads[i]
            new_grads.append(grad + momentum * prev_grad)
        
        # update previous gradients to new gradients
        prev_grads = new_grads
        
        # Apply the gradients
        new_spheres = list()
        for i in range(num_spheres):
            sphere = spheres[i]
            grad = new_grads[i]
            new_sphere = sphere_gd.spherical_gd_step(sphere,
                                                     grad,
                                                     alpha,
                                                     max_step)
            
            new_spheres.append(new_sphere)
        
        # update spheres to new spheres
        spheres = new_spheres
        
        # Print info
        if step % print_freq == 0:
            min_dist, overlaps = get_min_dist(spheres)
            # print_max_grad_norm(new_grads)
            print('{:5d}:  min dist: {:.8f}, overlaps: {}'.format(step, min_dist, overlaps))
            # print('')
    
    print('')
    print('Final Results:\n')
    for sphere in spheres:
        print(sphere)
    
    print('')
    min_dist, overlaps = get_min_dist(spheres)
    print('min dist: {:.8f}, overlaps: {}'.format(min_dist, overlaps))

def main():
    pack_spheres(dimensions=4,
                 num_spheres=22,
                 loss_power=1,
                 alpha=0.1,
                 max_step=np.pi/9,
                 momentum=0.9,
                 num_steps=30000,
                 print_freq=100)

if __name__ == '__main__':
    main()