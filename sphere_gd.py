# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 14:42:42 2021

@author: Robert Morton
"""

import numpy as np

def get_random_normal_vector(dimension, min_norm=1e-5):
    """
    Return a random normal vector with the given dimension.
    
    Parameters:
        dimension: int
            The dimension of the vector to return.
        min_norm: double
            The minimum norm of the unnormalize vector
            (to avoid dividing by very small numbers)
    
    Return:
        A random unit vector with the given dimension
    """
    
    while True:
        # Get a random vector
        vector = np.random.random_sample((dimension)) - 0.5
        
        # Normalize it to make calculations simpler
        norm = np.linalg.norm(vector)
        
        # Get a new vector if the first was too small
        # to avoid dividing by a very small number
        if norm < min_norm:
            continue
        
        return vector / norm

def get_orthonormal_vector(vectors, min_norm=1e-8):
    """
    Return a random normal vector orthogonal to all the given vectors.
    
    Parameters:
        vectors: list[numpy vector]
            The vectors for the new vector to be orthogonal to.
        min_norm: double
            The minimum norm of the orthogonalized vector before normalizing
            (to avoid dividing by very small numbers)
    """
    dimension = vectors[0].shape[0]
    
    while True:
        # Get a random vector
        vector = get_random_normal_vector(dimension)
        
        # Make it normal to all the previous vectors
        for v in vectors:
            vector -= v * (np.dot(vector, v))
        
        norm = np.linalg.norm(vector)
        
        if norm < min_norm:
            continue
        
        return vector / norm

def get_orthonormal_basis(x):
    """
    Return an orthonormal basis for the space normal to x.
    
    Parameters:
        x: numpy array
            The point for the space to be normal to.
    
    Return:
        A matrix containing the basis vectors as columns.
    """
    
    vectors = [x]
    length = x.shape[0]
    
    # Add vectors to the orthonormal basis
    for _ in range(length-1):
        # Get another basis vector
        vector = get_orthonormal_vector(vectors)
        vectors.append(vector)
    
    return np.stack(vectors[1:], axis=1)

def spherical_gd_step(x, gradient, alpha, max_step, min_norm=1e-13):
    """
    Return x after one step of gradient descent,
    with x constrained to be on the surface of a unit sphere (have norm 1)

    Parameters:
        x: numpy array
            The point to update.
        gradient: numpy array
            The partial derivatives of the loss with respect to x.
        alpha: double
            The learning rate.
        max_step: double
            The maximum step angle.
        min_norm: double
            The minimum gradient norm needed to take any step.

    Return:
        The updated position of x.
    """
    
    # First, get an orthonormal basis B for the plane normal to x
    basis = get_orthonormal_basis(x)
    
    # Then calculate the dot product of the gradient with each basis vector
    basis_grad = gradient @ basis # I can do this in one operation
    
    # Get the norm of the basis gradient
    # both to control step size and to normalize
    step_norm = np.linalg.norm(basis_grad)
    
    # If the step norm is too small, we can't safely divide by it
    # and it'll be fine to ignore it
    # so ignore it
    if step_norm < min_norm:
        return x
    
    # The point on the equator of the sphere to step towards
    step_target = basis_grad / (-step_norm)
    
    # Convert this into the full space
    step_target = basis @ step_target
    
    # print('step norm: {:5f}'.format(np.linalg.norm(step_target)))
    
    # The angle to step by
    angle = min(alpha * step_norm, max_step)
    
    # Maybe normalize the result to get rid of small error
    
    return np.cos(angle) * x + np.sin(angle) * step_target