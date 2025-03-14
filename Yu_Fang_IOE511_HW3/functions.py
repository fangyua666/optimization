"""IOE 511/MATH 562, University of Michigan
Code written by: Albert S. Berahas & Jiahao Shi


Define all the functions and calculate their gradients and Hessians, those functions include:
    (1) Rosenbrock function
    (2) Quadractic function
"""

import numpy as np
import scipy.io

def rosen_func(x):
    """Function that computes the function value for the Rosenbrock function

    Input:
        x
    Output:
        f(x)
    """

    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosen_grad(x):
    """Function that computes the gradient of the Rosenbrock function

    Input:
        x
    Output:
        g = nabla f(x)
    """
    gradient = np.zeros(2)
    gradient[0] = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
    gradient[1] = 200 * (x[1] - x[0] ** 2)
    
    return gradient


def rosen_Hess(x):
    """Function that computes the Hessian of the Rosenbrock function

    Input:
        x
    Output:
        H = nabla^2 f(x)
    """
    hessian = np.zeros((2,2))
    hessian[0][0] = 2 - 400 * x[1] + 1200 * x[0] ** 2
    hessian[0][1] = -400 * x[0]
    hessian[1][0] = -400 * x[0]
    hessian[1][1] = 200
    return hessian


def quad_func_quad2(x):
    """Function that computes the function value for the Quadractic function

    Input:
        x
    Output:
        f(x)
    """
    data = scipy.io.loadmat('quadratic2.mat')
    A = data['A']
    b = data['b']
    c = data['c']
    return (0.5 * x.T @ A @ x + b.T @ x + c)[0][0]   

def quad_grad_quad2(x):
    
    data = scipy.io.loadmat('quadratic2.mat')
    A = data['A']
    b = data['b']
    g = A @ x + b
    return g

def quad_Hess_quad2(x):
    data = scipy.io.loadmat('quadratic2.mat')
    A = data['A']
    return A

def quad_func_quad10(x):
    """Function that computes the function value for the Quadractic function

    Input:
        x
    Output:
        f(x)
    """
    data = scipy.io.loadmat('quadratic10.mat')
    A = data['A']
    b = data['b']
    c = data['c']
    
    return  (0.5 * x.T @ A @ x + b.T @ x + c)[0][0]

def quad_grad_quad10(x):
    data = scipy.io.loadmat('quadratic10.mat')
    A = data['A']
    b = data['b']
    g = A @ x + b
    return g

def quad_Hess_quad10(x):
    data = scipy.io.loadmat('quadratic10.mat')
    A = data['A']
    return A
