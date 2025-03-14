""" IOE 511/MATH 562, University of Michigan
Code written by: Albert S. Berahas & Jiahao Shi
"""

# [3]
import numpy as np

def GDStep(x, f, g, problem, method, options):
    """Function that: (1) computes the GD step; (2) updates the iterate; and,
         (3) computes the function and gradient at the new iterate

    Inputs:
        x, f, g, problem, method, options
    Outputs:
        x_new, f_new, g_new, d, alpha
    """
    # Set the search direction d to be -g
    d = -g  

    # determine step size
    match method.options["step_type"]:
        # Gradient descent with constant step size
        case "Constant":
            alpha = method.options["constant_step_size"]
            x_new = x + alpha * d
            f_new = problem.compute_f(x_new)
            g_new = problem.compute_g(x_new) 

        # Gradient descent with backtracking line search
        case "Backtracking":
            alpha = backtracking_line_search(x, f, g, d, problem, method, options)
            x_new = x + alpha * d
            f_new = problem.compute_f(x_new)
            g_new = problem.compute_g(x_new)

        case _:
            raise ValueError("step type is not defined")

    return x_new, f_new, g_new, d, alpha

def NewtonStep(x, f, g, problem, method, options):
    # define the search direction d
    H = problem.compute_H(x)
    d = -np.linalg.inv(H) @ g
    
    # determine the step size
    match method.options["step_type"]:
        # Newton's method with backtracking line search
        case "Backtracking":
            alpha = backtracking_line_search(x, f, g, d, problem, method, options)
            x_new = x + alpha * d
            f_new = problem.compute_f(x_new)
            g_new = problem.compute_g(x_new)
            H_new = problem.compute_H(x_new)
        
        case _:
            raise ValueError("step type is not defined")
    
    return x_new, f_new, g_new, d, alpha
    
    
def backtracking_line_search(x, f, g, d, problem, method, options):
    """Function that computes the step size using backtracking line search

    Inputs:
        x, f, g, d, problem, method, options
    Outputs:
        alpha
    """
    alpha_bar = 1
    tau = 0.5
    c_1 = 1e-4
    
    # set alpha to alpha_bar
    alpha = alpha_bar
    
    while problem.compute_f(x + alpha * d) > f + c_1 * alpha * (g.T @ d):
        alpha = tau * alpha

    return alpha