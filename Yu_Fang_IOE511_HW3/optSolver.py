"""IOE 511/MATH 562, University of Michigan
Code written by: Albert S. Berahas & Jiahao Shi
"""

import numpy as np

import algorithms
import functions


def optSolver_Yu_Fang(problem, method, options):
    """Function that runs a chosen algorithm on a chosen problem

    Inputs:
        problem, method, options (structs)
    Outputs:
        final iterate (x) and final function value (f)
    """

    # compute initial function/gradient/Hessian
    x = problem.x0
    f = problem.compute_f(x)
    g = problem.compute_g(x)
    H = problem.compute_H(x) if method.name == "Newton" else None
    norm_g = np.linalg.norm(g, ord=np.inf)
    norm_g_initial = norm_g

    # set initial iteration counter
    k = 0
    f_history = [f] # store the f(x_k)
    
    # termination criterion
    while (norm_g > options.term_tol * max(norm_g_initial, 1) )and (k <= options.max_iterations):  
        match method.name:
            case "GradientDescent":
                x_new, f_new, g_new, d, alpha = algorithms.GDStep(
                    x, f, g, problem, method, options
                )

            case "Newton":
                x_new, f_new, g_new, d, alpha = algorithms.NewtonStep(
                    x, f, g, problem, method, options
                )

            case _:
                raise ValueError("method is not implemented yet")

        # update old and new function values
        x_old = x
        f_old = f
        g_old = g
        norm_g_old = norm_g
        x = x_new
        f = f_new
        g = g_new
        norm_g = np.linalg.norm(g, ord=np.inf)
        f_history.append(f)

        # increment iteration counter
        k = k + 1
    return x, f, f_history
