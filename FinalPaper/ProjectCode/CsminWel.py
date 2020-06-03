import copy
from scipy import optimize
import numpy as np

def csminwell(fun, x0, args,
              options={'x_tol'        : 1e-10,
                       'f_tol'        : 1e-10,
                       'g_tol'        : 1e-10,
                       'show_trace'   : False,
                       'max_iter'     : 100},
              **kwargs):
    """
    The optimization algorithm originally provided by Chris Sims.
    See http://sims.princeton.edu/yftp/optimize/

    But this code is written to utilize `scipy.optimize.minimize`.
    That is, it can be used in callable method like
    ```
    scipy.optimize.minimize(fun, x0, args=(), method=csminwell)

    ```

    Return
    -------
    OptimizeResult object

    """

    xtol, ftol, grtol = options['x_tol'], options['f_tol'], options['g_tol']
    iterations = options['max_iter']
    show_trace = options['show_trace']

    H0 = 1e-5 * np.eye(len(x0))

    # Unpack dimensions
    nx = x0.shape[0]

    # Count function and gradient calls
    f_calls, g_calls, h_calls = 0, 0, 0

    # Maintain current state in x and previous state in x_previous
    x, x_previous = copy.copy(x0), copy.copy(x0)

    # start with Initial Hessian
    H = H0

    # start rc parameter at 0
    rc = 0

    f_x = fun(x0, args)
    f_calls += 1

    if (f_x == np.inf) | (f_x == np.nan):
        raise ArgumentError("Bad initial guess, 'f_x' returned Inf. Try again")
    elif f_x > 1e50:
        raise ArgumentError("Bad initial guess. Try again")

    # Create gradient
    def _grad(f, X, args, h = 1e-4):
        nx = len(X)
        I = np.eye(nx)
        results = np.zeros(nx)
        for i in range(nx):
            e = I[i]
            X_fh = X + h * e
            X_bh = X - h * e
            results[i] = (f(X_fh, args) - f(X_bh, args))/(2*h)
        return results

    def _gradwrap(f, x, args):
        # df = grad(f)
        # stor = df(x, args)
        stor = _grad(f, x, args)
        bad_grads = abs(stor) >= 1e15
        stor[bad_grads] = 0.0
        return stor, any(bad_grads)

    gr, badg = _gradwrap(fun, x0, args)
    g_calls += 1

    # Count interations
    iteration = 0

    # Maintain a trace
    # TBD

    # set objects to their starting values
    retcode3 = 101

    # set up return variables so they are available outside while loop
    fh = np.copy(f_x)
    xh = np.copy(x0)
    gh = np.copy(x0)
    retcodeh = 1000

    # Assess multiple types of convergence
    x_converged, f_converged, gr_converged = False, False, False

    # Iterate until convergence or exhaustion
    converged = False
    while (not converged) & (iteration < iterations):
        iteration += 1
        f1, x1, fc, retcode1 = csminit(fun, x, f_x, gr, badg, H, args, show_trace)
        f_calls += fc

        if retcode1 != 1:
            if (retcode1 == 2) | (retcode1 == 4):
                wall1, badg1 = True, True
            else:
                g1, badg1 = _gradwrap(fun, x1, args)
                g_calls += 1
                wall1 = badg1

            if wall1 & (len(H) > 1):

                Hcliff = H + np.diag(np.diag(H) * np.random.rand(nx))

                # print('Cliff.  Perturbing search direction.\n')

                f2, x2, fc, retcode2 = csminit(fun, x, f_x, gr, badg, Hcliff, args, show_trace)
                f_calls += fc

                if f2 < f_x:
                    if (retcode2 == 2) or (retcode2 == 4):
                        wall2, badg2 = True, True
                    else:
                        g2, badg2 = _gradwrap(fun, x2, args)
                        g_calls += 1
                        wall2 = badg2
                        badg2

                    if wall2:
                        # print("Cliff again.  Try traversing\n")
                        if np.linalg.norm(x2-x1) < 1e-13:
                            f3 = f_x
                            x3 = x
                            badg3 = True
                            retcode3 = 101
                        else:
                            gcliff = ((f2-f1) / ((np.linalg.norm(x2-x1))**2)) * (x2-x1)
                            if len(x0.shape) == 2:
                                gcliff = gcliff.conj().T
                            f3, x3, fc, retcode3 = csminit(fun, x, f_x, gcliff, False, np.eye(nx), args, show_trace)
                            f_calls += fc

                            if (retcode3 == 2) or (retcode3==4):
                                wall3 = True
                                badg3 = True
                            else:
                                g3, badg3 = _gradwrap(fun, x3, args)
                                g_calls += 1
                                wall3 = badg3
                    else:
                        f3 = f_x
                        x3 = x
                        badg3 = True
                        retcode3 = 101
                else:
                    f3 = f_x
                    x3 = x
                    badg3 = True
                    retcode3 = 101
            else:
                f2, f3 = f_x, f_x
                badg2, badg3 = True, True
                retcode2, retcode3 = 101, 101
        else:
            f1, f2, f3 = f_x, f_x, f_x
            retcode2, retcode3 = retcode1, retcode1
            #badg1, badg2, badg3 = False, False, False

        if (f3 < f_x - ftol) & (badg3 == 0):
            ih = 2
            fh = f3
            xh = x3
            gh = g3
            badgh = badg3
            retcodeh = retcode3
        elif (f2 < f_x - ftol) & (badg2 == 0):
            ih = 1
            fh = f2
            xh = x2
            gh = g2
            badgh = badg2
            retcodeh = retcode2
        elif (f1 < f_x - ftol) & (badg1 == 0):
            ih = 0
            fh = f1
            xh = x1
            gh = g1
            badgh = badg1
            retcodeh = retcode1
        else:
            fh = np.min([f1, f2, f3])
            ih = np.argmin([f1, f2, f3])

            if ih == 0:
                xh = x1
                retcodeh = retcode1
            elif ih == 1:
                xh = x2
                retcodeh = retcode2
            elif ih == 2:
                xh = x3
                retcodeh = retcode3

            try:
                nogh = not gh
            except:
                nogh = True

            if nogh:
                gh, badgh = _gradwrap(fun, xh, args)
                g_calls += 1

            badgh = True

        stuck = (abs(fh-f_x) < ftol)
        if (not badg) & (not badgh) & (not stuck):
                H = bfgsi(H, gh-gr, xh-x)

        if show_trace:
            print('Improvement on iteration {0} = {1:.9f}\n'.format(iteration, fh-f_x))

        x_previous = np.copy(x)
        # Update before next iteration
        f_x_previous, f_x = f_x, fh
        x = xh
        gr = gh
        badg = badgh

        x_converged, f_converged, gr_converged, converged = \
            assess_convergence(x, x_previous, f_x, f_x_previous, gr, xtol, ftol, grtol)

    result = optimize.OptimizeResult()
    result.initial_x  = x0
    result.x          = x
    result.success    = converged
    result.hess       = H
    result.iterations = iteration
    return result


def csminit(fun, x0, f0, g0, badg, H0, args, show_trace):
    x0 = x0.astype('float')
    angle = .005

    #(0<THETA<.5) THETA near .5 makes long line searches, possibly fewer iterations.
    theta   = .3
    fchange = 1000
    minlamb = 1e-9
    mindfac = .01
    f_calls = 0
    lambda_ = 1.0
    xhat    = x0
    f       = f0
    fhat    = f0
    gr      = g0
    gnorm   = np.linalg.norm(gr)

    if (gnorm < 1e-12) &  (not badg):
        # gradient convergence
        retcode = 1
        dxnorm  = 0.0
    else:
        # with badg true, we don't try to match rate of improvement to directional
        # derivative.  We're satisfied just to get some improvement in f.
        dx = (-H0 @ gr).flatten()
        dxnorm = np.linalg.norm(dx)

        if dxnorm > 1e12:
            #print('Near singular H problem.\n')
            dx = dx * fchange/dxnorm

        dfhat = dx @ g0

        if not badg:
            # test for alignment of dx with gradient and fix if necessary
            a = -dfhat / (gnorm*dxnorm)

            if a < angle:
                dx -= (angle*dxnorm/gnorm + dfhat/(gnorm*gnorm)) * gr
                dx *= dxnorm/np.linalg.norm(dx)
                dfhat = dx @ gr

        if show_trace:
            print('Predicted Improvement: {0:.9f}'.format(-dfhat/2))

        # Have OK dx, now adjust length of step (lambda) until min and max improvement
        # rate criteria are met.
        done        = False
        fact        = 3.0
        shrink      = True
        lambda_min  = 0.0
        lambda_max  = np.inf
        lambda_peak = 0.0
        f_peak      = f0
        lambda_hat  = 0.0

        while (not done):
            if len(x0.shape) == 2:
                dxtest = x0 + dx.conj().T * lambda_
            else:
                dxtest = x0 + dx * lambda_

            f = fun(dxtest, args)

            if show_trace:
                print('lambda = {0:.5f}; f = {1:.7f}'.format(lambda_,f))

            if f < fhat:
                fhat = f
                xhat = dxtest
                lambdahat = lambda_

            f_calls += 1
            shrink_signal = ((not badg) & (f0-f < np.max(-theta*dfhat*lambda_, 0))) | (badg & ((f0-f) < 0))

            grow_signal = (not badg) & (lambda_ > 0) & (f0-f > -(1-theta)*dfhat*lambda_)

            if shrink_signal & ((lambda_ > lambda_peak) | (lambda_ < 0)):
                if (lambda_ > 0) & ((not shrink) | (lambda_/fact <= lambda_peak)):
                    shrink = True
                    fact = fact**0.6
                    while lambda_ / fact <= lambda_peak:
                        fact = fact**0.6

                    if abs(fact-1.0) < mindfac:
                        if abs(lambda_) < 4:
                            retcode = 2
                        else:
                            retcode = 7

                        done = True

                if (lambda_ < lambda_max) & (lambda_ > lambda_peak):
                    lambda_max = lambda_

                lambda_ /=fact
                if abs(lambda_) < minlamb:
                    if (lambda_ > 0) & (f0 <= fhat):
                        lambda_ = -lambda_*fact**6
                    else:
                        if lambda_ < 0:
                            retcode = 6
                        else:
                            retcode = 3
                        done = True

            elif (grow_signal & (lambda_ > 0)) | (shrink_signal & ((lambda_ <= lambda_peak) & (lambda_ > 0))):
                if shrink:
                    shrink = False
                    fact = fact**0.6
                    if abs(fact-1) < mindfac:
                        if abs(lambda_) < 4:
                            retcode = 4
                        else:
                            retcode = 7

                        done = True

                if (f < f_peak) & (lambda_ > 0):
                    f_peak = f
                    lambda_peak = lambda_
                    if lambda_max <= lambda_peak:
                        lambda_max = lambda_peak * fact**2

                lambda_ *= fact
                if abs(lambda_) > 1e20:
                    retcode = 5
                    done = True

            else:
                done = True
                if fact < 1.2:
                    retcode = 7
                else:
                    retcode = 0
    if show_trace:
        print('Norm of dx {0:.5f}'.format(dxnorm))

    return fhat, xhat, f_calls, retcode

def bfgsi(H0, dg, dx):
    if len(dg.shape) == 2:
        dg = dg.conj().T

    if len(dx.shape) == 2:
        dx = dx.conj().T

    Hdg = H0 @ dg # vector
    dgdx = dx @ dg # scaler

    H = H0
    if abs(dgdx) > 1e-12:
        H += (dgdx + (dg@Hdg)) * (np.outer(dx,dx))/(dgdx**2) - (np.outer(Hdg,dx) + np.outer(dx,Hdg))/dgdx
    elif np.linalg.norm(dg) < 1e-7:
        pass
    else:
        pass
        #print("bfgs update failed")

    return H

def assess_convergence(x, x_previous, f_x, f_x_previous, gr, xtol, ftol, grtol):

    x_converged, f_converged, gr_converged = False, False, False

    if np.max(abs(x-x_previous)) < xtol:
        x_converged = True

    if abs(f_x-f_x_previous) < ftol:
        f_converged = True

    if np.linalg.norm(gr, np.inf) < grtol:
        gr_converged = True

    converged = x_converged | f_converged | gr_converged

    return x_converged, f_converged, gr_converged, converged

class ArgumentError(Exception):
    pass
