import numpy as np
from . import util
from .posterior import get_posterior


def hessian(model, x, data, check_neg_diag=True):
    """
    Compute Hessian of DSGE posterior function evaluated at x.
    Based on NYFED-DSGE's Julia-language implementation.

    """

    util.update_params(model, x)

    #para_free      = np.array([v.value if not v.fixed for v in model.params.values()])
    para_free_inds = np.where([not v.fixed for v in model.params.values()])

    # Compute hessian only for free parameters
    n_para  = len(x)
    hessian = np.zeros([n_para, n_para])

    x_model   = np.copy(x)               ## vector of all params
    x_hessian = x_model[para_free_inds]  ## vector of free params

    def f_hessian(x_hessian):
        x_model[para_free_inds] = x_hessian
        return -get_posterior(model, x_model, data)

    hessian_free, has_errors = hessizero(f_hessian, x_hessian, check_neg_diag)

    for row_free, row_full in enumerate(*para_free_inds):
        hessian[row_full, para_free_inds] = hessian_free[row_free, :]

    return hessian, has_errors

def hessizero(fcn, x, check_neg_diag=True):
    n_para  = len(x)
    hessian = np.zeros([n_para, n_para])

    ## Compute diagonal elements
    for i in range(n_para):
        hessian[i, i] = hess_diag_element(fcn, x, i, check_neg_diag = check_neg_diag)

    ## Compute off-diagonal elements
    invalid_corr   = dict()
    n_off_diag_els = int(n_para * (n_para - 1) / 2)
    off_diag_inds  = dict()
    k = 0
    for i in range(n_para - 1):
        for j in range(i+1,n_para):
            off_diag_inds[k] = (i, j)
            k += 1

    ### Iterate over off diag elements
    off_diag_out = dict()
    for k, (i,j) in off_diag_inds.items():
        sxsy            = np.sqrt(abs(hessian[i, i]*hessian[j, j]))
        off_diag_out[k] = hess_offdiag_element(fcn, x, i, j, sxsy)

    ## Fill in values
    for k in range(n_off_diag_els):
        (i, j)          = off_diag_inds[k]
        (value, rho_xy) = off_diag_out[k]

        hessian[i, j] = value
        hessian[j, i] = value

        if rho_xy < -1 | 1 < rho_xy:
            invalid_corr[(i, j)] = rho_xy

    has_errors = False
    if not invalid_corr == {}:
        print(f"  - HessianErrors: {invalid_corr}")
        has_errors = True

    return hessian, has_errors


def hess_diag_element(fcn, x, i, ndx=6, check_neg_diag=True):
    n_para   = len(x)
    dxscale  = np.ones(n_para)
    dx       = np.exp(-np.linspace(6, (6+(ndx-1)*2), 6))
    hessdiag = np.zeros(ndx)

    for k in range(2, 4):
        paradx    = np.copy(x)
        parady    = np.copy(x)
        paradx[i] = paradx[i] + dx[k]*dxscale[i]
        parady[i] = parady[i] - dx[k]*dxscale[i]

        fx  = fcn(x)
        fdx = fcn(paradx)
        fdy = fcn(parady)

        hessdiag[k]  = -(2*fx - fdx - fdy) / (dx[k]*dxscale[i])**2

    value = (hessdiag[2]+hessdiag[3])/2

    if check_neg_diag & (value < 0):
        raise ValueError("Negative diagonal in Hessian")

    return value

def hess_offdiag_element(fcn, x, i, j, sxsy, ndx=6):
    n_para   = len(x)
    dxscale  = np.ones(n_para)
    dx       = np.exp(-np.linspace(6, (6+(ndx-1)*2), 6))
    hessdiag = np.zeros(ndx)

    for k in range(2, 4):
        paradx      = np.copy(x)
        parady      = np.copy(x)
        paradx[i]   = paradx[i] + dx[k]*dxscale[i]
        parady[j]   = parady[j] - dx[k]*dxscale[j]
        paradxdy    = np.copy(paradx)
        paradxdy[j] = paradxdy[j] - dx[k]*dxscale[j]

        fx    = fcn(x)
        fdx   = fcn(paradx)
        fdy   = fcn(parady)
        fdxdy = fcn(paradxdy)

        hessdiag[k]  = -(fx - fdx - fdy + fdxdy) / (dx[k]*dx[k]*dxscale[i]*dxscale[j])

    value = (hessdiag[2]+hessdiag[3])/2

    if (value == 0) | (sxsy == 0):
        rho_xy = 0
    else:
        rho_xy = value / sxsy

    if (rho_xy < -1) | (1 < rho_xy):
        value = 0

    return value, rho_xy
