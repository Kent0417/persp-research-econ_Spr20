import numpy as np
from scipy.stats import multivariate_normal as MV
from . import util
from . import kfilter as kf
from . import StateSpace as SS


def get_posterior(model, para, data, phi_smc = 1.0,
                  sampler=False, catch_errors=False):
    """
    Compute log likelihood log(f(Y|theta)) + prior log pdf log(f(theta))
    Written based on NYFED-DSGE's Julia-language implementation.

    """

    catch_errors = catch_errors | sampler
    ## Parameter update
    if sampler:
        try:
            util.update_params(model, para)
        except util.ParamsBoundsError:
            return -np.inf

        ## Get systems
        try:
            SS.get_system(model)
        except util.GensysError:
            return -np.inf
    else:
        util.update_params(model, para)

    ## Calculate likelihood and prior logpdf
    kf_lval    = likelihood(model, data, sampler=sampler, catch_errors=catch_errors)
    prior_lval = phi_smc * prior(model)
    if (kf_lval == -np.inf) | (prior_lval == -np.inf):
        return -np.inf
    else:
        return kf_lval + prior_lval

def prior(model):
    para = util.get_params(model)
    return sum([v.prior.logpdf(p) for v, p in zip(model.params.values(), para) if not v.fixed])

def prior2(pvec):
    return sum([p.prior.logpdf(p.value) for p in pvec if not p.fixed])

def likelihood(model, data,
               sampler=False, catch_errors=False,
               use_chand_recursion=False, tol=0.0):

    ## Compute state-space system
    try:
        SS.get_system(model)
    except util.GensysError:
        return -np.inf

    ## return total log-likelihood
    try:
        if not use_chand_recursion:
            return sum(kf.kalman_filter(data, model, everything=False))
        else:
            return chand_recursion(data, model.System['TT'].value, model.System['RR'].value,
                                         model.System['C'].value,  model.System['ZZ'].value,
                                         model.System['D'].value,  model.System['QQ'].value,
                                         model.System['EE'].value, allout=True, tol=tol)[0]
    except Exception as e:
        if catch_errors:
            return -np.inf
        else:
            raise

def chand_recursion(y, T, R, C, Z ,D, Q, H, allout=True, tol=0.0):

    Ns     = T.shape[0]
    Ny, Nt = y.shape

    # Variable for `Fast Kalman` algorithm
    converged = False

    # Initialize s_pred and P_pred
    s_pred, P_pred = kf.initial_state(T, R, C, Q)

    V_pred    = Z @ P_pred @ Z.T + H
    V_pred    = 0.5 * (V_pred + V_pred.T)
    invV_pred = np.linalg.inv(V_pred)
    invV_t1   = invV_pred

    W_t      = T @ P_pred @ Z.T
    M_t      = -invV_pred
    kal_gain = W_t @ invV_pred

    loglh    = np.zeros(Nt)
    zero_vec = np.zeros(Ny)
    v_t      = zero_vec
    P        = P_pred

    for t in range(Nt):
        # Step 1: Compute forecast error, ν_t and evaluate likelihoood
        yhat     = Z @ s_pred + D
        v_t      = y[:, t] - yhat
        loglh[t] = MV.logpdf(v_t, zero_vec, V_pred)

        # Step 2: Compute s_{t+1} using Eq. 5
        if t < Nt:
            s_pred = T @ s_pred + kal_gain @ v_t
            if allout:
                P = P + W_t @ M_t @ W_t.T

        if not converged:
            # Intermediate calculations to re-use
            ZW_t   = Z @ W_t
            MWpZp  = M_t @ (ZW_t.T)
            TW_t   = T @ W_t

            # Step 3: Update forecast error variance F_{t+1} (Eq 19)
            V_t1      = V_pred
            invV_t1   = invV_pred
            V_pred    = V_pred + ZW_t @ MWpZp    # F_{t+1}
            V_pred    = 0.5 * (V_pred + V_pred.T)
            invV_pred = np.linalg.inv(V_pred)

            # Step 4: Update Kalman Gain (Eq 20). Recall kalgain = K_t * V_t^-1
            # Kalgain_{t+1} = (Kalgain_t*V_{t-1} + T*W_t*M_t*W_t'*Z')V_t^-1
            kal_gain1 = kal_gain
            kal_gain  = (kal_gain @ V_t1 + TW_t @ MWpZp) @ invV_pred

            # Step 5: Update W
            W_t = TW_t - kal_gain @ ZW_t

            # Step 6: Update M
            M_t = M_t + MWpZp @ invV_t1 * MWpZp.T # M_{t+1}
            M_t = 0.5 * (M_t + M_t.T)

            if (tol > 0.0) & (np.max(abs(kal_gain - kal_gain1)) < tol):
                converged = True

    if allout:
        s_TT = s_pred + P.T @ Z.T @ invV_t1 @ (v_t)
        P_TT = P - P.T @ Z.T @ invV_t1 @ Z @ P
        return sum(loglh), loglh, s_TT, P_TT
    else:
        return loglh
