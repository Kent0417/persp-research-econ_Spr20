import warnings
import numpy as np
from scipy.linalg import ordqz, svd, solve
from ProjectCode import util
from .import eqcond as eq

def solve_rank(model):

    GAM0, GAM1, C, PSI, PI = eq.eqcond(model)
    G1, C, impact, eu = gensys_rank(GAM0, GAM1, C, PSI, PI)

    if eu[0] != 1 | eu[1] != 1:
        raise util.GensysError()

    return G1, impact, C

def gensys_rank(GAM0, GAM1, C ,PSI, PI, div = 0.0):
    """
    Solve a liner rational expectations model by Sims(2002)
    Generate state-space solution to DSGE model system given as:

        GAM0 s(t) = GAM1 s(t-1) + c + PSI eps(t) + PI eta(t)

    Return system is:

        s(t) = G1 * s(t-1) + c + impact * eps(t)

    Input
    -----
    GAM0:  n x n matrix
    GAM1:  n x n matrix
    PSI: n x m matrix
    PI:  n x p

    Return
    -----
    G1: n x n matrix
    impact: n x m matrix
    eu:
        eu[0] = 1 for existence
        eu[1] = 1 for uniqueness
        eu  = [-2, -2] for coincident zeros
    """

    S, T, alpha, beta, _, _ = ordqz(GAM0, GAM1, output='complex')

    eu      = [0, 0]
    eps     = 1e-6
    nunstab = 0
    zxz     = 0
    a, b    = S, T
    n       = a.shape[0]
    if div == 0.0:
        div = new_div(a, b)

    if ((abs(alpha) < eps) * (abs(beta) < eps)).any():
        zxz = 1

    select = lambda alpha, beta: [not abs(be) > div*abs(al) for al, be in zip(alpha, beta)]

    nunstab = n - sum(select(alpha, beta))

    if zxz == 1:
        # warnings.warn('Coincident zeros. Indeterminancy and/or nonexistence')
        eu = [-2, -2]
        G1     = np.zeros([0,0])
        C      = np.zeros(0)
        impact = np.zeros([0,0])
        return G1, C ,impact, eu

    a, b, _, _, qt, z = ordqz(GAM0, GAM1, sort=select, output='complex')
    gev      = np.c_[np.diag(a), np.diag(b)]
    qt1, qt2 = qt[:, :(n-nunstab)], qt[:, (n-nunstab):]
    etawt    = qt2.conj().T @ PI
    neta     = PI.shape[1]

    ## Handling case of no stable roots
    if nunstab == 0:
        etawt = np.zeros([0, neta])
        ueta  = np.zeros([0, 0])
        deta  = np.zeros([0, 0])
        veta  = np.zeros([neta, 0])
        bigev = 0
    else:
        U, S, V = svd(etawt, full_matrices=False)
        V = V.conj().T
        bigev = S > eps
        ueta  = U[:, bigev]
        veta  = V[:, bigev]
        deta  = np.diag(S[bigev])

    existence = sum(bigev) >= nunstab
    if existence:
        eu[0] = 1
    # else:
    #     warnings.warn('Nonexistence')

    if nunstab == n:
        etawt1 = np.zeros([0, neta])
        ueta1  = np.zeros([0, 0])
        deta1  = np.zeros([0, 0])
        veta1  = np.zeros([neta, 0])
        bigev  = 0
    else:
        etawt1 = qt1.conj().T @ PI
        ndeta1 = min(n-nunstab, neta)
        U1, S1, V1 = svd(etawt1, full_matrices=False)
        V1    = V1.conj().T
        bigev = S1 > eps
        ueta1 = U1[:, bigev]
        veta1 = V1[:, bigev]
        deta1 = np.diag(S1[bigev])

    if veta1.size == 0:
        unique = True
    else:
        loose = veta1 - (veta @ veta.conj().T) @ veta1
        Ul, Sl, Vl = svd(loose, full_matrices=False)
        Vl     = Vl.conj().T
        nloose = (np.abs(Sl) > eps * n).sum()
        unique = (nloose == 0)

    if unique:
        eu[1] = 1
    # else:
    #     warnings.warn('Indeterminacy')

    tmat = np.hstack([np.eye(n-nunstab), \
                      -(ueta @ (solve(deta,veta.conj().T)) @ veta1 @ (deta1 @ ueta1.conj().T)).conj().T])

    G0   = np.vstack([tmat @ a, np.hstack([np.zeros([nunstab, n-nunstab]), \
                                            np.eye(nunstab)])])
    G1   = np.vstack([tmat @ b, np.zeros([nunstab, n])])

    G0_inv = np.linalg.inv(G0)
    G1     = G0_inv @ G1
    Busix  = b[(n-nunstab):, (n-nunstab):]
    Ausix  = a[(n-nunstab):, (n-nunstab):]
    C      = G0_inv @ np.hstack([tmat @ qt.conj().T @ C,
                                 solve((Ausix - Busix), qt2.conj().T @ C)])
    impact = G0_inv @ np.vstack([tmat @ (qt.conj().T @ PSI), \
                                 np.zeros([nunstab, PSI.shape[1]])])

    G1     = np.real(z @ (G1 @ z.conjugate().T))
    C      = np.real(z @ C)
    impact = np.real(z @ impact)

    return G1, C, impact, eu

def new_div(a, b):
    eps = 1e-6
    n   = b.shape[0]
    div = 1.01
    for i in range(n):
        if abs(a[i,i ]) > 0:
            divhat = abs(b[i, i]) / abs(a[i, i])
            if (1 + eps < divhat) & (divhat <= div):
                div = 0.5 * (1.0+divhat)
    return div

class GensysError(Exception):
    pass
