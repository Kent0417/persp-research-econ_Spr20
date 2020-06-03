import numpy as np
import warnings
from scipy.sparse import identity as speye
from scipy.sparse import diags as spdiag
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix
from scipy.linalg import ordqz, svd, schur, eigvals, solve

from ProjectCode import util
from . import reduction as red
from . import eqcond as eq

def solve_hank(model, sparse_mat=True):
    GAM0, GAM1, PSI, PI, C = eq.eqcond(model)

#     GAM0, GAM1, PSI, PI, C, basis_redundant, inv_basis_redundant= \
#                 red.solve_static_conditions(GAM0, GAM1, PSI, PI, C)

    inv_basis_redundant = speye(GAM0.shape[0], format='csc')
    basis_redundant     = speye(GAM0.shape[0], format='csc')

    if sparse_mat:
        GAM0 = csr_matrix(GAM0)
        GAM1 = csr_matrix(GAM1)
        PSI  = csr_matrix(PSI)
        PI   = csr_matrix(PI)
        C    = csr_matrix(C.reshape(len(C), 1, order='F'))

    # krylov reduction
    if model.settings['reduce_state_vars'].value:
        GAM0, GAM1, PSI, PI, C, basis_kry, inv_basis_kry = \
                red.krylov_reduction(model, GAM0, GAM1, PSI, PI, C)


    # value function reduction via spline projection
    if model.settings['reduce_v'].value:
        GAM0, GAM1, PSI, PI, C, basis_spl, inv_basis_spl = \
                red.valuef_reduction(model, GAM0, GAM1, PSI, PI, C)

    # Compute inverse basis for Z matrix and IRFs transformation
    if model.settings['reduce_v'].value & model.settings['reduce_state_vars'].value:
        inverse_basis = inv_basis_redundant @ inv_basis_kry * inv_basis_spl # from_spline
        basis = basis_spl @ basis_kry @ basis_redundant
    elif model.settings['reduce_state_vars'].value:
        inverse_basis = inv_basis_redundant @ inv_basis_kry
        basis = basis_kry @ basis_redundant
    else:
        inverse_basis = inv_basis_redundant
        basis = basis_redundant

    # Solve LRE model
    G1, C, impact, _, _, eu = \
            gensys_hank(GAM1.toarray(), C.toarray(),PSI.toarray(), PI.toarray())

    if eu[0] != 1 | eu[1] != 1:
        raise util.GensysError()

    G1 = np.real(G1)
    impact = np.real(impact)
    C = np.real(C).flatten()

    return G1, impact, C, inverse_basis, basis

def gensys_hank(GAM1, C, PSI, PI,
               check_existence=True, check_uniqueness=True,
               eps = np.sqrt(np.finfo(float).eps)*10, div=-1.0):
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

    eu          = [0, 0]
    T, U = schur(GAM1)
    n           = U.shape[0]
    g_eigs      = np.real(eigvals(T))
    stable_eigs = lambda eigs :eigs <= 0
    nunstab     = n - sum(stable_eigs(g_eigs))

    T, U, _ = schur(GAM1, sort=stable_eigs)

    U1 = U[:, :n - nunstab].T
    U2 = U[:, n - nunstab : n].T
    etawt = U2 @ PI

    _, ueta, deta, veta = decomposition_svdct(etawt, eps=eps)

    if check_existence:
        zwt = U2 @ PSI
        bigev, uz, dz, vz = decomposition_svdct(zwt, eps=eps)
        if all(bigev) == False:
            eu[0] = 1
        else:
            eu[0] = np.linalg.norm(uz-(ueta @ ueta.conj().T) @ uz) < eps * n

        # if (eu[0] == 0) & (div == -1):
        #     warnings.warn('Solution does not exist')
        impact = np. real(-PI @ veta @ solve(deta, ueta.T) @ uz @ dz @ vz.T + PSI)
    else:
        eu[0] = 1
        impact = np. real(-PI @ veta @ solve(deta, ueta.T) @ U2 + PSI)

    if check_uniqueness:
        etawt1 = U1 @ PI
        bigev, _, deta1, veta1 = decomposition_svdct(etawt1)
        if all(bigev) == False:
            eu[1] = 1
        else:
            eu[1] = np.linalg.norm(veta1 - (veta @ veta.conj().T) @ veta1) < eps * n

#     spdiag_internal = lambda n1, n2: (*spdiag(np.hstack([np.ones(n1),np.zeros(n2)])).nonzero(), \
#                                       np.hstack([np.ones(n1),np.zeros(n2)]))
#     I, J, V = spdiag_internal(n-nunstab, nunstab)
#     diag_m = csc_matrix((V, (I, J)), shape=(n, n))
    diag_m = spdiag(np.hstack([np.ones(n-nunstab),np.zeros(nunstab)]))
    G1 = np.real(U @ T @ diag_m @ U.T)
    F = U1[:, :nunstab].T @ np.linalg.inv(U1[:, nunstab:].T)
    impact = np.vstack([F @ PSI[nunstab:, :], PSI[nunstab:, :]])
    C = np.real(U @ C) * np.ones([U.shape[0], 1])

    return G1, C, impact, U, T, eu

def decomposition_svdct(A, eps=np.sqrt(np.finfo(float).eps)*10):
    U, S, V = svd(A, full_matrices=False)
    V = V.T.conj()
    bigev = S > eps
    Au    = U[:, bigev]
    Ad    = np.diag(S[bigev])
    Av    = V[:, bigev]

    return bigev, Au, Ad, Av
