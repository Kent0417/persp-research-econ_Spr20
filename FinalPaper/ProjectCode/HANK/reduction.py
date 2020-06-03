import numpy as np
from scipy.sparse import identity as speye
from scipy.sparse import diags as spdiag
from scipy.sparse import kron as spkron
from scipy.sparse import spmatrix, hstack, vstack, SparseEfficiencyWarning
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix
from scipy.sparse.linalg import lsqr as sp_lsqr
from scipy.sparse.linalg import spsolve
from scipy.linalg import svd, solve, null_space
from ProjectCode import util
import warnings

warnings.simplefilter('ignore', SparseEfficiencyWarning)

"""
Implement model reductions.

Written based on NYFED-DSGE's Julia-language implementation
and SeHyoun Ahn's Matlab code
See https://sehyoun.com/EXAMPLE_one_asset_HANK_web.html
"""

def solve_static_conditions(GAM0, GAM1, PSI, PI, C):

    redundant     = np.max(abs(np.hstack([GAM0, PSI])),axis=1) == 0
    redundant_list = [redundant[row] for row in range(len(redundant))]

    inv_state_red = null_space(GAM1[redundant_list,:])
    state_red     = inv_state_red.T

    g0  = state_red @ GAM0 @ inv_state_red
    g1  = state_red @ GAM1 @ inv_state_red
    g1  = solve(g0, g1)
    Psi = solve(g0, state_red @ PSI)
    Pi  = solve(g0, state_red @ PI)
    c   = solve(g0, state_red @ C)


    return g0, g1, Pi, Psi, c, state_red, inv_state_red

#------------------------------------------------------
# Distribution reduction

def krylov_reduction(model, GAM0, GAM1, PSI, PI, C):

    if not 'F' in model.settings.keys():
        model.settings.update({'F': type('setting', (object,),
                                       {'value': lambda x:x,
                                        'description':'Function applied during Krylov reduction'})})
    F = model.settings['F'].value

    # Grab Dimensions
    n_state_vars_unreduce = int(model.settings['n_state_vars_unreduce'].value)
    n_state_vars          = int(model.settings['n_state_vars'].value)
    n_jump_vars           = int(model.settings['n_jump_vars'].value)
    krylov_dim            = int(model.settings['krylov_dim'].value)
    n_total               = n_jump_vars + n_state_vars
    n_vars                = len(util.flatten_indices(model.endog_states))
    n_state_vars         -= n_state_vars_unreduce

    GAM1_full = GAM1.toarray()

    B_pv      = solve(-GAM1_full[n_total:n_vars, n_total:n_vars],
                       GAM1_full[n_total:n_vars,        :n_jump_vars])

    B_pg      = solve(-GAM1_full[n_total:n_vars,     n_total:n_vars],
                       GAM1_full[n_total:n_vars, n_jump_vars:n_jump_vars + n_state_vars])
    B_pZ      = solve(-GAM1_full[n_total:n_vars, n_total:n_vars],
                      GAM1_full[n_total:n_vars, n_jump_vars+n_state_vars:n_jump_vars+n_state_vars+n_state_vars_unreduce])
    B_gg      = GAM1_full[n_jump_vars:n_jump_vars + n_state_vars, n_jump_vars:n_jump_vars + n_state_vars]
    B_gv      = GAM1_full[n_jump_vars:n_jump_vars + n_state_vars, :n_jump_vars]
    B_gp      = GAM1_full[n_jump_vars:n_jump_vars + n_state_vars, n_total:n_vars]

    # Drop redundant equations
    obs        = B_pg
    _, d0, V_g = svd(obs, full_matrices=False)
    aux        = d0/d0[0]
    n_Bpg      = int(sum(aux > 10*np.finfo(float).eps))
    V_g        = V_g.conj().T
    V_g        = V_g[:, :n_Bpg] * aux[:n_Bpg]

    # Compute Krylov subspace
    A = lambda x: 0 + B_gg.conj().T @ x + B_pg.conj().T @ (B_gp.conj().T @ x)
    V_g, _, _  = deflated_block_arnoldi(A, V_g, krylov_dim)
    n_state_vars_red = V_g.shape[1]

    # Build state space reduction transform
    reduced_basis = lil_matrix(np.zeros((n_jump_vars+n_state_vars_red,n_vars)))
    reduced_basis[:n_jump_vars,:n_jump_vars] = speye(n_jump_vars, dtype=float, format='csc')
    reduced_basis[n_jump_vars: n_jump_vars+n_state_vars_red, \
                  n_jump_vars: n_jump_vars+n_state_vars    ] = V_g.conj().T
    reduced_basis[n_jump_vars + n_state_vars_red:n_jump_vars + n_state_vars_red + n_state_vars_unreduce,\
                  n_jump_vars + n_state_vars    :n_jump_vars + n_state_vars     + n_state_vars_unreduce] \
            = np.eye(n_state_vars_unreduce)

    # Build inverse transform
    inv_reduced_basis = lil_matrix(np.zeros((n_vars, n_jump_vars + n_state_vars_red)))
    inv_reduced_basis[:n_jump_vars,:n_jump_vars] = speye(n_jump_vars, dtype=float, format='csc')
    inv_reduced_basis[n_jump_vars:n_jump_vars+n_state_vars,\
                      n_jump_vars:n_state_vars_red+n_jump_vars] = V_g
    inv_reduced_basis[n_total:n_vars, :n_jump_vars] = B_pv

    inv_reduced_basis[n_total:n_vars,n_jump_vars:n_jump_vars + n_state_vars_red] = B_pg @ V_g
    inv_reduced_basis[n_total:n_vars,n_jump_vars+n_state_vars_red:n_jump_vars+n_state_vars_red+n_state_vars_unreduce] = B_pZ

    inv_reduced_basis[n_jump_vars+n_state_vars:n_total,\
                      n_jump_vars+n_state_vars_red:n_jump_vars+n_state_vars_red+n_state_vars_unreduce] \
        = speye(n_state_vars_unreduce, dtype=float, format='csc')

    model.settings.update({'n_state_vars_red': type('setting', (object,),
                                       {'value': n_state_vars_red + n_state_vars_unreduce,
                                        'description':'Number of state variables after reduction'})})

    # Change basis
    GAM0_kry, GAM1_kry, PSI_kry, PI_kry, C_kry = \
        change_basis(reduced_basis, inv_reduced_basis, GAM0, GAM1, PSI, PI, C, ignore_GAM0 = True)

    return GAM0_kry, GAM1_kry, PSI_kry, PI_kry, C_kry, reduced_basis, inv_reduced_basis

def deflated_block_arnoldi(A, B, m):
    q, r = np.linalg.qr(B, mode='complete')
    Q = q @ np.eye(*B.shape)
    basis = np.empty((Q.shape[0], 0))
    realsmall = np.sqrt(np.finfo(float).eps)

    if m == 1:
        basis = Q
    else:
        for i in range(m-1):
            # Manual Gram-Schmidt
            basis = np.hstack([basis,Q])
            aux = A(Q)
            for j in range(basis.shape[1]):
                aux -= np.outer(basis[:, j], basis[:, j]) @ aux

            # Check for potential deflation
            Q = np.empty(Q.shape[0]).reshape(Q.shape[0],-1)
            for j in range(aux.shape[1]):
                weight = np.sqrt((aux[:, j]**2).sum())
                if weight > realsmall:
                    if j == 0:
                        Q = (aux[:, j]/weight).reshape(aux.shape[0],-1)
                    else:
                        Q = np.hstack([Q, (aux[:, j]/weight).reshape(aux.shape[0],-1)])
                    for k in range(j+1, aux.shape[1]):
                        aux[:, k] -= Q[:, -1] * (Q[:, -1].conj().T @ aux[:, k])

            # More Gram-Schmidt
            for j in range(basis.shape[1]):
                Q -= np.outer(basis[:, j], basis[:, j]) @ Q
            Q = Q / np.sqrt((Q**2).sum(axis=0))
    err = np.linalg.qr(A(Q) - basis @ basis.conj().T @ A(Q))[1]

    return basis, Q, err

#------------------------------------------------------
# Value function reduction

def valuef_reduction(model, GAM0, GAM1, PSI, PI, C):

    n_jump_vars      = model.settings['n_jump_vars'].value
    n_state_vars_red = model.settings['n_state_vars_red'].value
    spline_grid      = model.settings['spline_grid'].value
    knots_dict       = model.settings['knots_dict'].value
    n_prior          = model.settings['n_prior'].value
    n_post           = model.settings['n_post'].value

    # Function calls to create basis reduction
    knots_dim        = sum([len(knots_dict[i])+1 for i in range(len(knots_dict.keys()))])
    spline_grid_dim  = np.prod(spline_grid.shape)
    from_spline      = lil_matrix(np.zeros((spline_grid_dim, knots_dim)), dtype=float)
    to_spline        = lil_matrix(np.zeros((knots_dim, spline_grid_dim)), dtype=float)

    # create spline basis
    from_spline[:,:], to_spline[:,:] = spline_basis(spline_grid, knots_dict)
    # extend to other dimensions along which we did not use a spline approximation.
    from_spline, to_spline = extend_to_nd(from_spline, to_spline, n_prior, n_post)
    # extra jump variables besides value function
    extra_jv = int(n_jump_vars - len(model.endog_states['value_function']))

    if extra_jv > 0:
        # Add additional dimensions for any jump variables we did not reduce b/c
        # spline reduction only for value function
        dim1_from, dim2_from = from_spline.shape
        from_spline = hstack([from_spline,
                              csc_matrix(np.zeros([dim1_from, extra_jv]))], format='csc')
        from_spline = vstack([from_spline,
                              csc_matrix(np.zeros((extra_jv, dim2_from+extra_jv)))], format='csc')

        to_spline   = hstack([to_spline,
                              csc_matrix(np.zeros((dim2_from, extra_jv)))], format='csc')
        to_spline   = vstack([to_spline,
                              csc_matrix(np.zeros((1, dim1_from+extra_jv)))], format='csc')

        for i in range(0,extra_jv):
            from_spline[-1-i, -1-i] = 1
            to_spline[-1-i, -1-i]   = 1

    n_splined = int(from_spline.shape[1])

    # Create projection matrix that projects value function onto spline basis
    from_spline, to_spline = projection_for_subset(from_spline, to_spline, 0, n_state_vars_red)

    GAM0_spl, GAM1_spl, PSI_spl, _, C_spl = \
            change_basis(to_spline, from_spline, GAM0, GAM1, PSI, PI, C, ignore_GAM0 = True)

    PI_spl = to_spline * PI * from_spline[:n_jump_vars, :n_splined]

    model.settings.update({'n_splined': type('setting', (object,),
                                       {'value': n_splined,
                                        'description':'Dimension of jump variables after spline basis reduction'})})
    return GAM0_spl, GAM1_spl, PSI_spl, PI_spl, C_spl, to_spline, from_spline

def spline_basis(x, knots_dict):
    knots = knots_dict[0]
    n_a = len(x)
    n_knots = len(knots)

    first_interp_mat = np.zeros([n_a, n_knots+1])
    aux_mat = np.zeros([n_a, n_knots])

    for i in range(n_a):
        loc = sum(knots <= x[i])
        if loc == n_knots:
            loc = n_knots - 1
        first_interp_mat[i, loc-1] = 1 - (x[i]-knots[loc-1])**2 / (knots[loc]-knots[loc-1])**2
        first_interp_mat[i, loc]   =     (x[i]-knots[loc-1])**2 / (knots[loc]-knots[loc-1])**2
        aux_mat[i, loc-1]          =     (x[i]-knots[loc-1]) - (x[i]-knots[loc-1])**2 / (knots[loc]-knots[loc-1])

    aux_mat2 = spdiag(np.ones(n_knots), offsets=0, shape=(n_knots, n_knots), format="csc") \
              +spdiag(np.ones(n_knots), offsets=1, shape=(n_knots, n_knots), format="csc")
    aux_mat2[-1,-1] = 0
    aux_mat2[n_knots-1,0] = 1
    aux_mat3 = spdiag(np.hstack([-2/np.diff(knots), 0.0]), offsets=0, shape=(n_knots, n_knots+1), format="csc") \
              +spdiag(np.hstack([ 2/np.diff(knots), 1.0]), offsets=1, shape=(n_knots, n_knots+1), format="csc")

    from_knots = csc_matrix(first_interp_mat)+csc_matrix(aux_mat)*(spsolve(aux_mat2, aux_mat3))
    to_knots = spsolve((from_knots.conj().T * from_knots), from_knots.conj().T) * speye(n_a, format="csc")

    return from_knots, to_knots

def extend_to_nd(from_small, to_small, n_prior, n_post):
    from_approx = spkron(spkron(speye(n_post), from_small), speye(n_prior))
    to_approx   = spkron(spkron(speye(n_post), to_small  ), speye(n_prior))

    return from_approx, to_approx

def projection_for_subset(from_small, to_small, n_pre, n_post):

    n_full, n_red = from_small.shape

    spdiag_internal = lambda n: (*spdiag(np.ones(n)).nonzero(), np.ones(n))
    I, J, V = spdiag_internal(n_red+n_pre)
    from_approx = csc_matrix((V, (I, J)), shape=(n_full+n_pre, n_pre+n_red))
    I, J, V = spdiag_internal(n_red+n_pre)
    to_approx   = csc_matrix((V, (I, J)), shape=(n_pre+n_red,  n_full+n_pre))

    from_approx[n_pre:n_pre+n_full, n_pre:n_pre+n_red] = from_small
    to_approx[n_pre:n_pre+n_red, n_pre:n_pre+n_full]   = to_small

    # Expand matrices and add needed values
    dim1_from_approx, dim2_from_approx = from_approx.shape
    from_approx = hstack([from_approx,
                          csc_matrix(np.zeros((dim1_from_approx, n_post)), dtype=float)], format='csc')
    from_approx = vstack([from_approx,
                          csc_matrix(np.zeros((n_post, dim2_from_approx+n_post)), dtype=float)], format='csc')

    to_approx   = hstack([to_approx,
                          csc_matrix(np.zeros((dim2_from_approx, n_post)), dtype=float)], format='csc')
    to_approx   = vstack([to_approx,
                          csc_matrix(np.zeros((n_post, dim1_from_approx+n_post)), dtype=float)], format='csc')
    from_approx[dim1_from_approx:, dim2_from_approx:] = speye(n_post)
    to_approx[dim2_from_approx:, dim1_from_approx:] = speye(n_post)

    return from_approx, to_approx


def change_basis(basis, inv_basis, GAM0, GAM1, PSI, PI, C, ignore_GAM0):

    g1 = basis @ GAM1 @ inv_basis

    if ignore_GAM0:
        g0 = speye(g1.shape[0], dtype=float, format='csc')
    else:
        g0 = basis @ GAM0 @ inv_basis

    c   = basis @ C
    Psi = basis @ PSI
    Pi  = basis @ PI

    return g0, g1, Psi, Pi, c
