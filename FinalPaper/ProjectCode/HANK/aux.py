import numpy as np
from scipy.sparse import identity as speye
from scipy.sparse import diags as spdiag
from scipy.sparse import kron as spkron
from scipy.sparse import spmatrix
from scipy.sparse.linalg import lsqr as sp_lsqr
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix

def construct_asset_grid(I, grid_param, grid_min, grid_max):
    a = np.linspace(0, 1, I)
    a = a**(1/grid_param)
    a = grid_min + (grid_max - grid_min)*a

    return a

def compute_stationary_income_distribution(P, n_income_states, iter_num=50):
    Pt = P.conj().T
    g_z = np.tile(1/n_income_states, n_income_states)
    for n in range(iter_num):
        g_z_new = np.linalg.solve((speye(n_income_states) - Pt * 1000), g_z)
        diff = np.max(abs(g_z_new - g_z))
        if diff < 1e-5:
            break
        g_z = g_z_new

    return g_z

def construct_labor_income_grid(initial_ygrid, income_distr, meanlabeff, n_gridpoints):
    z = np.exp(initial_ygrid)
    z_bar = z @ income_distr
    z = (meanlabeff/z_bar) * z
    zz = np.outer(np.ones([n_gridpoints, 1]), z)
    return zz

def construct_household_problem_functions(V, w, coefrra,frisch, labtax, labdisutil):

    def util(c, h):
        f = lambda x: np.log(x) if coefrra == 1.0 else x**(1-coefrra) / (1-coefrra)
        return f(c) - labdisutil * (h**(1+1/frisch)/(1+1/frisch))

    income = lambda h, z, profshare, lumptransfer, r, a: \
                            h * z * w * (1 - labtax) + lumptransfer + profshare + r * a
    labor  = lambda z, val:  (z * w * (1 - labtax) * val / labdisutil)**frisch

    return util, income, labor

def initialize_diff_grids(a, I, J):
#     daf    = np.empty_like(a) # forward difference for a
#     dab    = np.empty_like(a) # backward difference for a
    adelta = np.empty_like(a) # size of differences

    daf = np.array([a[-1]-a[-2] if i == I-1 else a[i+1]-a[i] for i in range(I)])
    dab = np.array([a[1]-a[0] if i == 0 else a[i]-a[i-1] for i in range(I)])

    adelta[0]  = 0.5 * daf[0]
    adelta[-1] = 0.5 * daf[-2]
    adelta[1:-1]  = 0.5 * (daf[:-2] + daf[1:-1])

#     for i in range(I):
#         # Create a grid of lengths of overlapping intervals in a dimension.
#         # The purpose is generally to compute Riemann integrals
#         # by taking midpoint Riemann sums and dividing by two to adjust for
#         # or average out the added Lebesgue measure given by using overlapping intervals.
#         daf[i] = a[-1] - a[-2] if i==I-1 else a[i+1] - a[i]
#         dab[i] = a[1] - a[0] if i==0 else a[i] - a[i-1]
#         if i==0:
#             adelta[0]  = 0.5 * daf[0]
#         elif i==I-1:
#             adelta[-1] = 0.5 * daf[-2]
#         else:
#             adelta[i]  = 0.5 * (daf[i-1] + daf[i])
    azdelta = np.tile(adelta, J)

    return daf, dab, azdelta

def construct_initial_diff_matrices(V:np.ndarray, Vaf:np.ndarray, Vab:np.ndarray,
                                    income:object, labor:object, h:np.ndarray, h0:np.ndarray,
                                    zz:np.ndarray, profshare:np.ndarray, lumptransfer:float,
                                    amax:float, amin:float, coefrra:float,r:float,daf:np.ndarray,
                                    dab:np.ndarray, maxhours:np.ndarray):
    I, J = V.shape
    #cf = np.empty_like(V)
    hf = np.empty_like(V)
    #cb = np.empty_like(V)
    hb = np.empty_like(V)

#     # foward difference
#     Vaf[:-1] = (V[1:] - V[:-1])/np.array([daf[:-1],daf[:-1]]).reshape(I-1,J)
#     Vaf[-1]  = income(h0[-1], zz[-1], profshare[-1], lumptransfer, r, amin)**(-coefrra)

#     Vab[1:] = (V[1:] - V[:-1]) / np.array([dab[1:],dab[1:]]).reshape(I-1,J)
#     Vab[0]  = income(h0[0], zz[0], profshare[0], lumptransfer, r, amin)**(-coefrra)

    Vaf[-1,:] = income(h[-1,:], zz[-1,:], profshare[-1,:], lumptransfer, r, amax)**(-coefrra)
    Vab[-1,:] = (V[-1,:] - V[-2,:]) / dab[-1]

    Vaf[0,:]  = (V[1,:] - V[0,:]) / daf[0]
    Vab[0,:]  = income(h0[0,:], zz[0,:], profshare[0,:], lumptransfer, r, amin)**(-coefrra)

    Vaf[1:-1] = (V[2:,:] - V[1:-1,:]) / daf[0]
    Vab[1:-1] = (V[1:-1,:] - V[:-2,:]) / dab[0]


    idx = ((i,j) for i in range(I)  for j in range(J))
    for t in idx:
        # if t[0]==I-1:
        #     Vaf[t] = income(h[t], zz[t], profshare[t], lumptransfer, r, amax)**(-coefrra)
        #     Vab[t] = (V[t] - V[t[0]-1, t[1]]) / dab[t[0]]
        # elif t[0]==0:
        #     Vaf[t] = (V[t[0]+1, t[1]] - V[t]) / daf[t[0]]
        #     Vab[t] = income(h0[t], zz[t], profshare[t], lumptransfer, r, amin)**(-coefrra)
        # else:
        #     Vaf[t] = (V[t[0]+1, t[1]] - V[t]) / daf[t[0]]
        #     Vab[t] = (V[t] - V[t[0]-1, t[1]]) / dab[t[0]]
        #
        # cf[t] = Vaf[t]**(-1/coefrra)
        # cb[t] = Vab[t]**(-1/coefrra)

        hf[t] = min(abs(labor(zz[t], Vaf[t])), maxhours)
        hb[t] = min(abs(labor(zz[t], Vab[t])), maxhours)

    # hf2 = np.minimum(abs(labor(zz, Vaf)), maxhours)
    # #hb2 = np.minimum(abs(labor(zz, Vab)), maxhours)
    # hf2 = convert_complex(hf2)
    # #hb2 = convert_complex(hb2)
    # print(hf- hf2)

    cf = Vaf**(-1/coefrra)
    cb = Vab**(-1/coefrra)

    # hf2 = np.minimum(abs(labor(zz.flatten(), Vaf.flatten())), maxhours).reshape(I,J)
    # hb2 = np.minimum(abs(labor(zz.flatten(), Vab.flatten())), maxhours).reshape(I,J)

    return Vaf, Vab, cf, hf, cb, hb

##
def hours_iteration(income:object, labor:object,
                    zz:np.ndarray, profshare:np.ndarray, lumptransfer:float,
                    aa:np.ndarray, coefrra:float, r:float,
                    cf:np.ndarray, hf:np.ndarray, cb:np.ndarray, hb:np.ndarray,
                    c0:np.ndarray, h0:np.ndarray, maxhours:float, niter_hours:int):
    I, J = zz.shape

    idx = ((i,j) for i in range(I) for j in range(J))
    for _ in range(niter_hours):
        for t in idx:
            if t[0]==I-1:
                cf[t] = income(hf[t], zz[t], profshare[t], lumptransfer, r, aa[t])
                hf[t] = labor(zz[t], cf[t]**(-coefrra))
                hf[t] = min(abs(hf[t]), maxhours)
            elif t[0]==0:
                cb[t] = income(hb[t], zz[t], profshare[t], lumptransfer, r, aa[t])
                hb[t] = labor(zz[t], cb[t]**(-coefrra))
                hb[t] = min(abs(hb[t]), maxhours)
#             c0[t] = income(h0[t], zz[t], profshare[t], lumptransfer, r, aa[t])
#             h0[t] = labor(zz[t], c0[t]**(-coefrra))
#             h0[t] = min(np.linalg.norm(h0[t]), maxhours)
        c0 = income(h0, zz, profshare, lumptransfer, r, aa)
        h0 = labor(zz, c0**(-coefrra))
        h0 = np.minimum(abs(h0.flatten()), maxhours).reshape(I,J)
    return cf, hf, cb, hb, c0, h0

def upwind(rho:float, V:np.ndarray, util:object, A_switch:spmatrix,
           cf:np.ndarray, cb:np.ndarray, c0:np.ndarray, hf:np.ndarray,
           hb:np.ndarray, h0:np.ndarray, sf:np.ndarray, sb:np.ndarray,
           Vaf:np.ndarray, Vab:np.ndarray, daf:np.ndarray, dab:np.ndarray,
           d_HJB:float = 1e6):

    T = sb[0,0].dtype
    I,J = sb.shape
#     h = np.empty_like(sb)
#     c = np.empty_like(sb)
#     s = np.empty_like(sb)
#     u = np.empty_like(sb)
#     X = np.empty_like(sb)
#     Z = np.empty_like(sb)
#     Y = np.empty_like(sb)

    Vf = (cf > 0) * (util(cf, hf) + sf * Vaf) + (cf <= 0) * (-1e12)
    Vb = (cb > 0) * (util(cb, hb) + sb * Vab) + (cb <= 0) * (-1e12)
    V0 = (c0 > 0) * util(c0, h0) + (c0 <= 0) * (-1e12)

    Iunique = (sb < 0) * (1 - (sf > 0)) + (1 - (sb < 0)) * (sf > 0)
    Iboth = (sb < 0) * (sf > 0)
    Ib = Iunique * (sb < 0) * (Vb > V0) + Iboth * (Vb == np.maximum(np.maximum(Vb, Vf), V0))
    If = Iunique * (sf > 0) * (Vf > V0) + Iboth * (Vf == np.maximum(np.maximum(Vb, Vf), V0))
    I0 = 1 - Ib - If

    h = hf * If + hb * Ib + h0 * I0
    c = cf * If + cb * Ib + c0 * I0
    s = sf * If + sb * Ib
    u = util(c, h)

    X = -Ib * sb / np.array([dab,dab]).reshape(I,J)
    Z = If * sf / np.array([daf,daf]).reshape(I,J)
    Y = -Z - X

    X[0,:]   = complex(0.) if T == np.complex else 0
    Z[I-1,:] = complex(0.) if T == np.complex else 0

    A = spdiag([X.reshape(I*J, order='F')[1:],
                Y.reshape(I*J, order='F'),
                Z.reshape(I*J, order='F')[:I*J-1]],
                offsets=[-1, 0, 1], shape=(I*J, I*J)) + A_switch

    I, J = u.shape
    B    = (1 / d_HJB + rho) * speye(I*J, dtype=T) - A
    b    = u.reshape(I*J,order='F') + V.reshape(I*J,order='F') / d_HJB

    V = spsolve(B, b).reshape(I, J, order='F')
    #V = scipy.sparse.linalg.lsqr(B,b)[0].reshape(I, J,order='F')

    return V, A, u, h, c, s

def solve_kfe(A:spmatrix, g0:np.ndarray, weight_mat:spmatrix,
              maxit_kfe:int=1000, tol_kfe:float=1e-12,
              d_kfe:float=1e6):

    if weight_mat.shape != A.shape:
        raise Exception('Dimension of weight matrix is incorrect.')

    weight_mat = csr_matrix(weight_mat)
    dim_size = A.shape[0]
    gg = g0.flatten(order='F') # stack distribution matrix into vector

    # Solve linear system
    for ikfe in range(maxit_kfe):
        gg_tilde = weight_mat @ gg # weight distribution points by their measure across wealth
        gg1_tilde = spsolve((speye(dim_size, dtype=np.complex) - d_kfe * A.conj().T), gg_tilde)

        gg1_tilde = gg1_tilde / gg1_tilde.sum()

        gg1 = spsolve(weight_mat, gg1_tilde)

        # Check iteration for convergence
        err_kfe = max(abs(gg1-gg))
        if err_kfe < tol_kfe:
            #print('converged!')
            break
        gg = gg1

    return gg.flatten(order='F')

def calculate_ss_equil_vars_init(zz:np.ndarray, m_ss:float, meanlabeff:float,
                                 lumptransferpc:float, govbondtarget:float):

    N_ss         = np.complex(1/3) # steady state hours: so that quarterly GDP = 1 in s.s
    Y_ss         = np.complex(1.)
    B_ss         = govbondtarget * Y_ss
    profit_ss    = np.complex((1 - m_ss) * Y_ss)
    profshare    = zz / meanlabeff * profit_ss
    lumptransfer = np.complex(lumptransferpc * Y_ss)

    return N_ss, Y_ss, B_ss, profit_ss, profshare, lumptransfer


def calculate_ss_equil_vars(zz:np.ndarray, h:np.ndarray, g:np.ndarray,
                            azdelta:np.ndarray, aa:np.ndarray, m_ss:float,
                            meanlabeff:float, lumptransferpc:float,
                            govbondtarget:float):
    # equilibrium objects
    Y_ss = N_ss  = sum(zz.flatten(order='F') * h.flatten(order='F') * g * azdelta)
    Y_ss         = N_ss
    B_ss         = sum(g * aa.flatten(order='F') * azdelta)
    profit_ss    = (1 - m_ss) * Y_ss
    profshare    = zz / meanlabeff * profit_ss
    lumptransfer = lumptransferpc * Y_ss
    bond_err     = B_ss / Y_ss - govbondtarget

    return N_ss, Y_ss, B_ss, profit_ss, profshare, lumptransfer, bond_err


def check_bond_market_clearing(bond_err:np.complex, crit_S:float, r:float,
                               r_min:float, r_max:float, r_rho:float, rho_min:float,
                               rho_max:float, iter_r:bool, iter_rho:bool):
    clearing_condition = False
    # Using the market clearing condition on bonds to determine whether or not
    # an equilibrium has been reached
    if abs(bond_err) > crit_S:
        if bond_err > 0:
            if iter_r:
                r_max  = r
                r      = 0.5 * (r + r_min)
            elif iter_rho:
                rho_min = r_rho
                r_rho   = 0.5 * (r_rho + rho_max)
        else:
            if iter_r:
                r_min  = r
                r      = 0.5 * (r + r_max)
            elif iter_rho:
                rho_max = r_rho
                r_rho   = 0.5 * (r_rho + rho_min)
    else:
        clearing_condition = True

    return r, r_min, r_max, r_rho, rho_min, rho_max, clearing_condition


def convert_complex(M):
    Mvec = M.flatten()
    return np.array(list(map(np.complex, Mvec))).reshape(M.shape)
