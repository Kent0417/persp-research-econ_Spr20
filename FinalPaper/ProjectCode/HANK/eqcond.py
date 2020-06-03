from ProjectCode.HANK import aux
import numpy as np
from scipy.sparse import identity as speye
from scipy.sparse import diags as spdiag
from scipy.sparse import kron as spkron
from scipy.sparse import spmatrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
from ProjectCode import util


def eqcond(model):
    """
    Compute equilibirum of HANK model
    Written based on NYFED-DSGE's Julia-language implementation.
    """
    nstates = len(util.flatten_indices(model.endog_states))     # 406
    n_s_exp = len(util.flatten_indices(model.expected_shocks))  # 201
    n_s_exo = len(util.flatten_indices(model.exog_shocks))      # 1

    x       = np.zeros(2 * nstates + n_s_exp + n_s_exo)

    # Read in parameters, settings
    niter_hours       = model.settings['niter_hours'].value
    n_v               = model.settings['n_jump_vars'].value
    n_g               = model.settings['n_state_vars'].value
    ymarkov_combined  = model.settings['ymarkov_combined'].value
    zz                = model.settings['zz'].value
    a                 = model.settings['a'].value
    I, J              = zz.shape
    amax              = model.settings['amax'].value
    amin              = model.settings['amin'].value
    aa                = np.repeat(a.reshape(-1,1), J, axis=1)
    A_switch          = spkron(ymarkov_combined, speye(I, dtype=float, format='csc'))
    daf, dab, azdelta = aux.initialize_diff_grids(a, I, J)

    # Set up steady state parameters
    V_ss     = model.steady_state['V_ss'].value
    g_ss     = model.steady_state['g_ss'].value
    G_ss     = model.steady_state['G_ss'].value
    w_ss     = model.steady_state['w_ss'].value
    N_ss     = model.steady_state['N_ss'].value
    C_ss     = model.steady_state['C_ss'].value
    Y_ss     = model.steady_state['Y_ss'].value
    B_ss     = model.steady_state['B_ss'].value
    r_ss     = model.steady_state['r_ss'].value
    rho_ss   = model.steady_state['rho_ss'].value
    sig_MP   = model.params['sig_MP'].value
    theta_MP = model.params['theta_MP'].value
    # sig_FP   = model.params['sig_FP'].value
    # theta_FP = model.params['theta_FP'].value
    # sig_PS   = model.params['sig_PS'].value
    # theta_PS = model.params['theta_PS'].value

    h_ss              = model.steady_state['h_ss'].value.reshape(I, J, order='F')
    ceselast          = model.params['ceselast'].value
    inflation_ss      = model.steady_state['inflation_ss'].value
    maxhours          = model.params['maxhours'].value
    govbcrule_fixnomB = model.params['govbcrule_fixnomB'].value
    priceadjust       = model.params['priceadjust'].value
    taylor_inflation  = model.params['taylor_inflation'].value
    taylor_outputgap  = model.params['taylor_outputgap'].value
    meanlabeff        = model.params['meanlabeff'].value

    # Necessary for construction of household problem functions
    labtax         = model.params['labtax'].value
    coefrra        = model.params['coefrra'].value
    frisch         = model.params['frisch'].value
    labdisutil     = model.params['labdisutil'].value

    TFP  = 1.0
    def _get_residuals(x):
        #x = x.toarray().flatten()
        # Prepare steady state deviations
        V           = (x[:n_v - 1] + V_ss).reshape(I, J, order='F')
        inflation   = x[n_v - 1]             + inflation_ss
        gg          = x[n_v : n_v + n_g - 1] + g_ss[:-1]
        MP          = x[n_v + n_g - 1]
        w           = x[n_v + n_g]           + w_ss
        hours       = x[n_v + n_g + 1]       + N_ss
        consumption = x[n_v + n_g + 2]       + C_ss
        output      = x[n_v + n_g + 3]       + Y_ss
        assets      = x[n_v + n_g + 4]       + B_ss
       #government  = x[n_v + n_g + 5]       + G_ss


        V_dot           = x[nstates            : nstates + n_v - 1]
        inflation_dot   = x[nstates + n_v - 1]
        g_dot           = x[nstates + n_v      : nstates + n_v + n_g - 1]
        mp_dot          = x[nstates + n_g + n_v - 1]
       #fp_dot          = x[nstates + n_g + n_v + 5]
       #ps_dot          = x[nstates + n_g + n_v + 3]
        VEErrors        = x[2*nstates          : 2*nstates + n_v - 1]
        inflation_error = x[2*nstates + n_v - 1]
        mp_shock        = x[2*nstates + n_v]
       #fp_shock        = x[2*nstates + n_v + 1]
       #ps_shock        = x[2*nstates + n_v + 2]


        g_end = (1 - gg @ azdelta[:-1]) / azdelta[-1]
        g     = np.append(gg, g_end)
        g[g < 1e-19] = 0.0

        #-----------------------------------------------------------------
        # Get equilibrium values, given steady state values
        normalized_wage = w / TFP
        profshare = (zz / meanlabeff) * ((1.0 - normalized_wage) * output)
        r_nominal = r_ss + taylor_inflation * inflation \
                    + taylor_outputgap * (np.log(output)-np.log(Y_ss)) + MP
        r         = r_nominal - inflation
        lumptransfer    = labtax * w * hours - G_ss - \
                            (r_nominal - (1-govbcrule_fixnomB) * inflation) * assets

        #-----------------------------------------------------------------
        # Compute one iteration of the HJB
        ## Get flow utility, income, and labor hour functions
        util, income, labor = \
            aux.construct_household_problem_functions(V, w, coefrra, frisch, labtax, labdisutil)

        ## Initialize other variables, using V to ensure everything is a dual number
        Vaf = np.copy(V)
        Vab = np.copy(V)
        #h0 = np.array([h_ss[i, j] for i in range(I) for j in range(J)]).reshape(I, J)
        h0  = np.copy(h_ss)

        #-----------------------------------------------------------------
        # Construct Initial Difference Matrices
#         hf = np.empty_like(V)
#         hb = np.empty_like(V)

        Vaf[-1,:] = income(h_ss[-1,:], zz[-1,:], profshare[-1,:], lumptransfer, r, amax)**(-coefrra)
        Vab[-1,:] = (V[-1,:] - V[-2,:]) / dab[-1]

        Vaf[0,:]  = (V[1,:] - V[0,:]) / daf[0]
        Vab[0,:]  = income(h_ss[0,:], zz[0,:], profshare[0,:], lumptransfer, r, amin)**(-coefrra)

        Vaf[1:-1] = (V[2:,:] - V[1:-1,:]) / daf[0]
        Vab[1:-1] = (V[1:-1,:] - V[:-2,:]) / dab[0]

        # idx = ((i,j) for i in range(I) for j in range(J))
        # for t in idx:
        #     hf[t] = min(abs(labor(zz[t], Vaf[t])), maxhours)
        #     hb[t] = min(abs(labor(zz[t], Vab[t])), maxhours)

        hf = np.minimum(abs(labor(zz, Vaf)), maxhours)
        hb = np.minimum(abs(labor(zz, Vab)), maxhours)

        cf = Vaf**(-1/coefrra)
        cb = Vab**(-1/coefrra)

        #-----------------------------------------------------------------
        # Hours Iteration
        idx = ((i,j) for i in range(I) for j in range(J))
        for ih in range(niter_hours):
            for t in idx:
                if t[0]==I-1:
                    cf[t] = income(hf[t], zz[t], profshare[t], lumptransfer, r, aa[t])
                    hf[t] = labor(zz[t], cf[t]**(-coefrra))
                    hf[t] = min(abs(hf[t]), maxhours)
                    if ih == niter_hours-1:
                        Vaf[t] = cf[t]**(-coefrra)

                elif t[0]==0:
                    cb[t] = income(hb[t], zz[t], profshare[t], lumptransfer, r, aa[t])
                    hb[t] = labor(zz[t], cb[t]**(-coefrra))
                    hb[t] = min(abs(hb[t]), maxhours)
                    if ih == niter_hours:
                        Vab[t] = cb[t]**(-coefrra)

            c0 = income(h0, zz, profshare, lumptransfer, r, aa)
            h0 = labor(zz, c0**(-coefrra))
            h0 = np.minimum(abs(h0.flatten()), maxhours).reshape(I,J)

        c0 = income(h0, zz, profshare, lumptransfer, r, aa)
        sf = income(hf, zz, profshare, lumptransfer, r, aa) - cf
        sb = income(hb, zz, profshare, lumptransfer, r, aa) - cb

        #-----------------------------------------------------------------
        # Upwind
        #T = sb[0,0].dtype

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

        X[0,  :] = 0.
        Z[I-1,:] = 0.

        A = spdiag([X.reshape(I*J, order='F')[1:],
                    Y.reshape(I*J, order='F'),
                    Z.reshape(I*J, order='F')[:I*J-1]],
                    offsets=[-1, 0, 1], shape=(I*J, I*J),format='csc') + A_switch

        #-----------------------------------------------------------------
        # Collect/calculate Residuals
        hjb_residual = u.flatten(order='F') + A * V.flatten(order='F') \
                        + V_dot + VEErrors  - rho_ss * V.flatten(order='F')

        pc_residual  = -((r - 0) * inflation - (ceselast / priceadjust * \
                                                (w / TFP - (ceselast-1) / ceselast) + \
                                                inflation_dot - inflation_error))

        g_azdelta      = g.flatten() * azdelta.flatten()
        g_intermediate = spdiag(1 / azdelta) * A.T @ g_azdelta
        g_residual     = g_dot - g_intermediate[:-1]

        mp_residual    = mp_dot - (-theta_MP * MP + sig_MP * mp_shock)

        realsav        = sum(aa.flatten(order='F') * g.flatten(order='F') * azdelta.flatten(order='F'))
        realsav_dot    = sum(s.flatten(order='F') * g.flatten(order='F') * azdelta.flatten(order='F'))
        bondmarket_residual  = realsav_dot/realsav + govbcrule_fixnomB * inflation

        labmarket_residual = sum(zz.flatten(order='F') * h.flatten(order='F') \
                                 * g.flatten(order='F') * azdelta.flatten(order='F')) - hours

        consumption_residual = sum(c.flatten(order='F') * g.flatten(order='F') \
                                   * azdelta.flatten(order='F')) - consumption

        output_residual = TFP * hours - output
       #output_residual = TFP * hours - (-theta_PS * output + sig_PS * ps_shock)


        assets_residual = assets - realsav

       #government_residual = fp_dot - (-theta_FP * government + sig_FP * fp_shock)

        # Return equilibrium conditions
        return np.hstack((hjb_residual, pc_residual, g_residual, mp_residual, bondmarket_residual,
                         labmarket_residual, consumption_residual, output_residual, assets_residual))
        # return np.hstack((hjb_residual, pc_residual, g_residual,
        #                  mp_residual,
        #                  bondmarket_residual, labmarket_residual, consumption_residual,
        #                  output_residual, assets_residual, government_residual))


    derivs = jacob_mat(_get_residuals, x, nstates, h=1e-10)

    GAM1 = -derivs[:,                     :   nstates]
    GAM0 =  derivs[:,   nstates           : 2*nstates]
    PI   = -derivs[:, 2*nstates           : 2*nstates + n_s_exp]
    PSI  = -derivs[:, 2*nstates + n_s_exp : 2*nstates + n_s_exp + n_s_exo]
    C    = np.zeros(nstates)

    return GAM0, GAM1, PSI, PI, C


def jacob_mat(f, x, nstates, h=1e-4):
    X  = np.eye(len(x))
    Xf = X*h
    Xb = X*-h
    forward  = np.array(list(map(f, Xf)))
    backward = np.array(list(map(f, Xb)))
    J = (forward-backward)/(2*h)

    return J.T
