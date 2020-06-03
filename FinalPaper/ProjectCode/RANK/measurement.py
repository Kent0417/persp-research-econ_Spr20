import numpy as np
from ProjectCode import util

def measurement(model, G1, impact, C):
    """
    Compute measurement equation
    Written based on NYFED-DSGE's Julia-language implementation.
    """
    
    endo = model.endog_states
    exo  = model.exog_shocks
    obs  = model.observables

    n_obs         = len(util.flatten_indices(obs))
    n_shocks_exo  = len(util.flatten_indices(exo))
    n_states      = len(util.flatten_indices(endo))

    ZZ = np.zeros([n_obs, n_states])
    DD = np.zeros(n_obs)
    EE = np.zeros([n_obs, n_obs])
    QQ = np.zeros([n_shocks_exo, n_shocks_exo])

    ## Output growth
    ZZ[obs['obs_gdp'], endo['y_t']]  = 1.0
    ZZ[obs['obs_gdp'], endo['y_t1']] = -1.0
    ZZ[obs['obs_gdp'], endo['z_t']]  = 1.0
    DD[obs['obs_gdp']]               = model.params['gam_Q'].value

    ## Inflation
    ZZ[obs['obs_inflation'], endo['pi_t']] = 4.0
    DD[obs['obs_inflation']]               = model.params['pi'].value

    ## FF rate
    ZZ[obs['obs_nominalrate'], endo['R_t']] = 4.0
    DD[obs['obs_nominalrate']]             = model.params['pi'].value + model.params['rA'].value \
                                             + 4.0 * model.params['gam_Q'].value

    ## Measurement error
    EE[obs['obs_gdp'], endo['y_t']]         = model.params['e_y'].value ** 2
    EE[obs['obs_inflation'], endo['pi_t']]  = model.params['e_pi'].value ** 2
    EE[obs['obs_nominalrate'], endo['R_t']] = model.params['e_R'].value ** 2

    ## Variance of innovations
    QQ[exo['z_sh'], exo['z_sh']]   = model.params['sig_z'].value ** 2
    QQ[exo['g_sh'], exo['g_sh']]   = model.params['sig_g'].value ** 2
    QQ[exo['rm_sh'], exo['rm_sh']] = model.params['sig_R'].value ** 2

    return ZZ, DD, QQ, EE
