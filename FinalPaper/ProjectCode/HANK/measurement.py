import numpy as np
from ProjectCode import util
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix

def measurement(model, G1, impact, C, inverse_basis):
    """
    Compute measurement equation
    Written based on NYFED-DSGE's Julia-language implementation.
    """
    
    endo = model.endog_states
    exo  = model.exog_shocks
    obs  = model.observables

    n_obs   = len(util.flatten_indices(obs))
    n_states = len(util.flatten_indices(endo))

    track_lag = model.settings['track_lag'].value
    freq      = model.settings['state_simulation_freq'].value

    ZZ = np.zeros([n_obs, n_states])
    DD = np.zeros(n_obs)
    #EE = np.zeros([n_obs, n_obs])
    EE = np.diag([0.20*0.579923,0.20*1.470832,0.20*2.237937])
    QQ = np.eye(freq) * 1/freq
   #QQ = np.eye(freq*exo) * 1/freq

    ZZ[obs['obs_gdp'], endo['output']] = 1.0
    ZZ[obs['obs_inflation'], endo['inflation']] = 4.0
    ZZ[obs['obs_nominalrate'], endo['monetary_policy']] = 4.0


    if track_lag:
        augmented_inverse_basis = np.zeros([n_states, (freq + 1) * G1.shape[0]])
        augmented_inverse_basis[:,-1 - inverse_basis.shape[1]+1:] = inverse_basis.toarray()
        ZZ = ZZ @ augmented_inverse_basis
    else:
        augmented_inverse_basis = np.zeros([n_states, freq * G1.shape[0]])
        augmented_inverse_basis[:,-1 - inverse_basis.shape[1]+1:] = inverse_basis.toarray()

        ZZ = ZZ @ augmented_inverse_basis

    return ZZ, DD, QQ, EE
