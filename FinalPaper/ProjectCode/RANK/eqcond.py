import numpy as np
from ProjectCode import util


def eqcond(model):
    """
    Compute equilibirum of RANK model
    Written based on NYFED-DSGE's Julia-language implementation.
    """

    endo = model.endog_states
    exo  = model.exog_shocks
    ex   = model.expected_shocks
    eq   = model.equilibrium_conditions

    n_states      = len(util.flatten_indices(model.endog_states))
    n_shocks_exog = len(util.flatten_indices(model.exog_shocks))
    n_shocks_exp  = len(util.flatten_indices(model.expected_shocks))

    GAM0 = np.zeros([n_states, n_states])
    GAM1 = np.zeros([n_states, n_states])
    C    = np.zeros(n_states)
    PSI  = np.zeros([n_states, n_shocks_exog])
    PI   = np.zeros([n_states, n_shocks_exp])

    #--- Endogenous States ---------------------------------------------------
    # 1. Consumption Euler Equation
    GAM0[eq['eq_euler'], endo['y_t']]    = 1.0
    GAM0[eq['eq_euler'], endo['R_t']]    = 1/model.params['tau'].value
    GAM0[eq['eq_euler'], endo['g_t']]    = -(1-model.params['rho_g'].value)
    GAM0[eq['eq_euler'], endo['z_t']]    = -model.params['rho_z'].value/model.params['tau'].value
    GAM0[eq['eq_euler'], endo['Ey_t1']]  = -1.0
    GAM0[eq['eq_euler'], endo['Epi_t1']] = -1/model.params['tau'].value

    # 2. NK Phillips Curve
    GAM0[eq['eq_phillips'], endo['y_t']]    = -model.params['kappa'].value
    GAM0[eq['eq_phillips'], endo['pi_t']]   = 1.0
    GAM0[eq['eq_phillips'], endo['g_t']]    = model.params['kappa'].value
    GAM0[eq['eq_phillips'], endo['Epi_t1']] = -1/(1+model.params['rA'].value/400)

    # 3. Monetary Policy Rule
    GAM0[eq['eq_mp'], endo['y_t']]   = -(1-model.params['rho_R'].value) * model.params['psi_2'].value
    GAM0[eq['eq_mp'], endo['pi_t']]  = -(1-model.params['rho_R'].value) * model.params['psi_1'].value
    GAM0[eq['eq_mp'], endo['R_t']]   = 1.0
    GAM0[eq['eq_mp'], endo['g_t']]   =  (1-model.params['rho_R'].value) * model.params['psi_2'].value
    GAM1[eq['eq_mp'], endo['R_t']]   = model.params['rho_R'].value
    PSI[eq['eq_mp'],  exo['rm_sh']]  = 1.0

    # 4. Output lag
    GAM0[eq['eq_y_t1'], endo['y_t1']] = 1.0
    GAM1[eq['eq_y_t1'], endo['y_t']]  = 1.0

    # 5. Government spending
    GAM0[eq['eq_g'], endo['g_t']] = 1.0
    GAM1[eq['eq_g'], endo['g_t']] = model.params['rho_g'].value
    PSI[eq['eq_g'], exo['g_sh']]  = 1.0

    # 6. Technology
    GAM0[eq['eq_z'], endo['z_t']] = 1.0
    GAM1[eq['eq_z'], endo['z_t']] = model.params['rho_z'].value
    PSI[eq['eq_z'], exo['z_sh']]  = 1.0

    # 7. Expected output
    GAM0[eq['eq_Ey'], endo['y_t']]   = 1.0
    GAM1[eq['eq_Ey'], endo['Ey_t1']] = 1.0
    PI[eq['eq_Ey'], ex['Ey_sh']]     = 1.0

    # 8. Expected inflation
    GAM0[eq['eq_Epi'], endo['pi_t']]   = 1.0
    GAM1[eq['eq_Epi'], endo['Epi_t1']] = 1.0
    PI[eq['eq_Epi'], ex['Epi_sh']]     = 1.0

    return GAM0, GAM1, C, PSI, PI
