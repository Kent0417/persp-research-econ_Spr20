# Written based on NYFED-DSGE's Julia-language implementation.

import numpy as np
from scipy.sparse import identity as speye
from scipy.sparse import diags as spdiag
from scipy.sparse import spmatrix
from scipy.sparse import kron as spkron
from scipy.sparse.linalg import lsqr as sp_lsqr
from scipy.sparse.linalg import spsolve

from dataclasses import dataclass
from collections import OrderedDict
import warnings

from tqdm import tqdm
import time

from . import aux
from ProjectCode import dist
from ProjectCode import util


warnings.simplefilter('ignore', np.ComplexWarning)

@dataclass()
class OneAssetHANK(object):
    model_type             : int
    params                 : OrderedDict
    steady_state           : OrderedDict
    keys                   : OrderedDict
    endog_states           : OrderedDict
    exog_shocks            : OrderedDict
    expected_shocks        : OrderedDict
    equilibrium_conditions : OrderedDict
    endog_states_augmented : OrderedDict
    observables            : OrderedDict
    observables_mappings   : OrderedDict
    settings               : dict

def init_model_indices(model):
    # Endogenous states
    endog_states = ['value_function','inflation', 'distribution',
                    'monetary_policy', 'w', 'N', 'C', 'output', 'B']

    # Exogenous shocks
    exog_shocks = ['mp_shock']
   #exog_shocks = ['mp_shock', 'fp_shock', output_shock]


    # Expectations shocks
    expected_shocks  = ['E_V','E_pi']

    # Equilibrium conditions
    equilibrium_conditions = ['value_function','inflation', 'distribution',
                              'monetary_policy', 'w', 'N', 'C', 'Y', 'B']

    # Additional states added after solving model
    # Lagged states and obvervables measurement error
    endog_states_augmented = []

    # Observables
    observables = model.observables_mappings.keys()

    # Pseudo-observables
    #pseudo_observables = model.pseudo_observable_mappings.keys()

    # Assign dimensions
    model.endog_states.update({'value_function' : np.arange(200)})   ## 0-199
    model.endog_states.update({'inflation' : 200})                   ## 200
    model.endog_states.update({'distribution' : np.arange(201,400)}) ## 201-399
    model.endog_states.update({'monetary_policy' : 400})             ## 400/monetary policy shock?
    model.endog_states.update({'w' : 401})                           ## 401
    model.endog_states.update({'N' : 402})                           ## 402
    model.endog_states.update({'C' : 403})                           ## 403
    model.endog_states.update({'output' : 404})                      ## 404
    model.endog_states.update({'B' : 405})                           ## 405
   #model.endog_states.update({'G' : 406})                           ## 406
    model.expected_shocks.update({'E_V': model.endog_states['value_function']})
    model.expected_shocks.update({'E_pi': model.endog_states['inflation']})

    for i, k in enumerate(endog_states_augmented):
        model.endog_states_augmented[k] = i + len(endog_states)
    for i, k in enumerate(observables):
        model.observables[k] = i
    for i, k in enumerate(exog_shocks):
        model.exog_shocks[k] = i
    for i, k in enumerate(observables):
        model.observables[k] = i


def init_parameters(model):
    # Initialize parameters
    model.params.update({'coefrra': type('params', (object,),
                                        {'value':1.0,
                                        'valuebounds': (1e-20, 1e5),
                                        'transform_parameterization':(1e-20, 1e5),
                                        'transform': "Exponential",
                                        'prior':dist.norm_dist(1.0,0.1),
                                         'description':'Relative risk aversion',
                                         'tex_label':r'coefrra',
                                         'fixed': True})})


    model.params.update({'frisch': type('params', (object,),
                                       {'value':0.5,
                                       'valuebounds': (1e-20, 1e5),
                                       'transform_parameterization':(1e-20, 1e5),
                                       'transform': "Exponential",
                                       'prior':dist.norm_dist(0.5,0.1),
                                        'description':'Frisch elasticity',
                                        'tex_label':r'frisch',
                                        'fixed': False})})

    model.params.update({'meanlabeff': type('params', (object,),
                                           {'value':3.0,
                                            'description':'meanlabeff: so that at h=1/3 output will be approximately = 1',
                                            'tex_label':r'meanlabeff',
                                            'fixed': True})})

    model.params.update({'maxhours': type('params', (object,),
                                          {'value':1.0,
                                           'description':'maxhours',
                                           'tex_label':r'maxhours',
                                           'fixed': True})})

    model.params.update({'ceselast': type('params', (object,),
                                          {'value':10.,
                                           'description':'CES elasticity',
                                           'tex_label':r'ceselast',
                                           'fixed': True})})

    model.params.update({'priceadjust': type('params', (object,),
                                        {'value':100.,
                                        'valuebounds': (1e-20, 1e5),
                                        'transform_parameterization':(1e-20, 1e5),
                                        'transform': "Exponential",
                                        'prior':dist.norm_dist(100,10),
                                         'description':'priceadjust..',
                                         'tex_label':r'priceadjust',
                                         'fixed': False})})

    # Production
    model.params.update({'taylor_inflation': type('params', (object,),
                                                  {'value':1.25,
                                                  'valuebounds': (1e-20, 1e5),
                                                  'transform_parameterization':(1e-20, 1e5),
                                                  'transform': "Exponential",
                                                  'prior':dist.norm_dist(1.25, 0.2),
                                                   'description':'Taylor rule coefficient on inflation',
                                                   'tex_label':r'taylor_inflation',
                                                   'fixed': False})})

    model.params.update({'taylor_outputgap': type('params', (object,),
                                                 {'value': 0.1,
                                                 'valuebounds': (1e-20, 1e5),
                                                 'transform_parameterization':(1e-20, 1e5),
                                                 'transform': "Exponential",
                                                 'prior':dist.norm_dist(0.1, 0.05),
                                                  'description':"Taylor rule coefficient on output",
                                                  'tex_label':"taylor_outputgap",
                                                  'fixed': False})})

    model.params.update({'labtax': type('params', (object,),
                                        {'value':0.2,
                                         'description':'Marginal tax rate on labor income',
                                         'tex_label':r'labtax',
                                         'fixed': True})})

    model.params.update({'govbondtarget': type('params', (object,),
                                               {'value':6.,
                                               'description':'govbondtarget: multiple of quarterly GDP',
                                               'tex_label':r'govbondtarget',
                                               'fixed': True})})

    model.params.update({'lumptransferpc': type('params', (object,),
                                                {'value':0.06,
                                                'description':'lumptransferepc: 6% of quarterly GDP in steady state',
                                                'tex_label':r'lumptransferpc',
                                                'fixed': True})})

    model.params.update({'govbcrule_fixnomB': type('params', (object,),
                                                   {'value':0.,
                                                   'description':'govbcrule_fixnomB',
                                                   'tex_label':r'govbcrule_fixnomB',
                                                   'fixed': True})})

    model.params.update({'labdisutil': type('params', (object,),
                                        {'value': model.params['meanlabeff'].value / \
                                                 ((0.75**(-model.params['coefrra'].value)) * \
                                                               ((1./3.)**(1/model.params['frisch'].value))),
                                        'description':'Coefficient of labor disutility',
                                        'tex_label':r'labdisutil',
                                        'fixed': True})})


    # Aggregate shocks
    model.params.update({'sig_MP': type('params', (object,),
                                        {'value':np.sqrt(0.05),
                                         'valuebounds': (1e-20, 1e5),
                                         'transform_parameterization':(1e-20, 1e5),
                                         'transform': "Exponential",
                                         'prior':dist.InvGamma(4.,0.4),
                                         'description':'Volatility of monetary policy shocks',
                                         'tex_label':r'$\sigma_{MP}$',
                                         'fixed': False})})

    model.params.update({'theta_MP': type('params', (object,),
                                          {'value':0.25,
                                          'valuebounds': (1e-20, 1-1e-7),
                                          'transform_parameterization':(1e-20, 1-1e-7),
                                          'transform': 'SquareRoot',
                                          'prior':dist.beta_dist(0.25, 0.1),
                                           'description':'Rate of mean reversion in monetary policy shocks',
                                           'tex_label':r'$\theta_{MP}$',
                                           'fixed': False})})

    # model.params.update({'sig_FP': type('params', (object,),
    #                                     {'value':np.sqrt(0.05),
    #                                      'valuebounds': (1e-20, 1e5),
    #                                      'transform_parameterization':(1e-20, 1e5),
    #                                      'transform': util.exponential_transform,
    #                                      'prior':dist.InvGamma(4.,0.4),
    #                                      'description':'Volatility of monetary policy shocks',
    #                                      'tex_label':r'$\sigma_{MP}$',
    #                                      'fixed': False})})
    #
    # model.params.update({'theta_FP': type('params', (object,),
    #                                       {'value':0.25,
    #                                        'description':'Rate of mean reversion in monetary policy shocks',
    #                                        'tex_label':r'$\theta_{MP}$',
    #                                        'fixed': True})})
    # model.params.update({'sig_PS': type('params', (object,),
    #                                     {'value':np.sqrt(0.05),
    #                                      'valuebounds': (1e-20, 1e5),
    #                                      'transform_parameterization':(1e-20, 1e5),
    #                                      'transform': util.exponential_transform,
    #                                      'prior':dist.InvGamma(4.,0.4),
    #                                      'description':'Volatility of monetary policy shocks',
    #                                      'tex_label':r'$\sigma_{MP}$',
    #                                      'fixed': False})})
    #
    # model.params.update({'theta_PS': type('params', (object,),
    #                                       {'value':0.25,
    #                                        'description':'Rate of mean reversion in monetary policy shocks',
    #                                        'tex_label':r'$\theta_{MP}$',
    #                                        'fixed': True})})
    #

     # Steady-State

    model.steady_state.update({'V_ss': type('steady_state', (object,),
                                           {'value':np.array([]),
                                            'description':'Stacked steady-state value function'})})

    model.steady_state.update({'inflation_ss': type('steady_state', (object,),
                                                    {'value':None,
                                                     'description':'Steady-state rate of inflation'})})

    model.steady_state.update({'g_ss': type('steady_state', (object,),
                                           {'value':np.array([]),
                                            'description':'Stacked steady-state distribution'})})

    model.steady_state.update({'r_ss': type('steady_state', (object,),
                                           {'value':None,
                                            'description':'Steady-state real interest rate'})})

    model.steady_state.update({'u_ss': type('steady_state', (object,),
                                           {'value':np.array([]),
                                            'description':'Steady-state flow utility'})})

    model.steady_state.update({'c_ss': type('steady_state', (object,),
                                           {'value':np.array([]),
                                            'description':'Steady-state individual consumption rate'})})

    model.steady_state.update({'h_ss': type('steady_state', (object,),
                                           {'value':np.array([]),
                                            'description':'Steady-state individual labor hours'})})

    model.steady_state.update({'s_ss': type('steady_state', (object,),
                                           {'value':np.array([]),
                                            'description':'Steady-state individual savings rate'})})

    model.steady_state.update({'rnom_ss': type('steady_state', (object,),
                                              {'value':None,
                                               'description':'Steady-state nominal interest rate'})})

    model.steady_state.update({'B_ss': type('steady_state', (object,),
                                           {'value':None,
                                            'description':'Steady-state bond supply'})})

    model.steady_state.update({'N_ss': type('steady_state', (object,),
                                          {'value':None,
                                           'description':'Steady-state labor supply'})})

    model.steady_state.update({'Y_ss': type('steady_state', (object,),
                                          {'value':None,
                                           'description':'Steady-state output'})})

    model.steady_state.update({'labor_share_ss': type('steady_state', (object,),
                                                     {'value':None,
                                                      'description':'Frictionless labor share of income'})})

    model.steady_state.update({'w_ss': type('steady_state', (object,),
                                           {'value':None,
                                            'description':'Steady-state wage rate'})})

    model.steady_state.update({'profit_ss': type('steady_state', (object,),
                                                {'value':None,
                                                 'description':'Steady-state profits'})})

    model.steady_state.update({'C_ss': type('steady_state', (object,),
                                           {'value':None,
                                            'description':'Steady-state aggregate consumption'})})

    model.steady_state.update({'T_ss': type('steady_state', (object,),
                                           {'value':None,
                                            'description':'Steady-state total taxes'})})

    model.steady_state.update({'G_ss': type('steady_state', (object,),
                                           {'value':None,
                                            'description':'Steady-state government spending'})})

    model.steady_state.update({'rho_ss': type('steady_state', (object,),
                                          {'value':None,
                                           'description':'Steady-state discount rate'})})


def model_settings(model):
    # Steady state r iteration
    model.settings.update({'r0'  : type('setting', (object,),
                                       {'value': 0.005,
                                        'description':'Initial guess for real interest rate'})})
    model.settings.update({'rmax': type('setting', (object,),
                                       {'value': 0.08,
                                        'description':'Maximum guess for real interest rate'})})
    model.settings.update({'rmin': type('setting', (object,),
                                       {'value': 0.001,
                                        'description':'Minimum guess for real interest rate'})})

    # Steady state rho iteration
    model.settings.update({'rho0'  : type('setting', (object,),
                                       {'value': 0.02,
                                        'description':'Initial guess for discount rate'})})
    model.settings.update({'rhomax': type('setting', (object,),
                                       {'value': 0.05,
                                        'description':'Maximum guess for discount rate'})})
    model.settings.update({'rhomin': type('setting', (object,),
                                       {'value': 0.005,
                                        'description':'Minimum guess for discount rate'})})

    # State space grid
    model.settings.update({'I'   : type('setting', (object,),
                                       {'value': 100,
                                        'description':'Number of grid points'})})
    model.settings.update({'J'   : type('setting', (object,),
                                       {'value': 2,
                                        'description':'Number of income states'})})
    model.settings.update({'amin': type('setting', (object,),
                                       {'value': 0.,
                                        'description':'Minimum asset grid value'})})
    model.settings.update({'amax': type('setting', (object,),
                                       {'value': 40.,
                                        'description':'Maximum asset grid value'})})
    model.settings.update({'agridparam': type('setting', (object,),
                                       {'value': 1,
                                        'description':'Bending coefficient: 1 for linear'})})
    model.settings.update({'a': type('setting', (object,),
                                       {'value': aux.construct_asset_grid(model.settings['I'].value,
                                                                          model.settings['agridparam'].value,
                                                                          model.settings['amin'].value,
                                                                          model.settings['amax'].value),
                                        'description':'Asset grid'})})
    model.settings.update({'ygrid_combined': type('setting', (object,),
                                        {'value': np.array([0.2, 1])})})
    model.settings.update({'ymarkov_combined': type('setting', (object,),
                                       {'value': np.array([[-0.5, 0.5],[0.0376, -0.0376]]),
                                        'description':'Markov transition parameters'})})
    model.settings.update({'g_z': type('setting', (object,),
                                        {'value': aux.compute_stationary_income_distribution(model.settings['ymarkov_combined'].value,
                                                                                             model.settings['J'].value),
                                        'description': 'Stationary income distribution'})})
    model.settings.update({'zz': type('setting', (object,),
                                       {'value': aux.construct_labor_income_grid(model.settings['ygrid_combined'].value,
                                                                                 model.settings['g_z'].value,
                                                                                 model.params['meanlabeff'].value,
                                                                                 model.settings['I'].value),
                                        'description':'Labor income grid repeated across asset dimension'})})
    model.settings.update({'z': type('setting', (object,),
                                       {'value': model.settings['zz'].value[0, :],
                                        'description':'Labor income grid'})})

    # Number of variables
    model.settings.update({'n_jump_vars': type('setting', (object,),
                                       {'value': model.settings['I'].value * model.settings['J'].value + 1,
                                        'description':'Number of jump variables'})})
    model.settings.update({'n_state_vars': type('setting', (object,),
                                       {'value': model.settings['I'].value * model.settings['J'].value,
                                        'description':'Number of state variables'})})
    model.settings.update({'n_state_vars_unreduce': type('setting', (object,),
                                       {'value': 0,
                                        'description':'Number of state variables not being reduced'})})
    model.settings.update({'n_static_relations': type('setting', (object,),
                                       {'value': 5,
                                        'description':"Number of static relations: bond-market clearing, labor market clearing, consumption, output, total assets"})})
    model.settings.update({'n_vars': type('setting', (object,),
                                       {'value': model.settings['n_jump_vars'].value \
                                                +model.settings['n_state_vars'].value \
                                                +model.settings['n_static_relations'].value,
                                        'description':'Number of variables, total'})})

    # Steady state approximation
    model.settings.update({'Ir': type('setting', (object,),
                                       {'value': 100,
                                        'description':'Max number of iterations for steady state'})})
    model.settings.update({'maxit_HJB': type('setting', (object,),
                                       {'value': 500,
                                        'description':'Max number of iterations for HJB'})})
    model.settings.update({'tol_HJB': type('setting', (object,),
                                       {'value': 1e-8,
                                        'description':'Tolerance for HJB error'})})
    model.settings.update({'d_HJB': type('setting', (object,),
                                       {'value': 1e6,
                                        'description':'Multiplier for implicit upwind scheme of HJB'})})
    model.settings.update({'maxit_kfe': type('setting', (object,),
                                       {'value': 1000,
                                        'description':'Max number of iterations for kfe'})})
    model.settings.update({'tol_kfe': type('setting', (object,),
                                       {'value': 1e-12,
                                        'description':'Tolerance for kfe error'})})
    model.settings.update({'d_kfe': type('setting', (object,),
                                       {'value': 1e6,
                                        'description':'Multiplier for implicit upwind scheme of kfe'})})
    model.settings.update({'niter_hours': type('setting', (object,),
                                       {'value': 10,
                                        'description':'Max number of iterations for finding hours worked'})})
    model.settings.update({'iterate_r': type('setting', (object,),
                                       {'value': False,
                                        'description':'Iterate on real interest rate or not'})})
    model.settings.update({'iterate_rho': type('setting', (object,),
                                       {'value': True,
                                        'description':'Iterate on discount rate or not'})})
    model.settings.update({'crit_S': type('setting', (object,),
                                       {'value': 1e-5,
                                        'description':'Tolerance for error in bond markets'})})

    # Reduction
    model.settings.update({'n_knots': type('setting', (object,),
                                       {'value': 12,
                                        'description':'Number of knot points'})})
    model.settings.update({'c_power': type('setting', (object,),
                                       {'value': 1,
                                        'description':'Amount of bending of knot point locations to make them nonuniform'})})
    model.settings.update({'n_post': type('setting', (object,),
                                       {'value': len(model.settings['zz'].value[0,:]),
                                        'description':'Number of dimensions that need to be approximated by spline basis'})})
    model.settings.update({'n_prior': type('setting', (object,),
                                       {'value': 1,
                                        'description':'Number of dimensions approximated by spline basis that \
                                                       were not used to compute the basis matrix'})})

    knots = np.linspace(model.settings['amin'].value, model.settings['amax'].value, num=model.settings['n_knots'].value-1)
    knots = (model.settings['amax'].value - model.settings['amin'].value)/ \
                (2**model.settings['c_power'].value-1) * ((knots - model.settings['amin'].value) / \
                                                     (model.settings['amax'].value \
                                                      - model.settings['amin'].value) + 1) ** model.settings['c_power'].value \
                                                 + model.settings['amin'].value - (model.settings['amax'].value - \
                                                                                   model.settings['amin'].value) / \
                                                                                   (2**model.settings['c_power'].value-1)

    model.settings.update({'knots_dict': type('setting', (object,),
                                       {'value': {0: knots},
                                        'description':'Location of knot points for each dimension for value function reduction'})})
    model.settings.update({'krylov_dim': type('setting', (object,),
                                       {'value': 20,
                                        'description':'Krylov reduction dimension'})})

    model.settings.update({'reduce_state_vars': type('setting', (object,),
                                       {'value': True,
                                        'description':'Reduce state variables or not'})})

    model.settings.update({'reduce_v': type('setting', (object,),
                                       {'value': True,
                                        'description':'Reduce value function or not'})})

    model.settings.update({'spline_grid': type('setting', (object,),
                                       {'value': model.settings['a'].value,
                                        'description':'Grid of knot points for spline basis'})})

    # Track lag
    model.settings.update({'track_lag': type('setting', (object,),
                                        {'value': False,
                                        'description':'Add first lag when constructing measurement equation'})})
    # Simulating states
    model.settings.update({'state_simulation_freq': type('setting', (object,),
                                        {'value': 2,
                                        'description':'How many states you want to simulate between states + 1'})})
    # Sequential Monte Carlo
    model.settings.update({'sampling_method'  : type('setting', (object,),
                                       {'value': 'SMC',
                                        'description':'Metropolis-Hastings'})})

    model.settings.update({'use_chand_recursion'  : type('setting', (object,),
                                       {'value': False,
                                        'description':'Use Chandrasekhar Recursions instead of standard Kalman filter'})})

    model.settings.update({'n_particles'  : type('setting', (object,),
                                       {'value': 3000,
                                        'description':'Number of particles for use in SMC'})})

    model.settings.update({'n_phi'  : type('setting', (object,),
                                       {'value': 10, 
                                        'description':'Number of stages in the tempering schedule'})})

    model.settings.update({'lambda'  : type('setting', (object,),
                                       {'value': 2.0,
                                        'description':"The 'bending coefficient' lambda in phi(n) = (n/N(phi))^lambda"})})

    model.settings.update({'n_smc_blocks'  : type('setting', (object,),
                                   {'value': 1,
                                    'description':'The number of parameter blocks in SMC'})})

    model.settings.update({'step_size_smc'  : type('setting', (object,),
                                   {'value': 0.5,
                                    'description':'Scaling factor for covariance of the particles. Controls size of steps in mutation step'})})

    model.settings.update({'n_mh_steps_smc'  : type('setting', (object,),
                                   {'value': 1.0,
                                    'description':'Number of Metropolis Hastings steps to attempt during the mutation step.'})})

    model.settings.update({'target_accept'  : type('setting', (object,),
                                   {'value': 0.25,
                                    'description':'The initial target acceptance rate for new particles during mutation'})})

    model.settings.update({'resampler_smc'  : type('setting', (object,),
                                   {'value': "polyalgo",
                                    'description':'Which resampling method to use in SMC'})})

    model.settings.update({'mixture_proportion'  : type('setting', (object,),
                                   {'value': 0.9,
                                    'description':"The mixture proportion for the mutation step's proposal distribution"})})

    model.settings.update({'use_fixed_schedule'  : type('setting', (object,),
                                   {'value': True,
                                    'description':"Boolean indicating whether or not to use a fixed tempering (phi) schedule"})})

    model.settings.update({'tempering_target'  : type('setting', (object,),
                                   {'value': 0.95,
                                    'description':"Coefficient of the sample size metric to be targeted when solving for an endogenous phi"})})

    model.settings.update({'resampling_threshold'  : type('setting', (object,),
                                   {'value': 0.5,
                                    'description':"Threshold s.t. particles will be resampled when the population drops below threshold * N"})})

    model.settings.update({'use_parallel_workers'  : type('setting', (object,),
                                   {'value': True,
                                    'description':"Use available parallel workers in computations"})})

    model.settings.update({'adaptive_tempering_target_smc'  : type('setting', (object,),
                                   {'value': 0.97,
                                    'description':"Either the adaptive tempering target or 0.0 if using fixed schedule"})})

    model.settings.update({'tempered_update_prior_weight'  : type('setting', (object,),
                                   {'value': 0.0,
                                    'description':"When bridging from old estimation, how much weight to put on prior."})})

    model.settings.update({'smc_iteration'  : type('setting', (object,),
                                   {'value': 1,
                                    'description':"The iteration index for the number of times smc has been run on the same data vintage."})})


    """
    # Forecast
    m <= Setting(:use_population_forecast, true,
                 "Whether to use population forecasts as data")
    m <= Setting(:forecast_zlb_value, 0.13,
        "Value of the zero lower bound in forecast periods, if we choose to enforce it")

    # Simulating states
    m <= Setting(:state_simulation_freq, 2,
                 "How many states you want to simulate between states + 1")

    """


def steadystate(model):
    # Read in parameters
    coefrra        = model.params['coefrra'].value
    frisch         = model.params['frisch'].value
    meanlabeff     = model.params['meanlabeff'].value
    maxhours       = model.params['maxhours'].value
    ceselast       = model.params['ceselast'].value
    labtax         = model.params['labtax'].value
    govbondtarget  = model.params['govbondtarget'].value
    labdisutil     = model.params['labdisutil'].value
    lumptransferpc = model.params['lumptransferpc'].value

    # Read in grids
    I                = model.settings['I'].value
    J                = model.settings['J'].value
    a                = model.settings['a'].value
    g_z              = model.settings['g_z'].value
    zz               = model.settings['zz'].value
    ymarkov_combined = model.settings['ymarkov_combined'].value

    # Set necessary variables
    aa = np.repeat(a.reshape(-1,1), J, axis=1)
    amax = np.max(a)
    amin = np.min(a)

    # Read in initial rates
    iterate_r   = model.settings['iterate_r'].value
    r           = model.settings['r0'].value
    rmin        = model.settings['rmin'].value
    rmax        = model.settings['rmax'].value
    iterate_rho = model.settings['iterate_rho'].value
    rho         = model.settings['rho0'].value
    rhomin      = model.settings['rhomin'].value
    rhomax      = model.settings['rhomax'].value

    # Read in approximation parameters
    Ir          = model.settings['Ir'].value
    maxit_HJB   = model.settings['maxit_HJB'].value
    tol_HJB     = model.settings['tol_HJB'].value
    d_HJB       = model.settings['d_HJB'].value
    maxit_kfe   = model.settings['maxit_kfe'].value
    tol_kfe     = model.settings['tol_kfe'].value
    d_kfe       = model.settings['d_kfe'].value
    niter_hours = model.settings['niter_hours'].value
    crit_S      = model.settings['crit_S'].value

    # Initializing equilibrium objects
    labor_share_ss = (ceselast - 1) / ceselast
    w       = w_ss = labor_share_ss

    # compute initial guesses at steady state values given zz, labor_share_ss, etc.
    N_ss, Y_ss, B_ss, profit_ss, profshare, lumptransfer = \
        aux.calculate_ss_equil_vars_init(zz, labor_share_ss,
                                         meanlabeff, lumptransferpc, govbondtarget)

    # Initialize matrices for finite differences
    Vaf = np.empty((I,J), dtype=np.complex64)
    Vab = np.empty((I,J), dtype=np.complex64)

    cf  = np.empty((I,J), dtype=np.complex64) # forward consumption difference
    hf  = np.empty((I,J), dtype=np.complex64) # forward hours difference
    sf  = np.empty((I,J), dtype=np.complex64) # forward saving difference
    cb  = np.empty((I,J), dtype=np.complex64) # backward consumption difference
    hb  = np.empty((I,J), dtype=np.complex64) # backward hours difference
    sb  = np.empty((I,J), dtype=np.complex64) # backward saving difference
    c0  = np.empty((I,J), dtype=np.complex64)
    A   = np.empty((I*J,J*J), dtype=np.complex64)

    Aswitch = spkron(ymarkov_combined, speye(I, dtype='complex64'))

    # Initialize steady state variables
    V  = np.empty((I,J), dtype=np.complex64) # value function
    u  = np.empty((I,J), dtype=np.complex64) # flow utility across state space
    s  = np.empty((I,J), dtype=np.complex64) # savings across state space
    c  = np.empty((I,J), dtype=np.complex64) # flow consumption
    h  = np.empty((I,J), dtype=np.complex64) # flow hours of labor
    h0 = np.empty((I,J), dtype=np.complex64) # guess of what h will be

    # Creates functions for computing flow utility, income earned, and labor done given
    # CRRA + frisch elasticity style labor disutility
    util, income, labor = \
        aux.construct_household_problem_functions(V, w, coefrra, frisch, labtax, labdisutil)

    # Setting up forward/backward difference grids for a.
    daf, dab, azdelta = aux.initialize_diff_grids(a, I, J)

    for ir in range(Ir):
        c.fill(np.complex(0.))
        h.fill(np.complex(1/3))
        h0.fill(np.complex(1.))

        # Initial guess
        inc = income(h, zz, profshare, lumptransfer, r, aa) # get income
        v   = util(inc, h) / rho                            # value function guess

        # Iterate HJB
        for ihjb in range(maxit_HJB):
            V = v
            Vaf, Vab, cf, hf, cb, hb = aux.construct_initial_diff_matrices(V, Vaf, Vab,
                                                                           income, labor, h, h0, zz,
                                                                           profshare, lumptransfer,
                                                                           amax, amin,
                                                                           coefrra, r, daf,
                                                                           dab, maxhours)
            # Iterative method to find consistent forward/backward/neutral
            # difference matrices for c and h
            cf, hf, cb, hb, c0, h0 = aux.hours_iteration(income, labor,
                                                         zz, profshare, lumptransfer,
                                                         aa, coefrra, r,
                                                         cf, hf, cb, hb,
                                                         c0, h0, maxhours, niter_hours)

            c0 = income(h0, zz, profshare, lumptransfer, r, aa)
            sf = income(hf, zz, profshare, lumptransfer, r, aa) - cf
            sb = income(hb, zz, profshare, lumptransfer, r, aa) - cb

            Vaf[I-1,:] = cf[I-1,:]**(-coefrra) # Forward difference for value function w.r.t. wealth
            Vab[0,:]   = cb[0,  :]**(-coefrra) # Backward difference for value function w.r.t. wealth

            V, A, u, h, c, s  = aux.upwind(rho, V, util, Aswitch,
                                           cf, cb, c0, hf, hb, h0, sf, sb,
                                           Vaf, Vab,  daf, dab, d_HJB = d_HJB)

            # Check for convergence
            Vchange = V-v
            v = V

            err_HJB = np.max(np.abs(Vchange))
            if err_HJB < tol_HJB:
                break
        # Create initial guess for g0
        g0 = np.zeros((I, J), dtype=np.complex64)
        # Assign stationary income distribution weight at a = 0, zero elsewhere
        g0[a==0,:] = g_z
        # g_z is marginal distribution, so re-weight by some multiplier of Lebesgue measure
        g0 = g0 / azdelta.reshape(I, J)

        # Solve for distribution
        g = aux.solve_kfe(A, g0, spdiag(azdelta), maxit_kfe=maxit_kfe,
                          tol_kfe=tol_kfe, d_kfe=d_kfe)
        # Back out static conditions/steady state values given our value function and distribution
        N_ss, Y_ss, B_ss, profit_ss, profshare, lumptransfer, bond_err = \
            aux.calculate_ss_equil_vars(zz, h, g, azdelta, aa, labor_share_ss, meanlabeff,
                                        lumptransferpc, govbondtarget)

        # Check bond market for market clearing
        r, rmin, rmax, rho, rhomin, rhomax, clear_cond = \
            aux.check_bond_market_clearing(bond_err, crit_S, r, rmin, rmax, rho, rhomin, rhomax,
                                           iterate_r, iterate_rho)
        if clear_cond:
            # Set steady state values
            model.steady_state['V_ss'].value           = np.real(V.flatten(order='F'))
            model.steady_state['inflation_ss'].value   = 0.
            model.steady_state['g_ss'].value           = np.real(g.flatten(order='F'))
            model.steady_state['r_ss'].value           = r
            model.steady_state['u_ss'].value           = np.real(u.flatten(order='F'))
            model.steady_state['c_ss'].value           = np.real(c.flatten(order='F'))
            model.steady_state['h_ss'].value           = np.real(h.flatten(order='F'))
            model.steady_state['s_ss'].value           = np.real(s.flatten(order='F'))
            model.steady_state['rnom_ss'].value        = model.steady_state['r_ss'].value \
                                                         + model.steady_state['inflation_ss'].value
            model.steady_state['B_ss'].value           = sum(model.steady_state['g_ss'].value * aa.flatten(order='F') * azdelta)
            model.steady_state['N_ss'].value           = np.real(sum(zz.flatten(order='F') * model.steady_state['h_ss'].value \
                                                                     * model.steady_state['g_ss'].value * azdelta))
            model.steady_state['Y_ss'].value           = model.steady_state['N_ss'].value
            model.steady_state['labor_share_ss'].value = (ceselast - 1) / ceselast
            model.steady_state['w_ss'].value           =  model.steady_state['labor_share_ss'].value
            model.steady_state['profit_ss'].value      = (1 - model.steady_state['labor_share_ss'].value) \
                                                          * model.steady_state['Y_ss'].value
            model.steady_state['C_ss'].value           = sum(model.steady_state['c_ss'].value * model.steady_state['g_ss'].value * azdelta)
            model.steady_state['T_ss'].value           = np.real(lumptransfer)
            model.steady_state['rho_ss'].value         = rho
            model.steady_state['G_ss'].value           = labtax * model.steady_state['w_ss'].value * model.steady_state['N_ss'].value \
                                                         - model.steady_state['T_ss'].value - model.steady_state['r_ss'].value * model.steady_state['B_ss'].value

            break
    return model

def get_model(data):
    """
    data: TBD
    TODO: connect to get_data or FRED directly
    data shape: variable * period
    """
    m = OneAssetHANK(model_type            = "HANK",
                    params                 = OrderedDict(),
                    steady_state           = OrderedDict(),
                    keys                   = OrderedDict(),
                    endog_states           = OrderedDict(),
                    exog_shocks            = OrderedDict(),
                    expected_shocks        = OrderedDict(),
                    equilibrium_conditions = OrderedDict(),
                    endog_states_augmented = OrderedDict(),
                    observables            = OrderedDict(),
                    observables_mappings   = OrderedDict(),
                    settings               = dict())

    # Set observable
    #m.observables_mappings['obs_gdp']           = data
    m.observables_mappings['obs_gdp']         = data[0]
    m.observables_mappings['obs_inflation']   = data[1]
    m.observables_mappings['obs_nominalrate'] = data[2]

    # Set settings
    init_model_indices(m)
    init_parameters(m)
    model_settings(m)

    # Calculate steady state
    steadystate(m)

    return m
