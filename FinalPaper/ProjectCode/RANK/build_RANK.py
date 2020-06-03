import numpy as np
from dataclasses import dataclass
from collections import OrderedDict
from ProjectCode import util
from ProjectCode import dist

@dataclass()
class RANK(object):
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
    endog_states = ['y_t', 'pi_t', 'R_t', 'y_t1', 'g_t','z_t','Ey_t1', 'Epi_t1']

    # Exogenous shocks
    exog_shocks  = ['z_sh','g_sh','rm_sh']

    # Expectations shocks
    expected_shocks = ['Ey_sh', 'Epi_sh']

    # Equilibrium conditions
    equilibrium_conditions = ['eq_euler', 'eq_phillips', 'eq_mp', 'eq_y_t1', 'eq_g', 'eq_z', 'eq_Ey', 'eq_Epi']

    # Observables
    observables = model.observables_mappings.keys()

    for i, k in enumerate(endog_states):
        model.endog_states[k] = i
    for i, k in enumerate(exog_shocks):
        model.exog_shocks[k] = i
    for i, k in enumerate(expected_shocks):
        model.expected_shocks[k] = i
    for i, k in enumerate(equilibrium_conditions):
        model.equilibrium_conditions[k] = i
    for i, k in enumerate(observables):
        model.observables[k] = i


def init_parameters(model):
    # Initialize parameters
    model.params.update({'tau': type('params', (object,),
                                    {'value':1.9937,
                                     'valuebounds': (1e-20, 1e5),
                                     'transform_parameterization':(1e-20, 1e5),
                                     'transform': "Exponential",
                                     'prior':dist.gamma_dist(2.0, 0.5),
                                     'description':'The inverse of the intemporal elasticity of substitution',
                                     'tex_label':r'$\tau$',
                                     'fixed': False})})

    model.params.update({'kappa': type('params', (object,),
                                    {'value':0.7306,
                                     'valuebounds': (1e-20, 1e5),
                                     'transform_parameterization':(1e-20, 1e5),
                                     'transform': "SquareRoot",
                                     'prior':dist.gamma_dist(0.2, 0.1),
                                     'description':'Composite parameter in New Keynesian Phillips Curve.',
                                     'tex_label':r'$\kappa$',
                                     'fixed': False})})

    model.params.update({'psi_1': type('params', (object,),
                                    {'value':1.1434,
                                     'valuebounds': (1e-20, 1e5),
                                     'transform_parameterization':(1e-20, 1e5),
                                     'transform': "Exponential",
                                     'prior':dist.gamma_dist(1.5, 0.25),
                                     'description':'The weight on inflation in the monetary policy rule.',
                                     'tex_label':r'$\psi_1$',
                                     'fixed': False})})

    model.params.update({'psi_2': type('params', (object,),
                                    {'value':0.4536,
                                     'valuebounds': (1e-20, 1e5),
                                     'transform_parameterization':(1e-20, 1e5),
                                     'transform': "Exponential",
                                     'prior':dist.gamma_dist(0.5, 0.25),
                                     'description':'The weight on the output gap in the monetary policy rule.',
                                     'tex_label':r'$\psi_2$',
                                     'fixed': False})})

    model.params.update({'rA': type('params', (object,),
                                    {'value':0.0313,
                                     'valuebounds': (1e-20, 1e5),
                                     'transform_parameterization':(1e-20, 1e5),
                                     'transform': "Exponential",
                                     'prior':dist.gamma_dist(0.5, 0.5),
                                     'description':'discount factor = 1/(1+rA/400).',
                                     'tex_label':r'$r_A$',
                                     'fixed': False})})

    model.params.update({'pi': type('params', (object,),
                                    {'value':8.1508,
                                     'valuebounds': (1e-20, 1e5),
                                     'transform_parameterization':(1e-20, 1e5),
                                     'transform': "Exponential",
                                     'prior':dist.gamma_dist(7.0, 2.0),
                                     'description':'Target inflation rate',
                                     'tex_label':r'$\pi^*$',
                                     'fixed': False})})

    model.params.update({'gam_Q': type('params', (object,),
                                    {'value':1.5,
                                     'valuebounds': (1e-20, 1e5),
                                     'transform_parameterization':(1e-20, 1e5),
                                     'transform': "Exponential",
                                     'prior':dist.norm_dist(0.4, 0.2),
                                     'description':'Steady state growth rate of technology',
                                     'tex_label':r'$\gamma_Q$',
                                     'fixed': False})})

    model.params.update({'rho_R': type('params', (object,),
                                    {'value':0.3847,
                                     'valuebounds': (1e-20, 1-1e-7),
                                     'transform_parameterization':(1e-20, 1-1e-7),
                                     'transform': "SquareRoot",
                                     'prior':dist.beta_dist(0.5, 0.2),
                                     'description':'AR(1) coefficient on interest rate',
                                     'tex_label':r'$\rho_R$',
                                     'fixed': False})})

    model.params.update({'rho_g': type('params', (object,),
                                    {'value':0.3777,
                                    'valuebounds': (1e-20, 1-1e-7),
                                    'transform_parameterization':(1e-20, 1-1e-7),
                                    'transform': "SquareRoot",
                                    'prior':dist.beta_dist(0.8, 0.1),
                                     'description':'AR(1) coefficient on government spending',
                                     'tex_label':r'$\rho_g$',
                                     'fixed': False})})

    model.params.update({'rho_z': type('params', (object,),
                                    {'value':0.9579,
                                     'valuebounds': (1e-20, 1-1e-7),
                                     'transform_parameterization':(1e-20, 1-1e-7),
                                     'transform': "SquareRoot",
                                     'prior':dist.beta_dist(0.66, 0.15),
                                     'description':'AR(1) coefficient on shocks to the technology growth rate',
                                     'tex_label':r'$\rho_z$',
                                     'fixed': False})})

    model.params.update({'sig_R': type('params', (object,),
                                    {'value':0.49,
                                     'valuebounds': (1e-20, 1e5),
                                     'transform_parameterization':(1e-20, 1e5),
                                     'transform': "Exponential",
                                     'prior':dist.inv_gamma_dist(4, 0.4),
                                     'description':'Standard deviation of shocks to the nominal interest rate.',
                                     'tex_label':r'$\sigma_R$',
                                     'fixed': False})})

    model.params.update({'sig_g': type('params', (object,),
                                    {'value':1.4594,
                                     'valuebounds': (1e-20, 1e5),
                                     'transform_parameterization':(1e-20, 1e5),
                                     'transform': "Exponential",
                                     'prior':dist.inv_gamma_dist(4, 1.0),
                                     'description':'Standard deviation of shocks to the government spending process.',
                                     'tex_label':r'$\sigma_g$',
                                     'fixed': False})})

    model.params.update({'sig_z': type('params', (object,),
                                    {'value':0.9247,
                                     'valuebounds': (1e-20, 1e5),
                                     'transform_parameterization':(1e-20, 1e5),
                                     'transform': "Exponential",
                                     'prior':dist.inv_gamma_dist(4, 0.5),
                                     'description':'Standard deviation of shocks to the technology growth rate process.',
                                     'tex_label':r'$\sigma_z$',
                                     'fixed': False})})

    model.params.update({'e_y': type('params', (object,),
                                    {'value':0.2 * 0.579923,
                                     'description':'Measurement error on GDP growth.',
                                     'tex_label':r'$e_y$',
                                     'fixed': True})})

    model.params.update({'e_pi': type('params', (object,),
                                    {'value':0.2 * 1.470832,
                                     'description':'Measurement error on inflatio.',
                                     'tex_label':r'$e_{\pi}$',
                                     'fixed': True})})

    model.params.update({'e_R': type('params', (object,),
                                    {'value':0.2 * 2.237937,
                                     'description':'Measurement error on the interest rate.',
                                     'tex_label':r'$e_R$',
                                     'fixed': True})})

def steadystate(model):
    return model

def model_settings(model):

    # Metropolis-Hastings
    model.settings.update({'sampling_method'  : type('setting', (object,),
                                       {'value': 'MH',
                                        'description':'Metropolis-Hastings'})})
    model.settings.update({'mh_cc0'  : type('setting', (object,),
                                       {'value': 0.01,
                                        'description':'Jump size for initialization of Metropolis-Hastings)'})})
    model.settings.update({'mh_cc'  : type('setting', (object,),
                                       {'value': 0.27,
                                        'description':'Jump size for Metropolis-Hastings (after initialization)'})})
    model.settings.update({'sampling'  : type('setting', (object,),
                                       {'value': 300000,
                                        'description':'Sampling size'})})
    model.settings.update({'burn-in'  : type('setting', (object,),
                                       {'value': 200000,
                                        'description':'Set burn-in period'})})
    model.settings.update({'adaptive_accept'  : type('setting', (object,),
                                       {'value': True,
                                        'description':'Whether to adaptively solve for acceptance rate in Metropolis-Hastings'})})
    model.settings.update({'mh_c'  : type('setting', (object,),
                                       {'value': 0.5,
                                        'description':'Step size used for adaptive acceptance rate in Metropolis-Hastings'})})
    model.settings.update({'mh_alpha'  : type('setting', (object,),
                                       {'value': 1.0,
                                        'description':'Mixture proportion for adaptive acceptance rate in Metropolis-Hastings'})})



def get_model(data):

    m = RANK(model_type            = "RANK",
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

    # Set settings
    model_settings(m)

    # Set observable
    m.observables_mappings['obs_gdp']         = data[0]
    m.observables_mappings['obs_inflation']   = data[1]
    m.observables_mappings['obs_nominalrate'] = data[2]

    init_parameters(m)
    init_model_indices(m)
    steadystate(m)

    return m
