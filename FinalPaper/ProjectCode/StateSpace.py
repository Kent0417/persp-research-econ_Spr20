from .RANK import measurement as meR
from .RANK import solve_RANK as solvR
from .HANK import measurement as meH
from .HANK import solve_HANK as solvH
from collections import OrderedDict


def get_system(model):
    """
    Generate state space system
    
    """

    model.System = OrderedDict()

    if model.model_type == 'RANK':
        TT, RR, C = solvR.solve_rank(model)
        ZZ, D, QQ, EE = meR.measurement(model, TT, RR, C)

    elif model.model_type == 'HANK':
        TT, RR, C, inverse_basis, basis = solvH.solve_hank(model)
        ZZ, D, QQ, EE = meH.measurement(model, TT, RR, C, inverse_basis)

    else:
        raise ValueError('The model type is invalid')




    model.System.update({'TT': type('Transition', (object,),
                                    {'value':TT,
                                     'description':'State transition matrix'})})
    model.System.update({'RR': type('Transition', (object,),
                                    {'value':RR,
                                     'description':'Shock term in the transition equation'})})
    model.System.update({'C': type('Transition', (object,),
                                    {'value':C,
                                     'description':'Constant vector in the transition equation'})})
    model.System.update({'ZZ': type('Measurement', (object,),
                                    {'value':ZZ,
                                     'description':'Mapping states to observables in the measurement equation'})})
    model.System.update({'D': type('Measurement', (object,),
                                    {'value':D,
                                     'description':'Constant vector in the measurement equation'})})
    model.System.update({'QQ': type('Measurement', (object,),
                                    {'value':QQ,
                                     'description':'Shock covariances'})})
    model.System.update({'EE': type('Measurement', (object,),
                                    {'value':EE,
                                     'description':'Measurement error covariances'})})

    return
