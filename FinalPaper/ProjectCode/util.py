import numpy as np
from .RANK import build_RANK as build_R
from .HANK import build_OneAssetHANK as build_H

def flatten_indices(dict_states):
    dict_var = dict_states.values()
    x = np.array([])
    for d in dict_var:
        x = np.append(x, d)
    return x

def transform_to_model_space(p, x, c=1):
    if not p.fixed:
        a, b = p.transform_parameterization
        if p.transform == "SquareRoot":
            return (a+b)/2 + (b-a)/2*c*x/np.sqrt(1 + c**2 * x**2)
        if p.transform == "Exponential":
            return a + np.exp(c*(x-b))
    else:
        return x

def transform_to_model_space_(model, values):
    new_values = np.array([transform_to_model_space(v, values[i]) \
                                        for i, v in enumerate(model.params.values())])
    update_params(model, new_values)

    return

def transform_to_real_line(p, c=1):
    x = p.value
    if not p.fixed:
        a, b = p.transform_parameterization
        if p.transform == "SquareRoot":
            cx = 2 * (x - (a+b)/2)/(b-a)
            if cx**2 > 1:
                raise ValueError()
            return (1/c)*cx/np.sqrt(1-cx**2)
        if p.transform == "Exponential":
            return b + (1 / c) * np.log(x-a)
    else:
        return x

def update_params(model, new_para):
    ## rename the function? maybe `update_model`
    pvec = model.params.values()
    if len(pvec) != len(new_para):
        raise ValueError("Vector of new params must be the same length of params")

    #update(pvec, new_para)

    for i, k in enumerate(model.params.keys()):
        if not model.params[k].fixed:
            bounds = model.params[k].valuebounds
            if (new_para[i] > bounds[0]) & (new_para[i] < bounds[1]):
                model.params[k].value = new_para[i]
            else:
                raise ParamsBoundsError("New parameter(s) out of bounce" )

    if model.model_type=='HANK':
        build_H.steadystate(model)
    else:
        build_R.steadystate(model)

    return

def update(pvec, values):
    if len(pvec) != len(values):
        raise ValueError("Vector of new params must be the same length of params")

    for i, k in enumerate(pvec):
        if not k.fixed:
            bounds = k.valuebounds
            if (values[i] > bounds[0]) & (values[i] < bounds[1]):
                k.value = values[i]
            else:
                raise ParamsBoundsError()

def rand(p):
    draw = np.zeros(len(p))
    for i, para in enumerate(p):
        if para.fixed:
            draw[i] = para.value
        else:
            random_state = np.random.randint(0,1000)
            prio = para.prior.rvs(random_state=random_state)
            while not ((para.valuebounds[0] < prio) & (prio < para.valuebounds[1])):
                random_state = np.random.randint(0,1000)
                prio = para.prior.rvs(random_state=random_state)
            draw[i] = prio
    return draw

def rand_params(p, n):
    priorsim = np.zeros((len(p), n))
    for i in range(n):
        priorsim[:, i] = rand(p)
    return priorsim

def get_params(model):
    return np.array([v.value for v in model.params.values()])

def get_params_full(model):
    return np.array([v for v in model.params.values()])

def get_setting(model, key):
    return model.settings[key].value

class ParamsBoundsError(Exception):
    pass

class GensysError(Exception):
    pass
