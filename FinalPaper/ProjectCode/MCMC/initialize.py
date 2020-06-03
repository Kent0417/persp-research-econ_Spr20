import numpy as np
from ProjectCode import util
from ProjectCode.posterior import prior2
from . import particle as ptcl
from joblib import Parallel, delayed

## for debug & test
from tqdm import tqdm
import time
# import warnings
# warnings.simplefilter('ignore')

def one_draw(loglikelihood, params, data):

    success = False
    draw    = util.rand_params(params, 1).flatten()
    draw_loglh = draw_logpost = 0.0
    while not success:
        try:
            util.update(params, draw)
            draw_loglh   = loglikelihood(params, data)
            draw_logpost = prior2(params)

            if (draw_loglh == -np.inf) | np.isnan(draw_loglh):
                draw_loglh = draw_logpost = -np.inf
        except (util.ParamsBoundsError, IndexError) as e:
            draw_loglh = draw_logpost = -np.inf

        if draw_loglh == -np.inf:
            draw    = util.rand_params(params, 1).flatten()
        else:
            success = True

    return vector_reshape(draw, draw_loglh, draw_logpost)


def draw(loglikelihood, params, data, c, free_para_inds, model, parallel = False):

    n_parts = c.particles.shape[0]

    one_draw_closure = lambda : one_draw(loglikelihood, model.params.values(), data)

    # For each particle, finds valid parameter draw and returns loglikelihood & posterior
    if parallel:
        result = Parallel(n_jobs=-1, verbose=2)([delayed(one_draw_closure)() for i in range(n_parts)])
        draws, loglh, logpost = vector_reduce(np.array(result))

    else:
        draws, loglh, logpost = vector_reduce(np.array([one_draw_closure() for i in tqdm(range(n_parts))]))

    # for i in tqdm(range(n_parts)):
    #     draws[i,:], loglh[i], logpost[i] = one_draw(loglikelihood, params, data)

    ptcl.update_draws(c, draws)
    ptcl.update_loglh(c, loglh)
    ptcl.update_logpost(c, logpost)
    ptcl.update_old_loglh(c, np.zeros(n_parts))
    ptcl.set_weights(c, np.ones(n_parts))

def cloud_settings(cloud, tempered_update=False,
                   n_parts=5000, n_phi=300, c=0.5, accept=0.25):

    if tempered_update:
        cloud.ESS = [cloud.ESS[-1]]
    else:
        cloud.ESS[0] = n_parts

    cloud.stage_index = 1
    cloud.n_phi       = n_phi
    cloud.resamples   = 0
    cloud.c           = c
    cloud.accept      = accept
    cloud.total_sampling_time = 0.
    cloud.tempering_schedule  = np.zeros(1)

def vector_reshape(*args):
    n_args = len(args)
    return_arg = []
    for i in range(n_args):
        if type(args[i]) == np.ndarray:
            arg = args[i]
        else:
            arg = np.array(args[i])
        return_arg.append(arg)
    return return_arg

def vector_reduce(args):
    n_args = len(args[0])
    return_arg = []
    for i in range(n_args):
        arg = np.vstack(args[:,i])
        if arg.shape[1] == 1:
            arg = arg.flatten()
        return_arg.append(arg)
    return return_arg
