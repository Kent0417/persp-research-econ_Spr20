import numpy as np
from dataclasses import dataclass

@dataclass()
class Cloud():
    particles           : np.ndarray
    tempering_schedule  : np.ndarray
    ESS                 : np.ndarray
    stage_index         : int
    n_phi               : int
    resamples           : int
    c                   : float
    accept              : float
    total_sampling_time : float


def set_Cloud(n_params, n_parts):
    return Cloud(particles           = np.empty((n_parts, n_params+5)), # 5 * 20
                 tempering_schedule  = np.zeros(1),
                 ESS                 = np.zeros(1),
                 stage_index         = 1,
                 n_phi               = 0,
                 resamples           = 0,
                 c                   = 0.0,
                 accept              = 0.25,
                 total_sampling_time = 0.0
                 )

"""
Find correct indices for accessing columns of cloud array.
"""
ind_para_end  = lambda N : N-6
ind_loglh     = lambda N : N-5
ind_logpost   = lambda N : N-4
ind_logprior  = lambda N : N-4 # ??Fix logprior/logpost shenanigans
ind_old_loglh = lambda N : N-3
ind_accept    = lambda N : N-2
ind_weight    = lambda N : N-1


def get_vals(c, transpose=True):
    if transpose:
        return c.particles[:, :ind_para_end(c.particles.shape[1])+1].T
    else:
        return c.particles[:, :ind_para_end(c.particles.shape[1])+1]

def update_draws(c, draws):
    I, J     = draws.shape # 5 * 15
    n_parts  = c.particles.shape[0] # 5
    n_params = ind_para_end(c.particles.shape[1]) + 1
    if (I, J) == (n_parts, n_params):
        for i in range(I):
            for j in range(J):
                c.particles[i, j] = draws[i, j]
    elif (I, J) == (n_params, n_parts):
        for i in range(I):
            for j in range(J):
                c.particles[j, i] = draws[i, j]
    else:
        raise ValueError("Draws are incorrectly sized!")

def update_loglh(c, loglh):
    assert c.particles.shape[0] == len(loglh), "Dimensional mismatch!"
    N = ind_loglh(c.particles.shape[1])
    for i in range(len(loglh)):
        c.particles[i, N] = loglh[i]

def update_logpost(c, logpost):
    assert c.particles.shape[0] == len(logpost), "Dimensional mismatch!"
    N = ind_logpost(c.particles.shape[1])
    for i in range(len(logpost)):
        c.particles[i, N] = logpost[i]

def update_old_loglh(c, old_loglh):
    assert c.particles.shape[0] == len(old_loglh), "Dimensional mismatch!"
    N = ind_old_loglh(c.particles.shape[1])
    for i in range(len(old_loglh)):
        c.particles[i, N] = old_loglh[i]

def update_weights(c, weights):
    assert c.particles.shape[0] == len(weights), "Dimensional mismatch!"
    N = ind_weight(c.particles.shape[1])
    for i in range(c.particles.shape[0]):
        c.particles[i, N] *= weights[i]

def set_weights(c, weights):
    assert c.particles.shape[0] == len(weights), "Dimensional mismatch!"
    N = ind_weight(c.particles.shape[1])
    for i in range(c.particles.shape[0]):
        c.particles[i, N] = weights[i]

def get_loglh(c):
    return c.particles[:, ind_loglh(c.particles.shape[1])]

def get_old_loglh(c):
    return c.particles[:, ind_old_loglh(c.particles.shape[1])]

def get_weights(c):
    return c.particles[:, ind_weight(c.particles.shape[1])]

def normalize_weights(c):
    sum_weights = sum(get_weights(c))
    c.particles[:, ind_weight(c.particles.shape[1])] *= c.particles.shape[0]
    c.particles[:, ind_weight(c.particles.shape[1])] /= sum_weights

def reset_weights(c):
    n_parts = c.particles.shape[0]
    c.particles[:, ind_weight(c.particles.shape[1])] = 1.0

def weighted_mean(c):
    return get_vals(c) @ get_weights(c) / sum(get_weights(c))

def weighted_cov(c):
    return np.cov(get_vals(c, transpose=False), rowvar=False, bias=True,
                  aweights=get_weights(c) / sum(get_weights(c)))

def update_cloud(cloud, new_particles):
    I, J = new_particles.shape
    if I == cloud.particles.shape[0]:
        cloud.particles = new_particles
    elif J == cloud.particles.shape[0]:
        for k in range(cloud.particles.shape[0]):
            cloud.particles[k, :] = new_particles[:, k]

    else:
        raise ValueError("draws are incorrect size!")

def update_acceptance_rate(c):
    c.accept = (c.particles[:, ind_accept(c.particles.shape[1])]).mean()

def update_mutation(p, para, like, post, old_like, accept):
    N = len(p)
    p[:ind_para_end(N)+1] = para
    p[ind_loglh(N)]       = like
    p[ind_logprior(N)]    = post
    p[ind_old_loglh(N)]   = old_like
    p[ind_accept(N)]      = accept
