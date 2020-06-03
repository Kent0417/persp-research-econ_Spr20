import numpy as np
from . import particle as ptcl
from ProjectCode import dist
from ProjectCode.posterior import prior2
from ProjectCode import util

import copy


def mutation(loglikelihood, params, data, p, d_mu, d_sig,
                blocks_free, blocks_all, phi_n, phi_n1,
                c=1.0, alpha=1.0, n_mh_steps=1):

    n_free_para = len([not v.fixed for v in params])
    step_prob   = np.random.rand()

    N         = len(p)
    para      = p[:ptcl.ind_para_end(N)+1]
    like      = p[ptcl.ind_loglh(N)]
    logprior  = p[ptcl.ind_logprior(N)]
    like_prev = p[ptcl.ind_old_loglh(N)]
    accept    = 0.0
    old_data  = np.array([])

    for step in range(n_mh_steps):
        for block_f, block_a in zip(blocks_free, blocks_all):
            # Index out parameters corresponding to given random block, create distribution
            # centered at weighted mean, with Σ corresponding to the same random block
            para_subset = para[block_a]
            d_subset    = dist.MvNormal(d_mu[block_f], d_sig[block_f][:,block_f])
            para_draw   = mvnormal_mixture_draw(para_subset, d_subset, c=c, alpha=alpha)

            q0, q1 = compute_proposal_densities(para_draw, para_subset,
                                                d_subset, c = c, alpha = alpha)

            para_new          = copy.deepcopy(para)
            para_new[block_a] = para_draw

            like_init, prior_init = like, logprior
            prior_new = like_new = like_old_data = -np.inf

            try:
                util.update(params, para_new)
                para_new  = np.array([v.value for v in params])
                prior_new = prior2(params)
                like_new  = loglikelihood(params, data)

                if like_new == -np.inf:
                    prior_new = like_old_data = -np.inf

                if len(old_data) == 0:
                    like_old_data = 0.0
                else:
                    like_old_data = loglikelihood(params, old_data)
            except (util.ParamsBoundsError, IndexError):
                prior_new = like_new = like_old_data = -np.inf
            # eta = np.exp(phi_n * (like_new-like_init) + (1 - phi_n) * (like_old_data - like_prev) + (prior_new - prior_init) + (q0 - q1))
            eta = np.exp(phi_n * (like_new-like_init) + (prior_new - prior_init) + (q0 - q1))
            if step_prob < eta:
                para      = para_new
                like      = like_new
                logprior  = prior_new
                like_prev = like_old_data
                accept += len(block_a)

            step_prob = np.random.rand()


    ptcl.update_mutation(p, para, like, logprior, like_prev, accept/n_free_para)

    return p

def mvnormal_mixture_draw(theta_old, d_prop, c=1.0, alpha=1.0):

    d_bar = dist.MvNormal(d_prop.mu, c**2 * d_prop.sig)

    # Create mixture distribution conditional on the previous parameter value, theta_old
    d_old      = dist.MvNormal(theta_old, c**2 * d_prop.sig)
    d_diag_old = dist.MvNormal(theta_old, np.diag(np.diag(c**2 * d_prop.sig)))
    d_mix_old  = dist.Mix3MVmodel(d_old, d_diag_old, d_bar, np.array([alpha, (1-alpha)/2, (1-alpha)/2]))
    random_state = np.random.randint(0,1000,3)
    theta_new = d_mix_old.rvs(random_state=random_state)

    return theta_new


def compute_proposal_densities(para_draw, para_subset, d_subset, c=1.0, alpha=1.0):
    d_sig = d_subset.sig
    q0  = alpha * np.exp(dist.DegenerateMvNormal(para_draw, c**2 * d_sig).logpdf(para_subset))
    q1  = alpha * np.exp(dist.DegenerateMvNormal(para_subset, c**2 * d_sig).logpdf(para_draw))

    ind_pdf = 1.0
    for i in range(len(para_subset)):
        sig_ii  = np.sqrt(d_sig[i,i])
        zstat   = (para_subset[i] - para_draw[i]) / sig_ii
        ind_pdf = ind_pdf / (sig_ii * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * zstat**2)

    q0 += (1.0-alpha)/2.0 * ind_pdf
    q1 += (1.0-alpha)/2.0 * ind_pdf

    q0 += (1.0-alpha)/2.0 * np.exp(dist.DegenerateMvNormal(d_subset.mu, c**2 * d_sig).logpdf(para_subset))
    q1 += (1.0-alpha)/2.0 * np.exp(dist.DegenerateMvNormal(d_subset.mu, c**2 * d_sig).logpdf(para_draw))

    q0 = np.log(q0)
    q1 = np.log(q1)

    if (q0 == np.inf) & (q1 == np.inf):
        q0 = 0.0

    return q0, q1
