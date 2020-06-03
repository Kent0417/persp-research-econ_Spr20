import numpy as np
from ProjectCode.posterior import get_posterior
from ProjectCode import dist
from tqdm import tqdm
import time


def metropolis_hastings(proposal_dist, model, data, result_path):
    ##---------------------------------------------------------------------------
    ## 0.  Set MH settings
    sampling        = model.settings['sampling'].value
    burn_in         = model.settings['burn-in'].value
    adaptive_accept = model.settings['adaptive_accept'].value
    cc0 = model.settings['mh_cc0'].value
    cc  = model.settings['mh_cc'].value  ## 0.27
    variance_adjust = True

    if type(burn_in) == float:
        burn_in_ind  = int(sampling * burn_in)
    elif type(burn_in) == int:
        burn_in_ind  = burn_in

    # para_full      = util.get_params(model)
    # para_free_inds = np.where([not v.fixed for v in model.params.values()])

    ##---------------------------------------------------------------------------
    ## 1.  Initialize starting values for drawing
    propdist     = dist.init_deg_mvnormal(proposal_dist.mu, proposal_dist.sig)
    para_old     = propdist.rvs(cc=cc0)
    # para_old                 = np.copy(para_full)
    # para_old_free            = propdist.rvs()
    # para_old[para_free_inds] = para_old_free

    params_accept = np.zeros([sampling+1, len(para_old)])
    initialized   = False
    while not initialized:
        post_old = get_posterior(model, para_old, data, sampler=True)
        if post_old > -np.inf:
            params_accept[0] = para_old
            initialized      = True
        else:
            # para_old_free            = propdist.rvs()
            # para_old[para_free_inds] = para_old_free
            para_old = propdist.rvs()
    print(' - Starting values:\n', para_old)
    time.sleep(1)
    ##---------------------------------------------------------------------------
    ## 2.  Begin drawing
    print('////////////// Start Algorithm! ///////////////')
    s2 = time.time()

    accept_count = 0
    #c            = 1.0
    #prop_para    = np.copy(para_full)
    for i in tqdm(range(sampling)):
        #HH        = c * HH_init
        para_old  = params_accept[i]

        inf_prior = True
        #para_old_free             = para_old[para_free_inds]
        #propdist                  = multivariate_normal(para_old_free, HH)
        d_subset = dist.DegenerateMvNormal(propdist.mu,
                                           (propdist.sig + propdist.sig.T)/2,
                                           np.linalg.pinv((propdist.sig + propdist.sig.T)/2),
                                           propdist.lamb_vals)
        # prop_para_free            = propdist.rvs()
        # prop_para[para_free_inds] = prop_para_free
        prop_para = d_subset.rvs(cc=cc)
        # while inf_prior:
        #     post_new = get_posterior(model, prop_para, data)
        #     if post_new > -np.inf:
        #         inf_prior = False
        #     else:
        #         # prop_para_free            = propdist.rvs()
        #         # prop_para[para_free_inds] = prop_para_free
        #         prop_para = d_subset.rvs(cc=cc)

        post_new = get_posterior(model, prop_para, data, sampler=True)


        if adaptive_accept:
            pass

        r = np.exp(post_new - post_old)

        if min(r, 1) > np.random.rand():
            params_accept[i+1] = prop_para
            accept_count += 1
            post_old = post_new
            propdist.mu = prop_para
        else:
            params_accept[i+1] = para_old

        accept_prob = accept_count/(i+1)

        if variance_adjust:
            if accept_prob < 0.25:
                if cc > 1e-10:
                    cc -= 1e-3
            else:
                cc += 1e-3

        if (i == 0) | ((i+1)%(sampling/10) == 0):
            print('-------- Epoch {} -----------'.format(i+1))
            print('Latest accepted params:\n', params_accept[i+1])
            print('acceptance rate: {:.2%}'.format(accept_prob))
            print('-----------------------------')

    e2 = time.time()-s2
    print('\n-------- Result -----------')
    print(' - Acceptance Rate: {:.2%}'.format(accept_prob))
    print(' - Take-in samples', sampling - burn_in_ind)
    print(' - Elapsed time for Algorithm: {0:.2f}'.format(e2/60),'min')

    posterior_dist = params_accept[burn_in_ind:]

    np.save(result_path, posterior_dist)
