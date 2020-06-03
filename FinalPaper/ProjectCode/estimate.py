import numpy as np
import time
import os
import datetime

from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from scipy.linalg import svd

from . import util
from . import dist
from .CsminWel import csminwell
from .posterior import get_posterior
from .hessian import hessian
from .MCMC import MH
from .MCMC import SMC

from .HANK import build_OneAssetHANK as build_hank
from .RANK import build_RANK as build_rank

def run(data, model="RANK", initial_guess=None,
        continue_intermediate=False,
        intermediate_stage_start=0,
        intermediate_stage_increment=10,
        save_intermediate=False):
    
    if model == "RANK":
        model = build_rank.get_model(data)
    elif model == "HANK":
        model = build_hank.get_model(data)

    #time_stamp  = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    output_path = os.path.join('output', model.model_type)
    os.makedirs(output_path, exist_ok=True)

    sampling_method = model.settings['sampling_method'].value

    #--------------------------------------------------------------------------------
    # Metropolis-Hastings algorithm
    #--------------------------------------------------------------------------------
    if sampling_method == 'MH':
        if initial_guess == None:
            pass
        else:
            util.update_params(model, initial_guess)
        print("Estimation Method: Metropolis-Hastings algorithm")
        ##---------------------------------------------------------------------------
        ## 1. Recalculate initial mu and sigma for proposal distribution
        if not continue_intermediate:
            print(' - Setting initial proposed distribution....')

            ## 1-1. Find optimal initial values via MLE with csminwell
            print('   - Looking for optimal initial values')
            s1 = time.time()

            para_free_inds = np.where([not v.fixed for v in model.params.values()])
            x_model        = np.array([util.transform_to_real_line(v) for v in model.params.values()])
            x_opt          = x_model[para_free_inds]

            def _crit(x_opt, args):
                model, data = args
                try:
                    x_model[para_free_inds] = x_opt
                    util.transform_to_model_space_(model, x_model)
                except:
                    return np.inf
                para = util.get_params(model)
                val = get_posterior(model, para, data, catch_errors=True)
                return -val

            out = minimize(_crit, x_opt, args=(model, data), method=csminwell)

            x_model[para_free_inds] = out.x
            util.transform_to_model_space_(model, x_model)
            para_init   = util.get_params(model)

            ## 1-2. Calculate Hessian inverse
            print('   - Calculating Hessian inverse')
            # hess        = out.hess
            # L           = generalized_Cholesky(hess)
            # hessian_inv = L @ L.T
            # hessian_inv = 0.5*(hessian_inv + hessian_inv.T)

            hess, _ = hessian(model, para_init, data)

            U, S, V = svd(hess, full_matrices=False)
            V = V.conj().T
            big_eig_vals = S > 1e-6
            hess_rank = sum(big_eig_vals)

            S_inv = np.zeros_like(hess)
            for i in range(hess_rank):
                S_inv[i, i] = 1/S[i]
            hessian_inv = V @ S_inv @ U.T

            propdist = dist.DegenerateMvNormal(para_init, hessian_inv,
                                               np.linalg.pinv(hessian_inv), np.diag(S_inv))

            e1 = time.time() - s1
            print(' - Set initial proposed distribution! | {0:.1f}'.format(e1),'sec')
        ##---------------------------------------------------------------------------
        ## 2. Reuse initial mu and sigma for proposal distribution from file
        else:
            print(' - Use preset initial proposed distribution.')
            optim_path = os.path.join(output_path, "optim_init.npz")
            optim_init  = np.load(optim_path)
            para_init   = optim_init['p']
            hessian_inv = optim_init['h']
            S_inv       = optim_init['s']
            util.update_params(model, para_init)
            propdist = dist.DegenerateMvNormal(para_init, hessian_inv,
                                               np.linalg.pinv(hessian_inv), np.diag(S_inv))

        ##---------------------------------------------------------------------------
        ## 2.5. Save  initial mu and sigma
        if save_intermediate:
            #np.savez('output/optim_init',  p = para_init, h = hessian_inv)
            optim_path = os.path.join(output_path, "optim_init")
            np.savez(optim_path, p = para_init, h = hessian_inv, s = S_inv)

        ##---------------------------------------------------------------------------
        ## 3. Start Metropolis-Hastings algorithm
        #para_init_free = para_init[para_free_inds]
        #propdist       = multivariate_normal(para_init_free, hessian_inv)
        result_path = os.path.join(output_path, "mh_result")
        MH.metropolis_hastings(propdist, model, data, result_path)

    #--------------------------------------------------------------------------------
    # Sequential Monte Carlo (SMC)
    #--------------------------------------------------------------------------------
    elif sampling_method == 'SMC':
        print("Estimation Method: Sequential Monte Carlo")
        SMC.sequential_mc(model, data, output_path,
                         continue_intermediate=continue_intermediate,
                         intermediate_stage_start=intermediate_stage_start,
                         intermediate_stage_increment=intermediate_stage_increment,
                         save_intermediate=save_intermediate)


    print('Finished estimation!')
