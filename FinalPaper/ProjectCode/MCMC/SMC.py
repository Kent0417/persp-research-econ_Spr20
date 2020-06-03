import numpy as np
import math
from ProjectCode import util
from ProjectCode.posterior import likelihood
from . import particle as ptcl
from . import initialize
from .resample import resample
from .mutation import mutation
from joblib import Parallel, delayed

import time
import copy
import pickle
import pandas as pd
from scipy import optimize
import os
from tqdm import tqdm


def sequential_mc(model, data, output_path,
                  continue_intermediate=False,
                  intermediate_stage_start=0, intermediate_stage_increment=10,
                  save_intermediate=False):

    # 0. Get Settings
    parallel    = util.get_setting(model, 'use_parallel_workers')
    n_parts     = util.get_setting(model, 'n_particles')
    n_blocks    = util.get_setting(model, 'n_smc_blocks')
    n_mh_steps  = util.get_setting(model, 'n_mh_steps_smc')

    smc_iteration = util.get_setting(model, 'smc_iteration')

    lamb  = util.get_setting(model, 'lambda')
    n_phi = util.get_setting(model, 'n_phi')

    # Define tempering settings
    #tempered_update_prior_weight = util.get_setting(model, 'tempered_update_prior_weight')
    tempering_target             = util.get_setting(model, 'adaptive_tempering_target_smc')
    #use_fixed_schedule           = tempering_target == 0.0
    use_fixed_schedule           = util.get_setting(model, 'use_fixed_schedule')


    # Step 2 (Correction) settings
    resampling_method = util.get_setting(model, 'resampler_smc')
    threshold_ratio   = util.get_setting(model, 'resampling_threshold')
    threshold         = threshold_ratio * n_parts

    # Step 3 (Mutation) settings
    c      = util.get_setting(model, 'step_size_smc')
    alpha  = util.get_setting(model, 'mixture_proportion')
    target = util.get_setting(model, 'target_accept')

    use_chand_recursion = util.get_setting(model, 'use_chand_recursion')

    def my_likelihood(pvec, data):
        para = np.array([p.value for p in pvec])
        util.update_params(model, para)
        likval = likelihood(model, data, sampler=False, catch_errors=True,
                   use_chand_recursion = use_chand_recursion)
        return likval


    ## Initialize path
    loadpath=""
    if continue_intermediate:
        print(f" - Using intermediate outputs. Restart estimation from stage {intermediate_stage_start:0>3}")
        loadpath = os.path.join(output_path, r'intermediate', f'stage_{intermediate_stage_start:0>3}')

    savepath = output_path
    particle_store_path = os.path.join(output_path, "smc_result")

    smc_main(my_likelihood, util.get_params_full(model), data, model,
            parallel = parallel,
            n_parts  = n_parts,
            n_blocks = n_blocks,
            n_mh_steps = n_mh_steps,

            lamb = lamb, n_phi = n_phi,

            resampling_method = resampling_method,
            threshold_ratio = threshold_ratio,

            c = c, alpha = alpha, target = target,

            use_fixed_schedule = use_fixed_schedule,
            tempering_target = tempering_target,
            smc_iteration = smc_iteration,
            loadpath = loadpath,
            savepath = savepath,
            particle_store_path = particle_store_path,
            output_path = output_path,

            save_intermediate = save_intermediate,
            continue_intermediate = continue_intermediate,
            intermediate_stage_start = intermediate_stage_start,
            intermediate_stage_increment = intermediate_stage_increment)

    return


def smc_main(loglikelihood, params, data, model,
             parallel=False, n_parts=5000, n_blocks=1, n_mh_steps=1,
             lamb=2.1, n_phi=300,
             resampling_method='systematic', threshold_ratio=0.5,
             c=0.5, alpha=1.0, target=0.25,
             use_fixed_schedule=True, tempering_target=0.97, smc_iteration=1,
             loadpath="", savepath="", particle_store_path="smc_result", output_path ="",
             continue_intermediate=False, intermediate_stage_start=0, save_intermediate=False,
             intermediate_stage_increment=10):

    def mutation_closure(p, d_mu, d_sig, blocks_free,
                         blocks_all, phi_n, phi_n1, c=1.0, alpha=1.0, n_mh_steps=1):
        return mutation(loglikelihood, model.params.values(), data, p, d_mu, d_sig, blocks_free,blocks_all, phi_n, phi_n1, c=1.0, alpha=1.0, n_mh_steps=1)

    # General
    i   = 0 # Index tracking the stage of the algorithm
    j   = 1 # Index tracking the fixed_schedule entry phiprop
    phi_n = phi_prop = 0. # Instantiate phi_n and phi_prop variables

    resampled_last_period = False # Ensures proper resetting of ESS_bar after resample
    #use_fixed_schedule = (tempering_target == 0.0)
    threshold          = threshold_ratio * n_parts

    fixed_para_inds = np.where([    p.fixed for p in params])[0]
    free_para_inds  = np.where([not p.fixed for p in params])[0]
    # para_symbols    = [θ.key for θ in parameters]

    n_para          = len(params)
    n_free_para     = len(free_para_inds)
    # Initialization of Particle Array Cloud
    cloud = ptcl.set_Cloud(n_para, n_parts)

    ## 0. Initialize Algorithm: Draws from prior
    if continue_intermediate:
        cloudpath = os.path.join(loadpath, "cloud.pkl")
        cloud = pd.read_pickle(cloudpath)
        # with open(cloudpath, "rb") as f:
        # cloud = pickle.load(f)
    else:
    # Instantiating Cloud object, update draws, loglh, & logpost
        print(" - Instantiating Cloud object, update draws, loglikelihood, & log posterior...")
        initialize.draw(loglikelihood, params, data, cloud, free_para_inds, model, parallel = parallel)
        initialize.cloud_settings(cloud, n_parts=n_parts, n_phi=n_phi, c=c, accept=target)

    # Fixed schedule for construction of phi_prop
    if use_fixed_schedule:
        cloud.tempering_schedule = ((np.arange(n_phi))/(n_phi-1))**lamb
    else:
        proposed_fixed_schedule  = ((np.arange(n_phi))/(n_phi-1))**lamb


    # Instantiate incremental and normalized weight matrices for logMDD calculation
    if continue_intermediate:
        weightpath = os.path.join(loadpath, "weight_mat.npz")
        weight_mat = np.load(weightpath)
        w_matrix   = weight_mat["w"]
        W_matrix   = weight_mat["W"]
        j          = weight_mat["j"]
        i          = cloud.stage_index
        c          = cloud.c
        #phi_prop   = proposed_fixed_schedule[j]
        #phi_prop   = cloud.tempering_schedule[j]

    else:
        w_matrix = np.zeros((n_parts,1))
        W_matrix = np.ones((n_parts,1))
    print(" - SMC Recursion starts...")
    #################################################################################
    ### Recursion
    #################################################################################
    while phi_n < 1.0:
        start_time = time.time()

        i += 1
        cloud.stage_index = i

        #---------------------------------------------------------------------------
        # Setting phi_n
        #---------------------------------------------------------------------------

        phi_n1 = cloud.tempering_schedule[i-1]

        if use_fixed_schedule:
            phi_n = cloud.tempering_schedule[i]
        else:
            phi_n, resampled_last_period, j, phi_prop = solve_adaptive_phi(cloud,
                                                           proposed_fixed_schedule,
                                                          i, j, phi_prop, phi_n1,
                                                        tempering_target, resampled_last_period)
        #print(phi_n)
        #---------------------------------------------------------------------------
        # Step 1: Correction
        #---------------------------------------------------------------------------

        # Calculate incremental weights (if no old data, get_old_loglh(cloud) = theta)
        incremental_weights = np.exp((phi_n1 - phi_n) * ptcl.get_old_loglh(cloud) \
                                    + (phi_n - phi_n1) * ptcl.get_loglh(cloud))

        ptcl.update_weights(cloud, incremental_weights)
        mult_weights = ptcl.get_weights(cloud)

        ptcl.normalize_weights(cloud)
        normalized_weights = ptcl.get_weights(cloud)
        #print('norm_w',normalized_weights)
        incremental_weights = incremental_weights.reshape(len(incremental_weights), -1)
        normalized_weights  = normalized_weights.reshape(len(normalized_weights), -1)
        w_matrix = np.hstack([w_matrix, incremental_weights])
        W_matrix = np.hstack([W_matrix, normalized_weights])

        #---------------------------------------------------------------------------
        # Step 2: Selection
        #---------------------------------------------------------------------------

        cloud.ESS = np.append(cloud.ESS, n_parts**2 / sum(normalized_weights**2))

        # If this assertion does not hold then there are likely too few particles
        #assert not np.isnan(cloud.ESS[i]) "No particles have non-zero weight."
        if cloud.ESS[i] < threshold:
        # Resample according to particle weights, uniformly reset weights to 1/n_parts
            new_inds = resample(normalized_weights/n_parts, method = resampling_method)
            #print('new_inds',new_inds)
            temp = copy.deepcopy(cloud.particles)
            for v, k in enumerate(new_inds):
                temp[v, :] = cloud.particles[k,:]
            cloud.particles = temp
            ptcl.reset_weights(cloud)
            cloud.resamples += 1
            resample_last_period = True
            W_matrix[:, i] = 1
            #print('resample particles',ptcl.get_vals(cloud, transpose=False))
        #---------------------------------------------------------------------------
        # Step 3: Mutation
        #---------------------------------------------------------------------------

        # Calculate adaptive c-step for use as scaling coefficient in mutation MH step
        c = c * (0.95 + 0.10 * np.exp(16.0 * (cloud.accept - target)) /
                (1.0 + np.exp(16.0 * (cloud.accept - target))))
        cloud.c = c
        theta_bar = ptcl.weighted_mean(cloud)
        R         = ptcl.weighted_cov(cloud)
        # Ensures marix is positive semi-definite symmetric
        # (not off due to numerical error) and values haven't changed
        #R_fr = (R[free_para_inds,free_para_inds] + R[free_para_inds, free_para_inds].T)/2
        R_fr = (R[free_para_inds][:,free_para_inds] + R[free_para_inds][:,free_para_inds].T)/2
        theta_bar_fr = theta_bar[free_para_inds]

        blocks_free = generate_free_blocks(n_free_para, n_blocks)
        blocks_all  = generate_all_blocks(blocks_free, free_para_inds)

        if parallel:
            result = Parallel(n_jobs=-1, verbose=2)([delayed(mutation_closure)(cloud.particles[k, :],
                                                                               theta_bar_fr, R_fr, blocks_free, blocks_all,
                                                                               phi_n, phi_n1, c=c, alpha=alpha,
                                                                               n_mh_steps=n_mh_steps) for k in range(n_parts)])
            new_particles = np.vstack(result)
        else:
            new_particles = np.vstack([mutation_closure(cloud.particles[k, :], theta_bar_fr, R_fr, blocks_free, blocks_all,
                                                        phi_n, phi_n1, c=c, alpha=alpha,
                                                        n_mh_steps=n_mh_steps) for k in tqdm(range(n_parts))]).T
        ptcl.update_cloud(cloud, new_particles)
        ptcl.update_acceptance_rate(cloud)

        #---------------------------------------------------------------------------
        # Timekeeping and Output Generation
        #---------------------------------------------------------------------------
        stage_sampling_time = (time.time() - start_time)
        cloud.total_sampling_time += stage_sampling_time

        if (np.mod(cloud.stage_index, intermediate_stage_increment) == 0) & save_intermediate:
            inter_path = os.path.join(output_path, r'intermediate', f'stage_{cloud.stage_index:0>3}')
            os.makedirs(inter_path, exist_ok=True)
            cloudpath  = os.path.join(inter_path, "cloud.pkl")
            weightpath = os.path.join(inter_path, "weight_mat")

            pd.to_pickle(cloud, cloudpath)
            # with open(cloudpath, "wb") as f:
            # pickle.dump(cloud, f)
            np.savez(weightpath, w=w_matrix, W=W_matrix, j=j)

    #---------------------------------------------------------------------------
    # Saving data
    #---------------------------------------------------------------------------
    particle_store = np.empty((n_parts,n_para))
    for i in range(n_parts):
        particle_store[i,:] = cloud.particles[i, :n_para]
    cloudpath  = os.path.join(savepath, "cloud.pkl")
    weightpath = os.path.join(savepath, "weight_mat")

    np.save(particle_store_path, particle_store)
    pd.to_pickle(cloud, cloudpath)
    np.savez(weightpath, w=w_matrix, W=W_matrix, j=j)

    return

def solve_adaptive_phi(cloud, proposed_fixed_schedule, i, j, phi_prop, phi_n1,
                       tempering_target, resampled_last_period):
    n_phi = len(proposed_fixed_schedule)

    if resampled_last_period:
        ESS_bar = tempering_target * cloud.particles.shape[0]
        resampled_last_period = False
    else:
        ESS_bar = tempering_target * cloud.ESS[i-1]

    optimal_phi_function = lambda phi: compute_ESS(ptcl.get_loglh(cloud), ptcl.get_weights(cloud),
                                                  phi, phi_n1,
                                                  old_loglh = ptcl.get_old_loglh(cloud)) - ESS_bar
    while (optimal_phi_function(phi_prop) >= 0) & (j <= n_phi):
        phi_prop = proposed_fixed_schedule[j]
        j += 1

    if (phi_prop != 1.0) | (optimal_phi_function(phi_prop) < 0):
        phi_n = optimize.bisect(optimal_phi_function, phi_n1, phi_prop)
        cloud.tempering_schedule = np.append(cloud.tempering_schedule, phi_n)
    else:
        phi_n = 1.0
        cloud.tempering_schedule = np.append(cloud.tempering_schedule, phi_n)

    return phi_n, resampled_last_period, j, phi_prop


def compute_ESS(loglh, current_weights, phi_n, phi_n1, old_loglh):
    N            = len(loglh)
    inc_weights  = np.exp((phi_n1-phi_n) * old_loglh + (phi_n - phi_n1)*loglh)
    new_weights  = current_weights * inc_weights
    norm_weights = N * new_weights / sum(new_weights)
    ESS          = N**2 / sum(norm_weights**2)

    return ESS


def generate_free_blocks(n_free_para, n_blocks):
    rand_inds = np.arange(n_free_para)
    np.random.shuffle(rand_inds)

    subset_length     = math.ceil(n_free_para/n_blocks)
    last_block_length = n_free_para - subset_length*(n_blocks - 1)

    blocks_free = []
    for i in range(n_blocks):
        if i < n_blocks:
            blocks_free.append(rand_inds[(i*subset_length):((i+1)*subset_length)])
        else:
            blocks_free.append(rand_inds[-1-last_block_length+1:])
    return blocks_free

def generate_all_blocks(blocks_free, free_para_inds):
    n_free_para  = len(free_para_inds)
    ind_mappings = dict()
    for k, v in zip(range(n_free_para), free_para_inds):
        ind_mappings[k] = v
    blocks_all = np.empty_like(blocks_free)
    for i, block in enumerate(blocks_free):
        blocks_all[i] = np.empty_like(block)
        for j, b in enumerate(block):
            blocks_all[i, j] = ind_mappings[b]

    return blocks_all
