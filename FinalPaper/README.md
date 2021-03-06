# DSGE model in Python

This code is written for the research project in MACS 30250 Perspectives on Computational Research for Economics (Spring 2020), University of Chicago.

This Python-language DSGE model implementation is created by referring to [New York Fed DSGE model](https://github.com/FRBNY-DSGE/DSGE.jl) written in Julia and [SeHyoun Ahn](https://sehyoun.com/EXAMPLE_one_asset_HANK_web.html)'s code written in Matlab.

## Contents
- Build DSGE model
  - Representative Agent New Keynesian model is based on An and Schorfheide (2007) (`build_RANK`)
  - One asset Heterogeneous Agent New Keynesian model is based on Ahn et al. (2017) (`build_OneAssetHANK`)
  - You can change parameters, sampling methods, prior distributions etc as you want
- Solve DSGE model via a solution method of Sims(2002) and Generate state space system (`solve_**`)
- Estimate structure parameters by Bayesian approach + Markov Chain Monte Carlo (`estimate`)

## Structure
```
├── HANK                             ## Construct and solve One asset HANK model
│   ├── aux.py
│   ├── build_OneAssetHANK.py
│   ├── eqcond.py
│   ├── measurement.py
│   ├── reduction.py
│   └── solve_HANK.py
├── RANK                             ## Construct and solve RANK model
│   ├── build_RANK.py
│   ├── eqcond.py
│   ├── measurement.py
│   └── solve_RANK.py
├── MCMC                             ## Implement Markov Chain Monte Carlo
│   ├── MH.py                         ### Metropolis hastings algorithm
│   ├── SMC.py                        ### Sequential Monte Carlo (SMC)
│   ├── initialize.py                  #### for SMC
│   ├── mutation.py                    #### for SMC
│   ├── particle.py                    #### for SMC
│   └── resample.py                    #### for SMC
├── estimate.py                      ## Implement Bayesian estimation with MCMC
├── CsminWel.py                      ## Optimization algorithm for finding initial states
├── hessian.py                       ## Calculate Hessian inverse
├── StateSpace.py                    ## Construct state space system
├── kfilter.py                       ## Implement Kalman filter
├── dist.py                          ## Provide statistical distributions
├── posterior.py                     ## Calculate log likelihood and prior log pdf
└── util.py                          ## Utility functions

```

## Example
```Python
from ProjectCode import estimate
import numpy as np
data = np.loadtxt('data/macro_data.csv')

# Estimate RANK model with MH algorithm (default)
# CAUTION: Take more than one hour to finish
estimate.run(data, "RANK",
            　continue_intermediate=False,
            　save_intermediate=True)

# Estimate one asset HANK model with SMC algorithm (default)
# CAUTION: Take more than 20-30 hours to finish
estimate.run(data, "HANK",
              continue_intermediate=False,
              save_intermediate=True,
              intermediate_stage_increment=1)

# Estimation result will be saved at `output` folder at current directory
```
