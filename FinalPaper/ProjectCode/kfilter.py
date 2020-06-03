import numpy as np
from scipy.linalg import solve_discrete_lyapunov

def kalman_filter(y, model, everything=False):
    """
    From the state space model as follows ,

    Transition equation
        x_t = TT * x_{t-1} + C + RR * eps_t, eps_t ~ N(0, QQ)

    Measurement equation
        y_t = ZZ * x_{t} + D + u_t, u_t ~ N(0, EE)

    Use Kalman Filter to estimate log likelihood.
    Written based on NYFED-StateSpaceRoutines' Julia-language implementation.


    Input
    ------
    TT : Ns x Ns state transition matrix
    RR : Ns x Ne shock term in the transition equation
    C  : Ns x 1  constant vector in the transition equation
    QQ : Ne x Ne shock covariances
    ZZ : Ny x Ns mapping states to observables in the measurement equation
    D  : Ny x 1  constant vector in the measurement equation
    EE : Ny x Ny measurement error covariances

    Output
    -------
    log_lik_vals : log likelihood values
    x_forecast   : forecast value of x
    P_forecast   : forecast value of P
    x_filter     : filtered value of x
    P_filter     : filtered value of P
    x_0          : initial value of x
    P_0          : initial value of P

    """
    TT, RR, C, ZZ, D, QQ, EE = [v.value for v in model.System.values()]

    if model.model_type == 'HANK':
        # Adjust TT to account for continuous time discretization
        TT = np.eye(TT.shape[0]) + TT * 1/model.settings['state_simulation_freq'].value
        track_lag = model.settings['track_lag'].value

        TT, RR, C= transform_transition_matrices(model, TT, RR, C, track_lag=track_lag)

    Ns = TT.shape[0] ## Num. of states
    Ny, Nt = y.shape  ## Periods

    x_forecast   = np.zeros([Ns, Nt])
    P_forecast   = np.zeros([Ns, Ns, Nt])
    x_filter     = np.zeros([Ns, Nt])
    P_filter     = np.zeros([Ns, Ns, Nt])
    log_lik_vals = np.zeros(Nt)

    x_0, P_0 = initial_state(TT, RR, C, QQ)

    x_t, P_t = x_0, P_0

    for t in range(Nt):
        ## Forecast
        x_filt, P_filt = x_t, P_t
        x_t = TT @ x_filt + C
        P_t = TT @ P_filt @ TT.T + RR @ QQ @ RR.T
        x_forecast[:, t]  = x_t
        P_forecast[:,:,t] = P_t

        ## Update
        y_obs = y[:, t]
        x_pred, P_pred = x_t, P_t
        y_pred = ZZ @ x_pred + D
        V_pred = ZZ @ P_pred @ ZZ.T + EE
        V_pred = (V_pred + V_pred.T)/2
        V_pred_inv = np.linalg.inv(V_pred)
        diff_y = y_obs - y_pred
        PZV = P_pred.T @ ZZ.T @ V_pred_inv
        x_t = x_pred + PZV @ diff_y
        P_t = P_pred - PZV @ ZZ @ P_pred
        x_filter[:, t] = x_t
        P_filter[:, :, t] = P_t

        log_lik_vals[t] = -(Ny * np.log(2 * np.pi) + np.log(np.linalg.det(V_pred)) \
                           + diff_y.T @ V_pred_inv @ diff_y) * 0.5

    if everything:
        return log_lik_vals, x_forecast, P_forecast, x_filter, P_filter, x_0, P_0, x_t, P_t

    return log_lik_vals


def initial_state(TT, RR, C, QQ):
    """
    Compute the initial state x_0 and state covariance matrix P_0
    """

    Ns = TT.shape[0]
    e = np.linalg.eigvals(TT)

    if all(np.abs(e) < 1.0):
        x_0 = np.linalg.inv(np.eye(Ns) - TT) @ C
        P_0 = solve_discrete_lyapunov(TT, RR @ QQ @ RR.T)
    else:
        x_0 = C
        P_0 = 1e+6 * np.eye(Ns)

    return x_0, P_0


def transform_transition_matrices(model, TT, R, C, track_lag=True):
    freq = model.settings['state_simulation_freq'].value
    if track_lag:
        TTT = np.zeros([TT.shape[0] * (freq+1), TT.shape[1] * (freq+1)])
        RRR = np.zeros([R.shape[0] * (freq+1), R.shape[1] * freq])
    else:
        TTT = np.zeros([TT.shape[0] * freq, TT.shape[1] * freq])
        RRR = np.zeros([R.shape[0] * freq, R.shape[1] * freq])

    # Construct powers of TT iteratively
    TT_powers = dict()
    for i in range(freq):
        if i == 0:
            TT_powers[i] = TT
        else:
            TT_powers[i] = TT_powers[i-1] @ TT

    # Insert directly into TTT
    if track_lag:
        for i in range(freq+1):
            if i == 0:
                TTT[:TT.shape[0], TT.shape[1]*freq:] = np.eye(TT.shape[0])
            else:
                TTT[TT.shape[0]*i:TT.shape[0]*(i+1), TT.shape[1]*freq:] = TT_powers[i-1]
    else:
        for i in range(freq):
            TTT[TT.shape[0]*i:TT.shape[0]*(i+1), TT.shape[1]*(freq-1):] = TT_powers[i]

    # Create RRR
    TR_powers = dict()
    TR_powers[-1] = R

    # Construct T^i * R iteratively
    for i in range(freq-1):
        TR_powers[i] = TT @ TR_powers[i-1]

    # Fill in RRR and CCC
    if track_lag:
        RRR[R.shape[0]:R.shape[0]*2, :R.shape[1]] = TR_powers[-1]
        for i in range(2,freq+1):
            for j in range(i-1, -1, -1):
                RRR[R.shape[0]*i:R.shape[0]*(i+1), R.shape[1]*(j-1):R.shape[1]*j] = TR_powers[i-j-1]
        CCC = np.vstack([np.zeros(C.shape[0]), np.repeat(C, freq)])
    else:
        RRR[:R.shape[0], :R.shape[1]] = TR_powers[-1]
        for i in range(1,freq):
            for j in range(i, -1, -1):
                RRR[R.shape[0]*i:R.shape[0]*(i+1), R.shape[1]*j:R.shape[1]*(j+1)] = TR_powers[i-j-1]
        CCC = np.repeat(C, freq)

    return TTT, RRR, CCC
