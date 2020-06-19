'''
Copyright (C) 2020  Daniel Duque (see license.txt)
Author: Daniel Duque
Date: Jun 19, 2020

This module runs a simulation of the SEIYARDH model for a single city,
considering different age groups and seven compartments. This model is
based of Zhanwei's Du model, Meyer's Group at UT and is distributed
under the GLPv3 license.

This module also contains functions to run the simulations in parallel
and a class to properly define a calendar (SimCalendar).
'''
import datetime as dt
import numpy as np
from utils import timeit, roundup


def simulate(epi, T, A, L, interventions, N, I0, z, policy, calendar, seed=-1, **kwargs):
    '''
    Simulates an SIR-type model with seven compartments, multiple age groups,
    and risk different age groups:
    Compartments
        S: Susceptible
        E: Exposed
        IY: Infected symptomatic
        IA: Infected asymptomatic
        IH: Infected hospitalized
        R: Recovered
        D: Death
    Connections between compartments:
        S-E, E-IY, E-IA, IY-IH, IY-R, IA-R, IH-R, IH-D

    Args:
        epi (EpiParams): instance of the parameterization
        T(int): number of time periods
        A(int): number of age groups
        L(int): number of risk groups
        F(int): frequency of the  interventions
        interventions (list of Intervention): list of inteventions
        N(ndarray): total population on each age group
        I0 (ndarray): initial infected people on each age group
        z (ndarray): interventions for each day
        policy (func): callabe to get intervention at time t
        calendar (SimCalendar): calendar of the simulation
        seed (int): seed for random generation. Defaults is -1 which runs
            the simulation in a deterministic fashion.
        kwargs (dict): additional parameters that are passed to the policy function
    '''
    # Random stream for stochastic simulations
    rnd_stream = np.random.RandomState(seed) if seed >= 0 else None
    epi.update_rnd_stream(rnd_stream)
    
    # Compartments
    types = 'int' if seed >= 0 else 'float'
    S = np.zeros((T, A, L), dtype=types)
    E = np.zeros((T, A, L), dtype=types)
    IA = np.zeros((T, A, L), dtype=types)
    IY = np.zeros((T, A, L), dtype=types)
    IH = np.zeros((T, A, L), dtype=types)
    R = np.zeros((T, A, L), dtype=types)
    D = np.zeros((T, A, L), dtype=types)
    
    # Additional tracking variables (for triggers)
    IYIH = np.zeros((T, A, L))
    
    # Initial Conditions (assumed)
    IY[0] = I0
    R[0] = 0
    S[0] = N - E[0] - IA[0] - IY[0] - D[0] - R[0]
    
    # Rates of change
    rate_IYR = np.array([[(1 - epi.pi[a, l]) * epi.gamma_IY[a] for l in range(L)] for a in range(A)])
    rate_IAR = np.tile(epi.gamma_IA, (L, 1)).transpose()
    rate_IYH = np.array([[(epi.pi[a, l]) * epi.Eta[a] for l in range(L)] for a in range(A)])
    rate_IHD = np.array([[epi.death[a, l] * epi.mu[a] for l in range(L)] for a in range(A)])
    rate_IHR = np.array([[(1 - epi.death[a, l]) * epi.gamma_IH[a] for l in range(L)] for a in range(A)])
    step_size = 1
    for t in range(T - 1):
        # Get dynamic intervention and corresponding contact matrix
        k_t = policy(t, z, IH=IH[:t], IYIH=IYIH[:t], **kwargs)
        phi_t = interventions[k_t].phi(calendar.is_weekday(t))
        # Epidemic dynamics
        
        # Dynamics for dS
        # TODO Vectorized dS might save 50% of the time
        dSprob = np.zeros((A, L), dtype='float')
        for a in range(A):
            for l in range(L):
                beta_t_a = {(a_, l_): epi.beta * phi_t[a, a_, l, l_] / step_size for a_ in range(A) for l_ in range(L)}
                dSprob[a, l] = sum(
                    beta_t_a[a_, l_]
                    * (epi.omega_E[a_, l_] * E[t, a_, l_] + epi.omega_IA * IA[t, a_, l_] + epi.omega_IY * IY[t, a_, l_])
                    / N[a_, l_] for a_ in range(A) for l_ in range(L))
        
        dS = rv_gen(rnd_stream, S[t], dSprob)
        S[t + 1] = S[t] - dS
        
        # Dynamics for E
        E_out = rv_gen(rnd_stream, E[t], epi.sigma_E / step_size)
        E[t + 1] = E[t] + dS - E_out
        
        # Dynamics for IA
        IAR = rv_gen(rnd_stream, IA[t], rate_IAR / step_size)
        EIA = rv_gen(rnd_stream, E_out, (1 - epi.tau))
        IA[t + 1] = IA[t] + EIA - IAR
        
        # Dynamics for IY
        EIY = E_out - EIA
        IYR = rv_gen(rnd_stream, IY[t], rate_IYR / step_size)
        IYIH[t] = rv_gen(rnd_stream, IY[t] - IYR, rate_IYH / step_size)
        IY[t + 1] = IY[t] + EIY - IYR - IYIH[t]
        
        # Dynamics for IH
        IHR = rv_gen(rnd_stream, IH[t], rate_IHR / step_size)
        IHD = rv_gen(rnd_stream, IH[t] - IHR, rate_IHD / step_size)
        IH[t + 1] = IH[t] + IYIH[t] - IHR - IHD
        
        # Dynamics for R
        R[t + 1] = R[t] + IHR + IYR + IAR
        
        # Dynamics for D
        D[t + 1] = D[t] + IHD
        
        # Validate simulation: checks we are not missing people
        # for a in range(A):
        #     for l in range(L):
        #         pop_dif = (
        #             np.sum(S[t, a, l] + E[t, a, l] + IA[t, a, l] + IY[t, a, l] + IH[t, a, l] + R[t, a, l] + D[t, a, l])
        #             - N[a, l])
        #         assert pop_dif < 1E2, f'Pop unbalanced {a} {l} {pop_dif}'
        total_imbalance = np.sum(S[t] + E[t] + IA[t] + IY[t] + IH[t] + R[t] + D[t]) - np.sum(N)
        assert np.abs(total_imbalance) < 1E2, f'fPop unbalanced {total_imbalance}'
    
    # Additional output
    # Change in compartment S, flow from S to E
    dS = S[1:, :] - S[:-1, :]
    # flow from IY to IH
    output = {
        'S': S,
        'E': E,
        'IA': IA,
        'IY': IY,
        'IH': IH,
        'R': R,
        'D': D,
        'dS': dS,
        'IYIH': IYIH,
        'z': np.array(z),
    }
    
    return output


def rv_gen(rnd_stream, n, p):
    if rnd_stream is None:
        return n * p
    else:
        return rnd_stream.binomial(n, p)


def system_simulation(mp_sim_input):
    '''
        Simulation function that gets mapped when running
        simulations in parallel.
    '''
    sim_setup, z, policy_fun, cost_fun, kwargs = mp_sim_input
    epi, T, A, L, interventions, N, I0, cal, seed = sim_setup
    out_sim = simulate(epi, T, A, L, interventions, N, I0, z, policy_fun, cal, seed, **kwargs)
    policy_cost = cost_fun(sim_setup, out_sim, **kwargs)
    return out_sim, policy_cost, z, kwargs


@timeit
def simulate_p(mp_pool, input_iter):
    '''
    Launches simulation in parallel
    Args:
        mp_pool (Pool): pool to parallelize
        input_ite (iterator): iterator with the inputs to parallelize.
            Input signature: sim_setup, z, policy_fun, cost_fun, kwargs
            sim_setup signature: epi, T, A, L, interventions, N, I0, cal, seed
    '''
    if mp_pool is None:
        results = []
        for sim_input in input_iter:
            results.append(system_simulation(sim_input))
        return results
    else:
        results = mp_pool.map(system_simulation, input_iter)
        return results


def dummy_cost(*args, **kwargs):
    return 0


def fix_policy(t, z, *args, **kwargs):
    '''
        Returns the intervention according to a
        fix policy z
        Args:
            t (int): time of the intervention
            z (ndarray): fix policy
    '''
    return z[t]


def hosp_based_policy(t, z, opt_phase, moving_avg_len, IYIH_threshold, hosp_level_release, baseline_enforcing_time,
                      lockdown_enforcing_time, feasible_interventions, SD_state, IYIH, IH, **kwargs):
    '''
        Lock-down and relaxation policy function. This function returns the
        intervention to be used at time t, according to the thresholds that
        are given as paramters.
        Args:
            t (int): time step of the simulation
            z (ndarray): vector with the interventions
            opt_phase (bool): True if optimization phase is happening, false otherwise
            moving_avg_len (int): number of days to compute IYIH moving average
            IYIH_threshold (list): threshold values for each time period
            hosp_level_release (float): value of the safety trigger for total hospitalizations
            baseline_enforcing_time (int): number of days relaxation is enforced
            lockdown_enforcing_time (int): number of days lockdown is enforced
            feasible_interventions (list of dict): list of feasible interventions. Dictionary
                has the signature of {'H': int, 'L': int}.
            SD_state (list): toggle history to keep track of whether at time t there is lockdown
                or relaxation.
            IYIH (ndarray): daily admissions, passed by the simulator
            IYIH (ndarray): hospitalizations admissions, passed by the simulator
            ** kwargs: additional parameters that are passed and are used elsewhere
    '''
    # If the state is already set, apply it right away
    if SD_state[t] is not None or t == 0:
        return z[t]
    
    # Compute daily admissions moving average
    moving_avg_start = np.maximum(0, t - moving_avg_len)
    IYIH_total = IYIH.sum((1, 2))
    IYIH_avg = IYIH_total[moving_avg_start:].mean()
    
    # Get threshold for time t
    hosp_rate_threshold = IYIH_threshold[t]
    # Get valid intervention at time t
    valid_t = feasible_interventions[t]
    
    if SD_state[t - 1] == 'L':  # If relaxation is currently in place
        if IYIH_avg >= hosp_rate_threshold:
            t_end = np.minimum(t + lockdown_enforcing_time, len(z))
            z[t:t_end] = valid_t['H']  # Turn on heavy SD from t to t_end
            SD_state[t:t_end] = 'H'
        else:
            z[t] = valid_t['L']  # Keep baseline
            SD_state[t] = 'L'
    elif SD_state[t - 1] == 'H':  # If lock-down is currently in place
        IH_total = IH[-1].sum()
        if IH_total <= hosp_level_release and IYIH_avg < hosp_rate_threshold:
            t_end = np.minimum(t + baseline_enforcing_time, len(z))
            z[t:t_end] = valid_t['L']  # Turn off heavy SD from t to t_end
            SD_state[t:t_end] = 'L'
        else:
            z[t] = valid_t['H']  # keep heavy social distance
            SD_state[t] = 'H'
    else:
        raise f'Unknown state/intervention {SD_state[t-1]} {z[t-1]}'
    
    return z[t]


class SimCalendar():
    '''
        A simulation calendar to map time steps to days. This class helps
        to determine whether a time step t is a weekday or a weekend, as well
        as school calendars.

        Attrs:
        start (datetime): start date of the simulation
        calendar (list): list of datetime for every time step
    '''
    def __init__(self, start_date, sim_length):
        '''
            Arg
        '''
        self.start = start_date
        self.calendar = [self.start + dt.timedelta(days=t) for t in range(sim_length)]
        self._is_weekday = [d.weekday() not in [5, 6] for d in self.calendar]
        self.lockdown = None
        self.schools_closed = None
        # TODO: Add school calendar from the input
    
    def is_weekday(self, t):
        '''
            True if t is a weekday, False otherwise
        '''
        return self._is_weekday[t]
    
    def load_initial_lockdown(self, start_date, end_date=dt.datetime.today()):
        # Deprecated
        # TODO update this function
        self.lockdown = [False if d < start_date else True for d in self.calendar if d <= end_date]
    
    def load_school_closure(self, start_date, end_date=dt.datetime.today()):
        # Deprecated
        # TODO update this function
        self.schools_closed = [False if d < end_date or d > end_date else True for d in self.calendar]
