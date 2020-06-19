'''
    Copyright (C) 2020  Daniel Duque (see license.txt)
    Module to compute threshold type policies
'''
import pickle
import numpy as np
import multiprocessing as mp
import datetime as dt
from collections import defaultdict
from itertools import product
from SEIYAHRD_sim import simulate, hosp_based_policy, fix_policy, simulate_p


def threshold_policy_search(instance,
                            interventions_train,
                            interventions_test,
                            sd_levels_train,
                            sd_levels_test,
                            cocooning,
                            school_closure,
                            mp_pool=None,
                            n_replicas_train=100,
                            n_replicas_test=100,
                            instance_name=None,
                            policy_class='constant',
                            policy=None):
    '''
        Launches the threshold policy search. A summary of the steps is:

        1. Create iterator for the thresholds to be evaluated
        2. Run deterministic simulations for each threshold
        3. Filter out policies that do not satisfy staffing rule
        4. Run stochastic simulation for survaving polices
        5. Filter out infeasible polices w.r.t to chance constraint
        6. Return best policy according to objective function

        Outputs a pickle file with the results of the simulation and the all the inputs.
    '''
    sim_configs = policy_input_iterator(instance,
                                        interventions_train,
                                        sd_levels_train,
                                        cocooning,
                                        school_closure,
                                        policy_class=policy_class,
                                        fixed_policy=policy)
    all_outputs = simulate_p(mp_pool, sim_configs)
    
    best_cost, best_sim, best_policy, best_params = np.Inf, None, None, None
    if len(all_outputs) == 1:
        sim_output, cost, z_out, kwargs_out = all_outputs[0]
        best_cost = cost
        best_sim = sim_output
        best_policy = z_out
        best_params = kwargs_out
    else:
        for system_i_out in all_outputs:
            sim_output, cost, z_out, kwargs_out = system_i_out
            if cost <= instance.T:
                # Staffing rule feasible
                kwargs_out['opt_phase'] = False  # not longer optimizing
                out_sample_configs = out_of_sample_iterator(n_replicas_train,
                                                            instance,
                                                            interventions_train,
                                                            sd_levels_train,
                                                            cocooning,
                                                            school_closure,
                                                            kwargs_out,
                                                            seed_shift=0)
                out_sample_outputs = simulate_p(mp_pool, out_sample_configs)
                stoch_replicas = [rep_i[0] for rep_i in out_sample_outputs]
                IH_feasible = np.sum([
                    np.any(stoch_replicas[rep_i]['IH'].sum(axis=(1, 2)) > instance.hosp_beds)
                    for rep_i in range(len(out_sample_outputs))
                ]) == 0
                Expected_cost = np.mean([
                    np.sum([interventions_train[k].social_distance >= 0.9 for k in stoch_replicas[rep_i]['z']])
                    for rep_i in range(len(out_sample_outputs))
                ])
                if Expected_cost < best_cost and IH_feasible:
                    best_cost = Expected_cost
                    best_sim = sim_output
                    best_policy = z_out
                    best_params = kwargs_out
            else:  # Staffing rule is infeasible.
                pass
    print_params = [
        'moving_avg_len', 'hosp_rate_threshold_1', 'hosp_rate_threshold_2', 'hosp_rate_threshold_3',
        'hosp_level_release'
    ]
    print(f'Simulations: {len(all_outputs)}')
    print(best_cost, [(k, best_params[k]) for k in print_params])
    
    best_params['opt_phase'] = False  # not longer optimizing
    out_sample_configs = out_of_sample_iterator(n_replicas_test,
                                                instance,
                                                interventions_test,
                                                sd_levels_test,
                                                cocooning,
                                                school_closure,
                                                best_params,
                                                seed_shift=1000)
    
    out_sample_outputs = simulate_p(mp_pool, out_sample_configs)
    stoch_replicas = [rep_i[0] for rep_i in out_sample_outputs]
    # Save solution
    instance_name = instance_name if instance_name is not None else f'threshold_sol_{instance.ins_name}.p'
    file_name = f'./InterventionsMIP/output/{instance_name}.p'
    with open(file_name, 'wb') as outfile:
        pickle.dump(
            (instance.summary, interventions_test, best_params, best_policy, best_sim, best_params, stoch_replicas),
            outfile, pickle.HIGHEST_PROTOCOL)
    
    return stoch_replicas, best_params


def run_calendar(instance, interventions, cal, T, sd_levels, cocooning, school_closure):
    '''
        Runs the calendar to fix decisions regarding school closures,
        cocooning and social distance. If decisions are not fixed,
        creates a list of feasible interventions for every time period.
        In principle, there are only two options, lock-down or relaxation,
        but the actual intervention related to either lock-down or relaxation
        changes across time in accordance to what is fixed (e.g., school calendar)

        Args:
        instance (module): python module with input data
        interventions (list): list of all interventions to be consider in the horizon
        cal (SimCalendar): an instance of the simulation calendar
        T (int): number of days being simulated
        sd_levels (dict): a map from lock-down/relaxation to transmission reduction (kappa)
        cocooning (float): level of transmission reduction, [0,1], for high risk and 65+ groups
        school_closure (int): 1 schools are closed, 0 schools are open unless is fixed otherwise

        Output:
            z_ini (ndarray): vector of the interventions id (position in the interventions list)
            SD_state (ndarray): vector of the lock-down state ("H" is lock-down, "L"' is relaxation)
            feasible_interventions (list of dic): feasible interventions on every time period for
                "H" and "L" states.
    '''
    
    # Run callendar and set what's already decided
    int_dict = {(i.SC, i.CO, i.SD): ix for ix, i in enumerate(interventions)}
    z_ini = np.array([None] * (T - 1))
    SD_state = np.array([None] * (T - 1))
    feasible_interventions = []
    for t in range(T - 1):
        d = cal.calendar[t]
        # School calendar  WARNING: THIS IS FOR AUSTIN ONLY
        # TODO: REMOVE school calendar from this function
        #       it should come from the instance instead.
        sc = school_closure
        sc_start_1 = dt.datetime(2020, 3, 14)
        sc_end_1 = dt.datetime(2020, 8, 18)
        sc_start_2 = dt.datetime(2021, 5, 26)
        sc_end_2 = dt.datetime(2021, 8, 23)
        
        if sc_start_1 <= d <= sc_end_1 or sc_start_2 <= d <= sc_end_2:
            sc = 1
        
        if d < instance.lockdown_start:
            z_ini[t] = int_dict[sc, 0, 0]
            feasible_interventions.append({'H': z_ini[t], 'L': z_ini[t]})
            SD_state[t] = 'NA'
        elif instance.lockdown_start <= d < instance.lockdown_end:
            z_ini[t] = int_dict[sc, 0, 0.95]
            feasible_interventions.append({'H': z_ini[t], 'L': z_ini[t]})
            SD_state[t] = 'H'
        elif instance.lockdown_end <= d < instance.school_closure_end:
            z_ini[t] = None
            feasible_interventions.append({
                'H': int_dict[sc, cocooning, sd_levels['H']],
                'L': int_dict[sc, cocooning, sd_levels['L']],
            })
            SD_state[t] = None
        elif d < instance.last_day_interventions:
            # Assumption: if High, schools need to be closed
            z_ini[t] = None
            feasible_interventions.append({
                'H': int_dict[1, cocooning, sd_levels['H']],
                'L': int_dict[sc, cocooning, sd_levels['L']],
            })
            SD_state[t] = None
        else:
            z_ini[t] = None
            feasible_interventions.append({
                'H': int_dict[0, 0, 0],
                'L': int_dict[0, 0, 0],
            })
            SD_state[t] = None
    return z_ini, SD_state, feasible_interventions


def out_of_sample_iterator(n_replicas, instance, interventions, sd_levels, cocooning, school_closure, params_policy,
                           seed_shift):
    '''
        Creates an iterator for different replicas, changing the seed for the random stream.
        The iterator will be used to map the simulator in parallel using the helper function
        simulate_p on the simulation module.
        
        Args:
            n_replicas (int): number of stochastic simulations
            instance (module): python module with input data
            interventions (list): list of all interventions to be consider in the horizon
            sd_levels (dict): a map from lock-down/relaxation to transmission reduction (kappa)
            cocooning (float): level of transmission reduction, [0,1], for high risk and 65+ groups
            school_closure (int): 1 schools are closed, 0 schools are open unless is fixed otherwise.
            params_policy (dict): paramters of the policy to be simulated. The signature is this
                dictionary comes from kwargs built in the function policy_input_iterator.
    '''
    epi, T, A, L, N, I0, hosp_beds, lambda_star, cal = instance.summary
    z_ini, SD_state, feasible_interventions = run_calendar(instance, interventions, cal, T, sd_levels, cocooning,
                                                           school_closure)
    for rep_i in range(-1, n_replicas):
        r_seed = rep_i + seed_shift if rep_i >= 0 else rep_i
        sim_setup = epi, T, A, L, interventions, N, I0, cal, r_seed
        kwargs = params_policy.copy()
        kwargs['hosp_beds'] = hosp_beds
        kwargs['lambda_star'] = lambda_star
        kwargs['SD_state'] = SD_state.copy()
        kwargs['feasible_interventions'] = feasible_interventions
        
        yield sim_setup, z_ini.copy(), hosp_based_policy, pos_deviation_lambda, kwargs


def policy_input_iterator(instance,
                          interventions,
                          sd_levels,
                          cocooning,
                          school_closure,
                          policy_class='constant',
                          fixed_policy=None):
    '''
        Creates an iterator of the candidate thresholds. The iterator will be used
        to map the simulator in parallel using the helper function simulate_p on the
        simulation module.
        
        Args:
        instance (module): a python module with all the required input
        interventions (list): list of all interventions to be consider in the horizon
        sd_levels (dict): a map from lock-down/relaxation to transmission reduction (kappa)
        cocooning (float): level of transmission reduction, [0,1], for high risk and 65+ groups
        school_closure (int): 1 schools are closed, 0 schools are open unless is fixed otherwise.
        policy_class (str): class of policy to optimize. Options are:
            "constant": optimizes one parameter and safety threshold
            "step": optimizes three parameters (first threshold, second threshold, and last month in
                    which it changes to the second threshold) and safety threshold
            "linear": optimize a threshold of the form a + b * t, and safety threshold
            "quad": optimize a threshold of the form a + b * t + c * t ^ 2, and safety threshold
        fixed_policy (dict): if provided, no search is excecuted and the iterator yields one policy.
            Signature of fixed_policy = { policy_class: "a class listed above",
                                          vals: [val1, val2, val3]
                                        }
    '''
    epi, T, A, L, N, I0, hosp_beds, lambda_star, cal = instance.summary
    sim_setup = epi, T, A, L, interventions, N, I0, cal, -1
    first_day_month_index = defaultdict(int)
    first_day_month_index.update({(d.month, d.year): t for t, d in enumerate(cal.calendar) if (d.day == 1)})
    
    z_ini, SD_state, feasible_interventions = run_calendar(instance, interventions, cal, T, sd_levels, cocooning,
                                                           school_closure)
    
    mov_avg_trial = [7]
    
    # Define the grid to search
    trials_a, trials_b, trials_c = [0], [0], [0]
    if fixed_policy is None:
        lambda_star_scaled = int(np.floor(np.mean(epi.gamma_IH) * lambda_star))
        if policy_class == 'constant':
            grid_size = 5
            trials_a = [grid_size * i for i in range(1, int(lambda_star_scaled / grid_size)) + 1] + [lambda_star_scaled]
        elif policy_class == 'step':
            grid_size = 10
            trials_a = [grid_size * i for i in range(1, int(lambda_star_scaled / grid_size) + 1)] + [lambda_star_scaled]
            trials_b = trials_a.copy()
            trials_c = [6, 7, 8, 9, 10]
            print(trials_a, trials_b)
        else:
            trials_a = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 110, 120]
            trials_b = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  # [0, 0.1, 0.2, 0.3, 0.4]
            trials_c = [0.0, 0.001, 0.002, 0.003, 0.004]  #, 0.001]
    else:
        # If a policy is given, we use the policy.
        policy_class = fixed_policy['class']
        if policy_class == 'constant':
            trials_a = [fixed_policy['vals'][0]]
        elif policy_class == 'step':
            trials_a = [fixed_policy['vals'][0]]
            trials_b = [fixed_policy['vals'][1]]
            trials_c = [fixed_policy['vals'][2]]
        else:
            raise 'Not implemented'
    
    release_level_trials = hosp_beds * np.array([0.6])  #[0.5, 0.6, 0.7, 0.8, 0.9])
    for ma, hrt_1, hrt_2, hrt_3, hlr in product(mov_avg_trial, trials_a, trials_b, trials_c, release_level_trials):
        kwargs = {
            'hosp_beds': hosp_beds,
            'lambda_star': lambda_star,
            'opt_phase': True,
            'first_day_month_index': first_day_month_index
        }
        kwargs['moving_avg_len'] = ma
        kwargs['hosp_rate_threshold_1'] = hrt_1
        kwargs['hosp_rate_threshold_2'] = hrt_2
        kwargs['hosp_rate_threshold_3'] = hrt_3
        kwargs['hosp_level_release'] = hlr
        kwargs['baseline_enforcing_time'] = 14
        kwargs['lockdown_enforcing_time'] = 1
        lambda_star_scaled = np.mean(epi.gamma_IH) * lambda_star
        
        if policy_class in ['linear', 'quad']:
            kwargs['IYIH_threshold'] = [
                np.minimum(lambda_star_scaled, hrt_1 + hrt_2 * t + hrt_3 * (t**2)) for t, d in enumerate(cal.calendar)
            ]
        elif policy_class == 'constant':
            kwargs['IYIH_threshold'] = [hrt_1 for t, d in enumerate(cal.calendar)]
        else:
            date_change = dt.datetime(year=2020, month=8, day=30)
            kwargs['IYIH_threshold'] = [
                np.minimum(lambda_star_scaled, hrt_1 if (d.year == 2020 and d.month <= hrt_3) else hrt_2)
                #np.minimum(lambda_star_scaled, hrt_1 if (d < date_change) else hrt_2)
                for t, d in enumerate(cal.calendar)
            ]
        kwargs['feasible_interventions'] = feasible_interventions
        kwargs['SD_state'] = SD_state.copy()
        
        yield sim_setup, z_ini.copy(), hosp_based_policy, pos_deviation_lambda, kwargs


def pos_deviation_lambda(sim_setup, sim_output, **kwargs):
    '''
        Evaluates the objective function for a simlation output
        Args:
            sim_setup (tuple): input of the simulation
            sim_output (dict): time series of the compartments
            ** kwargs: paramters passed along to compute cost
    '''
    epi = sim_setup[0]
    IYIH = np.sum(sim_output['IYIH'], axis=(-1, -2))
    z = sim_output['z']
    # root(lambda x: x + 4 * np.sqrt(x) - hosp_beds, hosp_beds) (for hosp_beds not normalized)
    lambda_star = kwargs['lambda_star']
    lambda_star_scaled = np.mean(epi.gamma_IH) * lambda_star
    diff_IYIH_above = 1E7 * np.maximum(0, IYIH - lambda_star_scaled)
    high_sd_day = np.sum(kwargs['SD_state'] == 'H')
    return diff_IYIH_above.sum() + high_sd_day


if __name__ == '__main__':
    from interventions import Intervension
    from epi_params import EpiSetup
    import instances.austin as instance
    
    n_proc = 4
    n_replicas_train = 350
    n_replicas_test = 350
    # Create the pool (Note: pool needs to be created only once to run on a cluster)
    mp_pool = mp.Pool(n_proc) if n_proc > 1 else None
    
    # Classes: step or constant
    policy_class = 'step'
    # If a specific policy is selected (insted of optimizing), uncomment
    my_policy = None
    # my_policy = {'class': policy_class, 'vals': [90, 216, 9]}
    
    # Set up of the optimization
    for sc in [0]:  # Schools closuse (1) or open (0)
        for co in [0.95]:  # Level of cocooning
            for base_line_train in [0.4]:  # Transmission reduction for training
                for base_line_test in [0.4]:  # Transmission reduction for testing
                    for const in ['opt']:  #
                        instance_name = f'local_{instance.city}_SC{sc}_CO{co}_BLTrain{base_line_train}_BLTest_{base_line_test}_{policy_class}_{const}'
                        print('\n============================================')
                        print(instance_name)
                        interventions_train = [
                            Intervension(0, 0, 0, instance.epi, instance.N),
                            Intervension(1, 0, 0, instance.epi, instance.N),
                            Intervension(0, 0, base_line_train, instance.epi, instance.N),
                            Intervension(1, 0, base_line_train, instance.epi, instance.N),
                            Intervension(1, 0, 0.9, instance.epi, instance.N),
                            Intervension(0, co, base_line_train, instance.epi, instance.N),
                            Intervension(1, co, base_line_train, instance.epi, instance.N),
                            Intervension(1, co, 0.9, instance.epi, instance.N),
                            Intervension(1, 0, 0.95, instance.epi, instance.N),
                            Intervension(0, 0, 0.95, instance.epi, instance.N)
                        ]
                        interventions_test = [
                            Intervension(0, 0, 0, instance.epi, instance.N),
                            Intervension(1, 0, 0, instance.epi, instance.N),
                            Intervension(0, 0, base_line_test, instance.epi, instance.N),
                            Intervension(1, 0, base_line_test, instance.epi, instance.N),
                            Intervension(1, 0, 0.9, instance.epi, instance.N),
                            Intervension(0, co, base_line_test, instance.epi, instance.N),
                            Intervension(1, co, base_line_test, instance.epi, instance.N),
                            Intervension(1, co, 0.9, instance.epi, instance.N),
                            Intervension(1, 0, 0.95, instance.epi, instance.N),
                            Intervension(0, 0, 0.95, instance.epi, instance.N)
                        ]
                        sd_level_train = {'H': 0.9, 'L': base_line_train}
                        sd_level_test = {'H': 0.9, 'L': base_line_test}
                        best_policy_replicas, policy_params = threshold_policy_search(instance,
                                                                                      interventions_train,
                                                                                      interventions_test,
                                                                                      sd_level_train,
                                                                                      sd_level_test,
                                                                                      cocooning=co,
                                                                                      school_closure=sc,
                                                                                      mp_pool=mp_pool,
                                                                                      n_replicas_train=n_replicas_train,
                                                                                      n_replicas_test=n_replicas_test,
                                                                                      instance_name=instance_name,
                                                                                      policy=my_policy,
                                                                                      policy_class=policy_class)
