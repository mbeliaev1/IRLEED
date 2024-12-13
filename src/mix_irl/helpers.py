import sys, os
sys.path.append(os.path.abspath(os.path.join('../src')))
from mix_irl import trajectory as T
from mix_irl import irl as I

from irl_maxent import gridworld as W
from irl_maxent import solver as S

import numpy as np

# HELPER METHODS #
def setup_mdp(setting=0):
    """
    Set-up our MDP/GridWorld
    """
    # deterministc 
    if setting == 0:
        size = 5
    elif setting == 1:
        size = 7
    elif setting == 2:
        size = 9

    if setting in [0,1,2]:
        world = W.GridWorld(size=size)
        # set up the reward function
        reward = np.zeros(world.n_states)
        reward[size**2-1] = 1.0
        reward[size-1] = 1.0
        reward[0] = 1.0
        terminal = [0,size-1,size**2-1]
        # remove 0,4,24 from initial states
        initial = np.concatenate([np.arange(1,size-1),np.arange(size,size**2-1)])

    elif setting == 3:
        size = 4
        world = W.GridWorld(size=size)
        # set up the reward function
        reward = np.zeros(world.n_states)
        reward[size**2-1] = 1.0
        terminal = [size**2-2,size**2-1]
        # remove 0,4,24 from initial states
        initial = np.concatenate([np.arange(0,size),np.arange(1,size)*size])

    return world, reward, terminal, initial

def get_traj(world, theta, initial, terminal, horizon, n_traj, discount, weight=None):
    """
    Generate some trajectories, 
    weight determines the suboptimality level
    if weight is None will retern exper trajectory (deterministic)
    """
    if weight is None:
        # deterministic policy
        policy = S.optimal_policy(world, theta, discount, eps=1e-3)
        policy_exec = T.policy_adapter(policy)
        tjs = list(T.generate_trajectories(n_traj, world, policy_exec, initial, terminal, horizon))
    else:
        # stochastic policy
        weighting = lambda x: x**weight
        value = S.stochastic_value_iteration(world.p_transition, theta, discount)
        policy = S.stochastic_policy_from_value(world, value, w=weighting)
        policy_exec = T.stochastic_policy_adapter(policy)
        tjs = list(T.generate_trajectories(n_traj, world, policy_exec, initial, terminal, horizon))

    return tjs, policy

def get_mix_traj(world, initial, terminal, horizon, n_trajs, discount, features, true_reward, weights, epsilons):
    """
    Generate mixture of trajectories, 
    weights determines the suboptimality level and number of trajectories to generate
    if weight is None will retern expert trajectory (deterministic)

    Outputs: list of trajectory & policy class containing len(weights) items each
    """
    mix_traj = []
    mix_policy = []
    for weight, epsilon, n_traj in zip(weights, epsilons, n_trajs):
        reward = features.dot(true_reward+epsilon)
        traj, policy = get_traj(world, reward, initial, terminal, horizon, n_traj, discount, weight) 
        mix_traj.append(traj)
        mix_policy.append(policy)
    return mix_traj, mix_policy

def eval_traj(reward, trajectories):
    '''
    Evaluates reward of given trajectory

    Returns mean and std
    '''
    rews = []
    lens = []
    for t in trajectories:
        lens.append(len(t))
        rew = 0
        for s in t.states():
            rew += reward[s]
        rews.append(rew)
    rews = np.array(rews)
    lens = np.array(lens)

    return rews.mean(), lens.mean()

def eval_theta(setup, theta):
    '''
    Evaluate the current theta by sampling trajectories and computing their performance
    '''
    n_eval_traj = setup['n_e_traj']
    world = setup['world']
    terminal = setup['terminal']
    true_reward = setup['true_reward']
    initial = setup['initial']
    horizon = setup['horizon']
    discount = setup['discount']
    features = setup['features']
    # basic reward and policy eval
    reward = features.dot(theta)
    traj, policy = get_traj(world, reward, initial, terminal, horizon, n_eval_traj, discount)
    
    mean_rew, mean_len = eval_traj(true_reward, traj)

    return mean_rew, mean_len, traj, policy

def get_setup(ratios, weights, lam, n_traj, options, epsilons=None):
    setup = {}
    # save init paramters
    setup['world'], setup['true_reward'], setup['terminal'], setup['initial'] = setup_mdp(options['env_id'])
    setup['lam'] = lam
    setup['ratios'] = ratios
    setup['weights'] = weights
    if epsilons is None:
        if lam>20:
            setup['epsilons']=np.zeros((len(ratios),setup['true_reward'].shape[0]))
        else:
            setup['epsilons'] = np.random.multivariate_normal(np.zeros_like(setup['true_reward']),np.eye(setup['true_reward'].shape[0])/(lam**2),len(ratios))
    setup['epsilons'] = np.array(epsilons)
    setup['n_traj'] = n_traj
    setup['discount'] = options['discount'] 
    setup['horizon'] = options['horizon'] 
    setup['n_e_traj'] = options['n_e_traj']
    setup['n_s_traj'] = options['n_s_traj']
    setup['causal'] = options['causal']
    setup['options'] = options

    # create needed params
    setup['p_transition'] = setup['world'].p_transition
    setup['features'] = W.state_features(setup['world'])
    setup['n_states'], _, setup['n_actions'] = setup['p_transition'].shape

    setup['n'] = len(ratios)
    setup['n_trajs'] = [int(ratio*n_traj) for ratio in ratios]

    setup['mix_traj'], setup['mix_policy'] = get_mix_traj(setup['world'],  
                                                            setup['initial'], 
                                                            setup['terminal'], 
                                                            setup['horizon'], 
                                                            setup['n_trajs'], 
                                                            setup['discount'],
                                                            setup['features'], 
                                                            setup['true_reward'],
                                                            weights,
                                                            setup['epsilons'])
    mix_e_features = []
    mix_p_initial = []
    rews, lens = [], []
    for trajectories in setup['mix_traj']:
        e_features = I.feature_expectation_from_trajectories(setup['features'], trajectories)
        p_initial = I.initial_probabilities_from_trajectories(setup['n_states'], trajectories)
        mix_e_features.append(e_features)
        mix_p_initial.append(p_initial)
        rew, length = eval_traj(setup['true_reward'],trajectories)
        rews.append(rew)
        lens.append(length)
    setup['mix_e_features'] = mix_e_features
    setup['mix_p_initial'] = mix_p_initial
    setup['dem_rews'] = np.mean(rews)
    setup['dem_lens'] = np.mean(lens)
    return setup