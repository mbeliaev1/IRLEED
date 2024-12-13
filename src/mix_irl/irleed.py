import sys, os
sys.path.append(os.path.abspath(os.path.join('../src/')))

import irl_maxent.gridworld as W
import irl_maxent.optimizer as O

import mix_irl.irl as I
import mix_irl.helpers as H

import numpy as np
from itertools import product       

class irleed:
    def __init__(self):
        # initialized in self.reset_data()
        self.setup  = None
        self.names  = None

        # initialized in self._reset_log()
        self.log = None

        # initialized in self._reset_params()
        self.params = None
        self.optims = None
        
        self.t = None
        

    def reset_data(self, ratios, weights, lam, n_traj, options):
        self.setup = H.get_setup(ratios, weights, lam, n_traj, options)
        # get names and norms
        self.setup['names'] = ['Dem_%d'%(i) for i in range(self.setup['n'])]
        self.setup['norms'] = [np.linalg.norm(self.setup['epsilons'][i]) for i in range(self.setup['n'])]

    def _reset_params(self):
        '''
        Initialize params we will be learning along with their optimizers
        NOTE: may want individual resets
        '''
        self.params = {}
        self.optims = {}
        _, n_features = self.setup['features'].shape

        # theta
        init = O.Constant(0.1)
        self.params['theta'] = init(n_features)
        self.optims['theta'] = O.ExpSga(lr=O.linear_decay(lr0=self.setup['options']['lr_theta']))
        self.optims['theta'].reset(self.params['theta'])

        # epsilons
        # init = O.Uniform(-0.01,0.01)
        init = O.Constant(0)
        self.params['epsilons'] = []
        self.optims['epsilons'] = []
        for i in range(self.setup['n']):
            self.params['epsilons'].append(init(n_features))
            # linear SGA allows negative values
            self.optims['epsilons'].append(O.Sga(lr=O.linear_decay(lr0=self.setup['options']['lr_epsilons'])))
            self.optims['epsilons'][i].reset(self.params['epsilons'][i])
        # betas
        init = O.Constant(1.0)
        self.params['betas'] = init(self.setup['n'])
        self.optims['betas'] = O.ExpSga(lr=O.linear_decay(lr0=self.setup['options']['lr_betas']))
        self.optims['betas'].reset(self.params['betas'])

    def _reset_log(self):
        self.log = {}
        self.log['betas'] = []
        self.log['delta_betas'] = []
        self.log['theta'] = []
        self.log['delta_theta'] = []
        self.log['epsilons'] = []

        self.log['norms'] = []
        self.log['rewards'] = []
        self.log['traj_lens'] = []
        
    def update_log(self, reward, traj_len, norms, delta_theta, delta_betas):
        # store params
        self.log['theta'].append(self.params['theta'].copy())
        self.log['betas'].append(self.params['betas'].copy())
        self.log['epsilons'].append([self.params['epsilons'][i].copy() for i in range(self.setup['n'])])
        # store deltas
        self.log['delta_theta'].append(delta_theta)
        self.log['delta_betas'].append(delta_betas)

        self.log['norms'].append(norms)
        self.log['rewards'].append(reward)
        self.log['traj_lens'].append(traj_len)

        if self.setup['options']['debug']:
            print('%d \t|%.5f \t| %.5f \t| %.5f \t| %.5f \t| %.5f \t|'%(self.t, 
                                                                        reward, 
                                                                        traj_len, 
                                                                        np.array(norms).mean(), 
                                                                        delta_betas, 
                                                                        delta_theta))
    
    def compute_grads(self):
        '''
        step one iteration of the modified algorithm
        returns gradient computations
        does not modify parameters, but uses set values (self.params)
        '''
        grads = {}
        grads['theta'] = np.zeros_like(self.params['theta'])
        grads['betas'] = np.zeros_like(self.params['betas'])
        grads['epsilons'] = np.zeros_like(self.params['epsilons'])

        p_initial = np.array(self.setup['mix_p_initial']).mean(axis=0)
        for i, e_features, m, beta, epsilon in zip(np.arange(self.setup['n']),
                                                   self.setup['mix_e_features'], 
                                                   self.setup['n_trajs'], 
                                                   self.params['betas'],
                                                   self.params['epsilons']):
            
            reward = self.setup['features'].dot(self.params['theta']+epsilon)
            # CAUSAL
            if self.setup['causal']:
                e_svf = I.compute_expected_causal_svf(self.setup['p_transition'], 
                                                      p_initial, 
                                                      self.setup['terminal'], 
                                                      reward*beta,
                                                      self.setup['discount'],)
                s_features = self.setup['features'].T.dot(e_svf)
            # SAMPLE BASED
            else:
                s_traj, _ = H.get_traj(self.setup['world'], 
                                    reward, 
                                    p_initial, 
                                    self.setup['terminal'], 
                                    self.setup['horizon'], 
                                    self.setup['n_s_traj'], 
                                    self.setup['discount'],
                                    beta)
                s_features = I.feature_expectation_from_trajectories(self.setup['features'], s_traj)
            # compute gradients
            div = e_features - s_features
            grads['theta'] += beta*(m/self.setup['n_traj'])*div
            grads['betas'][i] = div.dot(self.params['theta']+epsilon)
            # NOTE: may not need beta constant here
            grads['epsilons'][i] = beta*div
        return grads
        
    
    def run_irl(self,eps=1e-3,max_steps=100):
        '''
        runs regular irl, only updating theta
        returns log
        Inputs:
            eps       - threshold for delta theta per iteration
            max_steps - threshold for max steps til termination
        '''

        delta_theta = np.inf
        # setup logging
        inner_t = 0
        while (delta_theta > eps) and (inner_t<max_steps):
            inner_t += 1
            # print(inner_t)
            theta_old = self.params['theta'].copy()
            grads = self.compute_grads()
            # only step theta when running irl
            self.optims['theta'].step(grads['theta'])
            delta_theta = np.max(np.abs(theta_old - self.params['theta']))

    
    def run_irleed(self,outer_eps=1e-3,inner_eps=1e-3,max_steps=100):
        '''
        runs irleed
        Inputs:
            outer_eps - outer loop threshold for delta theta per iteration
            inner_eps - inner loop threshold for delta theta per iteration
            max_steps - threshold for max steps til termination
        NOTE: first iteration corresponds to vanilla IRL
        '''
        # reset everything
        self.t = 0
        self._reset_log()
        self._reset_params()

        # setup logging
        if self.setup['options']['debug']:
            print('Running IRLEED with Debug')
            print('Iter \t| Rew \t\t| Len \t\t|  eps-Norm \t| Del B \t| Del Th \t|')
            print('--------------------------'*4)

        delta_theta = np.inf
        while (delta_theta > outer_eps) and (self.t<max_steps):
            theta_old = self.params['theta'].copy()
            betas_old = self.params['betas'].copy()

            # first run irl to estimate theta under current paramters
            _ = self.run_irl(eps=inner_eps)
            # reset the optimizer's learning rate
            self.optims['theta'].reset(self.params['theta'])
            # compute the remaining gradients and update betas and epsilons
            grads = self.compute_grads()
            self.optims['betas'].step(grads['betas'])
            for i in range(self.setup['n']):
                self.optims['epsilons'][i].step(grads['epsilons'][i])

            # eval
            delta_theta = np.max(np.abs(theta_old - self.params['theta']))
            delta_betas = np.max(np.abs(betas_old - self.params['betas']))
            reward, traj_len, _, _ = H.eval_theta(self.setup, self.params['theta'])
            norms = [np.linalg.norm(self.params['epsilons'][i]) for i in range(self.setup['n'])]
            self.update_log(reward, 
                            traj_len, 
                            norms, 
                            delta_theta, 
                            delta_betas)

            self.t += 1
        return self.log
