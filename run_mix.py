import sys, os
ROOT = os.path.abspath(os.curdir)
sys.path.append(os.path.abspath(os.path.join(ROOT,'src')))

import mix_irl.irleed as I
import numpy as np
import pickle
import argparse
from tqdm import trange


def run_irleed(options):
    result = {}
    if ARGS.weight_scale > 20:
        weights = [None]*5
    else:
        weights = np.random.rand(5)*ARGS.weight_scale
    algo = I.irleed()
    algo.reset_data(options['ratios'], weights, options['lam'], options['n_traj'], options)
    result['log'] = algo.run_irleed(outer_eps=1e-4,inner_eps=1e-4,max_steps=options['max_steps'])
    result['dem_rews'] = algo.setup['dem_rews']
    result['dem_lens'] = algo.setup['dem_lens']
    result['mix_e_features'] = algo.setup['mix_e_features']
    return result

def main():
    # pathing 
    save_dir = ROOT+'/results/%s/env_%d/%.3f'%(ARGS.save_dir,ARGS.env_id,ARGS.weight_scale)
    try: os.mkdir(save_dir)
    except: pass
    # create place to save data and options
    options = {}
    # configure options
    options['discount'] = 0.9
    options['horizon'] = 100
    options['n_e_traj'] = 100
    options['n_s_traj'] = 100
    options['env_id'] = ARGS.env_id
    options['lr_betas'] = ARGS.lr_betas
    options['lr_theta'] = ARGS.lr_theta
    options['lr_epsilons'] = ARGS.lr_epsilons
    options['debug'] = ARGS.debug
    options['max_steps'] = ARGS.max_steps
    options['ratios'] = [0.2]*5
    options['n_traj'] = 200
    options['exp_key'] = ARGS.exp_key
    options['causal'] = ARGS.causal

    # values used in experiment
    for lam in [2,2.5,3,3.5,4,4.5,5,5.5,6,10,100]:
        data = []
        options['lam'] = lam
        save_path = save_dir + '/lam_%.3f.p'%lam
        if os.path.isfile(save_path):
            pass
        else:
            for seed in trange(ARGS.n_seeds):
                try:
                    result = run_irleed(options)
                except:
                    # only relevant for values of lambda lower than 1
                    print('skipped lam %.3f seed %d'%(lam,seed))
                    result = None
                data.append(result)
            pickle.dump([options,data], open(save_path,'wb'))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--weight_scale', type=float, default=1, help="scale of demonstrator accuracies")
    parser.add_argument('--lr_betas', type=float, default=0.05, help="learning rate for beta")
    parser.add_argument('--lr_theta', type=float, default=0.2, help="learning rate for theta")
    parser.add_argument('--lr_epsilons', type=float, default=0.1, help="learning rate for epsilon")
    parser.add_argument('--save_dir', type=str, default='irleed', help="directory to save to, will be created")
    parser.add_argument('--n_seeds', type=int, default=100, help="number of seeds to run")
    parser.add_argument('--env_id', type=int, default=1, help="env id")
    parser.add_argument('--max_steps', type=int, default=2, help="number of steps to run for")
    parser.add_argument('--exp_key', type=str, default='1-4', help="key of experiment to run")
    parser.add_argument('--debug', action='store_true', help="displays results while running")
    parser.add_argument('--causal', action='store_true', help="decides if we use causal IRL")
    ARGS = parser.parse_args()
    print(ARGS)
    
    main()