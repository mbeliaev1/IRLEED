# for original experiment results, run these 11 experiments corresponding to 11 settings of $\beta$. 
# note that $\lambda$ is set within the script. 
python run_mix.py --save_dir gridworld --weight_scale 0.4 
python run_mix.py --save_dir gridworld --weight_scale 0.5 
python run_mix.py --save_dir gridworld --weight_scale 1 
python run_mix.py --save_dir gridworld --weight_scale 1.5 
python run_mix.py --save_dir gridworld --weight_scale 2 
python run_mix.py --save_dir gridworld --weight_scale 2.5 
python run_mix.py --save_dir gridworld --weight_scale 3 
python run_mix.py --save_dir gridworld --weight_scale 3.5 
python run_mix.py --save_dir gridworld --weight_scale 4 
python run_mix.py --save_dir gridworld --weight_scale 4.5 
python run_mix.py --save_dir gridworld --weight_scale 5 

 