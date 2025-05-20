# IRLEED
This directory is supplementary material for our work presented at the Thirty-Ninth AAAI Conference on Artificial Intelligence (AAAI-25):

"Inverse Reinforcement Learning by Estimating Expertise of Demonstrators" Mark Beliaev and Ramtin Pedarsani.

All relevant citations for methods used are found in the paper's list of references. Note that this directory contains the implementation for the Gridworld experiments performed, as well as the code and data used to generate all figures in the report. For additional implementation details, please refer to the Appendix.

[I. Requirements](#i.-requirements)

[II. Instructions](#ii.-insctructions)

[III. Contents](#iiI.-contents)

## I. Requirements
We recommend using pacakge manager [pip](https://pip.pypa.io/en/stable/) as well as [conda](https://www.anaconda.com/products/individual) to install the relative packages:

**conda:**
- python-3.8.5 [python](https://www.python.org/downloads/release/python-385/)
- numpy-1.19.2 [numpy](https://numpy.org/devdocs/release/1.19.2-notes.html)

**pip:**
- tqdm
- matplotlib
- pandas

## II. Instructions

You can run the scripts directly using the bash script provided:

```bash
bash bash_scripts/run_mix.sh
```
You can also edit parameters accoridgly by using **src/run_mix.py** directly. All parameters have been set to the default used in the paper. Afterwards, use the notebook **analyze.ipynb** to generate the figures. 

## III. Contents

**bash_scripts/** - The bash script used to run the Gridworld experiment. Uses 121 dataset settings, with 100 random initializations for each.

**run_mix.py** script used for running IRLEED and MaxEnt IRL. 

**results/** - Where results are stored. Note that both the Gridoworld and Hopper results are added for reference, where the gridworld results can be generated with the corresponding bash script. 

**analyze.ipynb** - Main analysis file used to generate the figures used in the paper. 

**figures/** - Final storage place of figures.

**src/** - Main implementation of MaxEnt IRL as well as IRLEED. 

- **src/irl_maxent** contains a basic implementation of maximum entropy IRL that is used for reference. Only the plotting, gridworld, and stochastic gradient ascent implementations are used directly from this folder. The original code was forked from [this repository](https://github.com/qzed/irl-maxent), and the corresponding License has been attached. 

- **src/mix_irl** contains the implementation of IRLEED and maximum entroy IRL. Supporting files are included here.

Note that the main IRLEED implementation can be found in the 'irleed' class within the file **src/mix_irl/irleed.py**. This class is then used within the script **src/run_mix.py** to perform the experiment. The remaining files support this implementation. 
