# Introduction 

This is the codebase for our paper "The Impact of Non-stationarity on Generalisation in Deep Reinforcement Learning
" by M.Igl, G. Farquhar, J. Luketina, W. Boehmer and S. Whiteson.

It also includes an implementation of [IBAC-SNI](https://github.com/microsoft/IBAC-SNI) on ProcGen.

It comprises several sub-folders:
1. `gym-minigrid` contains the grid-world environment (for the Multiroom experiments) and is adapted from
   https://github.com/maximecb/gym-minigrid
   This environment is used together with `torch_rl`
3. `torch_rl` contains the agent to run on the `gym-minigrid` environment and is
   adapted from https://github.com/lcswillems/rl-starter-files
4. `multiroom_exps` contains the training code for the Multiroom experiments.
4. `train-procgen` contains the code for the results on the ProcGen domain. It is code adapted from 
   https://github.com/openai/train-procgen
5. `cifar` contains the code for the supervised experiments. It is code adapted from 
   https://github.com/kuangliu/pytorch-cifar

Plotting is explain at the very end.

# Preparation

All experiments can be run in the accompanying docker container. To build it, call 
```
./build.sh
```
in the root folder.

Then, an interactive docker session can be started with

```bash
./runi.sh <GPU-ID> <Containername>
./runi.sh 0 iter
```
where `<GPU-ID>` is the GPU you want to use and `<Containername>` can be anything or left empty.

# Supervised Learning

After starting interactive session (`./runi.sh` in root folder), move to `cifar` folder:
```bash
cd cifar
```

## Figure 2
Annealing the fraction of correct datapoints from 0 to 1

Run the baseline:
```bash
python main.py -p
```

Run with non-stationarity:
```bash
python main.py -p with annealing.every_n_epochs=1 annealing.type=<type>
```
where `<type>` can either be `size` (=Dataset size), `random` (=Noisy labels) or `consistent` (=Wrong labels). 

## Figure 3 left
Results for self-distillation

For the baseline (i.e. no non-stationarity)
```bash
python main.py -p with epochs=2500 annealing.every_n_epochs=1 self_distillation=1500 
```

And for non-stationarities:
```bash
python main.py -p with epochs=2500 annealing.every_n_epochs=1 self_distillation=1500 annealing.type=<type>
```
where again, please fill in type `<type>` as desired.

## Figure 3 middle
Two phase training

```bash
python main.py -p with epochs=1500 annealing.duration=700 frozen_test_epochs=800 annealing.type=<type> annealing.start_fraction=<fraction>
```

where `<type>` and `<fraction>` should be filled out as desired. 
In the experiments, we used the following values for `<fraction>`. 
```
For Wrong labels and Noisy Labels: 0.05, 0.1 0.2, 0.3, 0.4, 0.5, 0.75, 1.0
Additionally For Dataset Size: 0.005, 0.01, 0.02 + same as for the others
```

# Multiroom

Preparation
* Start the interactive docker session: `runi.sh` in root folder
* Install `gym-minigrid`: `pip install -e gym-minigrid` 
* Install `torch_rl`: `pip install -e torch_rl`
* Move to `multiroom_exps`: `cd multiroom_exps`

Running commands

```bash
python train.py -p with iter_type=<type>
```
where `type` can be either `none` (small caps!) or `distill`

# ProcGen

For ProcGen we need 4 GPUs at once, so we need to start the interactive docker container as 
```bash
./runi.sh 0,1,2,3 procgen
```
for GPUs 0,1,2,3.
Then go to subfolder `cd train-procgen/train_procgen/`

## Running the experiments

Baseline PPO:
```bash
mpiexec -np 4 python train.py -p with env_name=<env_name>
```

where `<env_name>` can be any of the ProcGen environments.
The ones used in this paper were starpilot, dodgeball, climber, ninja and bigfish.

PPO+ITER:
```bash
mpiexec -np 4 python train.py -p with env_name=<env_name> iter_loss.use=True
```

Baseline IBAC:
```bash
mpiexec -np 4 python train.py -p with env_name=<env_name> arch.reg=ibac
```
where we use selective noise injection as well.

IBAC+ITER:
```bash
mpiexec -np 4 python train.py -p with env_name=<env_name> arch.reg=ibac iter_loss.use=True
```

## Ablation studies

Sequential ITER
```bash
mpiexec -np 4 python train.py -p with env_name=<env_name> iter_loss.use=True \
iter_loss.v2=True \
iter_loss.timesteps_initial=71_000_000 \
iter_loss.timesteps_anneal=9_000_000 \
iter_loss.timesteps_free=71_000_000
```
**Careful:** This will generate about 500GB of data!

ITER without RL terms in distillation
```bash
mpiexec -np 4 python train.py -p with env_name=<env_name> iter_loss.use=True \
iter_loss.alpha_reg.schedule=const \
iter_loss.use_burnin_rl_loss=False 
```

# Plotting

The experiments are using [sacred](https://github.com/IDSIA/sacred) for configuration and logging.
For more thorough use of this codebase, I'd recommend setting up a MongoDB to store the results.
At the moment, results are logged using the FileStorageObserver into a `db` folder in the root directory.

There is a very simple plotting script included: `plot.py`:
```bash
python plot.py --id <id> --metric <metric>
```
where `<id>` is the unique experiment id assigned to each run by sacred. 
It is printed in stdout somewhere at the beginning when starting a new run.

`<metric>` is the name of what you want to plot.
 This is `train_acc` and `test_acc` for the supervised experiments, `rreturn_mean` for Multiroom and `eprewmean` for ProcGen.
 Many more things are also being logged, either check the code or `metrics.json` file to see what.
 
 **Special for Procgen:**
 The ProcGen experiments run 4 threads, three for training, one for testing. 
 *Each* of those 4 threads gets one unique id, but only two of those threads are acutally logging something, one the training, one the test performance.
 Just try plotting for each of those 4 ids, it will either crash (if that id wasn't logging) or the plotting script will actually print out whether that's the train or test performance.



