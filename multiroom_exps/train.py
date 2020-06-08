#!/usr/bin/env python3

import sys
import time

import gym
import gym_minigrid
import pymongo
import torch
import torch_rl
import utils
from model import ACModel
from sacred import Experiment
from sacred.observers import FileStorageObserver
# import torch_rl.experimental_config as exp_config
from torch_rl import exp_config

ex = Experiment('iter')
ex.add_config('./default.yaml')
ex.observers.append(FileStorageObserver('../db'))

@ex.config
def config(frames_initial, frames_free, iter_type, experimental):
    """Makes changes to configuration.

    Don't call manually, see sacred for further documentation.
    """
    assert iter_type in ['kickstarting', 'distill', 'smooth', 'none']
    frames_initial = frames_initial or frames_free

    if experimental['also_update_old_policy'] is None:
        experimental['also_update_old_policy'] = True
        if iter_type == "kickstarting":
            experimental['also_update_old_policy'] = False


@ex.capture
def scheduling(num_frames, frames_free, frames_anneal, frames_initial, schedule_function, iter_type):

    """
    Decides what phase we're in and what alpha should be.

    Args:
        num_frames: Current frame step
        frames_free: How many frames between distillation phases
        frames_anneal: How many frames during distillation phase
        frames_initial: How many frames before the first distillation phase
        schedule_function: (String) `linear` or `const`
        iter_type: (String) 'none' = no ITER, 'distill' = draw samples from teacher
            'kickstarting' = draw samples using student

    Returns:
        cycle_count: Starts at 0 in initial RL training phase. Stay at zero if ITER is switched off.
            Is increased by one at the beginning of each distillation phase.
        alpha: Is 0 during normal RL training, e.g. either during initial phase or
            when we have intermediate phases between distillations.
            In the rest of the code, the test alpha > 0 is used to check whether we're in a distillation phase.
            alpha is between 0 and 1 during distillation phase, based on the "schedule_function", e.g.
            "linear" means it goes from 1 to 0 linearly.
    """
    period = frames_free + frames_anneal
    if num_frames < frames_initial or iter_type == 'none':
        # Initial burn in period
        cycle_count = 0
        alpha = 0
    else:
        num_frames -= frames_initial
        cycle_count = (num_frames // period) + 1 # It's alreayd 0 for the burn in period
        adjusted_num_frames = num_frames % period

        x = min(adjusted_num_frames / float(frames_anneal), 1)  # goes from 0->1 for adjusted_num_frames == frame_anneal

        if schedule_function == "linear":
            alpha = 1 - x
        elif schedule_function == "const":
            alpha = 1 if x < 1 else 0
        else:
            raise NotImplementedError(f"schedule_function '{schedule_function}' not implemented")

    return cycle_count, alpha

def switch_training_model(algo, obs_space, envs):
    """Create new model, switch and reset env."""
    new_model = create_model(obs_space, envs)
    algo.switch_models(new_model)
    algo.reset_env()

def create_model(obs_space, envs):
    """Helper function to create new model faster."""
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model= ACModel(obs_space, envs[0].action_space)
    model = model.to(device)
    return model

@ex.capture
def create_algo(envs, pi_train, pi_old, preprocess_obss,
                iter_type, frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                policy_reg_coef, value_reg_coef, value_loss_coef,
                max_grad_norm, optim_eps, clip_eps, epochs, batch_size):
    """Function to be captured by sacred."""
    return torch_rl.PPOAlgo(envs, pi_train, pi_old, iter_type, frames_per_proc, discount, lr, gae_lambda,
                            entropy_coef, policy_reg_coef, value_reg_coef, value_loss_coef, max_grad_norm,
                            optim_eps, clip_eps, epochs, batch_size, preprocess_obss)

@ex.automain
def main(env_name, seed, meta, load_id, procs, fullObs, POfullObs, frames, log_interval, save_interval, experimental, _run):
    """Main function.

    Called by sacred with arguments filled in from default.yaml or command line.
    """

    # Make a bunch of experimental options available everywhere for easy change
    for cfg in experimental:
        setattr(exp_config, cfg, experimental[cfg])

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    model_name = meta['label'] + "_{}".format(_run._id)
    model_dir = utils.get_model_dir(model_name)

    # Define logger, CSV writer and Tensorboard writer
    logger = utils.get_logger(model_dir)
    csv_file, csv_writer = utils.get_csv_writer(model_dir)

    # Log command and all script arguments
    logger.info("{}\n".format(" ".join(sys.argv)))

    # Set seed for all randomness sources
    utils.seed(seed)

    # Generate environments
    envs = []
    for i in range(procs):
        env = gym.make(env_name)
        env.seed(seed + 10000 * i)
        if fullObs:
            env = gym_minigrid.wrappers.FullyObsWrapper(env)
        elif POfullObs:
            env = gym_minigrid.wrappers.PartialObsFullGridWrapper(env)
        envs.append(env)

    # Define obss preprocessor
    obs_space, preprocess_obss = utils.get_obss_preprocessor(env_name, envs[0].observation_space, model_dir)

    # Load training status
    if load_id is not None:
        model1, model2, status = utils.load_status_and_model_from_db(db_uri, db_name, model_dir, load_id)
        if model1 is not None:
            model1 = model1.to(device)
        model2 = model2.to(device)
        acmodels = model1, model2
        current_cycle_count, _ = scheduling(status['num_frames'])

        logger.info("Model successfully loaded\n")
        logger.info("Loaded status: {}".format(status))
    else:
        # First one is pi_old, second one is pi_train
        acmodels = [None, create_model(obs_space, envs)]
        status = {"num_frames": 0, "update": 0}
        current_cycle_count = 0

        logger.info("Model successfully created\n")
    logger.info("{}\n".format(acmodels[0]))

    logger.info("Used device: {}\n".format(device))

    # Define actor-critic algo
    algo = create_algo(envs, *acmodels, preprocess_obss)

    # Train model
    num_frames = status["num_frames"]
    total_start_time = time.time()
    update = status["update"]
    # current_cycle_count = 0

    while num_frames < frames:
        # Update model parameters

        cycle_count, alpha = scheduling(num_frames)

        if  cycle_count != current_cycle_count:
            current_cycle_count = cycle_count
            switch_training_model(algo, obs_space, envs)
            logger.info("Switched training model")

        update_start_time = time.time()
        logs = algo.update_parameters(alpha)
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs
        if update % log_interval == 0:
            fps = logs["num_frames"]/(update_end_time - update_start_time)
            duration = int(time.time() - total_start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value_train", "value_old", "policy_loss_train", "policy_loss_old", "value_loss_train", "value_loss_old"]
            data += [logs["entropy"], logs["value_train"], logs["value_old"], logs["policy_loss_train"], logs["policy_loss_old"], logs["value_loss_train"], logs["value_loss_old"]]
            header += ["grad_norm_train", "grad_norm_old", "alpha", "reg_loss_policy", "reg_loss_value"]
            data += [logs["grad_norm_train"], logs["grad_norm_old"], alpha, logs["reg_loss_policy"], logs["reg_loss_value"]]

            logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V:to {:.3f} {:.3f} "
                    .format(*data[:15]))
            logger.info(
                "pL:to {:.3f} {:.3f} | vL:to {:.3f} {:.3f} | ∇:to {:.3f} {:.3f} | alpha {:.2f} | rLpv {:.3f} {:.3f}\n"
                    .format(*data[15:]))


            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if status["num_frames"] == 0:
                csv_writer.writerow(header)
            csv_writer.writerow(data)
            csv_file.flush()

            for head, dat in zip(header, data):
                _run.log_scalar(head, dat, num_frames)

            status = {"num_frames": num_frames, "update": update}

        # Save vocabulary and model
        if save_interval > 0 and update % save_interval == 0:
            preprocess_obss.vocab.save()

            utils.save_model(algo.pi_old, algo.pi_train, model_dir)
            logger.info("Model successfully saved")
            utils.save_status(status, model_dir)

    utils.save_model_to_db(algo.pi_old, algo.pi_train, model_dir, num_frames, _run)
    utils.save_status_to_db({"num_frames": num_frames, "update": update}, model_dir, num_frames, _run)
