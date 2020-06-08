from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import tensorflow as tf
from baselines.common.mpi_util import setup_mpi_gpus
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecNormalize
)
import math
from baselines import logger
from mpi4py import MPI
from ppo_iter import ppo_iter
from ppo_iter.policies import build_impala_cnn_with_ibac
from procgen import ProcgenEnv
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred import SETTINGS
SETTINGS.CAPTURE_MODE = 'no'

LOG_DIR = './tmp/procgen'

ex = Experiment('procgen')
ex.add_config('./default.yaml')
ex.observers.append(FileStorageObserver('../../db'))



class SacredOutputFormat(logger.KVWriter):
    """
    The original train_procgen codebase uses a proprietary logger.
    This class extends this proprietary logger to log to sacred.
    """
    def __init__(self, _run, factor=1):
        self._run = _run
        self.factor = factor

    def writekvs(self, kvs):
        for k, v in sorted(kvs.items()):
            if k.startswith("distill_v2"):
                # itr = kvs.get("distill_v2/iteration", None)
                self._run.log_scalar(k, v)
            else:
                step = kvs.get("misc/total_timesteps", None)
                if step is not None:
                    step *= self.factor
                self._run.log_scalar(k, v, step)

@ex.config
def config(test_worker_interval, iter_loss, nsteps, num_envs, debug):
    """Sacred config. Don't call manually."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    timestep_factor = comm.Get_size()

    # Needed
    is_test_worker = False
    if test_worker_interval > 0:
        is_test_worker = comm.Get_rank() % test_worker_interval == (test_worker_interval - 1)
        timestep_factor -= 1

    if debug == True:
        nsteps=32
        num_envs=2
        iter_loss["timesteps_anneal"]=300
        iter_loss["timesteps_initial"]=200

    iter_loss["v2_buffer_size"] = math.floor(
        iter_loss["timesteps_anneal"] / nsteps / num_envs)

    del comm


@ex.automain
def main(env_name, paint_vel_info, distribution_mode, num_levels, start_level,
         log_interval, iter_loss, arch, eval,
         num_envs, learning_rate, lr_schedule, ent_coef, gamma, lam, nsteps, nminibatches, ppo_epochs,
         clip_range, timesteps_per_proc, use_vf_clipping, _run,
         is_test_worker, timestep_factor):

    comm = MPI.COMM_WORLD
    log_comm = comm.Split(1 if is_test_worker else 0, 0)
    logger._run = _run

    # Configure logger
    format_strs = ['csv', 'stdout'] if log_comm.Get_rank() == 0 else []
    logger.configure(dir="{}/id_{}".format(LOG_DIR, _run._id), format_strs=format_strs)

    # Add sacred logger:
    if log_comm.Get_rank() == 0:
        logger.get_current().output_formats.append(SacredOutputFormat(_run, timestep_factor))

    num_levels = 0 if is_test_worker else num_levels
    mpi_rank_weight = 0 if is_test_worker else 1
    logger.info("creating environment")
    venv = ProcgenEnv(num_envs=num_envs, env_name=env_name, paint_vel_info=paint_vel_info,
                      num_levels=num_levels, start_level=start_level, distribution_mode=distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)

    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    conv_fn = lambda x: build_impala_cnn_with_ibac(x, iter_loss=iter_loss, arch=arch,
                                                   depths=[16,32,32], emb_size=256)

    logger.info("training")
    ppo_iter.learn(
        env=venv,
        network=conv_fn,
        total_timesteps=timesteps_per_proc,
        ## Iter
        iter_loss=iter_loss,
        arch=arch,
        _run=_run,
        ## Rest
        nsteps=nsteps,
        nminibatches=nminibatches,
        lam=lam,
        gamma=gamma,
        noptepochs=ppo_epochs,
        log_interval=log_interval,
        ent_coef=ent_coef,
        mpi_rank_weight=mpi_rank_weight,
        clip_vf=use_vf_clipping,
        comm=comm,
        learning_rate=learning_rate,
        lr_schedule=lr_schedule,
        cliprange=clip_range,
        vf_coef=0.5,
        max_grad_norm=0.5,
        eval=eval,
    )


