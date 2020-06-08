import pymongo
import gridfs
import os
import time
import numpy as np
import os.path as osp
from baselines import logger
import tensorflow as tf
from baselines.common.mpi_util import sync_from_root
import psutil

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

## Requires MongoDB backend to work
db_uri = None
db_name = 'iter'

def get_docs(url, db, col):
    client = pymongo.MongoClient(url, ssl=True)
    return client[db][col]

def get_file_id(doc, file_name):
    """
    Helper function to access data when MongoObserver is used.
    Go through all files in doc and return the id of the file with file_name.
    """
    r = list(filter(lambda dic: dic['name'] == file_name, doc['artifacts']))
    assert len(r) == 1
    return r[0]['file_id']


def save_file_from_db(file_id, destination, db_uri, db_name):
    """
    Given a file_id (e.g. through get_file_id()) and a db_uri (a db connection string),
    save the corresponding file to `destination` (filename as string).
    """
    client = pymongo.MongoClient(db_uri, ssl=True)
    fs = gridfs.GridFSBucket(client[db_name])
    open_file = open(destination, 'wb+')
    fs.download_to_stream(file_id, open_file)

def constfn(val):
    def f(_):
        return val
    return f

def get_alpha(x, alpha_config):
    x = min(max(x, 0), 1)
    thresh = alpha_config['thresh']
    x = (x * alpha_config["nr_phases"]) % 1

    # alpha should go from 1 -> 0
    # alpha is 1 during free phase
    if x == 0:
        alpha = 1
    else:
        # We're guaranteed to be in the burnin phase
        if alpha_config['schedule'] == "linear":
            alpha = 1 - x
        elif alpha_config['schedule'] == "relu":
            # alpha = min(1, 2 * (1 - x))
            alpha = min(1, 1. / (1 - thresh) * (1 - x))
        elif alpha_config['schedule'] == "const":
            # If x == 0 we're still in the free phase and want alpha=1
            alpha = alpha_config['const']
        elif alpha_config['schedule'] == "step":
            alpha = 1 if x < thresh else 0
    return alpha


def scheduling(num_timesteps,
               iter_loss,
               alpha_name):
    initial_period = iter_loss['timesteps_initial'] + iter_loss['timesteps_anneal']
    period = iter_loss['timesteps_free'] + iter_loss['timesteps_anneal']
    if not iter_loss['use']:
        # Initial burn in period
        cycle_count = 0
        alpha = 0
        burnin_phase = False
    else:

        # One "Cycle" is the "free" training phase + the next burnin phase
        # The first cycle uses the "initial" timesteps instead of "free"
        if num_timesteps < initial_period:
            # period_timesteps is negative for "free" training and positive
            # during the burnin phase
            # It's maximum should be iter_loss["timesteps_anneal"]
            period_timesteps = num_timesteps - iter_loss['timesteps_initial']
            cycle_count = 0
        else:
            # Substract initial phase timesteps
            # num_timesteps -= iter_loss['timesteps_initial']
            num_timesteps -= initial_period
            cycle_count = (num_timesteps // period) + 1
            period_timesteps = (num_timesteps % period) - iter_loss["timesteps_free"]

        assert period_timesteps <= iter_loss["timesteps_anneal"], \
            "'period_timesteps' should be smaller than 'timesteps_anneal'"

        # period_timesteps goes from -free to +anneal
        # x is negative during free phase
        # x goes from 0->1 during anneal and is 1 for period_timesteps == frame_anneal
        x = period_timesteps / float(iter_loss['timesteps_anneal'])

        # During the 'free' phase, x=0, alhpa=1, but we still don't want regularization,
        # So we need this additional variable
        burnin_phase = period_timesteps >= 0

        alpha = get_alpha(x, alpha_config=iter_loss[alpha_name])
    # cycle_count tells us how many past cycles we already had. If that matches "number_cycles" or is higher
    # we just keep alpha=1 and burnin_phase=False at the current cycle count
    if iter_loss["number_cycles"] >= 0 and cycle_count >= iter_loss["number_cycles"]:
        cycle_count = iter_loss["number_cycles"]
        alpha = 1
        burnin_phase = False

    return cycle_count, alpha, burnin_phase



def get_lr_fn(lr_schedule, start_learning_rate, end_learning_rate=0):
    if lr_schedule is not None:
        if lr_schedule == "linear":
            # f goes from 1 to 0
            lr = lambda f: f * start_learning_rate + (1 - f) * end_learning_rate
    else:
        lr = constfn(start_learning_rate)
    return lr

def save_model(model, name, update, _run):
    checkdir = osp.join(logger.get_dir(), 'checkpoints')
    os.makedirs(checkdir, exist_ok=True)
    savepath = osp.join(checkdir, '{}_{}'.format(name, update))
    print('Saving to', savepath)
    model.save(savepath)
    _run.add_artifact(savepath)

def switch_training_model(update,  is_mpi_root, model_train, _run, iter_loss, session, comm,
                          save=True):
    if is_mpi_root and save:
        save_model(model_train, "model", update, _run)
    # Copy train -> Old, overwriting burnin "burnin" parameters
    vars_train = tf.get_collection(tf.GraphKeys.VARIABLES, scope="ppo_iter_train")
    vars_burnin = tf.get_collection(tf.GraphKeys.VARIABLES, scope="ppo_iter_burnin")
    if not iter_loss["dont_switch_just_reset_burnin"]:
        # Copy variables over from burnin to train
        print("Switching variables")
        for train_var in vars_train:
            # Get var name: Remove the first part of the name:
            var_name = "/".join(train_var.name.split("/")[1:])
            # Construct burnin var name by prepending the name with "ppo_iter_burnin"
            burnin_var_name = "/".join(["ppo_iter_burnin", var_name])
            # Find the burnin var
            burnin_var = [v for v in tf.global_variables() if v.name == burnin_var_name][0]
            # Assign it the "train" value
            session.run(tf.assign(train_var, burnin_var))
    else:
        print("NOT switching variables")

    print("Re-initialize burnin variables")
    # Reinitialize variables in "burnin". Should make them random again.
    re_init_train_op = tf.initialize_variables(vars_burnin)
    session.run(re_init_train_op)

    global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
    if MPI is not None:
        sync_from_root(session, global_variables, comm=comm) #pylint: disable=E1101


def get_all_burnin_data_dict(env, iter_loss, nsteps, comm):
    nr_datapoints = env.num_envs * iter_loss["v2_buffer_size"] * nsteps

    # Required data for _all_ processes
    required_memory = (nr_datapoints * 64 * 64 * 3
                       + nr_datapoints * (8 + 3 * 4)) * comm.Get_size()

    print("Required memory (GB): {}".format(required_memory / 2**30))
    print("Available memory (GB): {}".format(psutil.virtual_memory().available
                                             / 2**30))
    curr_available = comm.allgather(psutil.virtual_memory().available)
    print(curr_available)
    curr_available = min(curr_available)
    while curr_available < required_memory:
        print("Memory: {}/{} available".format(
            curr_available / 2**30,
            required_memory / 2**30
        ))
        time.sleep(60)
        curr_available = min(comm.allgather(psutil.virtual_memory().available))

    dd_size = nsteps * iter_loss["v2_buffer_size"]
    dd = {
        "obs": np.zeros(
            shape=(dd_size, env.num_envs, 64, 64, 3),
            dtype=np.uint8),
        "actions": np.zeros(
            shape=(dd_size, env.num_envs),
            dtype=np.int64),
        "values": np.zeros(
            shape=(dd_size, env.num_envs),
            dtype=np.float32),
        "neglogpacs": np.zeros(
            shape=(dd_size, env.num_envs),
            dtype=np.float32),
        "returns": np.zeros(
            shape=(dd_size, env.num_envs),
            dtype=np.float32),
    }
    # Fill with dummy values to request memory
    for i in range(dd_size):
        dd["obs"][i] = i % 255
        dd["actions"][i] = i % 255
        dd["values"][i] = i % 255
        dd["neglogpacs"][i] = i % 255
        dd["returns"][i] = i % 255
    return dd

# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def get_str_from_number(number):
    assert number < 1e6
    return f'{int(number):06}'

def get_path(number, id):
    stringnr = get_str_from_number(number)
    st = 3
    pathlist = ["./", f"{id}_data"] + [stringnr[i:i + st] for i in range(0, len(stringnr), st)]
    return os.path.join(*pathlist[:-1])

def save_data(burnin_data, start_idx, id, nsteps, prefix=""):
    # nsteps, nenvs, *obs_shape = burnin_data["obs"].shape
    for idx in range(nsteps):
        path = get_path(start_idx + idx, id)
        try:
            os.makedirs(path)
        except FileExistsError:
            pass
        save_dict = {key: value[idx] for key, value in burnin_data.items()}
        stringnr = get_str_from_number(start_idx + idx)
        np.save(os.path.join(path, f"{prefix}{stringnr}.npy"), save_dict)


def load_data(idx, id, prefix=""):
    path = get_path(idx, id)
    stringnr = get_str_from_number(idx)
    data_dict = np.load(os.path.join(path, f"{prefix}{stringnr}.npy"), allow_pickle=True)
    return data_dict[()]

def load_batch(batch_idxs, id, nenvs, prefix="", pi_size=None):
    batch_size = len(batch_idxs)
    if prefix == "":
        obs = np.empty(shape=(batch_size, nenvs, 64, 64, 3), dtype=np.uint8)
        actions = np.empty(shape=(batch_size, nenvs), dtype=np.int64)
        values = np.empty(shape=(batch_size, nenvs), dtype=np.float32)
        neglogpacs = np.empty(shape=(batch_size, nenvs), dtype=np.float32)
        returns = np.empty(shape=(batch_size, nenvs), dtype=np.float32)

        for i, idx in enumerate(batch_idxs):
            dd = load_data(idx, id, prefix)
            obs[i] = dd['obs']
            actions[i] = dd['actions']
            values[i] = dd['values']
            neglogpacs[i] = dd['neglogpacs']
            returns[i] = dd['returns']

        return {
            "obs": obs,
            "actions": actions,
            "values": values,
            "neglogpacs": neglogpacs,
            "returns": returns
        }
    else:
        teacher_values = np.empty(shape=(batch_size, nenvs), dtype=np.float32)
        teacher_pis = np.empty(shape=(batch_size, nenvs, pi_size), dtype=np.float32)

        for i, idx in enumerate(batch_idxs):
            dd = load_data(idx, id, prefix)
            teacher_values[i] = dd['teacher_values']
            teacher_pis[i] = dd['teacher_pis']

        return {
            "teacher_values": teacher_values,
            "teacher_pis": teacher_pis
        }

