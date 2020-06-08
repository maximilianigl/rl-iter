import os
import shutil
import time
import numpy as np
from baselines import logger
from collections import deque
import tensorflow as tf
from baselines.common import explained_variance, set_global_seeds
from ppo_iter.policies import build_policy
from baselines.common.tf_util import get_session
from baselines.common.mpi_util import sync_from_root
from ppo_iter.utils import get_docs, get_file_id, save_file_from_db, constfn, get_alpha
from ppo_iter.utils import scheduling, get_lr_fn, save_model, switch_training_model, get_all_burnin_data_dict
from ppo_iter.utils import safemean, save_data, load_batch
from ppo_iter.utils import db_uri, db_name
from ppo_iter.model import Model

try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from ppo_iter.runner import Runner



def learn(*, network, env, total_timesteps, iter_loss, arch, _run,
          seed=None, nsteps=2048, ent_coef=0.0, learning_rate=3e-4, lr_schedule=None,
          vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
          log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
          load_path=None, mpi_rank_weight=1, comm=None,
          eval=None, **network_kwargs):
    '''
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Parameters:
    ----------
    network:
       The network model. Will only work with the one in this repo because of IBAC
    env: baselines.common.vec_env.VecEnv
    total_timesteps: int
         number of timesteps (i.e. number of actions taken in the environment)
    iter_loss: dict
        the config dict as specified in default.yaml and/or overwritting by command line arguments
        see sacred for further documentation
    arch: dict
        config dict similar to iter_loss
    eval: dict
        config dict similar to iter_loss
    _run:
        sacred Experiment._run object. Used for logging
    ent_coef: float
        policy entropy coefficient in the optimization objective
    seed: float
        random seed
    nsteps: int
        number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
        nenv is number of environment copies simulated in parallel)
    ent_coef: float
        value function loss coefficient in the optimization objective
    learning_rate: float
        learning rate
    lr_schedule: None or str
        If None, use a const. learning rate. If string, only "linear" is implemented at the moment
    vf_coef: float
        Coefficient for vf optimisation
    max_grad_norm: flaot
        Max gradient norm before it's clipped
    gamma: float
        Discount factor
    lam: float
        For GAE
    log_interval: int
        number of timesteps between logging events
    nminibatches: int
        number of training minibatches per update. For recurrent policies,
        should be smaller or equal than number of environments run in parallel.
    noptepochs: int
        number of training epochs per update
    cliprange: float or function
        clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
        and 0 is the end of the training
    save_interval: int
        number of timesteps between saving events
    load_path: str
        path to load the model from
    **network_kwargs:
        keyword arguments to the policy / network builder.
        See baselines.common/policies.py/build_policy and arguments to a particular type of network
        For instance, 'mlp' network architecture has arguments num_hidden and num_layers.
    '''
    # Set learning rate schedule
    lr = get_lr_fn(lr_schedule, start_learning_rate=learning_rate)

    set_global_seeds(seed)
    session = get_session()

    # if isinstance(lr, float): lr = constfn(lr)
    # else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)
    model_fn = Model

    policy = build_policy(env, network, arch, **network_kwargs)

    # Instantiate the model object (that creates act_model and train_model)
    def create_model(scope_name, **kwargs):
        return model_fn(scope_name=scope_name, policy=policy, ob_space=ob_space, ac_space=ac_space,
                        nbatch_act=nenvs, nbatch_train=nbatch_train,
                        nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                        max_grad_norm=max_grad_norm, comm=comm, mpi_rank_weight=mpi_rank_weight,
                        iter_loss=iter_loss, arch=arch, **kwargs)

    # model_train is the teacher and always executed
    # model_burnin is trained. If teacher and student are swapped, the parameters from burnin are
    # copied into the teacher and burnin is re-initialized
    model_train = create_model("ppo_iter_train")
    model_burnin = create_model("ppo_iter_burnin",
                                target_vf=model_train.train_model.vf_run,
                                target_dist_param=model_train.train_model.pi_run)

    get_session().run(tf.variables_initializer(tf.global_variables()))
    global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
    if MPI is not None:
        sync_from_root(session, global_variables, comm=comm)  # pylint: disable=E1101

    if load_path is not None:
        print("Load model...")
    if eval["load_id"]:
        # Only works with mongodb as backend, not with tinydb
        raise NotImplementedError("Requires MongoDB backend to work")
        docs = get_docs(db_uri, db_name, "runs")
        projection = {'config': True}
        projection.update({'artifacts': True})

        doc = docs.find_one({'_id': eval["load_id"]}, projection)
        print("Loading model from db to disc")
        file_id = get_file_id(doc, eval["file_name"])
        load_path = os.path.join(logger.get_dir(), "loadmodel_{}".format(_run._id))
        save_file_from_db(file_id, load_path , db_uri, db_name)
        model_train.load(load_path)
        if eval["switch_after_load"]:
            switch_training_model(0, is_mpi_root, model_train, _run, iter_loss, session, comm,
                                  save=False)

    # Instantiate the runner object
    runner = Runner(env=env, model=model_train, model_burnin=model_burnin, nsteps=nsteps, gamma=gamma, lam=lam,
                    iter_loss=iter_loss, eval=eval)

    epinfobuf = deque(maxlen=100)

    burnin_data_idx = 0
    all_burnin_data = None

    assert iter_loss["timesteps_anneal"] > iter_loss["v2_buffer_size"] * env.num_envs * nsteps, \
    "{}, {}".format(iter_loss["timesteps_anneal"], iter_loss["v2_buffer_size"] * env.num_envs * nsteps)

    # Start total timer
    tfirststart = time.perf_counter()

    nupdates = total_timesteps//nbatch
    current_cycle_count = 0
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        num_timesteps = update * nbatch
        # Start timer
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)

        # Calculate the cliprange
        cliprangenow = cliprange(frac)

        # 'Burnin_phase' tells us whether we need regularization
        cycle_count, alpha_reg, burnin_phase = scheduling(num_timesteps, iter_loss, "alpha_reg")

        if  cycle_count != current_cycle_count:
            current_cycle_count = cycle_count
            if iter_loss["v2"]:
                logger.info("Training student")
                train_student(
                    teacher=model_train,
                    student=model_burnin,
                    data=all_burnin_data,
                    iter_loss=iter_loss,
                    lr=lrnow,
                    cliprange=cliprangenow,
                    nminibatches=nminibatches,
                    session=session,
                    max_idx=burnin_data_idx,
                    nenvs=env.num_envs,
                    nsteps=nsteps,
                    id=_run._id,
                )
            switch_training_model(update, is_mpi_root, model_train, _run, iter_loss, session, comm)
            # Resetting
            all_burnin_data = None
            burnin_data_idx = 0
            logger.info("Switched training model")

        tstart = time.perf_counter()

        if update % log_interval == 0 and is_mpi_root: logger.info('Stepping environment...')

        # Get minibatch
        obs, returns, b_returns, masks, actions, values, b_values, neglogpacs, states, b_states, epinfos, burnin_data= \
            runner.run(burnin_phase) #pylint: disable=E0632

        if burnin_phase and (iter_loss["v2"] or eval["save_latent"]):
            print("Saving data")
            if iter_loss["v2_use_files"] or eval["save_latent"]:
                # Burnin_data_idx is incremented by nsteps, which is nr. of files
                save_data(burnin_data, burnin_data_idx, _run._id, nsteps)
            else:
                if all_burnin_data is None:
                    all_burnin_data = get_all_burnin_data_dict(
                        env, iter_loss, nsteps, comm)
                for key, value in burnin_data.items():
                    all_burnin_data[key][burnin_data_idx:burnin_data_idx + nsteps] = value
            burnin_data_idx += nsteps

        if update % log_interval == 0 and is_mpi_root: logger.info('Done.')

        epinfobuf.extend(epinfos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mblossvals = []
        mblossvals_burnin = []
        if states is None: # nonrecurrent version
            # Index of each element of batch_size
            # Create the indices array
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):

                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices_train = (arr[mbinds] for arr in (obs, returns, actions, values, neglogpacs))
                    slices_burnin = (arr[mbinds] for arr in (obs, b_returns, actions, b_values, neglogpacs))
                    stats_train, train_op_train, feed = model_train.train(
                        lrnow, cliprangenow, *slices_train,
                    )

                    stats_burnin, train_op_burnin, feed_burnin = model_burnin.train(
                        lrnow, cliprangenow, *slices_burnin, alpha=alpha_reg,
                        )
                    feed.update(feed_burnin)  # Needs both!

                    fetches = {}
                    if eval["eval_only"]:
                        pass
                        session_outputs = {}
                    elif not burnin_phase or iter_loss["v2"]:
                        # For v2, normal PPO training is only the old policy,
                        # The student policy is trained differently
                        fetches.update({"stats_train": stats_train,})
                        fetches.update({"train_op": train_op_train})
                        session_outputs = session.run(fetches, feed)
                    elif (iter_loss["update_old_policy"] or
                          (iter_loss["update_old_policy_in_initial"] and cycle_count==0)):
                        fetches.update({"stats_burnin": stats_burnin})
                        fetches.update({"train_op": train_op_burnin})
                        session_outputs_burnin = session.run(fetches, feed)

                        fetches.update({"stats_train": stats_train,})
                        fetches.update({"train_op": train_op_train})
                        session_outputs = session.run(fetches, feed)

                        session_outputs.update(session_outputs_burnin)
                    else:
                        fetches.update({"stats_burnin": stats_burnin})
                        fetches.update({"train_op": train_op_burnin})
                        session_outputs = session.run(fetches, feed)

                    if "stats_train" in session_outputs.keys():
                        mblossvals.append(session_outputs["stats_train"])
                    else:
                        mblossvals.append(
                            [0 for loss in model_train.loss_names]
                        )

                    if "stats_burnin" in session_outputs.keys():
                        mblossvals_burnin.append(session_outputs["stats_burnin"])
                    else:
                        mblossvals_burnin.append(
                            [0 for loss in model_burnin.loss_names]
                        )

        else: # recurrent version
            raise NotImplementedError("Recurrent version not implemented")

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        lossvals_burnin = np.mean(mblossvals_burnin, axis=0)
        # End timer
        tnow = time.perf_counter()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))

        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("misc/serial_timesteps", update*nsteps)
            logger.logkv("misc/nupdates", update)
            logger.logkv("misc/total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("misc/explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('misc/time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model_train.loss_names):
                logger.logkv('loss/' + lossname, lossval)
            for (lossval, lossname) in zip(lossvals_burnin, model_burnin.loss_names):
                logger.logkv('loss_burnin/' + lossname, lossval)
            logger.logkv("schedule/alpha_reg", alpha_reg)
            logger.logkv("schedule/current_cycle_count", current_cycle_count)
            logger.logkv("schedule/burnin_phase", burnin_phase)

            logger.dumpkvs()

    if is_mpi_root:
        save_model(model_train, "model", update, _run)
    return model_train




def train_student(teacher, student, data, iter_loss, lr, cliprange,
                  nminibatches, session, max_idx, nenvs, nsteps, id):
    """Train student for sequential ITER (i.e. v2=True).

    Args:
        teacher: teacher model
        student: student model
        data: either a np array or None if we use files to store the data
        iter_loss: config dict
        lr: learning rate
        cliprange: cliprange used for gradients
        nminibatches: How many minibatches are used in PPO?
        session: TF session
        max_idx: How many frames have been stored? Need to know when things are stored in files
        nenvs: How many parallel envs are being exectued
        nsteps: How many steps per batch are executed
        id: Run id, needed to find the folder with the files

    Doesn't return anything, but updates the student
    """

    use_data = data is not None

    # In unit of steps
    num_processed_parallel = int(max(nsteps // nminibatches, 1))
    num_batches = int(max_idx // num_processed_parallel)
    max_idx = num_batches *  num_processed_parallel
    if use_data:
        obs = data["obs"][:max_idx]
        actions = data["actions"][:max_idx]
        returns = data["returns"][:max_idx]
        neglogpacs = data["neglogpacs"][:max_idx]
        values = data["values"][:max_idx]
        # Get example so I know dimensionality of pi
        test_obs = obs[0:num_processed_parallel]
        sa = test_obs.shape
        v, pi = teacher.train_model.value_and_pi(
            test_obs.reshape(-1, *sa[2:]))

        teacher_values = np.empty_like(values)
        teacher_pis = np.empty(
            shape=(nsteps * iter_loss["v2_buffer_size"], nenvs, pi.shape[-1]),
            dtype=pi.dtype)

    print("Re-evaluating")
    for batch_nr in range(num_batches):  # This leaves out the last (too small) batch
        current_idx = batch_nr * num_processed_parallel
        current_slice = slice(current_idx, current_idx + num_processed_parallel)
        batch_idxs = list(range(current_idx, current_idx + num_processed_parallel))

        batch_obs = (obs[current_slice]
                     if use_data
                     else load_batch(batch_idxs, id, nenvs)['obs'])

        sa = batch_obs.shape
        v, pi = teacher.train_model.value_and_pi(
            batch_obs.reshape(sa[0] * sa[1], *sa[2:]),
        )
        pi_size = pi.shape[1]
        v = v.reshape(*sa[:2])
        pi = pi.reshape(*sa[:2], pi_size)

        if use_data:
            teacher_values[current_slice] = v
            teacher_pis[current_slice] = pi
        else:
            save_data(
                {"teacher_values": v, "teacher_pis": pi},
                current_idx, id, num_processed_parallel, prefix="teacher_")

    if use_data:
        teacher_values = teacher_values.reshape(-1)
        teacher_pis = teacher_pis.reshape(-1, pi_size)
        obs = obs.reshape(-1, *sa[2:])
        actions = actions.reshape(-1)
        returns = returns.reshape(-1)
        neglogpacs = neglogpacs.reshape(-1)
        values = values.reshape(-1)

    # Each file contains nenvs datapoints that are loaded together for speed
    # On the other hand, when reading from memory, I can pick each data individually
    inds = (np.arange(max_idx * nenvs, dtype=np.int)
            if use_data
            else np.arange(max_idx, dtype=np.int))

    # Similarly, adapt `num_processed_parallel` if we're reading from memory
    if use_data: num_processed_parallel *= nenvs

    # Calculate the fps (frame per second)
    print("Training", flush=True)
    total_steps = iter_loss["v2_number_epochs"] * num_batches
    for iteration in range(iter_loss["v2_number_epochs"]):
        tstart = time.perf_counter()
        np.random.shuffle(inds)
        # losses = defaultdict(list)
        mblossvals_burnin = []
        time_data = 0
        time_train = 0

        for batch_nr in range(0, num_batches):
            current_idx = batch_nr * num_processed_parallel
            mbinds = inds[current_idx:current_idx + num_processed_parallel]

            current_step = iteration * num_batches + batch_nr
            alpha_reg = get_alpha(current_step / total_steps, iter_loss["alpha_reg"])

            it_start = time.perf_counter()
            if use_data:
                curr_obs = obs[mbinds]
                curr_returns = returns[mbinds]
                curr_actions = actions[mbinds]
                curr_values = values[mbinds]
                curr_neglogpacs = neglogpacs[mbinds]
                curr_teacher_values = teacher_values[mbinds]
                curr_teacher_pis = teacher_pis[mbinds]
            else:
                dd = load_batch(mbinds, id, nenvs)
                dd.update(
                    load_batch(mbinds, id, nenvs, prefix="teacher_", pi_size=pi_size)
                )
                curr_obs = dd['obs'].reshape(-1, *sa[2:])
                curr_returns = dd['returns'].reshape(-1)
                curr_actions = dd['actions'].reshape(-1)
                curr_values = dd['values'].reshape(-1)
                curr_neglogpacs = dd['neglogpacs'].reshape(-1)
                curr_teacher_values = dd['teacher_values'].reshape(-1)
                curr_teacher_pis = dd['teacher_pis'].reshape(-1, pi_size)

            it_mid = time.perf_counter()
            stats, op, feed = student.train(
                lr=lr,
                cliprange=cliprange,
                obs=curr_obs,
                returns=curr_returns,
                actions=curr_actions,
                values=curr_values,
                neglogpacs=curr_neglogpacs,
                teacher_values=curr_teacher_values,
                teacher_pis=curr_teacher_pis,
                alpha=alpha_reg,
            )

            fetches = {
                "stats_burnin": stats,
                "train_op": op
            }
            session_outputs = session.run(fetches, feed)
            mblossvals_burnin.append(session_outputs["stats_burnin"])
            it_end = time.perf_counter()
            time_data += it_mid - it_start
            time_train += it_end - it_mid

        lossvals_burnin = np.mean(mblossvals_burnin, axis=0)
        tnow = time.perf_counter()
        fps = int(len(inds) / (tnow - tstart))
        if not use_data: fps *= nenvs
        logger.logkv("distill_v2/iteration", iteration)
        logger.logkv("distill_v2/fps", fps)
        logger.logkv("distill_v2/data_loading_time", time_data / (time_data + time_train))
        logger.logkv("distill_v2/alpha_reg", alpha_reg)
        for (lossval, lossname) in zip(lossvals_burnin, student.loss_names):
            logger.logkv('distill_v2/' + lossname, lossval)
        logger.dumpkvs()

    logger.info("Trained v2 student")
    if iter_loss["v2_use_files"]:
        shutil.rmtree(f"{id}_data")
        logger.info(f"Removed data folder {id}_data")





