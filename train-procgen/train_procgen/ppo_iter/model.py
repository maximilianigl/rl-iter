import tensorflow as tf
import functools

from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize

# try:
from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
from mpi4py import MPI
import pprint

import numpy as np


# except ImportError:
#     MPI = None

class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the ac_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """
    def __init__(self, *, scope_name, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, mpi_rank_weight=1, comm=None, microbatch_size=None,
                 # For burn in:
                 target_vf=None, target_dist_param=None, iter_loss=None, arch=None):
        self.sess = sess = get_session()
        self.scope_name = scope_name
        self.mpi_rank_weight = mpi_rank_weight
        self.max_grad_norm = max_grad_norm

        # Is this the student(=burnin) or teacher?
        self.is_burnin_model = target_vf is not None

        if MPI is not None and comm is None:
            comm = MPI.COMM_WORLD

        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)
            self.train_model = train_model = policy(sess=sess)

        # CREATE THE PLACEHOLDERS
        self.A = train_model.pdtype.sample_placeholder([None])
        self.ADV = tf.placeholder(tf.float32, [None])
        self.R = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        self.OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = tf.placeholder(tf.float32, [None])
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = tf.placeholder(tf.float32, [])

        # For ITER losses
        self.alpha = tf.placeholder_with_default(tf.constant(0.), [])

        # For v2 (i.e. sequential ITER)
        self.teacher_v = tf.placeholder(tf.float32, [None])
        self.teacher_pi = tf.placeholder(tf.float32, [None, ac_space.n])


        if arch["reg"] == "ibac":
            # Special treatment to implement SNI for IBAC
            # Policy loss
            pg_loss, approxkl_train, clipfrac_train, entropy_train = self.get_pd_loss(train_model.pd_train)
            pg_loss_run, approxkl_run, clipfrac_run, entropy_run = self.get_pd_loss(train_model.pd_run)

            pg_loss = (pg_loss + pg_loss_run) / 2.
            entropy = tf.reduce_mean(train_model.mixture_categorical._components_distribution.entropy())
            entropy = (entropy + entropy_run) / 2.

            info_loss = train_model.info_loss
            vf_loss = self.get_vf_loss(train_model.vf_train)

        else:

            pd = train_model.pd_train
            vf = train_model.vf_train
            info_loss = train_model.info_loss

            pg_loss, approxkl_train, clipfrac_train, entropy = self.get_pd_loss(pd)
            vf_loss = self.get_vf_loss(vf)
            approxkl_run = clipfrac_run = tf.constant(0.)

        if not iter_loss["stochastic_vf_in_rl_update"]:
            # Only need this for IBAC:
            # Do I want to use the stochastic or deterministic latent for vf?
            # (vf_run is deterministic, vf_train is stochastic)
            vf_loss = self.get_vf_loss(train_model.vf_run)

        ############ Supervised losses for ITER
        if self.is_burnin_model and iter_loss['use_reg_loss_value']:
            vf_distill = (train_model.vf_train
                          if iter_loss["stochastic_vf_in_distill_update"]
                          else train_model.vf_run)

            if iter_loss["v2"]:
                target_vf = self.teacher_v

            reg_loss_value = tf.reduce_mean(
                tf.square(tf.stop_gradient(target_vf) - vf_distill))
        else:
            reg_loss_value = tf.constant(0.)

        if self.is_burnin_model and iter_loss['use_reg_loss_policy']:
            if iter_loss["v2"]:
                target_dist_param = self.teacher_pi

            reg_loss_policy = tf.reduce_mean(
                train_model.pdtype.pdfromflat(
                    tf.stop_gradient(target_dist_param)
                ).kl(train_model.pd_train))
        else:
            reg_loss_policy = tf.constant(0.)
        ############ End: Supervised losses for ITER

        params = tf.trainable_variables(self.scope_name)
        weight_params = [v for v in params if '/b' not in v.name]
        l2w_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in weight_params])

        pprint.pprint(params)
        num_parameters = np.sum([np.prod(v.get_shape().as_list()) for v in params])
        print(f"Number parameters: {num_parameters}")

        if self.is_burnin_model and not iter_loss["use_burnin_rl_loss"]:
            # For burnin model: Switch RL losses off
            pg_loss = tf.constant(0.)
            entropy = tf.constant(0.)
            vf_loss = tf.constant(0.)

        # Total loss
        rl_loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        reg_loss = (arch["l2w_weight"] * l2w_loss
                    + arch["info_loss_coef"] * info_loss)
        distill_loss = self.alpha * (reg_loss_value * iter_loss['value_reg_coef']
                                     + reg_loss_policy * iter_loss['policy_reg_coef'])

        loss = rl_loss + reg_loss + distill_loss

        self._train_op, self.grads_and_var = self.get_train_op(loss, params, comm)

        self.params = params
        self.loss = loss

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl_train', 'clipfrac_train',
                           'approxkl_run', 'clipfrac_run', "info_loss",
                           'reg_loss_value', 'reg_loss_policy', "l2w_loss"]
        self.stats_list = [pg_loss, vf_loss, entropy, approxkl_train, clipfrac_train,
                           approxkl_run, clipfrac_run, info_loss,
                           reg_loss_value, reg_loss_policy, l2w_loss]

        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

    def get_train_op(self, loss, params, comm):
        # 2. Build our trainer
        if comm is not None and comm.Get_size() > 1:
            trainer = MpiAdamOptimizer(comm, learning_rate=self.LR,
                                            mpi_rank_weight=self.mpi_rank_weight, epsilon=1e-5)
        else:
            trainer = tf.train.AdamOptimizer(learning_rate=self.LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if self.max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)

        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da
        grads_and_var = list(zip(grads, var))
        _train_op = trainer.apply_gradients(grads_and_var)
        return _train_op, grads_and_var

    def train(self, lr, cliprange, obs, returns, actions, values, neglogpacs,
              teacher_values=None, teacher_pis=None, alpha=None):

        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.train_model.X : obs,
            self.A : actions,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values
        }

        if teacher_values is not None:
            td_map.update({
                self.teacher_pi: teacher_pis,
                self.teacher_v: teacher_values,
            })

        if alpha is not None:
            td_map[self.alpha] = alpha

        return (self.stats_list, self._train_op, td_map)

    def get_pd_loss(self, pd):
        # Policy loss
        neglogpac_train = pd.neglogp(self.A)

        # Calculate ratio (pi current policy / pi old policy)
        ratio_train = tf.exp(self.OLDNEGLOGPAC - neglogpac_train)

        # Defining Loss = - J is equivalent to max J
        pg_losses_train = -self.ADV * ratio_train
        pg_losses2_train = -self.ADV * tf.clip_by_value(
            ratio_train, 1.0 - self.CLIPRANGE, 1.0 + self.CLIPRANGE)

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses_train, pg_losses2_train))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac_train - self.OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio_train - 1.0), self.CLIPRANGE)))
        entropy = tf.reduce_mean(pd.entropy())
        return pg_loss, approxkl, clipfrac, entropy

    def get_vf_loss(self, vpred):
        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        # vpred = train_model.vf_run
        vpredclipped = self.OLDVPRED + tf.clip_by_value(
            vpred - self.OLDVPRED, - self.CLIPRANGE, self.CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - self.R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - self.R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        return vf_loss
