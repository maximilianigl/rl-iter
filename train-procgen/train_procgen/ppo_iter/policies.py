"""
Modification from baselines to support ibac
"""
import tensorflow as tf
from baselines.common import tf_util
from baselines.a2c.utils import fc
from baselines.common.distributions import make_pdtype, _matching_fc, CategoricalPd
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.tf_util import adjust_shape
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.models import get_network_builder
ds = tf.contrib.distributions
import numpy as np

import gym


class PolicyWithValue(object):
    """
    Modified from original to incorporate IBAC-SNI, which requires us to return both the stochastic and
    deterministic latent so we can use the stochastic for training and deterministic for execution (and training).
    It also allows for other regularisations.
    """

    def __init__(self, env, observations, arch, latent, latent_mean, info_loss,
                 estimate_q=False, vf_latent=None, sess=None, **tensors):
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        arch            config dict

        latent          latent state from which policy distribution parameters should be inferred

        info_loss       config dict

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.X = observations
        self.state = tf.constant([])
        self.initial_state = None
        self.info_loss = info_loss
        self.__dict__.update(tensors)

        assert not vf_latent
        # vf_latent = tf.layers.flatten(vf_latent) if vf_latent is not None else latent_mean
        if estimate_q:
            raise NotImplementedError()
            # assert isinstance(env.action_space, gym.spaces.Discrete)
            # self.q = fc(vf_latent, 'q', env.action_space.n)
            # self.vf = self.q

        latent = tf.layers.flatten(latent)
        latent_mean = tf.layers.flatten(latent_mean)
        self.latent_mean = latent_mean  # For t-SNE

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(env.action_space)

        nr_samples = 1

        # Important: *_run means it's the deterministic, *_train means it's the stochastic version
        self.pd_run, self.pi_run = self.pdtype.pdfromlatent(latent_mean, init_scale=0.01)
        self.vf_run = fc(latent_mean, 'vf_run', 1)

        if arch["reg"] == "ibac":
            pdparam = _matching_fc(latent, 'pi', env.action_space.n, init_scale=1.0, init_bias=0)
            pdparam = tf.reshape(pdparam, shape=(nr_samples, -1, env.action_space.n))
            pdparam = tf.transpose(pdparam, perm=[1, 0, 2])

            # pdparam are logits [batch, nr_samples, action_dim]
            probs = tf.math.softmax(pdparam)
            # Mean across samples
            probs = tf.reduce_mean(probs, axis=1)
            self.pd_train = CategoricalPd(tf.math.log(probs))  # Need logits

            dists = ds.Categorical(logits=pdparam)
            self.mixture_categorical = ds.MixtureSameFamily(
                mixture_distribution=ds.Categorical(probs=[1. / nr_samples] * nr_samples),
                components_distribution=dists)
            # self.pd_train.neglogp = lambda a: - self.pd_train.log_prob(a)
            reshaped_latent = tf.reshape(latent, shape=(nr_samples, -1, 256))
            reshaped_latent = tf.transpose(reshaped_latent, perm=[1,0,2]) #[batch, nr_samples, 256]
            # self.vf_train = fc(latent, 'vf_train', 1)
            self.vf_train = fc(reshaped_latent[:,0,:], 'vf_train', 1)
        elif arch["reg"] in ["dropout", "vib"]:
            self.pd_train, _ = self.pdtype.pdfromlatent(latent, init_scale=0.01)
            self.vf_train = fc(latent, 'vf_train', 1)
        else:
            # self.pd_train, _ = self.pdtype.pdfromlatent(latent, init_scale=0.01)
            # If we're not using IBAC, then latent and latent_mean are the same
            self.pd_train = self.pd_run
            self.vf_train = self.vf_run


        # self.pd, self.pi = self.pdtype.pdfromlatent(latent, init_scale=0.01)
        # # Take an action
        self.action_run = self.pd_run.sample()
        self.neglogp_run = self.pd_run.neglogp(self.action_run)
        #
        # # Calculate the neg log of our probability
        self.sess = sess or tf.get_default_session()

        self.vf_run = self.vf_run[:,0]
        self.vf_train = self.vf_train[:,0]

        # For batched re-evaluation of teacher
        self.A = self.pdtype.sample_placeholder([None])
        self.neglogp_A = self.pd_run.neglogp(self.A)

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def value_and_pi(self, ob):

        v, pi = self._evaluate([
            self.vf_run,
            self.pi_run
        ], ob)
        return v, pi

    def step(self, observation, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """

        a, v, state, neglogp = self._evaluate([self.action_run, self.vf_run, self.state, self.neglogp_run], observation, **extra_feed)
        if state.size == 0:
            state = None
        return a, v, state, neglogp

    def value(self, ob, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        return self._evaluate(self.vf_run, ob, *args, **kwargs)

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)

def build_policy(env, policy_network, arch, value_network=None,  normalize_observations=False, estimate_q=False, **policy_kwargs):
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(**policy_kwargs)

    def policy_fn(nbatch=None, nsteps=None, sess=None, observ_placeholder=None):
        ob_space = env.observation_space

        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)

        extra_tensors = {}

        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X

        encoded_x = encode_observation(ob_space, encoded_x)

        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            policy_latent, policy_latent_mean, info_loss = policy_network(encoded_x)
            if isinstance(policy_latent, tuple):
                raise NotImplementedError()

        policy = PolicyWithValue(
            env=env,
            observations=X,
            arch=arch,
            latent=policy_latent,
            latent_mean=policy_latent_mean,
            info_loss=info_loss,
            # vf_latent=vf_latent,
            sess=sess,
            estimate_q=estimate_q,
            **extra_tensors
        )
        return policy

    return policy_fn


def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms


def build_impala_cnn_with_ibac(unscaled_images, arch, depths=[16,32,32], **conv_kwargs):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """

    layer_num = 0

    def get_layer_num_str():
        nonlocal layer_num
        num_str = str(layer_num)
        layer_num += 1
        return num_str

    def conv_layer(out, depth):
        return tf.layers.conv2d(out, depth, 3, padding='same', name='layer_' + get_layer_num_str())

    def residual_block(inputs):
        depth = inputs.get_shape()[-1].value

        out = tf.nn.relu(inputs)

        out = conv_layer(out, depth)
        out = tf.nn.relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        out = residual_block(out)
        return out

    out = tf.cast(unscaled_images, tf.float32) / 255.

    for depth in depths:
        out = conv_sequence(out, depth)

    out = tf.layers.flatten(out)
    out = tf.nn.relu(out)

    info_loss = tf.constant(0.)
    if arch["reg"] == "ibac":
        nr_samples = 1
        print("Creating VIB layer")
        params = tf.layers.dense(out, 256*2)
        out_mean, rho = params[:, :256], params[:, 256:]
        encoding = ds.NormalWithSoftplusScale(out_mean, rho - 5.0)

        with tf.variable_scope("info_loss", reuse=tf.AUTO_REUSE):
            prior = ds.Normal(0.0, 1.0)
            info_loss = tf.reduce_sum(tf.reduce_mean(
                ds.kl_divergence(encoding, prior), 0)) / np.log(2)

        batch_size = tf.shape(out)[0]
        out = tf.reshape(
            encoding.sample(nr_samples),
            shape=(batch_size * nr_samples, 256))
    elif arch["reg"] == "vib":
        params = tf.layers.dense(out, 256 * 2)
        out_mean, rho = params[:, :256], params[:, 256:]
        encoding = ds.NormalWithSoftplusScale(out_mean, rho - 5.0)
        prior = ds.Normal(0.0, 1.0)
        info_loss = tf.reduce_sum(tf.reduce_mean(
            ds.kl_divergence(encoding, prior), 0)) / np.log(2)
        out = encoding.sample()
    elif arch["reg"] == "dropout":
        out_mean = tf.layers.dense(out, 256, activation=None, name='layer_' + get_layer_num_str())
        out = tf.nn.dropout(out_mean, rate=arch["dropout_rate"])
    elif arch["reg"] == "noLayer":
        print("Removing fc layer")
        out_mean = out
    elif arch["reg"] is None:
        out_mean = tf.layers.dense(out, 256, activation=None, name='layer_' + get_layer_num_str())
        out = out_mean
    else:
        raise NotImplementedError("Wrong 'arch.reg' argument")

    if arch["add_extra_layer_before"]:
        print("Add extra Layer before")
        output_layer = tf.layers.Dense(256, name="optional_layer_before")
        out = output_layer(out)
        out_mean = output_layer(out_mean)

    if arch["nonlinearity"] == "relu":
        print("Use Relu")
        out = tf.nn.relu(out)
        out_mean = tf.nn.relu(out_mean)
    elif arch["nonlinearity"] == "tanh":
        print("Use Tanh")
        out = tf.math.tanh(out)
        out_mean = tf.math.tanh(out_mean)
    elif arch["nonlinearity"] is None:
        print("Don't use final nonlinearity")
    else:
        raise NotImplementedError("Wrong 'arch.nonlinearity' argument")

    if arch["add_extra_layer_after"]:
        print("Add extra Layer after")
        output_layer = tf.layers.Dense(256, name="optional_layer_after")
        out = output_layer(out)
        out_mean = output_layer(out_mean)


    return out, out_mean, info_loss

