import numpy as np
import tensorflow as tf
from baselines.common.runners import AbstractEnvRunner
from baselines.common.distributions import CategoricalPd
from baselines.common.tf_util import adjust_shape

def categorical(p):
    return (p.cumsum(-1) >= np.random.uniform(size=p.shape[:-1])[..., None]).argmax(-1)

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, model_burnin, nsteps, gamma, lam, iter_loss, eval):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        self.iter_loss = iter_loss
        self.eval = eval
        # Discount rate
        self.gamma = gamma
        self.model_burnin = model_burnin
        self.b_states = self.model_burnin.initial_state
        self.pd = self.model.act_model.pd_run


    def reset_env(self):
        self.obs[:] = self.env.reset()
        self.states = self.model.initial_state
        self.b_states = self.model_burnin.initial_state
        self.dones = [False for _ in range(self.nenv)]

    def act_interpolate(self, obs, states, b_states, dones):
        sess = self.model.sess
        act_model = self.model.act_model
        b_act_model = self.model_burnin.act_model
        feed_dict = {
            act_model.X: adjust_shape(act_model.X, obs),
            b_act_model.X: adjust_shape(b_act_model.X, obs),
        }

        if states is not None:
            feed_dict.update({
                act_model.S: adjust_shape(act_model.S, states),
                b_act_model.S: adjust_shape(b_act_model.S, b_states),
            })

        variables = [
            self.model.act_model.action_run,
            self.model.act_model.vf_run,
            self.model.act_model.state,
            self.model_burnin.act_model.vf_run,
            self.model_burnin.act_model.state,
            self.model.act_model.neglogp_run,
            self.model.act_model.latent_mean,
            self.model_burnin.act_model.latent_mean,
            self.model_burnin.act_model.action_run
        ]

        a, v, state, b_v, b_state, neglogp, lm, b_lm, b_a = sess.run(variables, feed_dict)

        if state.size == 0:
            state = None
        if b_state.size == 0:
            b_state = None

        return a, v, b_v, state, b_state, neglogp, lm, b_lm, b_a



    def run(self, burnin_phase):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_b_values, mb_dones, mb_neglogpacs, mb_lms, mb_b_lms, mb_b_actions = \
            [],[],[],[],[],[],[],[],[],[]
        mb_states = self.states
        mb_b_states = self.states
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init

            if not burnin_phase:
                # Don't need burnin outputs as I'm only training the training network
                actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
                b_values, self.b_states, latent_mean, b_latent_mean, b_actions = np.zeros_like(values), None, None, None, None
            else:
                # Generate Actions from mixed distribution
                actions, values, b_values, self.state, self.b_state, neglogpacs, latent_mean, b_latent_mean, b_actions = \
                    self.act_interpolate(self.obs, self.states, self.b_states, self.dones)

            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_b_values.append(b_values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            mb_lms.append(latent_mean)
            mb_b_lms.append(b_latent_mean)
            mb_b_actions.append(b_actions)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_b_values = np.asarray(mb_b_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_lms = np.asarray(mb_lms, dtype=np.float32)
        mb_b_lms = np.asarray(mb_b_lms, dtype=np.float32)
        mb_b_actions = np.asarray(mb_b_actions, dtype=np.float32)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)
        b_last_values = self.model_burnin.value(self.obs, S=self.b_states, M=self.dones)

        # discount/bootstrap off value fn
        # mb_returns = np.zeros_like(mb_rewards)
        # mb_b_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        mb_b_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        b_lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
                b_nextvalues = b_last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
                b_nextvalues = mb_b_values[t + 1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            b_delta = mb_rewards[t] + self.gamma * b_nextvalues * nextnonterminal - mb_b_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            mb_b_advs[t] = b_lastgaelam = b_delta + self.gamma * self.lam * nextnonterminal * b_lastgaelam
        mb_returns = mb_advs + mb_values
        mb_b_returns = mb_b_advs + mb_b_values

        if self.iter_loss["v2"]:
            assert not self.eval["save_latent"]
            burnin_data = {
                "obs": mb_obs,
                "actions": mb_actions,
                "returns": mb_returns,
                "neglogpacs": mb_neglogpacs,
                "values": mb_values,
            }
        elif self.eval["save_latent"]:
            burnin_data = {
                "obs": mb_obs,
                "actions": mb_actions,
                "latent_means": mb_lms,
                "b_latent_means": mb_b_lms,
                "b_actions": mb_b_actions,
            }
        else:
            burnin_data = None
        return (*map(sf01, (mb_obs, mb_returns, mb_b_returns, mb_dones, mb_actions, mb_values, mb_b_values,
                            mb_neglogpacs)), mb_states, mb_b_states, epinfos, burnin_data)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


