from abc import ABC, abstractmethod

import torch
from torch_rl.format import default_preprocess_obss
from torch.distributions.categorical import Categorical

from torch_rl.utils import DictList, ParallelEnv
# import torch_rl.experimental_config as exp_config
from .. import exp_config


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, pi_old, pi_train, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 policy_reg_coef, value_reg_coef,
                 value_loss_coef, max_grad_norm, preprocess_obss, reshape_reward, iter_type):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        pi_old : torch.Module
            the old model (=teacher)
        pi_train : torch.Module
            the new model (=student).
            During 'normal' RL training, we execute this model, not the 'old' one.
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        iter_type :  string
            which type of ITER to use "none", "distill" (the normal one) or
            "kickstarting" (executing the student during distillation!
        """

        # Store parameters

        self.env = ParallelEnv(envs)
        self.pi_train = pi_train
        self.pi_train.train()
        self.pi_old = pi_old
        if self.pi_old is not None:
            self.pi_old.train()
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.policy_reg_coef = policy_reg_coef
        self.value_reg_coef = value_reg_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward
        self.iter_type = iter_type

        # Store helpers values

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.reset_env()
        self.obss = [None]*(shape[0])
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values_old = torch.zeros(*shape, device=self.device)
        self.values_train = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages_old = torch.zeros(*shape, device=self.device)
        self.advantages_train = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        # Initialize log values
        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

    def switch_models(self, new_pi):
        self.pi_old = self.pi_train
        self.pi_train = new_pi
        self.pi_train.train()
        # if self.pi_old is not None:
        #     self.pi_old.eval()

        parameters = list(self.pi_train.parameters())
        if exp_config.also_update_old_policy:
            parameters += list(self.pi_old.parameters())

        self.optimizer = torch.optim.Adam(parameters, self.lr, eps=self.adam_eps)

    def execute_model(self, alpha, **kwargs):
        """Execute model.

        Args:
            alpha: float between 0 and 1. If it's 0, we know we're not distilling and
                I don't need to execute old policy
            **kwargs: Other arguments for the `compute` function of the model. Should at least contain `obs`.

        Returns:
            dict containing 'dist' and 'value' for 'old' and 'train', as well es 'execute', which could be
            either, depending on whether iter_type=='distill' or iter_type=='kickstarting'
        """

        if alpha == 0:
            # If alpha == 0, we're not currently distilling -> Don't need to execute old policy
            dist_train, value_train = self.pi_train.compute(**kwargs)
            dist_old, value_old = None, None
            dist_execute = dist_train

        else:
            dist_old, value_old = self.pi_old.compute(**kwargs)
            dist_train, value_train = self.pi_train.compute(**kwargs)

            assert self.iter_type in ["kickstarting", "distill"]
            dist_execute = dist_train if (self.iter_type == "kickstarting" or self.pi_old is None) else dist_old

            # Return new distribution, old value and weighted? sum of KL's
        return {"dist_execute": dist_execute,
                "dist_old": dist_old, "value_old": value_old,
                "dist_train": dist_train, "value_train": value_train}

    def reset_env(self):
        self.obs = self.env.reset()
        self.mask = torch.ones(self.num_procs, device=self.device)

    def collect_experiences(self, alpha):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.
        Args
        ------
        alpha: float between 0 and 1
            used to determine which policy to execute, based on whether
            we're currently distilling or not and what iter_type is

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                model_results = self.execute_model(alpha=alpha, obs=preprocessed_obs)
                dist = model_results['dist_execute']
                value_old = model_results['value_old']
                value_train = model_results['value_train']
            action = dist.sample()

            obs, reward, done, _ = self.env.step(action.cpu().numpy())

            # Update experiences values
            self.obss[i] = self.obs
            self.obs = obs
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)

            self.actions[i] = action
            self.values_train[i] = value_train

            if alpha > 0:
                self.values_old[i] = value_old

            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            # Update log values
            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            model_results = self.execute_model(alpha=alpha, obs=preprocessed_obs)
            next_value_old = model_results['value_old']
            next_value_train = model_results['value_train']

        # For self.advantages_old
        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask

            if alpha > 0:
                next_value_old = self.values_old[i+1] if i < self.num_frames_per_proc - 1 else next_value_old
                next_advantage_old = self.advantages_old[i+1] if i < self.num_frames_per_proc - 1 else 0
                delta_old = self.rewards[i] + self.discount * next_value_old * next_mask - self.values_old[i]
                self.advantages_old[i] = delta_old + self.discount * self.gae_lambda * next_advantage_old * next_mask

            next_value_train = self.values_train[i+1] if i < self.num_frames_per_proc - 1 else next_value_train
            next_advantage_train = self.advantages_train[i+1] if i < self.num_frames_per_proc - 1 else 0
            delta_train = self.rewards[i] + self.discount * next_value_train * next_mask - self.values_train[i]
            self.advantages_train[i] = delta_train + self.discount * self.gae_lambda * next_advantage_train * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.advantage_old = self.advantages_old.transpose(0, 1).reshape(-1)
        exps.advantage_train = self.advantages_train.transpose(0, 1).reshape(-1)

        if alpha > 0:
            exps.value_old = self.values_old.transpose(0, 1).reshape(-1)
            exps.returnn_old = exps.value_old + exps.advantage_old
        exps.value_train = self.values_train.transpose(0, 1).reshape(-1)
        exps.returnn_train = exps.value_train + exps.advantage_train

        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences
        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values
        keep = max(self.log_done_counter, self.num_procs)

        log = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, log

    @abstractmethod
    def update_parameters(self):
        pass
