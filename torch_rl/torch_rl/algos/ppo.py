import numpy
import torch
from torch.distributions.kl import kl_divergence
from torch_rl.algos.base import BaseAlgo
from torch.distributions.categorical import Categorical
# import torch_rl.experimental_config as exp_config
from torch_rl import exp_config


class PPOAlgo(BaseAlgo):
    """PPO with implemented ITER."""

    def __init__(self, envs, pi_old, pi_train, iter_type, num_frames_per_proc=None, discount=0.99, lr=7e-4, gae_lambda=0.95,
                 entropy_coef=0.01, policy_reg_coef=1., value_reg_coef=0.5, value_loss_coef=0.5, max_grad_norm=0.5,
                 adam_eps=1e-5, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, pi_old, pi_train, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         policy_reg_coef, value_reg_coef,
                         value_loss_coef, max_grad_norm, preprocess_obss, reshape_reward, iter_type)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.adam_eps = adam_eps
        self.lr = lr

        self.optimizer = torch.optim.Adam(self.pi_train.parameters(), self.lr, eps=self.adam_eps)
        self.batch_num = 0

    def policy_loss(self, sb, dist, advantage):
        ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage
        return -torch.min(surr1, surr2).mean()

    def value_loss(self, sbvalue, sbreturnn, value):
        value_clipped = sbvalue + torch.clamp(value - sbvalue, -self.clip_eps, self.clip_eps)
        surr1 = (value - sbreturnn).pow(2)
        surr2 = (value_clipped - sbreturnn).pow(2)
        return torch.max(surr1, surr2).mean()

    def update_parameters(self, alpha):
        # Collect experiences

        exps, logs = self.collect_experiences(alpha)
        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values_train = []
            log_values_old = []
            log_policy_loss_train = []
            log_policy_loss_old = []
            log_value_loss_train = []
            log_value_loss_old = []
            log_grad_norms_train = []
            log_grad_norms_old = []
            log_reg_loss_policy = []
            log_reg_loss_value = []

            device = next(self.pi_train.parameters()).device


            for inds in self._get_batches_starting_indexes():
                # Initialize batch values

                # Create a sub-batch of experience
                sb = exps[inds]

                # Compute loss
                model_results = self.execute_model(alpha=alpha, obs=sb.obs)

                # Policy loss (train)
                if alpha > 0 and exp_config.no_rl_loss_during_anneal:
                    value_loss_train = torch.Tensor([0]).to(device)
                    policy_loss_train = torch.Tensor([0]).to(device)
                    entropy = torch.Tensor([0]).to(device)
                else:
                    # Value loss (train)
                    policy_loss_train = self.policy_loss(sb, model_results['dist_train'],
                                                         advantage=sb.advantage_train)
                    value_loss_train = self.value_loss(sbvalue=sb.value_train,
                                                  sbreturnn=sb.returnn_train,
                                                  value=model_results['value_train'])
                    entropy = model_results['dist_train'].entropy().mean()

                if alpha > 0:
                    # This means we're in the distillation (or 'annealing') phase

                    # During annealing also update old policy?
                    if exp_config.also_update_old_policy:
                        policy_loss_old = self.policy_loss(sb, model_results['dist_old'],
                                                        advantage=sb.advantage_old)
                        entropy += model_results['dist_old'].entropy().mean()

                        # Value loss (old): Towards value_old
                        value_loss_old = self.value_loss(sbvalue=sb.value_old,
                                                     sbreturnn=sb.returnn_old,
                                                     value=model_results['value_old'])
                    else:
                        kl_old = torch.Tensor([0]).to(device)
                        policy_loss_old = torch.Tensor([0]).to(device)
                        value_loss_old = torch.Tensor([0]).to(device)

                    # Regularization Terms (during annealing)
                    detached_dist_old = Categorical(probs=model_results['dist_old'].probs.detach())

                    if exp_config.use_reg_loss_policy:
                        reg_loss_policy = kl_divergence(detached_dist_old, model_results['dist_train']).mean()
                    else:
                        reg_loss_policy = torch.Tensor([0]).to(device)

                    if exp_config.use_reg_loss_value:
                        reg_loss_value = (model_results['value_old'].detach() - model_results['value_train']).pow(2).mean()
                    else:
                        reg_loss_value = torch.Tensor([0]).to(device)
                else:
                    # If alpha = 0, we're in the 'free' phase
                    reg_loss_policy = torch.Tensor([0]).to(device)
                    reg_loss_value = torch.Tensor([0]).to(device)
                    kl_old = torch.Tensor([0]).to(device)
                    policy_loss_old = torch.Tensor([0]).to(device)
                    value_loss_old = torch.Tensor([0]).to(device)

                loss = (policy_loss_old + policy_loss_train
                        + self.value_loss_coef * (value_loss_train + value_loss_old)
                        - self.entropy_coef * entropy
                        + alpha * (self.policy_reg_coef * reg_loss_policy + self.value_reg_coef * reg_loss_value))

                # Update actor-critic
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm_train = sum(p.grad.data.norm(2).item() ** 2 for p in self.pi_train.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.pi_train.parameters(), self.max_grad_norm)
                if alpha > 0 and exp_config.also_update_old_policy and self.pi_old is not None:
                    grad_norm_old = sum(p.grad.data.norm(2).item() ** 2 for p in self.pi_old.parameters()) ** 0.5
                    torch.nn.utils.clip_grad_norm_(self.pi_old.parameters(), self.max_grad_norm)
                else:
                    grad_norm_old = 0
                self.optimizer.step()

                # Update log values

                log_entropies.append(entropy.item())
                log_values_train.append(model_results['value_train'].mean().item())
                if alpha > 0:
                    log_values_old.append(model_results['value_old'].mean().item())
                else:
                    log_values_old.append(0)

                log_policy_loss_train.append(policy_loss_train.item())
                log_policy_loss_old.append(policy_loss_old.item())
                log_value_loss_train.append(value_loss_train.item())
                log_value_loss_old.append(value_loss_old.item())
                log_grad_norms_train.append(grad_norm_train)
                log_grad_norms_old.append(grad_norm_old)
                log_reg_loss_policy.append(reg_loss_policy.item())
                log_reg_loss_value.append(reg_loss_value.item())

        # Log some values

        logs["entropy"] = numpy.mean(log_entropies)
        logs["value_train"] = numpy.mean(log_values_train)
        logs["value_old"] = numpy.mean(log_values_old)
        logs["policy_loss_train"] = numpy.abs(numpy.mean(log_policy_loss_train))
        logs["policy_loss_old"] = numpy.abs(numpy.mean(log_policy_loss_old))
        logs["value_loss_train"] = numpy.mean(log_value_loss_train)
        logs["value_loss_old"] = numpy.mean(log_value_loss_old)
        logs["grad_norm_train"] = numpy.mean(log_grad_norms_train)
        logs["grad_norm_old"] = numpy.mean(log_grad_norms_old)
        logs["reg_loss_policy"] = numpy.mean(log_reg_loss_policy)
        logs["reg_loss_value"] = numpy.mean(log_reg_loss_value)

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + 1) % self.num_frames_per_proc != 0]
        self.batch_num += 1

        batches_starting_indexes = [indexes[i:i+self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        return batches_starting_indexes
