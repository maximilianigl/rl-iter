import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import gym

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class ACModel(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()

        # Decide which components are enabled
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        print("Image size:", n,m)

        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 32, (2, 2)),
            nn.ReLU(),
        )
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*32
        print("Image embedding size: ", self.image_embedding_size)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size

        # Define actor's model
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("Unknown action space: " + str(action_space))

        self.reg_layer = nn.Linear(self.embedding_size, 64)

        self.actor = nn.Sequential(
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def encode(self, obs):
        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        x = self.image_conv(x)
        embedding = x.reshape(x.shape[0], -1)
        bot_mean = self.reg_layer(embedding)
        return bot_mean

    # When I'm not using vib
    def compute(self, obs):
        bot_mean = self.encode(obs)
        x_dist = self.actor(bot_mean)
        dist = Categorical(logits=F.log_softmax(x_dist, dim=1))
        value = self.critic(bot_mean).squeeze(1)
        return dist, value

