import torch
import torch.nn as nn
from utils.networks import DQN, DuelingDQN
from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from .networks import MLPNetwork
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise
import random
import math

#from utils.utils import weight_init
#from utils.utils import fanin_init

HIDDEN_DIM = 64 #300

# 各种的DQN系列的agent的合集爱尔

class DQNsAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, algs = 'dqn', num_in_critic=0, hidden_dim=64, hidden_dim_critic=128,
                 lr=0.01, discrete_action=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.algs = algs
        if algs == 'dqn':
            self.policy = DQN(num_in_pol, num_out_pol)
        elif algs == 'double_dqn':
            self.policy = DQN(num_in_pol, num_out_pol)
            self.target_policy = DQN(num_in_pol, num_out_pol)
            hard_update(self.target_policy, self.policy)
        else:
            self.policy = DuelingDQN(num_in_pol, num_out_pol)
            self.target_policy = DuelingDQN(num_in_pol, num_out_pol)
            hard_update(self.target_policy, self.policy)

        self.num_out_pol = num_out_pol

        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        epsilon_start = 1.0
        epsilon_final = 0.01
        epsilon_decay = 500
        self.frame_idx = 0.0
        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
            -1. * frame_idx / epsilon_decay)

    def reset_noise(self):
        self.frame_idx = 0.0

    def scale_noise(self, scale):
        self.frame_idx = 0.0

    def step(self, obs, explore=False):
        self.frame_idx  += 1.0/25.0
        if random.random() <= self.epsilon_by_frame(self.frame_idx) and explore==True:
            action = random.randrange(self.num_out_pol)
            action = torch.tensor([action])
        else:
            q_value = self.policy.forward(obs)
            action = q_value.max(1)[1].data
        if action.device != 'cpu':
            action = action.to('cpu')
        return action


    def get_params(self):
        if self.algs == 'dqn':
            return {'policy': self.policy.state_dict(),
                    'policy_optimizer': self.policy_optimizer.state_dict()}
        else:
            return {'policy': self.policy.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict()}

    def load_params(self, params):
        if self.algs == 'dqn':
            self.policy.load_state_dict(params['policy'])
            self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        else:
            self.policy.load_state_dict(params['policy'])
            self.target_policy.load_state_dict(params['target_policy'])
            self.policy_optimizer.load_state_dict(params['policy_optimizer'])

