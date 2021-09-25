import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pchs

# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size

# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    return F.softmax(y / temperature, dim=1)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
        'brown', 'orange', 'teal',  'lightblue', 'lime', 'lavender', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold',  'darkred', 'darkblue']
def label(xy, text):
    y = xy[1] - 0.15  # shift y-value for label so that it's below the artist
    plt.text(xy[0], y, text, ha="center", family='sans-serif', size=6)

def draw_comm_results(agents, landmarks, comm_list, action_before, action_after):
    # first darw agents:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, pos_i in enumerate(agents):
        # add a circle
        circle = pchs.Circle(pos_i.state.p_pos, 0.05, color='green', ec="none")
        ax.add_patch(circle)
        label(pos_i.state.p_pos, str(i))

    for i, pos_i in enumerate(landmarks):
        # add a circle
        circle = pchs.Rectangle(pos_i.state.p_pos - [0.025, 0.025], 0.05, 0.05, ec="none")
        #circle = pchs.Circle(pos_i.state.p_pos, 0.05,color='pink', ec="none")
        ax.add_patch(circle)
        label(pos_i.state.p_pos, str(i))

    for i, agent_i, in enumerate(comm_list):
        # draw arrow
        action_before_temp = action_before[i,0]/torch.norm(action_before[i,0])/5
        action_before_temp = action_before_temp.detach().numpy()
        pos_i = agents[agent_i]
        arrow = pchs.Arrow(pos_i.state.p_pos[0],pos_i.state.p_pos[1], action_before_temp[0],action_before_temp[1], color='red', width=0.02)
        ax.add_patch(arrow)
        # draw arrow
        action_after_temp = action_after[i,0]/torch.norm(action_after[i,0])/5
        action_after_temp = action_after_temp.detach().numpy()
        pos_i = agents[agent_i]
        arrow = pchs.Arrow(pos_i.state.p_pos[0],pos_i.state.p_pos[1], action_after_temp[0],action_after_temp[1], color='green',  width=0.02)
        ax.add_patch(arrow)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()



def plot_curve_with_label(data, label, color):
    """
    :param data: shape is (num_curves, dim_data)
    :param label: shape is (num_curves,)
    :param color: shape is (num_curves,)
    :return: none
    """
    train_sizes = range(data.shape[1])
    if data.shape[0]==1:
        #fig = plt.gcf()

        #plt.plot(train_sizes, data[0,:], 'o-', color=color,
                     #label=label)
        #fig.set_size_inches(5, 10)
        plt.figure(figsize=(6, 6))
        plt.grid()

        #plt.plot(train_sizes, data[0, :], '-', color=color,
        #         label=label)
        plt.plot(train_sizes, data[0, :], '-', color=color)
        #plt.legend(loc="best")
        plt.xlabel('Training episodes', fontsize='large')
        plt.ylabel('Mean reward', fontsize='large')
        #plt.savefig('test2png.png', dpi=100)
        plt.savefig('learning_curve.pdf')
    else:

        plt.grid()

        train_scores_mean = np.mean(data,0)
        train_scores_std = np.std(data,0)
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.legend(loc="best")
    plt.show()

if __name__ == '__main__':
    data = np.random.randn(1,100)
    plot_curve_with_label(data,'Gaussian','r')