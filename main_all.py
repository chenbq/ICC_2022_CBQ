import argparse
import torch
import time
import os
import numpy as np
import pickle
from types import SimpleNamespace
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer, ReplayBufferATOC, ReplayBufferAttention
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from utils.misc import plot_curve_with_label
from algorithms.maddpg import MADDPG
from algorithms.DTPC import DTPC
from algorithms.dqn import DQNs
from tqdm import tqdm
import shutil

USE_CUDA = True# torch.cuda.is_available()
to_gpu=USE_CUDA
continue_train = False # 是否从当前训练结果，继续训练（注意buff没有保存，必须从新获取）
episode_num = 0
max_links = 3

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def transf_params(params):
    save_params = dict()
    for var_name in params:
        if len(var_name)==0:
            continue
        if 'optim' in var_name:
            save_params.update({var_name:params[var_name]})
        else:
            temp_dict = {}
            for in_name in params[var_name]:
                temp_dict.update({in_name:params[var_name][in_name].shape})
            save_params.update({var_name:temp_dict})
    return save_params

def save_code(run_dir):
    code_dir = run_dir / 'codes'
    src_dir = Path('.')
    os.makedirs(code_dir)

    # file list
    file_list = [src_dir/ 'main_all.py', src_dir/ 'main_eval.py', src_dir/ 'plot_rewards.py']
    # dir list
    dir_list = [ 'algorithms', 'configs', 'multiagent_com', 'utils']
    for file_i in file_list:
        shutil.copy2(file_i, code_dir)  # complete target filename given

    for dir_i in dir_list:
        shutil.copytree(src_dir /dir_i, code_dir/ dir_i)
        # os.makedirs(dir_i)
        # for src_file in (dir_i).iterdir():
        #     if src_file.is_file():
        #         shutil.copy2(src_file, code_dir)  # target filename is /dst/dir/file.ext

def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run

    save_code(run_dir)

    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    with open(run_dir/'config.txt', 'w', encoding='utf-8') as f:
        f.write(str(vars(config)))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    ReplayBufferx = ReplayBuffer
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                            config.discrete_action)
    # setting the types of action space
    if config.agent_alg in ['dqn', 'double_dqn', 'dueling_dqn']:
        env.envs[0].discrete_action_space = True
        env.envs[0].discrete_action_input = True
    else:
        env.envs[0].discrete_action_space = False
        env.envs[0].discrete_action_input = False


    if config.agent_alg == 'DTPC':
        AgentNet = DTPC.init_from_env(env, tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)
    elif config.agent_alg == 'maddpg' or config.agent_alg == 'ddpg':
        AgentNet = MADDPG.init_from_env(env,agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)
    elif config.agent_alg == 'dqn' or config.agent_alg == 'double_dqn' or config.agent_alg == 'dueling_dqn':
        AgentNet = DQNs.init_from_env(env,agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)

    network_params = transf_params(AgentNet.get_params())
    with open(run_dir/'network.txt', 'w', encoding='utf-8') as f:
        f.write(str(network_params))
    replay_buffer = ReplayBufferx(config.buffer_length, AgentNet.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])

    episode_rewards = [0.0]  # sum of rewards for all agents

    t = 0
    d_Q_all = [] # recording all d_Q for ATOC
    comm_all = []

    # 初始化noise
    AgentNet.scale_noise(config.init_noise_scale)
    AgentNet.reset_noise()

    path_temp = '{}/all_reward.csv'.format(log_dir)
    mode_temp = 'w'
    if os.path.exists(path_temp):
        mode_temp = 'a'
    with open(path_temp, mode_temp) as f:
        f.write(','.join(['t_'+str(t) for t in range(config.episode_length)]) + ',mean' + '\n')
    for ep_i in tqdm(range(0, config.n_episodes, config.n_rollout_threads)):
        episode_num = ep_i/5000.0
        # 每一次循环，跳过n_rollout_threads次。
        # 因为对并行的环境采样后，会重复训练n_rollout_threads次。
        if ep_i ==4602:
            print('hold on')
        if ep_i % 1000 == 0:
            print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs_raw = env.reset()
        obs = np.array([[obs_raw[:, :, 0][0, i] for i in range(AgentNet.nagents)]])
        if config.agent_alg == 'ATOC':
            nearby_agents = np.array([[obs_raw[:, :, 1][0, i][:max_links] for i in range(AgentNet.nagents)]])
            AgentNet.update_episode_num(episode_num)
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        if USE_CUDA:
            AgentNet.prep_rollouts(device='gpu')
        else:
            AgentNet.prep_rollouts(device='cpu')


        # scale noise to proper scale
        # 2021-09-24 added
        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        AgentNet.scale_noise(
            config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        AgentNet.reset_noise()

        for et_i in range(config.episode_length):
            #if et_i%5==0 and config.agent_alg == 'ATOC': #first
            #if config.agent_alg == 'ATOC': #second


            if config.agent_alg=='maddpg' or config.agent_alg == 'ddpg' or config.agent_alg in ['dqn','double_dqn', 'dueling_dqn']:
                # torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                #                   requires_grad=False)
                #          for i in range(AgentNet.nagents)]
                if USE_CUDA:
                    torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])).to('cuda'),
                                          requires_grad=False)
                                 for i in range(AgentNet.nagents)]
                else:
                    torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                          requires_grad=False)
                                 for i in range(AgentNet.nagents)]
            else:
                torch_obs = Variable(torch.Tensor([obs[:,agent_idx,:] for agent_idx in range(AgentNet.nagents)]),
                                    requires_grad=False)
                if USE_CUDA:
                    torch_obs = torch_obs.to('cuda')


            torch_agent_actions = AgentNet.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            obs_raw, rewards, dones, infos = env.step(actions)
            next_obs = np.array([[obs_raw[:, :, 0][0, i] for i in range(AgentNet.nagents)]])


            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)

            if os.path.exists(path_temp):
                mode_temp = 'a'
            with open(path_temp, mode_temp) as f:
                f.write(str(rewards[0][0]) + ',')

            for i, rew in enumerate(rewards[0]):
                episode_rewards[-1] += rew
            obs = next_obs
            t += config.n_rollout_threads
            #if len(replay_buffer) >= config.batch_size * config.episode_length and t % config.steps_per_update == 0:
            if len(replay_buffer) >= config.batch_size and t % config.steps_per_update == 0:
                if USE_CUDA:
                    AgentNet.prep_training(device='gpu')
                else:
                    AgentNet.prep_training(device='cpu')
                #for i_net in range(config.n_rollout_threads):
                # #重复采样n次，然后做n次训练。update
                if config.agent_alg=='maddpg'  or config.agent_alg == 'ddpg' or config.agent_alg in ['dqn', 'double_dqn', 'dueling_dqn']:
                    for a_i in range(AgentNet.nagents):
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA)
                        AgentNet.update(sample, a_i, logger=logger)
                else:
                    sample = replay_buffer.sample(config.batch_size,
                                                    to_gpu=USE_CUDA)
                    AgentNet.update(sample, logger=logger)
                    
                #if ep_i % config.steps_per_update == 0:
                AgentNet.update_all_targets()
                if USE_CUDA:
                    AgentNet.prep_rollouts(device='gpu')
                else:
                    AgentNet.prep_rollouts(device='cpu')

        if os.path.exists(path_temp):
            mode_temp = 'a'
        with open(path_temp, mode_temp) as f:
            f.write(','+str(episode_rewards[-1])+ '\n')
        # Append one element after each episode
        episode_rewards.append(0)


        # evaluate after given period
        if ep_i % 100 ==0 :
            AgentNet.prep_training(device='cpu')
            mean_reward, mean_data= evaluate_alg(env, AgentNet, config, run_dir, explore=False, f_name='random')

            logger.add_scalar('agent%i/mean_rewards' % -1, mean_reward, ep_i)
            logger.add_scalar('agent%i/mean_data_reate' % -1, mean_data, ep_i)
            print('mean_rewards',mean_reward)
            print('mean_data_reate', mean_data)

        # Save model for given interval
        if ep_i % config.save_interval < config.n_rollout_threads:
            #os.makedirs(run_dir / 'incremental', exist_ok=True)
            #AgentNet.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            AgentNet.save(run_dir / 'model.pt')


    AgentNet.save(run_dir / 'model.pt')

    # static evaluation
    evaluate_alg(env, AgentNet, config, run_dir, written = True)
    # random evaluation
    evaluate_alg(env, AgentNet, config, run_dir,explore=True, f_name='random', written = True)

    # save current data
    with open(run_dir / 'epsode_rewards.pkl', 'wb') as fp:
        pickle.dump(episode_rewards, fp)
    # save current data
    with open(run_dir / 'comm_status_list.pkl', 'wb') as fp:
        pickle.dump(comm_all, fp)
    with open(run_dir / 'dq_list.pkl', 'wb') as fp:
        pickle.dump(d_Q_all, fp)
    # plot current curves
    plot_curve_with_label(np.expand_dims(episode_rewards,0), 'episode_reward', 'r')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


def evaluate_alg(env, AgentNet, config, run_dir, explore = False, f_name = 'static', written = False):
    Eval_episode = 20
    #path_temp = Path('./models') / config.env_id /config.model_name/ (f_name+'_algorithm_reward.csv')
    path_temp = run_dir / (f_name+'_algorithm_reward.csv')
    mode_temp = 'w'
    if os.path.exists(path_temp):
        mode_temp = 'a'

    # for ep_i in tqdm(range(0, config.n_episodes, config.n_rollout_threads)):
    mean_data_vs_time = np.zeros((Eval_episode, config.episode_length))
    mean_trans_p_vs_time = np.zeros((Eval_episode, config.episode_length))
    mean_vel_vs_time = np.zeros((Eval_episode, config.episode_length))
    mean_fly_p_vs_time = np.zeros((Eval_episode, config.episode_length))
    mean_reward_vs_time = np.zeros((Eval_episode, config.episode_length))
    for ep_i in range(Eval_episode):
        #print(ep_i)
        obs_raw = env.reset()
        obs = np.array([[obs_raw[:, :, 0][0, i] for i in range(AgentNet.nagents)]])
        for et_i in range(config.episode_length):

            # rearrange observations to be per agent, and convert to torch Variable
            if config.agent_alg == 'maddpg' or config.agent_alg == 'ddpg' or config.agent_alg in ['dqn', 'double_dqn',
                                                                                                  'dueling_dqn']:
                torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                      requires_grad=False)
                             for i in range(AgentNet.nagents)]
            else:
                torch_obs = Variable(torch.Tensor([obs[:, agent_idx, :] for agent_idx in range(AgentNet.nagents)]),
                                     requires_grad=False)
            # get actions as torch Variables
            torch_agent_actions = AgentNet.step(torch_obs, explore=explore)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            if config.agent_alg in ['dqn', 'double_dqn', 'dueling_dqn']:
                env.envs[0].discrete_action_space = True
                env.envs[0].discrete_action_input = True
            else:
                env.envs[0].discrete_action_space = False
                env.envs[0].discrete_action_input = False
            # if et_i == 0:
            #     if config.agent_alg in ['dqn', 'double_dqn', 'dueling_dqn']:
            #         env.envs[0].discrete_action_space = True
            #         env.envs[0].discrete_action_input = True
            #         actions = [[9, 9, 9, 9]]
            #     else:
            #         env.envs[0].discrete_action_space = False
            #         env.envs[0].discrete_action_input = False
            #         actions = [[np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]),
            #                     np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0])]]
            # else:
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]

            obs_raw, rewards, dones, infos = env.step(actions)
            next_obs = np.array([[obs_raw[:, :, 0][0, i] for i in range(AgentNet.nagents)]])

            obs = next_obs
            # env.render()
            (num_sat_u, fly_p, trans_p, vel) = env.get_info()[0]
            mean_data_vs_time[ep_i, et_i] = num_sat_u
            mean_trans_p_vs_time[ep_i, et_i] = np.mean(trans_p)
            mean_fly_p_vs_time[ep_i, et_i] = np.mean(fly_p)
            mean_vel_vs_time[ep_i, et_i] = np.mean(vel)
            mean_reward_vs_time[ep_i, et_i] = np.mean(rewards)
    #

    # data rate/ trans_power/fly_power/velocity/reward
    if written == True:
        with open(path_temp, mode_temp) as f:
            f.write(','.join([str(t) for t in np.mean(mean_data_vs_time, axis=0)]) + '\n')
            f.write(','.join([str(t) for t in np.mean(mean_trans_p_vs_time, axis=0)]) + '\n')
            f.write(','.join([str(t) for t in np.mean(mean_fly_p_vs_time, axis=0)]) + '\n')
            f.write(','.join([str(t) for t in np.mean(mean_vel_vs_time, axis=0)]) + '\n')
            f.write(','.join([str(t) for t in np.mean(mean_reward_vs_time, axis=0)]) + '\n')

    return np.mean(mean_reward_vs_time), np.mean(mean_data_vs_time)

if __name__ == '__main__':

    model_dir = Path('./configs')
    config_dir = model_dir/ 'maddpg_config.txt'
    with open(config_dir, 'r', encoding='utf-8') as f:
        #config = f.read()
        a = eval(f.read())
        config = SimpleNamespace(**a)
    run(config)
