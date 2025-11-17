import numpy as np
import torch
import os


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        if getattr(self.args, 'replay_dir', '') != '':
            self.save_path = self.args.result_dir + '/' + \
                args.alg + '/' + getattr(args, 'map', 'default')
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
        print('Init RolloutWorker')

    def generate_episode(self, episode_num=None, evaluate=False, skip_reset: bool = False):
        """
        返回:
          episode, episode_reward, info["time"], win_tag, for_gantt, move_time, for_devices
        其中 for_devices 与 for_gantt(=episodes_situation) 在长度和顺序上保持一一对应。
        """
        # 开始收集与环境交互的情况
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        # Optionally skip env.reset to allow starting from an externally restored state
        if not skip_reset:
            self.env.reset(self.n_agents)
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # NEW: 始终保存“最后一次”info（即使未自然 terminated）
        last_info = {"episodes_situation": [],
                     "devices_situation": [], "time": 0.0}
        for_gantt = []
        for_devices = None

        while not terminated and step < self.episode_limit:
            obs = self.env.get_obs()         # [n_agents, obs_dim]
            state = self.env.get_state()     # [state_dim]
            actions, avail_actions, actions_onehot = [], [], []

            # 选择动作
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(
                    agent_id)   # one-hot 可用动作
                action = self.agents.choose_action(
                    obs[agent_id], last_action[agent_id], agent_id,
                    avail_action, epsilon, evaluate
                )

                # 提前占位，降低并发冲突
                if action < len(self.env.sites):
                    self.env.has_chosen_action(action, agent_id)

                # action 的 one-hot 向量
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            # 环境前进一步
            reward, terminated, info = self.env.step(actions)
            # NEW: 记录“最后一次”info（保证达到 episode_limit 也有记录）
            last_info = info

            # 采样缓存
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])

            episode_reward += reward
            step += 1

            # 按 step 退火
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

            # NEW: 每步都更新 for_gantt/for_devices，循环结束后自然是“最后一次”
            for_gantt = info.get("episodes_situation", [])
            for_devices = info.get("devices_situation", None)

        win_tag = terminated

        # 统计移动时间（每架飞机的平均移动分钟）
        epi_sit = last_info.get("episodes_situation", [])
        move_time = sum(job_trans[5]
                        for job_trans in epi_sit) 

        # last obs
        # 注意：此处按原代码逻辑，把末尾的 obs/state 再补一帧
        obs = self.env.get_obs()
        state = self.env.get_state()
        o.append(obs)
        s.append(state)

        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]

        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # padding 到 episode_limit
        if step < self.episode_limit:
            for _ in range(step, self.episode_limit):
                o.append(np.zeros((self.n_agents, self.obs_shape)))
                u.append(np.zeros([self.n_agents, 1]))
                s.append(np.zeros(self.state_shape))
                r.append([0.])
                o_next.append(np.zeros((self.n_agents, self.obs_shape)))
                s_next.append(np.zeros(self.state_shape))
                u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
                avail_u.append(np.zeros((self.n_agents, self.n_actions)))
                avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
                padded.append([1.])
                terminate.append([1.])

        episode = dict(
            o=o.copy(),
            s=s.copy(),
            u=u.copy(),
            r=r.copy(),
            avail_u=avail_u.copy(),
            o_next=o_next.copy(),
            s_next=s_next.copy(),
            avail_u_next=avail_u_next.copy(),
            u_onehot=u_onehot.copy(),
            padded=padded.copy(),
            terminated=terminate.copy()
        )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon

        # CHANGED: 增加返回 for_devices
        return episode, episode_reward, last_info.get("time", 0.0), win_tag, for_gantt, move_time, for_devices


# RolloutWorker for communication
class CommRolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init CommRolloutWorker')

    def generate_episode(self, episode_num=None, evaluate=False):
        """
        返回:
          episode, episode_reward, info["time"], win_tag, for_gantt, move_time, for_devices
        """
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset(self.n_agents)
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)

        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        last_info = {"episodes_situation": [],
                     "devices_situation": [], "time": 0.0}
        for_gantt = []
        for_devices = None

        while not terminated and step < self.episode_limit:
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []

            # 获取所有智能体动作权重（通信策略）
            weights = self.agents.get_action_weights(
                np.array(obs), last_action)

            # choose action for each agent
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                action = self.agents.choose_action(
                    weights[agent_id], avail_action, epsilon, evaluate)

                if action < len(self.env.sites):  # 选择即占位
                    self.env.has_chosen_action(action, agent_id)

                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            reward, terminated, info = self.env.step(actions)
            last_info = info  # NEW

            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1

            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

            # NEW: 每步更新，循环后即最后一次
            for_gantt = info.get("episodes_situation", [])
            for_devices = info.get("devices_situation", None)

        win_tag = terminated
        print("step:", step)

        epi_sit = last_info.get("episodes_situation", [])
        move_time = sum(job_trans[5]
                        for job_trans in epi_sit) 

        # last obs
        obs = self.env.get_obs()
        state = self.env.get_state()
        o.append(obs)
        s.append(state)

        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]

        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        if step < self.episode_limit:
            for _ in range(step, self.episode_limit):
                o.append(np.zeros((self.n_agents, self.obs_shape)))
                u.append(np.zeros([self.n_agents, 1]))
                s.append(np.zeros(self.state_shape))
                r.append([0.])
                o_next.append(np.zeros((self.n_agents, self.obs_shape)))
                s_next.append(np.zeros(self.state_shape))
                u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
                avail_u.append(np.zeros((self.n_agents, self.n_actions)))
                avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
                padded.append([1.])
                terminate.append([1.])

        episode = dict(
            o=o.copy(),
            s=s.copy(),
            u=u.copy(),
            r=r.copy(),
            avail_u=avail_u.copy(),
            o_next=o_next.copy(),
            s_next=s_next.copy(),
            avail_u_next=avail_u_next.copy(),
            u_onehot=u_onehot.copy(),
            padded=padded.copy(),
            terminated=terminate.copy()
        )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon

        # CHANGED: 增加返回 for_devices
        return episode, episode_reward, last_info.get("time", 0.0), win_tag, for_gantt, move_time, for_devices
