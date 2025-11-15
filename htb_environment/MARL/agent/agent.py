import numpy as np
import torch
from MARL.policy.qmix import QMIX
from torch.distributions import Categorical


# Agent no communication
class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        if args.alg != 'qmix':
            raise Exception("Only 'qmix' is supported in this build.")
        self.policy = QMIX(args)

        self.args = args
        print('Init Agents')

    # 有可能分布选出来不符合要求的动作,目前貌似用不到这个东西
    def random_choice_with_mask(self, avail_actions):
        """更健壮的掩码随机：保证有解、尽量不选“等待”动作"""
        avail_actions = np.asarray(avail_actions).astype(np.float32)
        idxs = np.where(avail_actions > 0.5)[0]
        if len(idxs) == 0:
            # 兜底：如果掩码意外全0，返回最后一个动作（也可以改成“等待”）
            return self.n_actions - 1
        wait = self.n_actions - 3          # 你项目里定义的“等待”索引（保持不变）
        if wait in idxs and len(idxs) > 1:  # 尽量避免等
            idxs = idxs[idxs != wait]
        return int(np.random.choice(idxs, 1, replace=False)[0])


    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, evaluate=False):
        inputs = obs.copy()
        agent_id = np.zeros(self.n_agents, dtype=np.float32)
        agent_id[agent_num] = 1.0

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))

        # hidden state
        hidden_state = self.policy.eval_hidden[:, agent_num, :]

        # to tensor
        inputs_t = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_t = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)

        if self.args.cuda:
            inputs_t = inputs_t.cuda()
            hidden_state = hidden_state.cuda()
            avail_t = avail_t.cuda()

        # 前向：返回当前 agent 的 Q 向量
        q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(
            inputs_t, hidden_state)
        # q_value: [1, n_actions] or [n_actions]，统一成 [n_actions]
        q_value = q_value.squeeze(0)

        # 掩码：不可行动作设为 -inf
        q_value = q_value.clone()  # 避免原张量就地修改带来梯度问题
        q_value[avail_t.squeeze(0) == 0.0] = -float("inf")

        # ε-greedy
        if (np.random.rand() < epsilon) and (not evaluate):
            action = self.random_choice_with_mask(avail_actions)
        else:
            action = int(torch.argmax(q_value).item())
        return action


    # 获取bach中最大的结束step数
    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        # 获取这些episode中最大的结束step数
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:  # 如果episodelimit的长度内没有terminal==1，导致max_episode_len == 0
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break   # 若已经终止则跳出这个episode检查下一个episode
            
            if terminated[episode_idx, transition_idx, 0] == 0:
                max_episode_len = self.args.episode_limit   # 若该episode没有在episode_limit步前完成调度，设max_episode_len为episode_limit
                break
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training

        # different episode has different length, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]    # 取所有episode前max_episode_len个step的信息
        loss = self.policy.learn(batch, max_episode_len, train_step, epsilon)

        if train_step > 0 and train_step % self.args.save_cycle == 0:
            print("\n开始保存模型", train_step, self.args.save_cycle)
            self.policy.save_model(train_step)
        
        return loss


# Agent for communication
# class CommAgents:
#     def __init__(self, args):
#         self.n_actions = args.n_actions
#         self.n_agents = args.n_agents
#         self.state_shape = args.state_shape
#         self.obs_shape = args.obs_shape
#         alg = args.alg
#         if alg.find('reinforce') > -1:
#             self.policy = Reinforce(args)
#         elif alg.find('coma') > -1:
#             self.policy = COMA(args)
#         elif alg.find('central_v') > -1:
#             self.policy = CentralV(args)
#         else:
#             raise Exception("No such algorithm")
#         self.args = args
#         print('Init CommAgents')

#     # 根据weights得到概率，然后再根据epsilon选动作
#     def choose_action(self, weights, avail_actions, epsilon, evaluate=False):
#         weights = weights.unsqueeze(0)
#         avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
#         action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # 可以选择的动作的个数
#         # 先将Actor网络的输出通过softmax转换成概率分布
#         prob = torch.nn.functional.softmax(weights, dim=-1)
#         # 在训练的时候给概率分布添加噪音
#         prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
#         prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

#         """
#         不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
#         会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
#         """
#         prob = prob / prob.sum()
#         if epsilon == 0 and evaluate:
#             # 测试时直接选最大的
#             action = torch.argmax(prob)
#         else:
#             # action = Categorical(prob.squeeze(0)).sample().long()
#             action = torch.multinomial(prob, num_samples=1)
#             while prob[0][action] == 0:
#                 action = torch.multinomial(prob, num_samples=1)
#         return action

#     def get_action_weights(self, obs, last_action):
#         obs = torch.tensor(obs, dtype=torch.float32)
#         last_action = torch.tensor(last_action, dtype=torch.float32)
#         inputs = list()
#         inputs.append(obs)
#         # 给obs添加上一个动作、agent编号
#         if self.args.last_action:
#             inputs.append(last_action)
#         if self.args.reuse_network:
#             inputs.append(torch.eye(self.args.n_agents))
#         inputs = torch.cat([x for x in inputs], dim=1)
#         if self.args.cuda:
#             inputs = inputs.cuda()
#             self.policy.eval_hidden = self.policy.eval_hidden.cuda()
#         weights, self.policy.eval_hidden = self.policy.eval_rnn(inputs, self.policy.eval_hidden)
#         weights = weights.reshape(self.args.n_agents, self.args.n_actions)
#         return weights.cpu()

#     def _get_max_episode_len(self, batch):
#         terminated = batch['terminated']
#         episode_num = terminated.shape[0]
#         max_episode_len = 0
#         # 获取这些episode中最大的结束step数
#         for episode_idx in range(episode_num):
#             for transition_idx in range(self.args.episode_limit):
#                 if terminated[episode_idx, transition_idx, 0] == 1:  # 如果episodelimit的长度内没有terminal==1，导致max_episode_len == 0
#                     if transition_idx + 1 >= max_episode_len:
#                         max_episode_len = transition_idx + 1
#                     break   # 若已经终止则跳出这个episode检查下一个episode
#             if terminated[episode_idx, transition_idx, 0] == 0:
#                 max_episode_len = self.args.episode_limit   # 若该episode没有在episode_limit步前完成调度，设max_episode_len为episode_limit
#                 break
#         return max_episode_len

#     def train(self, batch, train_step, epsilon=None):  # coma在训练时也需要epsilon计算动作的执行概率
#         # 每次学习时，各个episode的长度不一样，因此取其中最长的episode作为所有episode的长度
#         max_episode_len = self._get_max_episode_len(batch)
#         for key in batch.keys():
#             batch[key] = batch[key][:, :max_episode_len]
#         loss = self.policy.learn(batch, max_episode_len, train_step, epsilon)
#         if train_step > 0 and train_step % self.args.save_cycle == 0:
#             self.policy.save_model(train_step)
#         return loss
