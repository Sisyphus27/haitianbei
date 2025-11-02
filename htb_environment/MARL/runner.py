import numpy as np
import os
from MARL.common.rollout import RolloutWorker
from MARL.agent.agent import Agents
from MARL.common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import sys
import json
from datetime import datetime
import csv

# NEW: 导入转码器
from utils.schedule_converter import convert_schedule_with_fixed_logic


class NumpyEncoder(json.JSONEncoder):
    """把 numpy.* 类型安全转为 python 基元，便于 json.dump"""

    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
        except Exception:
            pass
        return super().default(obj)


class Runner:
    def __init__(self, env, args):
        self.env = env
        self.args = args

        self.agents = Agents(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)

        if args.learn:
            self.buffer = ReplayBuffer(args)

        self.win_rates = []
        self.episode_rewards = []
        self.results = {
            "evaluate_reward": [],
            "average_reward": [],
            "evaluate_makespan": [],
            "average_makespan": [],
            "evaluate_move_time": [],
            "average_move_time": [],
            "schedule_results": [],
            "devices_results": [],   
            "win_rates": [],
            "train_reward": [],
            "train_makespan": [],
            "train_move_time": [],
            "loss": []
        }

        self.save_path = self.args.result_dir + '/' + args.alg + \
            '/' + str(args.n_agents)+'_agents' + '/' + args.result_name
        os.makedirs(self.save_path, exist_ok=True)

    def run(self, alg):
        start_time = datetime.now()
        file_path = os.path.join(self.save_path, "info.json")
        plan_path = os.path.join(self.save_path, "plan.json")

        train_steps = 0
        evaluate_times = 1

        for epoch in range(self.args.n_epoch):

            if epoch % self.args.evaluate_cycle == 0 and epoch != 0:
                print('\nevaluate times:', evaluate_times, end=' ')
                win_rate, reward, time, move_time = self.evaluate()
                print('Evaluate win_rate: {}, reward: {}, makespan: {}, move_times: {}'.format(
                    win_rate, reward, time, move_time))
                self.win_rates.append(win_rate)
                self.episode_rewards.append(reward)
                evaluate_times += 1

            episodes = []
            r_s = []
            t_s = []

            for episode_idx in range(self.args.n_episodes):
                episode, train_reward, train_time, _, for_gant, train_move_time, for_devices = self.rolloutWorker.generate_episode(
                    episode_idx)
                self.results['train_reward'].append(train_reward)
                self.results['train_makespan'].append(train_time)
                self.results['train_move_time'].append(train_move_time)
                episodes.append(episode)
                r_s.append(sum(episode['r'][0])[0])
                t_s.append(train_time)

            # 拼 batch
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate(
                        (episode_batch[key], episode[key]), axis=0)

            # off-policy 训练（QMIX）
            self.buffer.store_episode(episode_batch)
            for _ in range(self.args.train_steps):
                mini_batch = self.buffer.sample(
                    min(self.buffer.current_size, self.args.batch_size))
                loss = self.agents.train(mini_batch, train_steps)
                self.results['loss'].append(loss)
                train_steps += 1

            # 训练日志
            avg_r = np.mean(
                self.results['train_reward'][-self.args.n_episodes:])
            avg_t = np.mean(
                self.results['train_makespan'][-self.args.n_episodes:])
            text = '\rRun {}, train epoch {}, ave_rewards {:.2f}, ave_makespan {:.2f}'
            sys.stdout.write(text.format(alg, epoch + 1, avg_r, avg_t))
            sys.stdout.flush()

        # 保存 info.json + plan.json
        end_time = datetime.now()
        self.results["running_time"] = str(end_time - start_time)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4, cls=NumpyEncoder)
        convert_schedule_with_fixed_logic(
            file_path, plan_path, self.args.n_agents)

    def export_schedule_csv(self, episodes_situation, devices=None, filename="schedule.csv"):
        os.makedirs(self.save_path, exist_ok=True)
        fpath = os.path.join(self.save_path, filename)
        with open(fpath, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time_min", "job_code", "job_id", "site_id", "plane_id",
                    "proc_min", "move_min", "FixedDevices", "MobileDevices"])
            for idx, row in enumerate(sorted(episodes_situation, key=lambda x: x[0])):
                t, jid, sid, pid, pmin, mmin = row
                code = self.env.jobs_obj.id2code()[jid]
                dev = devices[idx] if (devices and idx < len(devices)) else {
                    "FixedDevices": [], "MobileDevices": []}
                w.writerow([f"{t:.2f}", code, jid, sid, pid, f"{pmin:.2f}", f"{mmin:.2f}",
                            ";".join(map(str, dev.get("FixedDevices", []))), ";".join(map(str, dev.get("MobileDevices", [])))])

    def evaluate(self):
        file_path = os.path.join(self.save_path, "evaluate.json")
        plan_path = os.path.join(self.save_path, "plan_eval.json")

        win_number = 0
        reward = 0
        time = 0
        move_time = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, episode_time, win_tag, for_gant, episode_move_time, for_devices = self.rolloutWorker.generate_episode(
                epoch, evaluate=True)
            self.results['evaluate_reward'].append(episode_reward)
            self.results['evaluate_makespan'].append(episode_time)
            self.results['evaluate_move_time'].append(episode_move_time)
            self.results['schedule_results'].append(for_gant)
            self.results['devices_results'].append(for_devices)
            reward += episode_reward
            time += episode_time
            move_time += episode_move_time
            if win_tag:
                win_number += 1

        win_rate = win_number / self.args.evaluate_epoch
        reward = reward / self.args.evaluate_epoch
        time = time / self.args.evaluate_epoch
        move_time = move_time / self.args.evaluate_epoch
        self.results['average_reward'].append(reward)
        self.results['average_makespan'].append(time)
        self.results['average_move_time'].append(move_time)
        self.results['win_rates'].append(win_rate)

        # 保存 evaluate.json + plan_eval.json（若只加载模型进行评估）
        if self.args.load_model and not self.args.learn:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=4, cls=NumpyEncoder)
            convert_schedule_with_fixed_logic(
                file_path, plan_path, self.args.n_agents)

        # 可选导出 CSV（取最后一组）
        if getattr(self.args, "export_csv", True) and len(self.results["schedule_results"]) > 0:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.export_schedule_csv(self.results["schedule_results"][-1],
                                    self.results.get(
                                        "devices_results", [None])[-1],
                                    filename=f"schedule_{stamp}.csv")
        return win_rate, reward, time, move_time
