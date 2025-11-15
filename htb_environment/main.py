from datetime import datetime
import json
import copy
from snapshot_scheduler import (
    infer_schedule_from_snapshot,
    restore_env_from_snapshot,
    summarize_plane_completion,
)
from utils.knowledgeGraph_test import KGPrior
from utils.PDRs.shortestDistence import SDrules
from MARL.common.arguments import get_common_args, get_mixer_args
from MARL.runner import Runner
import sys
from environment import ScheduleEnv
import pickle
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# from os.path import dirname, abspath
# sys.path.append(dirname(dirname(abspath(__file__))))

np.random.seed(2)


def _load_snapshot_payload(source: str):
    spec = (source or "").strip()
    if not spec:
        return None
    if os.path.exists(spec):
        with open(spec, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(spec)


def _jsonify_events(events):
    out = []
    for evt in events or []:
        if isinstance(evt, tuple):
            out.append(list(evt))
        else:
            out.append(evt)
    return out


def _dump_snapshot_result(args, snapshot, info):
    result_dir = getattr(args, "result_dir", "result")
    alg = getattr(args, "alg", "snapshot")
    n_agents = getattr(args, "n_agents", len(snapshot.get("planes") or []))
    result_name = getattr(args, "result_name", "snapshot")
    out_dir = os.path.join(result_dir, alg,
                           f"{n_agents}_agents", result_name, "snapshot")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _scalar(value):
        try:
            import numpy as np
            if isinstance(value, (np.integer, np.floating)):
                return value.item()
        except Exception:
            pass
        return value

    payload = {
        "input_snapshot": snapshot,
        "result": {
            "reward": _scalar(info.get("reward")),
            "time": _scalar(info.get("time")),
            "win_rate": _scalar(info.get("win_rate")),
            "move_time": _scalar(info.get("move_time")),
            "episodes_situation": _jsonify_events(info.get("episodes_situation")),
            "devices_situation": info.get("devices_situation") or [],
            "completion": info.get("completion")
        }
    }
    if "disturbance" in info:
        payload["result"]["disturbance"] = info["disturbance"]

    out_path = os.path.join(out_dir, f"snapshot_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    return out_path


def _maybe_run_snapshot_mode(args) -> bool:
    spec = getattr(args, "snapshot_json", "")
    if not spec:
        return False
    try:
        snapshot = _load_snapshot_payload(spec)
    except Exception as exc:
        print(f"[Snapshot] Failed to parse {spec}: {exc}")
        sys.exit(1)
    if not isinstance(snapshot, dict):
        print("[Snapshot] snapshot_json must be a JSON object")
        sys.exit(1)
    if snapshot.get("planes"):
        args.n_agents = len(snapshot["planes"])

    # 创建环境并从快照恢复状态
    env = ScheduleEnv(args)
    env.reset(args.n_agents)
    restore_env_from_snapshot(env, snapshot)

    # 挂载先验（如果启用）
    if args.use_prior:
        prior = KGPrior(ds=args.prior_dim_site, dp=args.prior_dim_plane)
        env.attach_prior(prior, args.prior_dim_site, args.prior_dim_plane)

    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    args = get_mixer_args(args)

    # 使用 Runner 进行评估（生成甘特/plan_eval 等与批次流程一致）
    runner = Runner(env, args)
    win_rate, reward, makespan, move_time = runner.evaluate_from_snapshot()
    episodes = runner.results.get("schedule_results", [])
    devices = runner.results.get("devices_results", [])
    last_schedule = episodes[-1] if episodes else []
    last_devices = devices[-1] if devices else []
    completion = summarize_plane_completion(env)
    summary = {
        "reward": reward,
        "time": makespan,
        "win_rate": win_rate,
        "episodes_situation": last_schedule,
        "devices_situation": last_devices or [],
        "move_time": move_time,
        "completion": completion
    }
    if env.enable_disturbance:
        summary["disturbance"] = {
            "active_stands": sorted(env.disturbance_blocked_stands),
            "history": copy.deepcopy(env.disturbance_history)
        }

    out_path = _dump_snapshot_result(args, snapshot, summary)
    episodes_cnt = len(last_schedule or [])
    print('[Snapshot] Finished snapshot evaluation, win_rate: {}, reward: {}, makespan: {}, events: {}, saved: {}'.format(
        win_rate, reward, makespan, episodes_cnt, out_path))
    if completion.get("unfinished"):
        unfinished_ids = [entry["plane_id"]
                          for entry in completion["unfinished"]]
        print("[Snapshot] WARNING: unfinished planes detected: {}".format(
            unfinished_ids))
    return True


# 强化学习决策函数，带入来自DRL的强化学习agent
def marl_agent_wrapper():
    args = get_common_args()
    args.alg = 'qmix'

    if _maybe_run_snapshot_mode(args):
        return

    # 如果使用 batch_mode 便默认以评估模式运行（避免 argparse 的 bool 解析陷阱导致无法通过命令行禁用 learn/load_model）
    if getattr(args, 'batch_mode', False):
        args.learn = False
        args.load_model = False
    # 先构造带 args 的环境
    env = ScheduleEnv(args)

    # 如开启先验，挂到环境（注意缩进到 if 中）
    if args.use_prior:
        # 你导入的是 utils.knowledgeGraph_test.KGPrior，就用它；
        # 如果你用了我给的 utils.kg_prior.KGPrior，确保 import 对应即可。
        prior = KGPrior(ds=args.prior_dim_site, dp=args.prior_dim_plane)
        env.attach_prior(prior, args.prior_dim_site, args.prior_dim_plane)

    args = get_mixer_args(args)

    # 初始化/训练
    env.reset(args.n_agents)
    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]

    print("是否加载模型（测试必须）：", args.load_model, "是否训练：", args.learn)
    runner = Runner(env, args)
    if args.learn:
        runner.run(args.alg)
    else:
        start_time = datetime.now()
        win_rate, reward, time, move_time = runner.evaluate()
        end_time = datetime.now()
        running_time = end_time - start_time
        print('Evaluate win_rate: {}, reward: {}, makespan: {}, move_times: {}, running_time: {}'.format(
            win_rate, reward, time, move_time, running_time))


# 随机决策函数，用于测试环境
def random_agent_wrapper():

    results = {
        "makespan": [],
        "schedule_results": [],
        "win_tag": [],
        "win_rates": []
    }
    episodes = 100
    env = ScheduleEnv()
    n_agents = 16
    env.reset(n_agents)
    env_info = env.get_env_info()
    episode_limit = env_info["episode_limit"]
    win_num = 0
    start_time = datetime.now()
    for episode in range(episodes):

        env.reset(n_agents)
        is_terminal = False
        step = 0
        while not is_terminal and step < episode_limit:
            actions = []
            # 每次只调运空闲的agent
            for i in range(len(env.planes)):
                avail_actions = env.get_avail_agent_actions(i)
                tem_choose = []
                for k, eve in enumerate(avail_actions):
                    if eve == 1:
                        tem_choose.append(k)
                if len(env.sites)+1 in tem_choose or len(env.sites)+2 in tem_choose:  # i正忙
                    actions.append(len(env.sites))
                else:
                    if len(env.sites) in tem_choose:
                        tem_choose.remove(len(env.sites))
                    if tem_choose == []:
                        action = len(env.sites)
                    else:
                        action = np.random.choice(tem_choose, 1, False)[0]
                    if action < len(env.sites):
                        env.has_chosen_action(action, i)
                    actions.append(action)
            reward, is_terminal, info = env.step(actions)
            step += 1
        if is_terminal:
            win_num += 1
            results["schedule_results"].append(info["episodes_situation"])
        results["win_tag"].append(is_terminal)
        results["makespan"].append(info["time"])
        print("episode ", episode+1, " makespan: ", info["time"])
    win_rate = win_num/episodes
    average_makespan = sum(results["makespan"])/episodes
    end_time = datetime.now()
    running_time = end_time - start_time
    results["running_time"] = str(running_time)
    results["win_rates"].append(win_rate)
    print('Evaluate win_rate: {}, makespan: {}, running_time: {}, step: {}'.format(
        win_num/episodes, average_makespan, running_time, step))
    # 存储中间结果
    with open("result/random/test.json", 'w') as f:
        json.dump(results, f, indent=4)


def SDrules_agent_wrapper():
    n_agents = 8
    episodes = 20
    results = {
        "makespan": [],
        "schedule_results": [],
        "win_tag": [],
        "win_rates": []
    }
    sd_rules = SDrules()
    env = ScheduleEnv()
    env.reset(n_agents)
    env_info = env.get_env_info()
    episode_limit = env_info["episode_limit"]
    sites_locations = [
        env.sites[i].absolute_position for i in range(len(env.sites))]
    win_num = 0
    actions = []
    start_time = datetime.now()
    for episode in range(episodes):
        done = False
        env.reset(n_agents)
        speed = env.planes_obj.plane_speed
        step = 0
        while not done and step < episode_limit:
            actions = []
            agents_id_sequence = sd_rules.FIFO_generate_agents_sequence(
                len(env.planes))  # 智能体选择顺序
            # agents_id_sequence = sd_rules.MLF_generate_agents_sequence(env.planes)
            # agents_id_sequence = sd_rules.LLF_generate_agents_sequence(env.planes)

            for agent_id in agents_id_sequence:
                avail_actions = env.get_avail_agent_actions(agent_id)
                current_plane_location = env.planes[agent_id].position
                action = sd_rules.choose_action(
                    agent_id, avail_actions, current_plane_location, sites_locations, speed)
                actions.append(action)
                if action < len(env.sites):
                    env.has_chosen_action(action, agent_id)
            # 按照action的顺序进行重新整理，因为环境需要按照顺序接受actions
            reorder_actions = [-1 for i in range(len(env.planes))]
            for i in range(len(env.planes)):
                reorder_actions[agents_id_sequence[i]] = actions[i]

            _, done, info = env.step(reorder_actions)
            step += 1
        if done:
            win_num += 1
            results["schedule_results"].append(info["episodes_situation"])
        results["win_tag"].append(done)
        results["makespan"].append(info["time"])
        print("episode ", episode+1, " makespan: ", info["time"])
    win_rate = win_num/episodes
    average_makespan = sum(results["makespan"])/episodes
    end_time = datetime.now()
    running_time = end_time - start_time
    results["running_time"] = str(running_time)
    results["win_rates"].append(win_rate)
    print('Evaluate win_rate: {}, makespan: {}, running_time: {}'.format(
        win_num/episodes, average_makespan, running_time))
    # 存储中间结果
    save_path = "result/SDrules/" + str(n_agents) + "/info.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)


def Genetic_agent_wrapper():
    from GA import GeneticAlgorithm
    episodes = 20
    results = {
        "makespan": [],
        "average_makespan": [],
        "move_time": [],
        "average_move_time": [],
        "schedule_results": [],
        "win_tag": [],
        "win_rate": []
    }
    n_agents = 12
    env = ScheduleEnv()
    env.reset(n_agents)
    env_info = env.get_env_info()
    episode_limit = env_info["episode_limit"]
    win_num = 0
    start_time = datetime.now()
    for episode in range(episodes):
        done = False
        env.reset(n_agents)
        step = 0
        while not done and step < episode_limit:
            ga = GeneticAlgorithm(env, population_size=10,
                                  generations=5, mutation_rate=0.1)
            actions = ga.run()
            for i, action in enumerate(actions):
                if action < len(env.sites):
                    env.has_chosen_action(action, i)
            reward, done, info = env.step(actions)
            step += 1
        if done:
            win_num += 1
            results["schedule_results"].append(info["episodes_situation"])
            move_time = sum(job_trans[5]
                            for job_trans in info["episodes_situation"])
            move_time = move_time / n_agents
        results["win_tag"].append(done)
        results["makespan"].append(info["time"])
        results["move_time"].append(move_time)
        print("episode ", episode+1, " makespan: ", info["time"])

    win_rate = win_num/episodes
    average_makespan = sum(results["makespan"])/episodes
    results["average_makespan"].append(average_makespan)
    average_move_time = sum(results["move_time"])/episodes
    results["average_move_time"].append(average_move_time)
    results["win_rate"].append(win_num/episodes)
    end_time = datetime.now()
    running_time = end_time - start_time
    results["running_time"] = str(running_time)
    print('Evaluate win_rate: {}, makespan: {}, move_times: {}, running_time: {}'.format(
        win_rate, average_makespan, average_move_time, running_time))
    # 存储中间结果
    save_path = "result/GA/" + str(n_agents) + "/info.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)


def Diff_Evolution_agent_wrapper():
    from DE import DifferentialEvolution
    episodes = 300
    results = {
        "makespan": [],
        "average_makespan": [],
        "move_time": [],
        "average_move_time": [],
        "schedule_results": [],
        "win_tag": [],
        "win_rate": []
    }
    n_agents = 8
    env = ScheduleEnv()
    env.reset(n_agents)
    env_info = env.get_env_info()
    episode_limit = env_info["episode_limit"]
    win_num = 0
    start_time = datetime.now()
    for episode in range(episodes):
        done = False
        env.reset(n_agents)
        step = 0
        # episode_density = 0
        while not done and step < episode_limit:
            de = DifferentialEvolution(env, population_size=10, generations=5)
            actions = de.run()
            for i, action in enumerate(actions):
                if action < len(env.sites):
                    env.has_chosen_action(action, i)
            reward, done, info = env.step(actions)
            step += 1
        if done:
            win_num += 1
            results["schedule_results"].append(info["episodes_situation"])
            move_time = sum(job_trans[5]
                            for job_trans in info["episodes_situation"])
            move_time = move_time / n_agents
        results["win_tag"].append(done)
        results["makespan"].append(info["time"])
        results["move_time"].append(move_time)
        print("episode ", episode+1, " makespan: ", info["time"])

    win_rate = win_num/episodes
    average_makespan = sum(results["makespan"])/episodes
    results["average_makespan"].append(average_makespan)
    average_move_time = sum(results["move_time"])/episodes
    results["average_move_time"].append(average_move_time)
    results["win_rate"].append(win_num/episodes)
    end_time = datetime.now()
    running_time = end_time - start_time
    results["running_time"] = str(running_time)
    print('Evaluate win_rate: {}, makespan: {}, move_times: {}, running_time: {}'.format(
        win_rate, average_makespan, average_move_time, running_time))
    # 存储中间结果
    save_path = "result/DE/" + str(n_agents) + "/info.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    marl_agent_wrapper()
    # random_agent_wrapper()
    # SDrules_agent_wrapper()
    # Genetic_agent_wrapper()
    # Diff_Evolution_agent_wrapper()
