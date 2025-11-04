# pipeline.py（新增/覆盖同名函数）
import os
import json
import numpy as np
from datetime import datetime

from environment import ScheduleEnv
from MARL.agent.agent import Agents
from MARL.common.rollout import RolloutWorker
from MARL.common.replay_buffer import ReplayBuffer

from kg_bridge import T1KGPriorAdapter, schedule_to_kg_triples  # NEW：使用任务一接口的适配器
from data_provider.data_loader import Dataset_KG  # NEW：直接导入任务一核心类
from utils.schedule_converter import convert_schedule_with_fixed_logic


class NpEncoder(json.JSONEncoder):
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


def _episodes_makespan(episodes_situation):
    if not episodes_situation:
        return float("inf")
    return max((t + pmin + mmin) for (t, _, _, _, pmin, mmin) in episodes_situation)


def run_kg_epoch_pipeline(args):
    """
    完整闭环：
      1) 创建并连接任务一的 KG 客户端 Dataset_KG
      2) 用 KG 构造先验适配器 T1KGPriorAdapter，并 attach 到环境
      3) 标准 MARL 训练：rollout -> buffer -> agent.train
      4) 选出本 epoch 最优/最后一组调度，转三元组 -> 调用任务一接口 update_with_triples 回写
      5) 下一 epoch 自动使用更新后的图谱统计（先验随图谱变化）
    """
    save_path = os.path.join(args.result_dir, args.alg,
                             f"{args.n_agents}_agents", args.result_name)
    os.makedirs(save_path, exist_ok=True)
    info_path = os.path.join(save_path, "info.json")
    plan_path = os.path.join(save_path, "plan.json")

    # (1) —— 任务一：初始化 Dataset_KG（连接 Neo4j）
    kg = Dataset_KG(
        root_path=os.getcwd(),
        load_data=False,
        neo4j_uri=getattr(args, "neo4j_uri", os.environ.get("NEO4J_URI")),
        neo4j_user=getattr(args, "neo4j_user", os.environ.get("NEO4J_USER")),
        neo4j_password=getattr(args, "neo4j_password",
                               os.environ.get("NEO4J_PASSWORD")),
        neo4j_database=getattr(args, "neo4j_database",
                               os.environ.get("NEO4J_DATABASE", None)),
    )

    # (2) —— 环境 + 先验（用任务一图谱）
    env = ScheduleEnv(args)
    prior = T1KGPriorAdapter(
        kg, ds=args.prior_dim_site, dp=args.prior_dim_plane)
    env.attach_prior(prior, args.prior_dim_site,
                     args.prior_dim_plane)  # 环境会在 get_obs 时融合先验

    agents = Agents(args)
    rollout = RolloutWorker(env, agents, args)
    buffer = ReplayBuffer(args)

    results = {
        "evaluate_reward": [], "average_reward": [],
        "evaluate_makespan": [], "average_makespan": [],
        "evaluate_move_time": [], "average_move_time": [],
        "win_rates": [], "train_reward": [], "train_makespan": [],
        "train_move_time": [], "loss": [], "schedule_results": [], "devices_results": []
    }

    start_time = datetime.now()
    train_steps = 0
    evaluate_times = 0

    for epoch in range(args.n_epoch):
        # === 采样若干 episodes ===
        episodes = []
        best_gantt, best_devices, best_ms = None, None, float("inf")
        for epi_idx in range(args.n_episodes):
            episode, train_reward, train_time, _, for_gantt, move_time, for_devices = rollout.generate_episode(
                epi_idx)
            results["train_reward"].append(train_reward)
            results["train_makespan"].append(train_time)
            results["train_move_time"].append(move_time)
            episodes.append(episode)

            # 选 makespan 最小的一组（你也可以换成“最后一组”或“回报最大一组”）
            ms = _episodes_makespan(for_gantt)
            if ms < best_ms:
                best_ms, best_gantt, best_devices = ms, list(
                    for_gantt), (list(for_devices) if for_devices else None)

        # === 拼 batch & 写回放池 ===
        batch = episodes[0]
        for e in episodes[1:]:
            for k in batch.keys():
                batch[k] = np.concatenate((batch[k], e[k]), axis=0)
        buffer.store_episode(batch)

        # === 训练（off-policy: QMIX 等）===
        for _ in range(args.train_steps):
            mini = buffer.sample(min(buffer.current_size, args.batch_size))
            loss = agents.train(mini, train_steps)
            results["loss"].append(loss)
            train_steps += 1

        # === 每个 epoch 的 KG 闭环：把“最好的一组调度”写回任务一图谱 ===
        if best_gantt:
            triples = schedule_to_kg_triples(best_gantt, env)
            kg.update_with_triples(triples)  # —— 任务一核心接口：不改任务一代码

        # === 可选评估 ===
        if args.evaluate_cycle and (epoch + 1) % args.evaluate_cycle == 0:
            evaluate_times += 1
            # 直接复用 Runner.evaluate() 的思路：生成一组评估 episode，记录 reward/makespan/move_time
            from MARL.runner import Runner  # 只为复用已有 evaluate 与 plan 导出逻辑
            tmp_runner = Runner(env, args)
            win_rate, reward, time, move_time = tmp_runner.evaluate()
            results["win_rates"].append(win_rate)
            results["average_reward"].append(reward)
            results["average_makespan"].append(time)
            results["average_move_time"].append(move_time)

        # 进度输出
        avg_r = np.mean(results["train_reward"][-args.n_episodes:])
        avg_t = np.mean(results["train_makespan"][-args.n_episodes:])
        print(
            f"\r[KG-Epoch] {epoch+1}/{args.n_epoch}  ave_reward={avg_r:.2f}  ave_makespan={avg_t:.2f}", end="")

    # === 结束：保存信息并导出标准 plan.json（你现有的转换器）===
    end_time = datetime.now()
    results["running_time"] = str(end_time - start_time)
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NpEncoder)

    try:
        move_jid = int(env.jobs_obj.code2id().get("ZY_M", 1))
    except Exception:
        move_jid = 1
    convert_schedule_with_fixed_logic(
        info_path, plan_path, args.n_agents, move_job_id=move_jid)

    print(f"\n[KG-Pipeline] Done. Saved: {save_path}")
