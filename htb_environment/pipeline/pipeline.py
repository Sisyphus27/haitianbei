# pipeline.py（新增/覆盖同名函数）
import os
import json
import logging
import numpy as np
from datetime import datetime

from environment import ScheduleEnv
from MARL.agent.agent import Agents
from MARL.common.rollout import RolloutWorker
from MARL.common.arguments import get_mixer_args
from MARL.common.replay_buffer import ReplayBuffer

from pipeline.kg_bridge import T1KGPriorAdapter, schedule_to_kg_triples  # NEW：使用任务一接口的适配器
from data_provider.data_loader import Dataset_KG  # NEW：直接导入任务一核心类
from exp.kg_service import KGServiceLocal  # 使用本地KG服务封装
from utils.schedule_converter import convert_schedule_with_fixed_logic


class NpEncoder(json.JSONEncoder):
    def default(self, o):
        try:
            import numpy as np
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.ndarray,)):
                return o.tolist()
        except Exception:
            pass
        return super().default(o)


def _load_event_texts(event_jsonl: str | None) -> list[str]:
    if not event_jsonl:
        return []
    logger = logging.getLogger(__name__)
    texts: list[str] = []
    try:
        with open(event_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                ln = line.strip()
                if not ln:
                    continue
                payload: str | None = None
                try:
                    obj = json.loads(ln)
                    if isinstance(obj, dict):
                        for key in ("text", "event", "input"):
                            val = obj.get(key)
                            if isinstance(val, str) and val.strip():
                                payload = val.strip()
                                break
                    elif isinstance(obj, str) and obj.strip():
                        payload = obj.strip()
                except json.JSONDecodeError:
                    payload = ln
                if payload:
                    texts.append(payload)
    except FileNotFoundError:
        logger.warning("[KG-Pipeline] 事件JSONL文件不存在: %s", event_jsonl)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[KG-Pipeline] 加载事件JSONL失败 (%s): %s", event_jsonl, exc)
    return texts


class KGAwareEnv:
    """包装原始环境，使其在观测与动作后与任务一KG保持同步。"""

    def __init__(
        self,
        env,
        kg_service: KGServiceLocal | None,
        event_texts: list[str],
        *,
        events_per_step: int = 1,
        loop_events: bool = False,
    ) -> None:
        self._env = env
        self._kg_service = kg_service
        self._event_texts = list(event_texts or [])
        self._events_per_step = max(1, int(events_per_step))
        self._loop_events = bool(loop_events)
        self._event_iter = iter(self._event_texts)
        self._last_event_len = 0
        self._log = logging.getLogger(__name__)

    def __getattr__(self, item):
        return getattr(self._env, item)

    def reset(self, n_agents):
        if self._loop_events:
            self._event_iter = iter(self._event_texts)
        self._last_event_len = 0
        return self._env.reset(n_agents)

    def get_obs(self):
        self._inject_events_into_kg()
        return self._env.get_obs()

    def step(self, actions):
        reward, terminated, info = self._env.step(actions)
        self._sync_env_events_to_kg(info)
        return reward, terminated, info

    # ---- internal helpers -------------------------------------------------
    def _next_event_text(self) -> str | None:
        if not self._event_texts:
            return None
        try:
            return next(self._event_iter)
        except StopIteration:
            if not self._loop_events:
                return None
            self._event_iter = iter(self._event_texts)
            try:
                return next(self._event_iter)
            except StopIteration:
                return None

    def _inject_events_into_kg(self) -> None:
        if not self._kg_service:
            return
        for _ in range(self._events_per_step):
            text = self._next_event_text()
            if not text:
                break
            try:
                self._kg_service.extract_and_update(text)
            except Exception as exc:  # noqa: BLE001
                self._log.warning("[KG-Pipeline] 事件写入KG失败: %s", exc)

    def _sync_env_events_to_kg(self, info) -> None:
        if not self._kg_service:
            return
        if not isinstance(info, dict):
            return
        episodes = info.get("episodes_situation")
        if not episodes:
            return
        new_events = episodes[self._last_event_len :]
        if not new_events:
            return
        try:
            triples = schedule_to_kg_triples(new_events, self._env)
            if triples:
                self._kg_service.kg.update_with_triples(triples)
        except Exception as exc:  # noqa: BLE001
            self._log.warning("[KG-Pipeline] 调度结果回写KG失败: %s", exc)
        self._last_event_len = len(episodes)


def _episodes_makespan(episodes_situation):
    if not episodes_situation:
        return float("inf")
    return max((t + pmin + mmin) for (t, _, _, _, pmin, mmin) in episodes_situation)

# LOG： 完整的 KG 闭环训练流程
def run_kg_epoch_pipeline(args, kg_service=None, event_jsonl: str | None = None):
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

    # (0) —— 补充算法默认超参（rnn_hidden_dim 等），避免上游未调用时缺参
    try:
        args = get_mixer_args(args)
    except Exception:
        pass

    # (1) —— 任务一：优先使用调用方传入的 KGServiceLocal；否则创建（连接 Neo4j）并封装
    if kg_service is None:
        kg_core = Dataset_KG(
            root_path=os.getcwd(),
            load_data=False,
            neo4j_uri=getattr(args, "neo4j_uri", os.environ.get("NEO4J_URI")),
            neo4j_user=getattr(args, "neo4j_user", os.environ.get("NEO4J_USER")),
            neo4j_password=getattr(args, "neo4j_password",
                                os.environ.get("NEO4J_PASSWORD")),
            neo4j_database=getattr(args, "neo4j_database",
                                os.environ.get("NEO4J_DATABASE", None)),
        )
        kg_service = KGServiceLocal(kg_core, ctx_ttl_sec=2)

    # (2) —— 环境 + 先验（用任务一图谱）
    env = ScheduleEnv(args)
    # 先验从底层 Dataset_KG 读取统计（T1KGPriorAdapter 依赖 neighbors 等底层接口）
    prior = T1KGPriorAdapter(
        kg_service.kg, ds=args.prior_dim_site, dp=args.prior_dim_plane)
    env.attach_prior(prior, args.prior_dim_site,
                    args.prior_dim_plane)  # 环境会在 get_obs 时融合先验

    event_texts = _load_event_texts(event_jsonl)
    if event_texts and kg_service is not None:
        events_per_step = getattr(args, "kg_events_per_step", 1)
        loop_events = getattr(args, "kg_events_loop", False)
        env = KGAwareEnv(
            env,
            kg_service,
            event_texts,
            events_per_step=max(1, int(events_per_step)
                                if events_per_step else 1),
            loop_events=bool(loop_events),
        )
        logging.getLogger(__name__).info(
            "[KG-Pipeline] 已加载事件文本 %d 条, events_per_step=%s, loop=%s",
            len(event_texts),
            getattr(args, "kg_events_per_step", 1),
            getattr(args, "kg_events_loop", False),
        )
    elif event_texts and kg_service is None:
        logging.getLogger(__name__).warning(
            "[KG-Pipeline] 存在事件文本但未提供KG服务, 将忽略事件驱动更新"
        )
    elif not event_texts:
        logging.getLogger(__name__).info("[KG-Pipeline] 未启用事件数据驱动KG同步")

    # 补齐 Agents 所需的环境维度信息
    try:
        env.reset(args.n_agents)
        info = env.get_env_info()
        args.n_actions = int(info["n_actions"])  # type: ignore[attr-defined]
        args.state_shape = int(info["state_shape"])  # type: ignore[attr-defined]
        args.obs_shape = int(info["obs_shape"])  # type: ignore[attr-defined]
        args.episode_limit = int(info["episode_limit"])  # type: ignore[attr-defined]
    except Exception:
        # 若失败，继续抛给后续模块以便定位
        pass

    agents = Agents(args)
    rollout = RolloutWorker(env, agents, args)
    buffer = ReplayBuffer(args)

    results = {
        "evaluate_reward": [], "average_reward": [],
        "evaluate_makespan": [], "average_makespan": [],
        "evaluate_move_time": [], "average_move_time": [],
        "win_rates": [], "train_reward": [], "train_makespan": [],
        "train_move_time": [], "loss": [], "schedule_results": [], "devices_results": [],
        "running_time": ""
    }

    start_time = datetime.now()
    train_steps = 0
    evaluate_times = 0
    # NOTE: 结合事件流与KG同步的海天杯训练循环
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

            # 选 makespan 最小的一组
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
            # 通过本地服务封装触达底层更新接口
            kg_service.kg.update_with_triples(triples)

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
