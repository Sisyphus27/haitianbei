# arguments.py

import argparse
import os
import logging


def get_common_args():
    parser = argparse.ArgumentParser()
    # 环境
    parser.add_argument('--map', type=str, default='boatschedule')
    parser.add_argument('--n_agents', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--alg', type=str, default='qmix')
    parser.add_argument('--last_action', type=bool, default=True)
    parser.add_argument('--reuse_network', type=bool, default=True)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--optimizer', type=str, default="RMS")
    parser.add_argument('--evaluate_epoch', type=int, default=20)
    parser.add_argument('--model_dir', type=str, default='./MARL/model')
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--result_name', type=str, default='test')
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--learn', type=bool, default=True)
    parser.add_argument('--cuda', type=bool, default=True)

    # 先验
    parser.add_argument('--replay_dir', type=str, default='')
    parser.add_argument('--use_prior', action='store_true', default=True)
    parser.add_argument('--prior_dim_site', type=int, default=8)
    parser.add_argument('--prior_dim_plane', type=int, default=3)
    parser.add_argument('--obs_pad', type=int, default=32)

    # 训练参数
    parser.add_argument('--n_epoch', type=int, default=None)
    parser.add_argument('--n_episodes', type=int, default=None)
    parser.add_argument('--train_steps', type=int, default=None)
    parser.add_argument('--evaluate_cycle', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--buffer_size', type=int, default=None)
    parser.add_argument('--save_cycle', type=int, default=None)
    parser.add_argument('--target_update_cycle', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)

    # 探索率
    parser.add_argument('--epsilon_start', type=float, default=None)
    parser.add_argument('--epsilon_end', type=float, default=None)
    parser.add_argument('--epsilon_anneal_steps', type=int, default=None)
    parser.add_argument('--epsilon_anneal_scale', type=str,
                        default=None, choices=['step', 'episode', 'epoch'])
    

    # neo4j 连接参数（支持从 NEO4J_AUTH 解析）
    parser.add_argument("--use_task1_kg", action="store_true",
                        help="使用任务一的Dataset_KG作为先验，并在每个epoch回写三元组")
    parser.add_argument("--prior_dim_site", type=int, default=8)
    parser.add_argument("--prior_dim_plane", type=int, default=3)
    parser.add_argument("--neo4j_uri", type=str,
                        default=os.environ.get("NEO4J_URI"))
    parser.add_argument("--neo4j_user", type=str,
                        default=os.environ.get("NEO4J_USER"))
    parser.add_argument("--neo4j_password", type=str,
                        default=os.environ.get("NEO4J_PASSWORD"))
    parser.add_argument("--neo4j_database", type=str,
                        default=os.environ.get("NEO4J_DATABASE"))


    args = parser.parse_args()
    return args


def get_mixer_args(args):
    """补齐 QMIX 训练所需的缺省参数。

    调试改进：统一 epsilon 逻辑，避免因为全部为 None 导致进入 else 分支时看起来像是"跳出"函数；
    同时添加 DEBUG 日志，单步跟踪时更易观察。始终在函数末尾返回 args，不存在早退。
    """
    log = logging.getLogger('marl.args')

    def _setdefault(name, value):
        if getattr(args, name, None) is None:
            setattr(args, name, value)

    # network (QMIX)
    _setdefault('rnn_hidden_dim', 64)
    _setdefault('qmix_hidden_dim', 32)
    _setdefault('two_hyper_layers', False)
    _setdefault('hyper_hidden_dim', 64)
    _setdefault('lr', 5e-4)

    # ===== epsilon （统一逻辑，不分支早退） =====
    any_provided = any(v is not None for v in (
        getattr(args, 'epsilon_start', None),
        getattr(args, 'epsilon_end', None),
        getattr(args, 'epsilon_anneal_steps', None),
    ))
    if any_provided:
        eps = 1.0 if getattr(args, 'epsilon_start', None) is None else args.epsilon_start
        mine = 0.05 if getattr(args, 'epsilon_end', None) is None else args.epsilon_end
        steps = 50000 if getattr(args, 'epsilon_anneal_steps', None) is None else args.epsilon_anneal_steps
    else:
        # 使用已有 (可能已经传入) 或缺省值
        eps = getattr(args, 'epsilon', None)
        if eps is None:
            eps = 1.0
        mine = getattr(args, 'min_epsilon', None)
        if mine is None:
            mine = 0.05
        steps = 50000  # 未显式提供时使用缺省 anneal 步数
    # 统一写回
    args.epsilon = eps
    args.min_epsilon = mine
    args.anneal_epsilon = (eps - mine) / float(max(1, steps))
    if getattr(args, 'epsilon_anneal_scale', None) is None:
        args.epsilon_anneal_scale = 'step'
    log.debug(
        f"[ARGS] epsilon_start={getattr(args,'epsilon_start',None)} epsilon_end={getattr(args,'epsilon_end',None)} steps={getattr(args,'epsilon_anneal_steps',None)} -> epsilon={args.epsilon:.4f} min={args.min_epsilon:.4f} anneal={args.anneal_epsilon:.8f} scale={args.epsilon_anneal_scale} provided={any_provided}"
    )

    # loop
    _setdefault('n_epoch', 5)
    _setdefault('n_episodes', 5)
    _setdefault('train_steps', 2)
    _setdefault('evaluate_cycle', 5)
    _setdefault('batch_size', 32)
    _setdefault('buffer_size', 1000)
    _setdefault('save_cycle', 50)
    _setdefault('target_update_cycle', 200)

    # training misc
    _setdefault('grad_norm_clip', 10)
    log.debug(
        f"[ARGS] loop defaults: n_epoch={args.n_epoch} n_episodes={args.n_episodes} train_steps={args.train_steps} batch_size={args.batch_size} buffer_size={args.buffer_size}"
    )
    return args
