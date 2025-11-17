# arguments.py

import argparse
import os


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
    parser.add_argument('--evaluate_epoch', type=int, default=1)
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
    parser.add_argument('--epsilon_anneal_scale',
                        type=str,
                        default=None,
                        choices=['step', 'episode', 'epoch'])

    # 环境设置
    parser.add_argument('--landing_sep_min', type=int, default=0)
    parser.add_argument('--arrival_gap_min', type=int, default=0)
    parser.add_argument('--plane_speed', type=int, default=5)
    # 批次测试场景（用于构造按批次到达的 arrival_plan）
    parser.add_argument('--batch_mode', action='store_true', default=False,
                        help='启用批次到达模式（用于评估指定批次/间隔的着陆场景）')
    parser.add_argument('--batch_start_time_min', type=int, default=7*60,
                        help='批次场景第一个航班的起始时间（分钟），例如 07:00 -> 420')
    parser.add_argument('--batch_size_per_batch', type=int, default=12,
                        help='每批飞机数（重命名以避免与训练参数 batch_size 冲突）')
    parser.add_argument('--batches_count', type=int, default=1,
                        help='批次数（总飞机数 = batch_size * batches_count；如评估用设置 n_agents=总飞机数）')
    parser.add_argument('--intra_gap_min', type=int, default=2,
                        help='同一批内相邻飞机的到达间隔（分钟）')
    parser.add_argument('--inter_batch_gap_min', type=int, default=60,
                        help='批次间隔（上一批最后一架落地到下一批首架落地的分钟数）')

    parser.add_argument('--episode_limit', type=int, default=None,
                        help='显式指定环境的 episode_limit，避免自动扩张导致的内存膨胀')

    parser.add_argument('--template_seed_dir', type=str, default='',
                        help='若提供路径，则在训练前加载该目录中的 template 计划，生成成功调度的经验并注入经验回放池')
    parser.add_argument('--template_seed_repeat', type=int, default=1,
                        help='模板经验注入次数（可用于多次重复注入同一成功调度以增强效果）')
    parser.add_argument('--epsilon_after_seed', type=float, default=None,
                        help='完成模板经验注入后，强制将 epsilon 设置为该值，降低后续探索率')
    parser.add_argument('--template_pretrain_steps', type=int, default=0,
                        help='模板注入完成后额外在经验回放池上预训练多少次，以便快速复现成功策略')

    # 扰动事件配置
    parser.add_argument('--enable_disturbance', action='store_true', default=False,
                        help='开启扰动事件调度逻辑')
    parser.add_argument('--disturbance_events', type=str, default='',
                        help='扰动事件配置，支持 JSON 字符串或 JSON 文件路径，格式见 readme.md')

    # 快照调度
    parser.add_argument('--snapshot_json', type=str, default='',
                        help='若提供 JSON 文件路径或 JSON 字符串，则跳过训练/评估，直接基于快照推理后续调度计划')
    parser.add_argument('--snapshot_use_agent', action='store_true', default=False,
                        help='在快照模式下使用 MARL agent 的决策流程（可与 --load_model 一起使用以加载训练模型）')
    parser.add_argument('--snapshot_mode', action='store_true', default=False,
                        help='在快照模式下使用 MARL agent 的决策流程（可与 --load_model 一起使用以加载训练模型）')
    args = parser.parse_args()
    return args


def get_mixer_args(args):
    def _setdefault(name, value):
        if getattr(args, name, None) is None:
            setattr(args, name, value)

    # network (QMIX)
    _setdefault('rnn_hidden_dim', 64)
    _setdefault('qmix_hidden_dim', 32)
    _setdefault('two_hyper_layers', False)
    _setdefault('hyper_hidden_dim', 64)
    _setdefault('lr', 5e-4)

    # epsilon
    if args.epsilon_start is not None or args.epsilon_end is not None or args.epsilon_anneal_steps is not None:
        eps = 1.0 if args.epsilon_start is None else args.epsilon_start
        mine = 0.05 if args.epsilon_end is None else args.epsilon_end
        steps = 50000 if args.epsilon_anneal_steps is None else args.epsilon_anneal_steps
        args.epsilon = eps
        args.min_epsilon = mine
        args.anneal_epsilon = (eps - mine) / float(max(1, steps))
        if getattr(args, 'epsilon_anneal_scale', None) is None:
            args.epsilon_anneal_scale = 'step'
    else:
        _setdefault('epsilon', 1.0)
        _setdefault('min_epsilon', 0.05)
        _setdefault('anneal_epsilon',
                    (args.epsilon - args.min_epsilon) / 50000.0)
        _setdefault('epsilon_anneal_scale', 'step')

    # loop
    _setdefault('n_epoch', 2)
    _setdefault('n_episodes', 1)
    _setdefault('train_steps', 1)
    _setdefault('evaluate_cycle', 1)
    _setdefault('batch_size', 16)
    _setdefault('buffer_size', 20)
    _setdefault('save_cycle', 50)
    _setdefault('target_update_cycle', 200)

    # training misc
    _setdefault('grad_norm_clip', 10)
    return args
