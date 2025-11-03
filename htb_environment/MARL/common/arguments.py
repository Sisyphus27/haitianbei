# arguments.py

import argparse


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
    _setdefault('n_epoch', 6)
    _setdefault('n_episodes', 5)
    _setdefault('train_steps', 2)
    _setdefault('evaluate_cycle', 5)
    _setdefault('batch_size', 32)
    _setdefault('buffer_size', 2000)
    _setdefault('save_cycle', 50)
    _setdefault('target_update_cycle', 200)

    # training misc
    _setdefault('grad_norm_clip', 10)
    return args
