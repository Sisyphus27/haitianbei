'''
Author: zy
Date: 2025-10-22 17:30:41
LastEditTime: 2025-10-22 17:34:04
LastEditors: zy
Description: 
FilePath: haitianbei/exp/exp_basic.py

'''
import os
import numpy as np
from typing import Any, cast

# 使 torch 变为可选依赖：若不可用则走 CPU/空模型逻辑
try:
    import torch as torch_mod
    TORCH_AVAILABLE = True
except Exception:
    torch_mod = None
    TORCH_AVAILABLE = False
# 为类型检查提供宽松的 Any
torch = cast(Any, torch_mod)


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        # 设备/模型在不依赖 torch 的情况下也能运行
        self.device = self._acquire_device()
        self.model, self.parameters1 = self._build_model()
        if TORCH_AVAILABLE and hasattr(torch, 'device'):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if hasattr(self.model, 'to'):
                self.model.to(self.device)
            # 若 parameters1 为空或元素无 to 方法，保持原样
            new_params = []
            for param in self.parameters1:
                try:
                    new_params.append(param.to(self.device))
                except Exception:
                    new_params.append(param)
            self.parameters1 = new_params

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if TORCH_AVAILABLE:
            if self.args.use_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(
                    self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
                device = torch.device('cuda:{}'.format(self.args.gpu))
                print('Use GPU: cuda:{}'.format(self.args.gpu))
            else:
                device = torch.device('cpu')
                print('Use CPU')
            return device
        else:
            print('Torch not available, fallback to CPU (no-ops)')
            return 'cpu'

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass