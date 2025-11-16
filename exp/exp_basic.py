"""
基础实验类：提供模型加载和设备管理的基础功能

功能：
1. 设备管理：自动检测并使用GPU或CPU
2. 模型构建：定义模型构建接口（由子类实现）
3. 参数初始化：管理实验参数和设备配置

在stream-judge模式中的作用：
- 为Exp_main提供基础框架，处理设备选择、模型初始化等通用功能
"""

import os
import numpy as np
from typing import Any, cast
import logging as _logging
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
                if not _logging.getLogger().handlers:
                    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
                _logging.info('Use GPU: cuda:{}'.format(self.args.gpu))
            else:
                device = torch.device('cpu')
                if not _logging.getLogger().handlers:
                    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
                _logging.info('Use CPU')
            return device
        else:
            if not _logging.getLogger().handlers:
                _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
            _logging.info('Torch not available, fallback to CPU (no-ops)')
            return 'cpu'

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass