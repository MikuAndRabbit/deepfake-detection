import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.nn.modules.loss import _Loss
import numpy as np
from typing import Tuple, Union, List, Optional
from model.cross_vit import CrossEfficientViT
import yaml


def check_correct(preds, labels):
    preds = preds.cpu()
    labels = labels.cpu()
    # 使用激活函数计算模型预测结果
    preds = [np.asarray(torch.sigmoid(pred).detach().numpy()).round()
             for pred in preds]
    # 统计预测正确的样本数和正负预测个数
    correct = 0
    positive_class = 0
    negative_class = 0
    for i in range(len(labels)):
        pred = int(preds[i])
        if labels[i] == pred:
            correct += 1
        if pred == 1:
            positive_class += 1
        else:
            negative_class += 1
    return correct, positive_class, negative_class


def _load_model(config_path: str, checkpoint_path: Optional[str] = None) -> CrossEfficientViT:
    """加载模型

    Args:
        config_path (str): 配置文件地址
        checkpoint_path (Optional[str], optional): 参数文件. Defaults to None.

    Returns:
        CrossEfficientViT: 模型
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    model = CrossEfficientViT(config=config)
    if checkpoint_path is not None:
        para_dict = torch.load(checkpoint_path)
        model.load_state_dict(para_dict)
    return model


def load_model(path: str, model: nn.Module, optimizer: Optional[Optimizer] = None, loss_fn: Optional[_Loss] = None, lr_scheduler: Optional[_LRScheduler] = None) -> Tuple[int, Optional[Optimizer], Optional[_Loss], Optional[_LRScheduler]]:
    """从指定路径加载模型和优化器等状态。

    Args:
        path (str): 保存模型和优化器等状态的文件路径
        model (nn.Module): 要恢复状态的模型
        optimizer (Optional[Optimizer]): 要恢复状态的优化器，默认为 None
        loss_fn (Optional[_Loss]): 要恢复状态的损失函数，默认为 None
        lr_scheduler (Optional[_LRScheduler]): 要恢复状态的学习率调度器，默认为 None
    Returns:
        Tuple[int, Optional[Optimizer], Optional[_Loss], Optional[_LRScheduler]]: 包含 epoch、optimizer、loss_fn 和 lr_scheduler 等状态的元组
    """
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if loss_fn is not None and 'loss_fn' in checkpoint:
        loss_fn.load_state_dict(checkpoint['loss_fn'])
    if lr_scheduler is not None and 'lr_scheduler' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    return epoch, optimizer, loss_fn, lr_scheduler


def save_model(model: nn.Module, idx: int, path: str, optimizer: Optional[Optimizer], loss_fn: Optional[_Loss], lr_scheduler: Optional[_LRScheduler]):
    """保存模型

    Args:
        model (nn.Module): 模型
        idx (int): 训练的代数，从0开始
        path (str): 保存文件的位置
        optimizer (Optional[Optimizer]): 优化方法
        loss_fn (Optional[_Loss]): 损失函数
        lr_scheduler (Optional[_LRScheduler]): 学习率调整方法
    """
    checkpoint = {'model': model.state_dict(), 'epoch': idx,
                  'optimizer': optimizer.state_dict() if optimizer else None,
                  'loss_fn': loss_fn.state_dict() if loss_fn else None,
                  'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None}
    torch.save(checkpoint, path)
