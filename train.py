import os
import time
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.nn.modules.loss import _Loss
import pandas as pd
from typing import Optional, Final
from tqdm import tqdm
from loguru._logger import Logger
from loguru import logger
import sys

from util.train_val import check_correct, save_model
from util.video import crop_faces_videos, detect_faces_in_videos

def train(model: nn.Module, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader], optimizer: Optimizer, lr_scheduler: _LRScheduler, loss_fn: _Loss, device: str, s_epoch: int, e_epoch: int, lr: float, logger: Logger, checkpoint: bool = True, checkpoint_interval: int = 5, checkpoint_pos: str = './checkpoint/'):
    need_val_epoch = True if val_dataloader is not None else False
    if not need_val_epoch:
        logger.warning('If no validation set is provided, this training will not be validated!')
    epoch_train_losses = []
    epoch_val_losses = []
    
    # 检查保存点路径设置是否正确
    if checkpoint:
        # 获取绝对路径
        checkpoint_pos = os.path.abspath(checkpoint_pos)
        if not os.path.isdir(checkpoint_pos):
            logger.error('Checkoint position shoule be a folder path!')
            return
        if os.path.exists(checkpoint_pos):
            logger.warning('Checkpoint position exist! It maybe contain files!')
        else:
            os.mkdir(checkpoint_pos)
            logger.info('Creat checkpoint folder at {}'.format(checkpoint_pos))
            
    
    model = model.to(deivce=device) # type: ignore
    for epoch_idx in tqdm(range(s_epoch, e_epoch + 1)):
        logger.info('Start train of {} epoch'.format(str(epoch_idx)))
        # 训练
        model.train()
        epoch_train_correct = 0
        epoch_train_loss = 0.
        epoch_train_pos = 0
        epoch_train_neg = 0
        for data_batch_idx, (images, labels) in enumerate(tqdm(train_dataloader)):
            images = images.to(device)
            # 模型计算结果
            output = model(images).cpu()
            loss = loss_fn(output, labels)
            # 结果信息统计
            correct, pos, neg = check_correct(output, labels)
            epoch_train_correct += correct
            epoch_train_pos += pos
            epoch_train_neg += neg
            epoch_train_loss += loss.item()
            # 模型参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info('{} epoch training result:'.format(str(epoch_idx)))
        logger.info('loss: {}'.format(str(epoch_train_loss)))
        logger.info('correct: {}'.format(str(epoch_train_correct), str()))
        logger.info('positive: {}'.format(str(epoch_train_pos)))
        logger.info('negative: {}'.format(str(epoch_train_neg)))
        epoch_train_losses.append(epoch_train_loss)
        
        # 验证
        if need_val_epoch:
            logger.info('Start valuate of {} epoch'.format(str(epoch_idx)))
            model.eval()
            epoch_val_correct = 0
            epoch_val_pos = 0
            epoch_val_neg = 0
            epoch_val_loss = 0.
            for data_batch_idx, (images, labels) in enumerate(tqdm(val_dataloader)):
                images = images.to(device)
                # 模型计算结果
                output = model(images).cpu()
                loss = loss_fn(output, labels)
                # 结果信息统计
                correct, pos, neg = check_correct(output, labels)
                epoch_val_correct += correct
                epoch_val_pos += pos
                epoch_val_neg += neg
                epoch_val_loss += loss.item()
            logger.info('{} epoch valuate result:'.format(str(epoch_idx)))
            logger.info('loss: {}'.format(str(epoch_val_loss)))
            logger.info('correct: {}'.format(str(epoch_val_correct)))
            logger.info('positive: {}'.format(str(epoch_val_pos)))
            logger.info('negative: {}'.format(str(epoch_val_neg)))
            epoch_val_losses.append(epoch_val_loss)
        
        # 学习率调整
        lr_scheduler.step()
        
        # 模型保存
        if checkpoint and (epoch_idx + 1) % checkpoint_interval == 0:
            file_path = os.path.join(checkpoint_pos, str(epoch_idx) + '-' + str(int(time.time() * 1000)))
            save_model(model, epoch_idx, file_path, optimizer, loss_fn, lr_scheduler)


if __name__ == "__main__":
    # TODO 进行训练
    # 常量
    # 路径数据
    VIDEO_DATASET_RELPATH = './data/video'
    VIDEO_LABEL_RELPATH = './data/label/label.csv'
    FRAME_JSON_RELPATH = './data/frameinfo'
    FRAME_IMG_RELPATH = './data/frame'
    # 数据集控制
    HAVE_FACE_JSON = True
    HAVE_FACE_IMG = True
    # 日志路径配置
    TRAIN_LOG_POS = './log'
    
    # 配置日志信息
    # 配置日志记录器
    logger.add(os.path.join(TRAIN_LOG_POS, 'train_{time:x}.log'), rotation="100 MB", compression="zip", enqueue=True, format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{level}</level> {message}')
    # 添加一个处理器，将日志同时输出到控制台和文件
    logger.add(sink=sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{level}</level> {message}")
    
    # 处理视频数据集
    video_dataset_path = os.path.abspath(VIDEO_DATASET_RELPATH)
    frame_json_path = os.path.abspath(FRAME_JSON_RELPATH)
    frame_img_path = os.path.abspath(FRAME_IMG_RELPATH)
    if not HAVE_FACE_JSON:
        detect_faces_in_videos(video_dataset_path, load_worker=3, out_dir=frame_json_path)
    if not HAVE_FACE_IMG:
        crop_faces_videos(video_dataset_path, frame_json_path, frame_img_path)
    
    # 获取标签
    video_label_path = os.path.abspath(VIDEO_LABEL_RELPATH)
    labels_path = os.path.abspath(VIDEO_LABEL_RELPATH)    
    data = pd.read_csv('data.csv', header=None, names=['filename', 'label'])
    data_dict = dict(zip(data['filename'], data['label']))
    print(len(data_dict))
    
    # 构建DataLoader

    