import os
import time
import torch.nn as nn
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.nn.modules.loss import _Loss
import pandas as pd
from typing import Optional, Tuple
from tqdm import tqdm
from loguru import logger
from loguru._logger import Logger
import sys
from model.cross_vit import CrossEfficientViT
from util.augmentation import DeepFakesDataset
from util.image import get_all_imgpath

from util.train_val import check_correct, save_model
from util.video import crop_faces_videos, detect_faces_in_videos

def train(model: nn.Module, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader], optimizer: Optimizer, 
          lr_scheduler: _LRScheduler, loss_fn: _Loss, device, s_epoch: int, e_epoch: int, lr: float, logger: 
              Logger, checkpoint: bool = True, checkpoint_interval: int = 5, checkpoint_pos: str = './checkpoint/') -> Tuple[Tuple, Tuple]: 
    need_val_epoch = True if val_dataloader is not None else False
    if not need_val_epoch:
        logger.warning('If no validation set is provided, this training will not be validated!')
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_train_correctes = []
    epoch_val_correctes = []
    epoch_train_correct_percentes = []
    epoch_val_correctes_percentes = []
    
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
            
    # 开始训练
    model = model.to('cuda:0') # type: ignore
    for epoch_idx in tqdm(range(s_epoch, e_epoch + 1)):
        logger.info('Start train of {} epoch'.format(str(epoch_idx)))
        # 训练
        model.train()
        epoch_train_correct = 0
        epoch_train_loss = 0.
        epoch_train_pos = 0
        epoch_train_neg = 0
        for data_batch_idx, (images, labels) in enumerate(tqdm(train_dataloader)):
            labels = labels.float()
            images = images.to(device)
            # 模型计算结果
            output = model(images).cpu()
            output = output.squeeze(1)
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
        epoch_train_correct_percent = epoch_train_correct / (epoch_train_pos + epoch_train_neg)
        logger.info('{} epoch training result:'.format(str(epoch_idx)))
        logger.info('train loss: {}'.format(str(epoch_train_loss)))
        logger.info('train correct: {} {}'.format(str(epoch_train_correct_percent), str(epoch_train_correct)))
        logger.info('train positive: {}'.format(str(epoch_train_pos)))
        logger.info('train negative: {}'.format(str(epoch_train_neg)))
        epoch_train_losses.append(epoch_train_loss)
        epoch_train_correctes.append(epoch_train_correct)
        epoch_train_correct_percentes.append(epoch_train_correct_percent)
        
        # 验证
        if need_val_epoch:
            logger.info('Start valuate of {} epoch'.format(str(epoch_idx)))
            model.eval()
            epoch_val_correct = 0
            epoch_val_pos = 0
            epoch_val_neg = 0
            epoch_val_loss = 0.
            for data_batch_idx, (images, labels) in enumerate(tqdm(val_dataloader)):
                labels = labels.float()
                images = images.to(device)
                # 模型计算结果
                output = model(images).cpu()
                output = output.squeeze(1)
                loss = loss_fn(output, labels)
                # 结果信息统计
                correct, pos, neg = check_correct(output, labels)
                epoch_val_correct += correct
                epoch_val_pos += pos
                epoch_val_neg += neg
                epoch_val_loss += loss.item()
            epoch_val_correctes_percent = epoch_val_correct / (epoch_val_pos + epoch_val_neg)
            logger.info('{} epoch valuate result:'.format(str(epoch_idx)))
            logger.info('valuate loss: {}'.format(str(epoch_val_loss)))
            logger.info('valuate correct: {} {}'.format(str(epoch_val_correctes_percent), str(epoch_val_correct)))
            logger.info('valuate positive: {}'.format(str(epoch_val_pos)))
            logger.info('valuate negative: {}'.format(str(epoch_val_neg)))
            epoch_val_losses.append(epoch_val_loss)
            epoch_val_correctes.append(epoch_val_correct)
            epoch_val_correctes_percentes.append(epoch_val_correctes_percent)
        
        # 学习率调整
        lr_scheduler.step()
        
        # 模型保存
        if checkpoint and (epoch_idx + 1) % checkpoint_interval == 0:
            file_path = os.path.join(checkpoint_pos, str(epoch_idx) + '-' + str(int(time.time() * 1000)))
            save_model(model, epoch_idx, file_path, optimizer, loss_fn, lr_scheduler)
            
    # 返回点东西
    return (epoch_train_losses, epoch_train_correctes, epoch_train_correct_percentes), \
        (epoch_val_losses, epoch_val_correctes, epoch_val_correctes_percentes)


if __name__ == "__main__":
    # TODO 进行训练
    # 配置文件读取
    CONFIG_PATH = './architecture.yaml'
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    
    # 常量
    # 路径数据
    VIDEO_DATASET_RELPATH = config['path']['video-dataset']
    VIDEO_LABEL_RELPATH = config['path']['video-label']
    FRAME_JSON_RELPATH = config['path']['frame-json']
    FRAME_IMG_RELPATH = config['path']['frame-img']
    TRAIN_CHECKPOINT = config['path']['checkpoint']
    TRAIN_LOG_POS = config['path']['log']
    # 数据集控制
    HAVE_FACE_JSON = config['control']['have-face-json']
    HAVE_FACE_IMG = config['control']['have-face-img']    
    # 训练验证集
    TRAIN_PERCENT = config['training']['train-data-percent']
    # DataLoader
    DATALOAD_BATCH_SIZE = config['training']['bs']
    DATALOAD_WORKER_NUM = config['training']['worker-num']
    # 测试控制
    TESTING = config['control']['testing']
    TEST_DATASET_BEGIN = config['control']['test-data-begin']
    # 训练变量
    LEARNING_RATE = config['training']['lr']
    WEIGHT_DECAY = config['training']['weight-decay']
    LRS_STEP_SIZE = config['training']['step-size']
    LRS_GAMMA = config['training']['gamma']
    CHECK_INTERVAL = config['training']['checkpoint-interval']
    NEED_CHECK = config['training']['checkpoint']
    START_EPOCH = config['training']['start-epoch']
    END_EPOCH = config['training']['end-epoch']
    # 设备
    DEVICE = config['device']
    
    
    # 配置日志信息
    # 配置日志记录器
    logger.add(os.path.join(TRAIN_LOG_POS, 'train_{time:x}.log'), rotation="100 MB", compression="zip", enqueue=True, format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{level}</level> {message}')
    # 添加一个处理器，将日志同时输出到控制台和文件
    # logger.add(sink=sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{level}</level> {message}")
    
    
    if TESTING:
        logger.warning('IN TEST MODE, PLEASE PAY ATTENTION TO THIS!')
    
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
    data = pd.read_csv(VIDEO_LABEL_RELPATH, header=None, names=['filename', 'label'])
    # data_dict['xxx.mp4'] = '1' / '0'
    data_dict = {}
    for _, row in data.iterrows():
        data_dict[row['filename']] = row['label']
    data_dict.pop('filename')
    
    # 获取所有图片路径并根据其所在视频确定其label
    img_paths = []
    img_labels = []
    frame_folders = os.listdir(frame_img_path)
    _pos = 0
    _neg = 0
    for folder_path in frame_folders:
        # 获取对应label
        id = folder_path
        folder_path = os.path.join(frame_img_path, folder_path)
        label = data_dict.get(id + '.mp4')
        if label is None:
            logger.warning('Can\'t find label of video {} in label.csv, please check!'.format(id))
            continue
        label = int(label)
        # 获取所有图片路径
        count = 0
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                img_paths.append(os.path.join(root, file))
                count += 1
            break
        img_labels.extend([label] * count)
    
    if TESTING:
        img_paths = img_paths[TEST_DATASET_BEGIN:]
        img_labels = img_labels[TEST_DATASET_BEGIN:]
    
    # 构建训练、验证数据集
    dataset = DeepFakesDataset(img_paths, img_labels, 224)
    _dataset_len = len(dataset)
    train_sample_num = int(_dataset_len * TRAIN_PERCENT)
    val_sample_num = _dataset_len - train_sample_num
    train_dataset, val_dataset = random_split(dataset, [train_sample_num, val_sample_num])
    
    # 构建DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=DATALOAD_BATCH_SIZE, shuffle=True, sampler=None, 
                                  batch_sampler=None, num_workers=DATALOAD_WORKER_NUM, pin_memory=False, drop_last=False, 
                                  timeout=0, prefetch_factor=2, persistent_workers=False)
    del train_dataset
    val_dataloader = DataLoader(val_dataset, batch_size=DATALOAD_BATCH_SIZE, shuffle=True, sampler=None, 
                                batch_sampler=None, num_workers=DATALOAD_WORKER_NUM, pin_memory=False, drop_last=False, 
                                timeout=0, prefetch_factor=2, persistent_workers=False)
    del val_dataset
    del dataset
    
    '''
    for img, lab in train_dataloader:
        # img.shape = torch.Size([16, 3, 224, 224])
        # lab.shape = torch.Size([16])
    '''
    
    # 计算正样本权重
    sum_cnt = 0
    pos_cnt = 0
    for img, labels in train_dataloader:
        sum_cnt += len(labels)
        pos_cnt += torch.sum(labels == 1).item()
    class_weights = pos_cnt / sum_cnt
    logger.info('Training dataset class weight: {}'.format(class_weights))
    
    # 相关准备    
    model = CrossEfficientViT(config=config)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LRS_STEP_SIZE, gamma=LRS_GAMMA)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]))
    device = torch.device(DEVICE)
    
    # 开始训练
    train_info, val_info = \
        train(model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, optimizer=optimizer, lr_scheduler=scheduler, 
          loss_fn=loss_function, device=device, s_epoch=START_EPOCH, e_epoch=END_EPOCH, lr=LEARNING_RATE, logger=logger, 
          checkpoint=NEED_CHECK, checkpoint_interval=CHECK_INTERVAL, checkpoint_pos=TRAIN_CHECKPOINT)
    
    # 分析结果
    train_losses, train_cor, train_cor_per = train_info
    logger.info('/**************** Train Result ****************/')
    logger.info(train_losses)
    logger.info(train_cor)
    logger.info(train_cor_per)
    logger.info('/**********************************************/')
    
    val_losses, val_cor, val_cor_per = val_info
    logger.info('/*************** Valuate Result ***************/')
    logger.info(val_losses)
    logger.info(val_cor)
    logger.info(val_cor_per)
    logger.info('/**********************************************/')
