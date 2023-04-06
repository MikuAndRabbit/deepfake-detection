import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import cv2
import numpy as np
from typing import List, Literal, Union, Optional
from torchvision.transforms import ToTensor

import uuid
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate, Resize


class IsotropicResize(Resize):
    def __init__(self, max_size: int, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR, always_apply=False, p=1):
        """图片等比例缩放

        Args:
            max_size (int): 变换后图片的最大大小
            interpolation_down (_type_, optional): 缩小图片时所用的插值方法. Defaults to cv2.INTER_AREA.
            interpolation_up (_type_, optional): 放大图片时所用的插值方法. Defaults to cv2.INTER_LINEAR.
            always_apply (bool, optional): _description_. Defaults to False.
            p (int, optional): 应用这一变换的概率. Defaults to 1.
        """
        # 继承父类的构造函数，并将width和height都设置为max_size，这样就保证了等比例缩放
        super(IsotropicResize, self).__init__(
            max_size, max_size, interpolation_up, always_apply, p)
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(self, img, **params):
        # 获取原图的高度和宽度
        h, w = img.shape[0], img.shape[1]
        # 判断哪一个是更小的，这样就可以保证缩放后的图像不会超出max_size的范围
        if h > w:
            size = self.height
            resize_h, resize_w = size, int(w * size / h)
        else:
            size = self.width
            resize_h, resize_w = int(h * size / w), size
        # 根据缩放比例选择不同的插值方法
        if resize_h < h or resize_w < w:
            interpolation = self.interpolation_down
        else:
            interpolation = self.interpolation_up
        # 使用cv2.resize函数将原图缩放到指定的大小，并返回缩放后的图像
        resized = cv2.resize(img, (resize_w, resize_h),
                             interpolation=interpolation)
        return resized

    def get_transform_init_args_names(self):
        # 返回__init__方法中的参数名，这样就可以在序列化和反序列化时使用这些参数
        return ('max_size', 'interpolation_down', 'interpolation_up')


def traindata_augmentation_transformer(img_size: int):
    """针对训练集数据的增强方法

    Args:
        img_size (int): 图片大小

    Returns:
        Compose: 返回一个转换器对象
    """
    return Compose([
        ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
        GaussNoise(p=0.3),
        HorizontalFlip(),
        OneOf([
            IsotropicResize(max_size=img_size, interpolation_down=cv2.INTER_AREA,
                            interpolation_up=cv2.INTER_CUBIC),
            IsotropicResize(max_size=img_size, interpolation_down=cv2.INTER_AREA,
                            interpolation_up=cv2.INTER_LINEAR),
            IsotropicResize(max_size=img_size, interpolation_down=cv2.INTER_LINEAR,
                            interpolation_up=cv2.INTER_LINEAR),
        ], p=1),
        PadIfNeeded(min_height=img_size, min_width=img_size,
                    border_mode=cv2.BORDER_CONSTANT),
        OneOf([RandomBrightnessContrast(), FancyPCA(),
              HueSaturationValue()], p=0.4),
        ToGray(p=0.2),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                         rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    ])


def valdata_augmentation_transformer(img_size: int):
    """针对验证集数据的增强方法

    Args:
        img_size (int): 图片大小

    Returns:
        Compose: 返回一个转换器对象
    """
    return Compose([
        IsotropicResize(max_size=img_size, interpolation_down=cv2.INTER_AREA,
                        interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=img_size, min_width=img_size,
                    border_mode=cv2.BORDER_CONSTANT),
    ])


class DeepFakesDataset(Dataset):
    def __init__(self, images_path: List[str], labels: Union[np.ndarray, List[int]], image_size: int, mode: str = 'train'):
        """需要增强的数据集

        Args:
            images_path (List[str]): 图片的路径
            labels (Union[np.ndarray, List[int]]): 图片的标签（是否经过深度伪造）
            image_size (int): 图片大小
            mode (str, optional): 当前的模式，训练集还是验证集. Defaults to 'train'.
        """
        self.x = np.array([cv2.cvtColor(cv2.imread(
            image_path), cv2.COLOR_BGR2RGB) for image_path in images_path])
        self.y = torch.tensor(labels, dtype=torch.int)
        self.image_size = image_size
        self.mode = mode
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        image = np.asarray(self.x[index])
        # 获取增强器
        if self.mode == 'train':
            transformer = traindata_augmentation_transformer(self.image_size)
        else:
            transformer = valdata_augmentation_transformer(self.image_size)
        # 获取增强后图片
        image = transformer(image=image)['image']
        return torch.tensor(image).float(), self.y[index]

    def __len__(self):
        return self.n_samples
