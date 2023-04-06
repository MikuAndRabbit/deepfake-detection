import glob
import os
import numpy as np
import cv2
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from typing import List, Optional, Type
from albumentations import Compose, Resize

from util.video import FacenetDetector, VideoFaceDetector
from util.augmentation import traindata_augmentation_transformer, valdata_augmentation_transformer


class ImageDataset(Dataset):
    """图片数据集
    """    
    def __init__(self, imgs_path: List[str], img_size: int = 224, augmentation: bool = False, mode: Optional[str] = 'train') -> None:
        """构造函数

        Args:
            imgs_path (List[str]): 图片地址
            augmentation (bool, optional): 是否进行图片增强. Defaults to False.
            mode (Optional[str], optional): 如果进行增强，该数据集将用于哪种模式. Defaults to 'train'.
        """
        self.img_size = img_size
        self.paths = imgs_path
        self.augmentation = augmentation
        self.mode = mode
        
    def __getitem__(self, index: int):
        img_path = self.paths[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # resize image
        size = self.img_size
        if img.shape[:2] != (size, size):
            img = resize_image(img, size)
            
        # augmentation
        if self.augmentation:
            if self.mode == 'train':
                transformer = traindata_augmentation_transformer(size)
            else:
                transformer = valdata_augmentation_transformer(size)
            img = transformer(image = img)['image']    
        
        return img

    def __len__(self):
        return len(self.paths)
    

def resize_image(img: np.ndarray, image_size: int):
    """修改图片大小至宽高相等

    Args:
        img (Image.Image): 图片
        image_size (int): 修改后的大小

    Returns:
        ndarray: 修改后的图片
    """
    transform = Compose([
        Resize(height = image_size, width = image_size)
    ])
    return transform(image = img)['image']


def get_all_imgpath(folder_path: str) -> List[str]:
    """获取文件夹中所有的图片路径

    Args:
        folder_path (str): 文件夹路径

    Returns:
        List[str]: 图片路径集合
    """    
    imgs_path = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        imgs_path.extend(glob.glob(os.path.join(folder_path, ext)))
    return imgs_path
    
# TODO
def detect_faces_in_imgs(dataset_path: str, detector_cls: Type[VideoFaceDetector] = FacenetDetector, device: str = 'cpu', out_dir: Optional[str] = None, load_worker: int = 3):
    detector = detector_cls(device=device)
    dataset = ImageDataset(get_all_imgpath(dataset_path))
    loader = DataLoader(dataset, shuffle=False, num_workers=load_worker, batch_size=1)
    missed_img_paths = []
    
    # 每次处理一个图片
    for img in loader:
        pass